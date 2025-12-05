//! Optimized REML gradient computations with aggressive caching
//!
//! This module provides optimized versions of REML gradient/Hessian calculations
//! that cache expensive operations like QR decompositions.

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
#[cfg(feature = "blas")]
use ndarray_linalg::{QR, SolveTriangular, UPLO, Diag};
use crate::linalg::solve;

/// Cached QR factorization and related matrices for REML optimization
#[cfg(feature = "blas")]
pub struct REMLCache {
    /// Upper triangular R from QR decomposition of augmented matrix
    pub r_upper: Array2<f64>,
    /// Transpose of R (lower triangular)
    pub r_t: Array2<f64>,
    /// Beta coefficients
    pub beta: Array1<f64>,
    /// X'WX matrix
    pub xtwx: Array2<f64>,
    /// X'Wy vector
    pub xtwy: Array1<f64>,
    /// Residuals
    pub residuals: Array1<f64>,
    /// RSS (Residual Sum of Squares)
    pub rss: f64,
    /// Scale parameter phi
    pub phi: f64,
    /// Total rank (sum of penalty ranks)
    pub total_rank: usize,
    /// Penalty ranks
    pub penalty_ranks: Vec<usize>,
    /// Square root penalties
    pub sqrt_penalties: Vec<Array2<f64>>,
    /// Current lambda values
    pub lambdas: Vec<f64>,
}

#[cfg(feature = "blas")]
impl REMLCache {
    /// Create a new cache by computing QR factorization and related quantities
    pub fn new(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        lambdas: &[f64],
        penalties: &[Array2<f64>],
        sqrt_penalties: &[Array2<f64>],
    ) -> Result<Self> {
        let n = y.len();
        let p = x.ncols();
        let m = lambdas.len();

        // Compute X'WX and X'Wy
        let mut xtwx = Array2::<f64>::zeros((p, p));
        let mut xtwy = Array1::<f64>::zeros(p);

        // Optimized computation using column-wise operations
        for j in 0..p {
            for i in 0..n {
                let wi_sqrt = w[i].sqrt();
                let xi_j_weighted = x[[i, j]] * wi_sqrt;

                // Accumulate X'Wy
                xtwy[j] += xi_j_weighted * y[i] * wi_sqrt;

                // Accumulate X'WX (symmetric, only compute upper triangle)
                for k in j..p {
                    xtwx[[j, k]] += xi_j_weighted * x[[i, k]] * wi_sqrt;
                }
            }
        }

        // Fill in lower triangle by symmetry
        for j in 0..p {
            for k in 0..j {
                xtwx[[j, k]] = xtwx[[k, j]];
            }
        }

        // Get penalty ranks
        let penalty_ranks: Vec<usize> = sqrt_penalties.iter()
            .map(|sp| sp.ncols())
            .collect();
        let total_rank: usize = penalty_ranks.iter().sum();

        // Build augmented matrix for QR
        let mut total_rows = n;
        for sqrt_pen in sqrt_penalties.iter() {
            total_rows += sqrt_pen.ncols();
        }

        // Compute sqrt(W) * X
        let mut sqrt_w_x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let weight_sqrt = w[i].sqrt();
            for j in 0..p {
                sqrt_w_x[[i, j]] = x[[i, j]] * weight_sqrt;
            }
        }

        // Build Z matrix
        let mut z = Array2::<f64>::zeros((total_rows, p));

        // Fill in sqrt(W)X
        for i in 0..n {
            for j in 0..p {
                z[[i, j]] = sqrt_w_x[[i, j]];
            }
        }

        // Fill in scaled square root penalties
        let mut row_offset = n;
        for (sqrt_pen, &lambda) in sqrt_penalties.iter().zip(lambdas.iter()) {
            let sqrt_lambda = lambda.sqrt();
            let rank = sqrt_pen.ncols();
            for i in 0..rank {
                for j in 0..p {
                    z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];
                }
            }
            row_offset += rank;
        }

        // QR decomposition
        let (_, r) = z.qr()
            .map_err(|e| GAMError::InvalidParameter(format!("QR decomposition failed: {:?}", e)))?;

        // Extract upper triangular part
        let mut r_upper = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                r_upper[[i, j]] = r[[i, j]];
            }
        }

        let r_t = r_upper.t().to_owned();

        // Compute beta
        let mut a = xtwx.clone();
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            a.scaled_add(*lambda, penalty);
        }

        // Add ridge for stability
        let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
        for i in 0..p {
            a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
        }

        let beta = solve(a, xtwy.clone())?;

        // Compute residuals and RSS
        let fitted = x.dot(&beta);
        let mut rss = 0.0;
        let mut residuals = Array1::<f64>::zeros(n);
        for i in 0..n {
            residuals[i] = y[i] - fitted[i];
            rss += residuals[i] * residuals[i] * w[i];
        }

        let phi = rss / (n as f64 - total_rank as f64);

        Ok(Self {
            r_upper,
            r_t,
            beta,
            xtwx,
            xtwy,
            residuals,
            rss,
            phi,
            total_rank,
            penalty_ranks: penalty_ranks.clone(),
            sqrt_penalties: sqrt_penalties.to_vec(),
            lambdas: lambdas.to_vec(),
        })
    }

    /// Compute REML gradient using cached QR factorization
    pub fn gradient(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        lambdas: &[f64],
        penalties: &[Array2<f64>],
    ) -> Result<Array1<f64>> {
        let n = y.len();
        let p = x.ncols();
        let m = lambdas.len();

        let inv_phi = 1.0 / self.phi;
        let phi_sq = self.phi * self.phi;
        let n_minus_r = (n as f64) - (self.total_rank as f64);

        // Pre-compute P = RSS + Σλⱼ·β'·Sⱼ·β
        let mut penalty_sum = 0.0;
        for j in 0..m {
            let s_j_beta = penalties[j].dot(&self.beta);
            let beta_s_j_beta: f64 = self.beta.iter().zip(s_j_beta.iter())
                .map(|(bi, sbi)| bi * sbi)
                .sum();
            penalty_sum += lambdas[j] * beta_s_j_beta;
        }
        let p_value = self.rss + penalty_sum;

        // Compute gradient for each penalty
        let mut gradient = Array1::<f64>::zeros(m);

        for i in 0..m {
            let lambda_i = lambdas[i];
            let penalty_i = &penalties[i];
            let rank_i = self.penalty_ranks[i] as f64;
            let sqrt_penalty = &self.sqrt_penalties[i];

            // Term 1: tr(A^{-1}·λᵢ·Sᵢ) using cached R
            let rank = sqrt_penalty.ncols();

            let x_batch = self.r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
                .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

            let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
            let trace = lambda_i * trace_term;

            // Term 2: -rank(Sᵢ)
            let rank_term = -rank_i;

            // Compute ∂β/∂ρᵢ using cached R
            let s_i_beta = penalty_i.dot(&self.beta);
            let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

            let y = self.r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
                .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

            let dbeta_drho = self.r_upper.solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
                .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
                .mapv(|x| -x);

            // Compute ∂RSS/∂ρᵢ
            let x_dbeta = x.dot(&dbeta_drho);
            let drss_drho: f64 = -2.0 * self.residuals.iter().zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

            let dphi_drho = drss_drho / n_minus_r;

            // Compute ∂P/∂ρᵢ
            let beta_s_i_beta: f64 = self.beta.iter().zip(s_i_beta.iter())
                .map(|(bi, sbi)| bi * sbi)
                .sum();
            let explicit_pen = lambda_i * beta_s_i_beta;

            let mut implicit_pen = 0.0;
            for j in 0..m {
                let s_j_beta = penalties[j].dot(&self.beta);
                let s_j_dbeta = penalties[j].dot(&dbeta_drho);
                let term1: f64 = s_j_beta.iter().zip(dbeta_drho.iter())
                    .map(|(sj, dbi)| sj * dbi)
                    .sum();
                let term2: f64 = self.beta.iter().zip(s_j_dbeta.iter())
                    .map(|(bi, sjd)| bi * sjd)
                    .sum();
                implicit_pen += lambdas[j] * (term1 + term2);
            }

            let dp_drho = drss_drho + explicit_pen + implicit_pen;

            // Term 3: ∂(P/φ)/∂ρᵢ
            let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;

            // Term 4: ∂[(n-r)·log(2πφ)]/∂ρᵢ
            let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

            // Total gradient (divide by 2)
            gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
        }

        Ok(gradient)
    }

    /// Update cache for new lambda values (recomputes QR and related quantities)
    pub fn update(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        lambdas: &[f64],
        penalties: &[Array2<f64>],
    ) -> Result<()> {
        // Only rebuild if lambdas changed significantly
        let max_change = lambdas.iter().zip(&self.lambdas)
            .map(|(new, old)| (new / old).ln().abs())
            .fold(0.0f64, f64::max);

        // If change is small, don't rebuild (saves computation)
        if max_change < 0.01 {
            return Ok(());
        }

        *self = Self::new(y, x, w, lambdas, penalties, &self.sqrt_penalties)?;
        Ok(())
    }
}
