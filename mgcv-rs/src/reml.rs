use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::{Determinant, Inverse, Solve, Eigh, UPLO};
use crate::errors::{MgcvError, Result};

/// REML (Restricted Maximum Likelihood) for smoothing parameter selection
pub struct REML;

impl REML {
    /// Compute REML score for given smoothing parameters
    ///
    /// REML score: -0.5 * (log|X^T X + S| + log|X^T X| + y^T (X^T X + S)^{-1} y + (n - p) log σ²)
    ///
    /// where σ² is the residual variance
    pub fn score(
        X: &Array2<f64>,
        S: &Array2<f64>,
        y: &ArrayView1<f64>,
        lambda: f64,
    ) -> Result<f64> {
        let n = X.nrows();
        let m = X.ncols();

        if n != y.len() {
            return Err(MgcvError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }

        // X^T X
        let XtX = X.t().dot(X);

        // X^T X + λS
        let A = &XtX + &(S * lambda);

        // Solve for coefficients: β = (X^T X + λS)^{-1} X^T y
        let Xty = X.t().dot(y);
        let beta = A.solve_into(Xty.clone())
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve for beta: {:?}", e)))?;

        // Residuals
        let y_hat = X.dot(&beta);
        let residuals = y - &y_hat;
        let rss = residuals.dot(&residuals);

        // Effective degrees of freedom
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to invert A: {:?}", e)))?;

        let F = XtX.dot(&A_inv);
        let edf = F.diag().sum();

        // Residual variance estimate
        let sigma2 = rss / (n as f64 - edf);

        // Log determinants
        let det_A = A.det()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to compute det(A): {:?}", e)))?;
        let det_XtX = XtX.det()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to compute det(X^T X): {:?}", e)))?;

        if det_A <= 0.0 || det_XtX <= 0.0 {
            return Err(MgcvError::LinAlgError("Non-positive determinant".to_string()));
        }

        // REML score (negative for minimization)
        let reml = 0.5 * (
            det_A.ln() - det_XtX.ln()
            + (n as f64 - m as f64) * sigma2.ln()
            + rss / sigma2
        );

        Ok(reml)
    }

    /// Optimize smoothing parameter using golden section search
    pub fn optimize(
        X: &Array2<f64>,
        S: &Array2<f64>,
        y: &ArrayView1<f64>,
        lambda_min: f64,
        lambda_max: f64,
        tol: f64,
    ) -> Result<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
        let resphi = 2.0 - phi;

        let mut a = lambda_min.ln();
        let mut b = lambda_max.ln();
        let mut c = a + resphi * (b - a);
        let mut d = b - resphi * (b - a);

        let mut fc = Self::score(X, S, y, c.exp())?;
        let mut fd = Self::score(X, S, y, d.exp())?;

        let max_iter = 100;
        for _ in 0..max_iter {
            if (b - a).abs() < tol {
                break;
            }

            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = a + resphi * (b - a);
                fc = Self::score(X, S, y, c.exp())?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = b - resphi * (b - a);
                fd = Self::score(X, S, y, d.exp())?;
            }
        }

        let lambda_opt = ((a + b) / 2.0).exp();
        Ok(lambda_opt)
    }

    /// Optimize multiple smoothing parameters using gradient descent
    pub fn optimize_multiple(
        X_mats: &[Array2<f64>],
        S_mats: &[Array2<f64>],
        y: &ArrayView1<f64>,
        initial_lambdas: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_smooth = X_mats.len();
        if n_smooth != S_mats.len() || n_smooth != initial_lambdas.len() {
            return Err(MgcvError::InvalidParameter(
                "Inconsistent number of smooth terms".to_string(),
            ));
        }

        // Use log scale for numerical stability
        let mut log_lambdas: Vec<f64> = initial_lambdas.iter().map(|&l| l.ln()).collect();

        for iter in 0..max_iter {
            // Compute REML score for current lambdas
            let lambdas: Vec<f64> = log_lambdas.iter().map(|&l| l.exp()).collect();
            let current_score = Self::score_multiple(X_mats, S_mats, y, &lambdas)?;

            // Compute gradient using finite differences
            let eps = 1e-5;
            let mut gradient = vec![0.0; n_smooth];

            for i in 0..n_smooth {
                let mut lambdas_plus = lambdas.clone();
                lambdas_plus[i] *= (eps).exp();

                let score_plus = Self::score_multiple(X_mats, S_mats, y, &lambdas_plus)?;
                gradient[i] = (score_plus - current_score) / eps;
            }

            // Update with line search
            let mut step_size = 0.1;
            let mut improved = false;

            for _ in 0..10 {
                let mut new_log_lambdas = log_lambdas.clone();
                for i in 0..n_smooth {
                    new_log_lambdas[i] -= step_size * gradient[i];
                }

                let new_lambdas: Vec<f64> = new_log_lambdas.iter().map(|&l| l.exp()).collect();
                let new_score = Self::score_multiple(X_mats, S_mats, y, &new_lambdas)?;

                if new_score < current_score {
                    log_lambdas = new_log_lambdas;
                    improved = true;
                    break;
                }

                step_size *= 0.5;
            }

            if !improved || gradient.iter().map(|&g| g * g).sum::<f64>().sqrt() < tol {
                break;
            }
        }

        let lambdas: Vec<f64> = log_lambdas.iter().map(|&l| l.exp()).collect();
        Ok(lambdas)
    }

    /// Compute REML score for multiple smooth terms
    fn score_multiple(
        X_mats: &[Array2<f64>],
        S_mats: &[Array2<f64>],
        y: &ArrayView1<f64>,
        lambdas: &[f64],
    ) -> Result<f64> {
        let n = y.len();
        let m: usize = X_mats.iter().map(|X| X.ncols()).sum();

        // Combine all basis matrices
        let mut X_full = Array2::zeros((n, m));
        let mut col_offset = 0;
        for X in X_mats {
            let cols = X.ncols();
            for i in 0..n {
                for j in 0..cols {
                    X_full[[i, col_offset + j]] = X[[i, j]];
                }
            }
            col_offset += cols;
        }

        // Combine all penalty matrices
        let mut S_full = Array2::zeros((m, m));
        col_offset = 0;
        for (S, &lambda) in S_mats.iter().zip(lambdas) {
            let size = S.nrows();
            for i in 0..size {
                for j in 0..size {
                    S_full[[col_offset + i, col_offset + j]] = S[[i, j]] * lambda;
                }
            }
            col_offset += size;
        }

        // X^T X
        let XtX = X_full.t().dot(&X_full);

        // X^T X + S
        let A = &XtX + &S_full;

        // Solve for coefficients
        let Xty = X_full.t().dot(y);
        let beta = A.solve_into(Xty.clone())
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve: {:?}", e)))?;

        // Residuals
        let y_hat = X_full.dot(&beta);
        let residuals = y - &y_hat;
        let rss = residuals.dot(&residuals);

        // Effective degrees of freedom
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to invert: {:?}", e)))?;

        let F = XtX.dot(&A_inv);
        let edf = F.diag().sum();

        let sigma2 = rss / (n as f64 - edf);

        // Log determinants
        let det_A = A.det()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to compute det: {:?}", e)))?;
        let det_XtX = XtX.det()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to compute det: {:?}", e)))?;

        if det_A <= 0.0 || det_XtX <= 0.0 {
            return Err(MgcvError::LinAlgError("Non-positive determinant".to_string()));
        }

        let reml = 0.5 * (
            det_A.ln() - det_XtX.ln()
            + (n as f64 - m as f64) * sigma2.ln()
            + rss / sigma2
        );

        Ok(reml)
    }
}
