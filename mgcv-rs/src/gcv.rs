use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::{Inverse, Solve};
use crate::errors::{MgcvError, Result};

/// GCV (Generalized Cross-Validation) for smoothing parameter selection
pub struct GCV;

impl GCV {
    /// Compute GCV score for given smoothing parameter
    ///
    /// GCV = (n * RSS) / (n - tr(H))²
    ///
    /// where H is the hat matrix and RSS is the residual sum of squares
    pub fn score(
        X: &Array2<f64>,
        S: &Array2<f64>,
        y: &ArrayView1<f64>,
        lambda: f64,
        gamma: f64,  // Extra penalty on df (typically 1.0 to 1.4)
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
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve: {:?}", e)))?;

        // Fitted values and residuals
        let y_hat = X.dot(&beta);
        let residuals = y - &y_hat;
        let rss = residuals.dot(&residuals);

        // Effective degrees of freedom: tr(H) = tr(X (X^T X + λS)^{-1} X^T)
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to invert: {:?}", e)))?;

        let F = XtX.dot(&A_inv);  // F = X^T X (X^T X + λS)^{-1}
        let edf = F.diag().sum();

        // GCV score with gamma penalty
        let gcv = (n as f64 * rss) / (n as f64 - gamma * edf).powi(2);

        Ok(gcv)
    }

    /// Optimize smoothing parameter using golden section search
    pub fn optimize(
        X: &Array2<f64>,
        S: &Array2<f64>,
        y: &ArrayView1<f64>,
        lambda_min: f64,
        lambda_max: f64,
        gamma: f64,
        tol: f64,
    ) -> Result<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
        let resphi = 2.0 - phi;

        // Search in log space for numerical stability
        let mut a = lambda_min.ln();
        let mut b = lambda_max.ln();
        let mut c = a + resphi * (b - a);
        let mut d = b - resphi * (b - a);

        let mut fc = Self::score(X, S, y, c.exp(), gamma)?;
        let mut fd = Self::score(X, S, y, d.exp(), gamma)?;

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
                fc = Self::score(X, S, y, c.exp(), gamma)?;
            } else {
                a = c;
                c = d;
                fc = fd;
                d = b - resphi * (b - a);
                fd = Self::score(X, S, y, d.exp(), gamma)?;
            }
        }

        let lambda_opt = ((a + b) / 2.0).exp();
        Ok(lambda_opt)
    }

    /// Optimize multiple smoothing parameters
    pub fn optimize_multiple(
        X_mats: &[Array2<f64>],
        S_mats: &[Array2<f64>],
        y: &ArrayView1<f64>,
        initial_lambdas: &[f64],
        gamma: f64,
        tol: f64,
        max_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_smooth = X_mats.len();
        if n_smooth != S_mats.len() || n_smooth != initial_lambdas.len() {
            return Err(MgcvError::InvalidParameter(
                "Inconsistent number of smooth terms".to_string(),
            ));
        }

        // Use log scale
        let mut log_lambdas: Vec<f64> = initial_lambdas.iter().map(|&l| l.ln()).collect();

        for iter in 0..max_iter {
            // Compute GCV score
            let lambdas: Vec<f64> = log_lambdas.iter().map(|&l| l.exp()).collect();
            let current_score = Self::score_multiple(X_mats, S_mats, y, &lambdas, gamma)?;

            // Compute gradient using finite differences
            let eps = 1e-5;
            let mut gradient = vec![0.0; n_smooth];

            for i in 0..n_smooth {
                let mut lambdas_plus = lambdas.clone();
                lambdas_plus[i] *= (eps).exp();

                let score_plus = Self::score_multiple(X_mats, S_mats, y, &lambdas_plus, gamma)?;
                gradient[i] = (score_plus - current_score) / eps;
            }

            // Line search
            let mut step_size = 0.1;
            let mut improved = false;

            for _ in 0..10 {
                let mut new_log_lambdas = log_lambdas.clone();
                for i in 0..n_smooth {
                    new_log_lambdas[i] -= step_size * gradient[i];
                }

                let new_lambdas: Vec<f64> = new_log_lambdas.iter().map(|&l| l.exp()).collect();
                let new_score = Self::score_multiple(X_mats, S_mats, y, &new_lambdas, gamma)?;

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

    /// Compute GCV score for multiple smooth terms
    fn score_multiple(
        X_mats: &[Array2<f64>],
        S_mats: &[Array2<f64>],
        y: &ArrayView1<f64>,
        lambdas: &[f64],
        gamma: f64,
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

        // GCV score
        let gcv = (n as f64 * rss) / (n as f64 - gamma * edf).powi(2);

        Ok(gcv)
    }

    /// UBRE (Un-Biased Risk Estimator) - alternative to GCV
    ///
    /// UBRE = RSS/n + 2σ² tr(H)/n - σ²
    pub fn ubre_score(
        X: &Array2<f64>,
        S: &Array2<f64>,
        y: &ArrayView1<f64>,
        lambda: f64,
        sigma2: f64,  // Known or estimated variance
    ) -> Result<f64> {
        let n = X.nrows();

        // X^T X + λS
        let XtX = X.t().dot(X);
        let A = &XtX + &(S * lambda);

        // Solve for coefficients
        let Xty = X.t().dot(y);
        let beta = A.solve_into(Xty.clone())
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve: {:?}", e)))?;

        // Residuals
        let y_hat = X.dot(&beta);
        let residuals = y - &y_hat;
        let rss = residuals.dot(&residuals);

        // Effective degrees of freedom
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to invert: {:?}", e)))?;

        let F = XtX.dot(&A_inv);
        let edf = F.diag().sum();

        // UBRE score
        let ubre = rss / (n as f64) + 2.0 * sigma2 * edf / (n as f64) - sigma2;

        Ok(ubre)
    }
}
