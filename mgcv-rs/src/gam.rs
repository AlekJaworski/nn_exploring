use ndarray::{Array1, Array2, ArrayView1};
use crate::basis::Basis;
use crate::family::Family;
use crate::reml::REML;
use crate::gcv::GCV;
use crate::errors::{MgcvError, Result};

/// Smoothing parameter selection method
#[derive(Debug, Clone, Copy)]
pub enum SmoothingMethod {
    REML,
    GCV { gamma: f64 },
    Manual,
}

/// A smooth term in the GAM
pub struct SmoothTerm {
    pub basis: Box<dyn Basis>,
    pub lambda: f64,
    pub name: String,
}

/// Generalized Additive Model
pub struct GAM {
    /// Smooth terms
    smooth_terms: Vec<SmoothTerm>,
    /// Family
    family: Box<dyn Family>,
    /// Smoothing method
    smoothing_method: SmoothingMethod,
    /// Coefficients
    coefficients: Option<Array1<f64>>,
    /// Fitted values (on response scale)
    fitted_values: Option<Array1<f64>>,
    /// Linear predictor (on link scale)
    linear_predictor: Option<Array1<f64>>,
    /// Effective degrees of freedom
    edf: Option<f64>,
    /// Deviance
    deviance: Option<f64>,
}

impl GAM {
    /// Create a new GAM
    pub fn new(family: Box<dyn Family>, smoothing_method: SmoothingMethod) -> Self {
        Self {
            smooth_terms: Vec::new(),
            family,
            smoothing_method,
            coefficients: None,
            fitted_values: None,
            linear_predictor: None,
            edf: None,
            deviance: None,
        }
    }

    /// Add a smooth term
    pub fn add_smooth(&mut self, name: String, basis: Box<dyn Basis>, lambda: f64) {
        self.smooth_terms.push(SmoothTerm {
            basis,
            lambda,
            name,
        });
    }

    /// Fit the GAM using PIRLS (Penalized Iteratively Reweighted Least Squares)
    pub fn fit(
        &mut self,
        X_data: &[ArrayView1<f64>],  // One array per smooth term
        y: &ArrayView1<f64>,
        max_iter: usize,
        tol: f64,
    ) -> Result<()> {
        let n = y.len();

        // Check dimensions
        if X_data.len() != self.smooth_terms.len() {
            return Err(MgcvError::DimensionMismatch {
                expected: self.smooth_terms.len(),
                actual: X_data.len(),
            });
        }

        for (i, x) in X_data.iter().enumerate() {
            if x.len() != n {
                return Err(MgcvError::DimensionMismatch {
                    expected: n,
                    actual: x.len(),
                });
            }
        }

        // Initialize mu
        let mut mu = self.family.initialize(y);
        let mut eta = self.family.link(&mu.view());

        // Construct basis matrices
        let X_mats: Result<Vec<Array2<f64>>> = X_data.iter()
            .zip(&self.smooth_terms)
            .map(|(x, term)| term.basis.basis_matrix(x))
            .collect();
        let X_mats = X_mats?;

        // Get penalty matrices
        let S_mats: Result<Vec<Array2<f64>>> = self.smooth_terms.iter()
            .map(|term| term.basis.penalty_matrix())
            .collect();
        let S_mats = S_mats?;

        // Optimize smoothing parameters if needed
        match self.smoothing_method {
            SmoothingMethod::REML => {
                let initial_lambdas: Vec<f64> = self.smooth_terms.iter()
                    .map(|t| t.lambda)
                    .collect();

                let optimal_lambdas = REML::optimize_multiple(
                    &X_mats,
                    &S_mats,
                    y,
                    &initial_lambdas,
                    1e-6,
                    50,
                )?;

                for (term, &lambda) in self.smooth_terms.iter_mut().zip(&optimal_lambdas) {
                    term.lambda = lambda;
                }
            }
            SmoothingMethod::GCV { gamma } => {
                let initial_lambdas: Vec<f64> = self.smooth_terms.iter()
                    .map(|t| t.lambda)
                    .collect();

                let optimal_lambdas = GCV::optimize_multiple(
                    &X_mats,
                    &S_mats,
                    y,
                    &initial_lambdas,
                    gamma,
                    1e-6,
                    50,
                )?;

                for (term, &lambda) in self.smooth_terms.iter_mut().zip(&optimal_lambdas) {
                    term.lambda = lambda;
                }
            }
            SmoothingMethod::Manual => {
                // Use provided lambdas
            }
        }

        // PIRLS iterations
        for iter in 0..max_iter {
            let eta_old = eta.clone();

            // Compute working weights and working response
            let mu_eta_vals = self.family.mu_eta(&eta.view());
            let var_vals = self.family.variance(&mu.view());

            let mut w = Array1::zeros(n);
            let mut z = Array1::zeros(n);

            for i in 0..n {
                w[i] = mu_eta_vals[i].powi(2) / var_vals[i].max(1e-10);
                z[i] = eta[i] + (y[i] - mu[i]) / mu_eta_vals[i].max(1e-10);
            }

            // Combine all basis matrices
            let m: usize = X_mats.iter().map(|X| X.ncols()).sum();
            let mut X_full = Array2::zeros((n, m));
            let mut col_offset = 0;
            for X in &X_mats {
                let cols = X.ncols();
                for i in 0..n {
                    for j in 0..cols {
                        X_full[[i, col_offset + j]] = X[[i, j]];
                    }
                }
                col_offset += cols;
            }

            // Combine penalty matrices
            let mut S_full = Array2::zeros((m, m));
            col_offset = 0;
            for (S, term) in S_mats.iter().zip(&self.smooth_terms) {
                let size = S.nrows();
                for i in 0..size {
                    for j in 0..size {
                        S_full[[col_offset + i, col_offset + j]] = S[[i, j]] * term.lambda;
                    }
                }
                col_offset += size;
            }

            // Weighted penalized least squares: (X^T W X + S) β = X^T W z
            let mut XtWX = Array2::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += X_full[[k, i]] * w[k] * X_full[[k, j]];
                    }
                    XtWX[[i, j]] = sum;
                }
            }

            let A = &XtWX + &S_full;

            let mut XtWz = Array1::zeros(m);
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += X_full[[j, i]] * w[j] * z[j];
                }
                XtWz[i] = sum;
            }

            use ndarray_linalg::Solve;
            let beta = A.solve_into(XtWz)
                .map_err(|e| MgcvError::LinAlgError(format!("PIRLS solve failed: {:?}", e)))?;

            // Update eta and mu
            eta = X_full.dot(&beta);
            mu = self.family.linkinv(&eta.view());

            // Check for convergence
            let max_diff = eta.iter()
                .zip(eta_old.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);

            if max_diff < tol {
                break;
            }
        }

        // Compute effective degrees of freedom
        use ndarray_linalg::Inverse;
        let m: usize = X_mats.iter().map(|X| X.ncols()).sum();
        let mut X_full = Array2::zeros((n, m));
        let mut col_offset = 0;
        for X in &X_mats {
            let cols = X.ncols();
            for i in 0..n {
                for j in 0..cols {
                    X_full[[i, col_offset + j]] = X[[i, j]];
                }
            }
            col_offset += cols;
        }

        let mut S_full = Array2::zeros((m, m));
        col_offset = 0;
        for (S, term) in S_mats.iter().zip(&self.smooth_terms) {
            let size = S.nrows();
            for i in 0..size {
                for j in 0..size {
                    S_full[[col_offset + i, col_offset + j]] = S[[i, j]] * term.lambda;
                }
            }
            col_offset += size;
        }

        let XtX = X_full.t().dot(&X_full);
        let A = &XtX + &S_full;
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to compute edf: {:?}", e)))?;

        let F = XtX.dot(&A_inv);
        let edf = F.diag().sum();

        // Compute deviance
        let wt = Array1::ones(n);
        let dev_resids = self.family.dev_resids(&y, &mu.view(), &wt.view());
        let deviance = dev_resids.sum();

        // Get final coefficients
        let m: usize = X_mats.iter().map(|X| X.ncols()).sum();
        let mut S_full = Array2::zeros((m, m));
        col_offset = 0;
        for (S, term) in S_mats.iter().zip(&self.smooth_terms) {
            let size = S.nrows();
            for i in 0..size {
                for j in 0..size {
                    S_full[[col_offset + i, col_offset + j]] = S[[i, j]] * term.lambda;
                }
            }
            col_offset += size;
        }

        let XtX = X_full.t().dot(&X_full);
        let A = &XtX + &S_full;
        let Xty = X_full.t().dot(y);

        use ndarray_linalg::Solve;
        let beta = A.solve_into(Xty)
            .map_err(|e| MgcvError::LinAlgError(format!("Final solve failed: {:?}", e)))?;

        self.coefficients = Some(beta);
        self.fitted_values = Some(mu);
        self.linear_predictor = Some(eta);
        self.edf = Some(edf);
        self.deviance = Some(deviance);

        Ok(())
    }

    /// Predict for new data
    pub fn predict(&self, X_data: &[ArrayView1<f64>]) -> Result<Array1<f64>> {
        let beta = self.coefficients.as_ref()
            .ok_or(MgcvError::InvalidParameter("Model not fitted".to_string()))?;

        if X_data.len() != self.smooth_terms.len() {
            return Err(MgcvError::DimensionMismatch {
                expected: self.smooth_terms.len(),
                actual: X_data.len(),
            });
        }

        let n = X_data[0].len();

        // Construct basis matrices
        let X_mats: Result<Vec<Array2<f64>>> = X_data.iter()
            .zip(&self.smooth_terms)
            .map(|(x, term)| term.basis.basis_matrix(x))
            .collect();
        let X_mats = X_mats?;

        // Combine matrices
        let m: usize = X_mats.iter().map(|X| X.ncols()).sum();
        let mut X_full = Array2::zeros((n, m));
        let mut col_offset = 0;
        for X in &X_mats {
            let cols = X.ncols();
            for i in 0..n {
                for j in 0..cols {
                    X_full[[i, col_offset + j]] = X[[i, j]];
                }
            }
            col_offset += cols;
        }

        // Compute linear predictor
        let eta = X_full.dot(beta);

        // Transform to response scale
        let mu = self.family.linkinv(&eta.view());

        Ok(mu)
    }

    /// Get model summary
    pub fn summary(&self) -> Option<GAMSummary> {
        Some(GAMSummary {
            edf: self.edf?,
            deviance: self.deviance?,
            n_smooth_terms: self.smooth_terms.len(),
            lambdas: self.smooth_terms.iter().map(|t| t.lambda).collect(),
        })
    }

    /// Get coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get fitted values
    pub fn fitted_values(&self) -> Option<&Array1<f64>> {
        self.fitted_values.as_ref()
    }
}

/// Summary of fitted GAM
#[derive(Debug)]
pub struct GAMSummary {
    pub edf: f64,
    pub deviance: f64,
    pub n_smooth_terms: usize,
    pub lambdas: Vec<f64>,
}
