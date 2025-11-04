use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Inverse, Solve};
use crate::basis::Basis;
use crate::errors::{MgcvError, Result};

/// Represents a smooth term in a GAM
pub struct Smooth {
    /// Basis function
    basis: Box<dyn Basis>,
    /// Smoothing parameter (lambda)
    lambda: f64,
    /// Coefficients
    coefficients: Option<Array1<f64>>,
    /// Fitted values
    fitted_values: Option<Array1<f64>>,
}

impl Smooth {
    /// Create a new smooth with given basis
    pub fn new(basis: Box<dyn Basis>) -> Self {
        Self {
            basis,
            lambda: 1.0,
            coefficients: None,
            fitted_values: None,
        }
    }

    /// Set the smoothing parameter
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Fit the smooth using penalized least squares
    ///
    /// Solves: (X^T W X + λS) β = X^T W y
    pub fn fit(
        &mut self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        weights: Option<&ArrayView1<f64>>,
    ) -> Result<()> {
        let n = x.len();
        if n != y.len() {
            return Err(MgcvError::DimensionMismatch {
                expected: n,
                actual: y.len(),
            });
        }

        // Get basis matrix
        let X = self.basis.basis_matrix(x)?;
        let m = X.ncols();

        // Get penalty matrix
        let S = self.basis.penalty_matrix()?;

        // Weight matrix
        let W = if let Some(w) = weights {
            if w.len() != n {
                return Err(MgcvError::DimensionMismatch {
                    expected: n,
                    actual: w.len(),
                });
            }
            w.to_owned()
        } else {
            Array1::ones(n)
        };

        // Construct weighted normal equations: (X^T W X + λS) β = X^T W y
        let mut XtWX = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += X[[k, i]] * W[k] * X[[k, j]];
                }
                XtWX[[i, j]] = sum;
            }
        }

        // Add penalty
        let A = &XtWX + &(&S * self.lambda);

        // Right hand side: X^T W y
        let mut XtWy = Array1::zeros(m);
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += X[[j, i]] * W[j] * y[j];
            }
            XtWy[i] = sum;
        }

        // Solve the system
        let beta = A.solve_into(XtWy)
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve system: {:?}", e)))?;

        // Compute fitted values
        let mut fitted = Array1::zeros(n);
        for i in 0..n {
            for j in 0..m {
                fitted[i] += X[[i, j]] * beta[j];
            }
        }

        self.coefficients = Some(beta);
        self.fitted_values = Some(fitted);

        Ok(())
    }

    /// Predict for new data
    pub fn predict(&self, x: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let beta = self.coefficients.as_ref()
            .ok_or(MgcvError::InvalidParameter("Model not fitted yet".to_string()))?;

        let X = self.basis.basis_matrix(x)?;
        let n = X.nrows();
        let m = X.ncols();

        let mut y_pred = Array1::zeros(n);
        for i in 0..n {
            for j in 0..m {
                y_pred[i] += X[[i, j]] * beta[j];
            }
        }

        Ok(y_pred)
    }

    /// Get the hat matrix (smoother matrix)
    pub fn hat_matrix(
        &self,
        x: &ArrayView1<f64>,
        weights: Option<&ArrayView1<f64>>,
    ) -> Result<Array2<f64>> {
        let X = self.basis.basis_matrix(x)?;
        let n = X.nrows();
        let m = X.ncols();
        let S = self.basis.penalty_matrix()?;

        let W = if let Some(w) = weights {
            w.to_owned()
        } else {
            Array1::ones(n)
        };

        // Construct X^T W X + λS
        let mut XtWX = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += X[[k, i]] * W[k] * X[[k, j]];
                }
                XtWX[[i, j]] = sum;
            }
        }

        let A = &XtWX + &(&S * self.lambda);

        // Hat matrix: H = X (X^T W X + λS)^{-1} X^T W
        let A_inv = A.inv()
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to invert matrix: {:?}", e)))?;

        let mut H = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..m {
                    for l in 0..m {
                        sum += X[[i, k]] * A_inv[[k, l]] * X[[j, l]] * W[j];
                    }
                }
                H[[i, j]] = sum;
            }
        }

        Ok(H)
    }

    /// Compute effective degrees of freedom: tr(H)
    pub fn edf(&self, x: &ArrayView1<f64>, weights: Option<&ArrayView1<f64>>) -> Result<f64> {
        let H = self.hat_matrix(x, weights)?;
        let trace = H.diag().sum();
        Ok(trace)
    }

    /// Get coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get fitted values
    pub fn fitted_values(&self) -> Option<&Array1<f64>> {
        self.fitted_values.as_ref()
    }

    /// Get current lambda
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Set lambda
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }
}

/// Penalized regression solver
pub struct PenalizedRegression;

impl PenalizedRegression {
    /// Fit penalized regression with multiple smooth terms
    pub fn fit_multiple(
        X_mats: &[Array2<f64>],
        S_mats: &[Array2<f64>],
        lambdas: &[f64],
        y: &ArrayView1<f64>,
        weights: Option<&ArrayView1<f64>>,
    ) -> Result<Array1<f64>> {
        let n = y.len();
        if X_mats.is_empty() {
            return Err(MgcvError::InvalidParameter("No basis matrices provided".to_string()));
        }

        // Total number of coefficients
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

        let W = if let Some(w) = weights {
            w.to_owned()
        } else {
            Array1::ones(n)
        };

        // Solve (X^T W X + S) β = X^T W y
        let mut XtWX = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += X_full[[k, i]] * W[k] * X_full[[k, j]];
                }
                XtWX[[i, j]] = sum;
            }
        }

        let A = &XtWX + &S_full;

        let mut XtWy = Array1::zeros(m);
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += X_full[[j, i]] * W[j] * y[j];
            }
            XtWy[i] = sum;
        }

        let beta = A.solve_into(XtWy)
            .map_err(|e| MgcvError::LinAlgError(format!("Failed to solve system: {:?}", e)))?;

        Ok(beta)
    }
}
