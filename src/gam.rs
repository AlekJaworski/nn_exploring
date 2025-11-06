//! Main GAM model structure and fitting

use ndarray::{Array1, Array2};
use crate::{
    Result, GAMError,
    basis::{BasisFunction, CubicSpline, BoundaryCondition},
    penalty::compute_penalty,
    pirls::{fit_pirls, Family, PiRLSResult},
    smooth::{SmoothingParameter, OptimizationMethod},
};

/// A smooth term in a GAM
pub struct SmoothTerm {
    /// Name of the covariate
    pub name: String,
    /// Basis function
    pub basis: Box<dyn BasisFunction>,
    /// Penalty matrix
    pub penalty: Array2<f64>,
    /// Smoothing parameter
    pub lambda: f64,
}

impl SmoothTerm {
    /// Create a new smooth term with cubic spline basis (evenly-spaced knots)
    pub fn cubic_spline(
        name: String,
        num_basis: usize,
        x_min: f64,
        x_max: f64,
    ) -> Result<Self> {
        let basis = CubicSpline::with_num_knots(
            x_min,
            x_max,
            num_basis - 2,
            BoundaryCondition::Natural
        );

        let knots = basis.knots().unwrap();
        let penalty = compute_penalty("cubic", num_basis, Some(knots), 1)?;

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
        })
    }

    /// Create a new smooth term with quantile-based knots (like mgcv)
    pub fn cubic_spline_quantile(
        name: String,
        num_basis: usize,
        x_data: &Array1<f64>,
    ) -> Result<Self> {
        let basis = CubicSpline::with_quantile_knots(
            x_data,
            num_basis - 2,
            BoundaryCondition::Natural
        );

        let knots = basis.knots().unwrap();
        let penalty = compute_penalty("cubic", num_basis, Some(knots), 1)?;

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
        })
    }

    /// Evaluate the basis functions for this smooth term
    pub fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.basis.evaluate(x)
    }

    /// Get number of basis functions
    pub fn num_basis(&self) -> usize {
        self.basis.num_basis()
    }
}

/// Generalized Additive Model
pub struct GAM {
    /// Smooth terms
    pub smooth_terms: Vec<SmoothTerm>,
    /// Distribution family
    pub family: Family,
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Fitted values
    pub fitted_values: Option<Array1<f64>>,
    /// Linear predictor
    pub linear_predictor: Option<Array1<f64>>,
    /// Smoothing parameters
    pub smoothing_params: Option<SmoothingParameter>,
    /// IRLS weights
    pub weights: Option<Array1<f64>>,
    /// Deviance
    pub deviance: Option<f64>,
    /// Whether model has been fitted
    pub fitted: bool,
}

impl GAM {
    /// Create a new GAM with specified family
    pub fn new(family: Family) -> Self {
        Self {
            smooth_terms: Vec::new(),
            family,
            coefficients: None,
            fitted_values: None,
            linear_predictor: None,
            smoothing_params: None,
            weights: None,
            deviance: None,
            fitted: false,
        }
    }

    /// Add a smooth term to the model
    pub fn add_smooth(&mut self, smooth: SmoothTerm) {
        self.smooth_terms.push(smooth);
    }

    /// Fit the GAM using PiRLS with automatic smoothing parameter selection
    ///
    /// # Arguments
    /// * `x` - Covariate matrix (each column is one covariate)
    /// * `y` - Response vector
    /// * `opt_method` - Optimization method for smoothing parameters (REML or GCV)
    /// * `max_outer_iter` - Maximum iterations for smoothing parameter optimization
    /// * `max_inner_iter` - Maximum iterations for PiRLS
    /// * `tolerance` - Convergence tolerance
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        let n = y.len();

        if x.nrows() != n {
            return Err(GAMError::DimensionMismatch(
                format!("X has {} rows but y has {} elements", x.nrows(), n)
            ));
        }

        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(
                format!("X has {} columns but model has {} smooth terms",
                    x.ncols(), self.smooth_terms.len())
            ));
        }

        // Construct design matrix by evaluating all basis functions
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in self.smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
        }

        // Combine all design matrices
        let mut full_design = Array2::zeros((n, total_basis));
        let mut col_offset = 0;

        for design in &design_matrices {
            let num_cols = design.ncols();
            for i in 0..n {
                for j in 0..num_cols {
                    full_design[[i, col_offset + j]] = design[[i, j]];
                }
            }
            col_offset += num_cols;
        }

        // Construct penalty matrices
        let mut penalties: Vec<Array2<f64>> = Vec::new();
        col_offset = 0;

        for smooth in &self.smooth_terms {
            let num_basis = smooth.num_basis();
            let mut penalty_full = Array2::zeros((total_basis, total_basis));

            // Place this smooth's penalty in the appropriate block
            for i in 0..num_basis {
                for j in 0..num_basis {
                    penalty_full[[col_offset + i, col_offset + j]] = smooth.penalty[[i, j]];
                }
            }

            penalties.push(penalty_full);
            col_offset += num_basis;
        }

        // Initialize smoothing parameters
        let mut smoothing_params = SmoothingParameter::new(
            self.smooth_terms.len(),
            opt_method
        );

        // Outer loop: optimize smoothing parameters
        let mut weights = Array1::ones(n);

        for outer_iter in 0..max_outer_iter {
            // Inner loop: PiRLS with current smoothing parameters
            let pirls_result = fit_pirls(
                y,
                &full_design,
                &smoothing_params.lambda,
                &penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;

            weights = pirls_result.weights.clone();

            // Update smoothing parameters using REML/GCV
            let old_lambda = smoothing_params.lambda.clone();

            smoothing_params.optimize(
                y,
                &full_design,
                &weights,
                &penalties,
                10, // max iterations for lambda optimization
                tolerance,
            )?;

            // Check convergence of smoothing parameters
            let max_lambda_change = old_lambda.iter()
                .zip(smoothing_params.lambda.iter())
                .map(|(old, new)| ((old.ln() - new.ln()).abs()))
                .fold(0.0f64, f64::max);

            if max_lambda_change < tolerance {
                // Converged - do final fit
                let final_result = fit_pirls(
                    y,
                    &full_design,
                    &smoothing_params.lambda,
                    &penalties,
                    self.family,
                    max_inner_iter,
                    tolerance,
                )?;

                self.coefficients = Some(final_result.coefficients);
                self.fitted_values = Some(final_result.fitted_values);
                self.linear_predictor = Some(final_result.linear_predictor);
                self.weights = Some(final_result.weights);
                self.deviance = Some(final_result.deviance);
                self.smoothing_params = Some(smoothing_params);
                self.fitted = true;

                return Ok(());
            }
        }

        // Reached max outer iterations - use current fit
        let final_result = fit_pirls(
            y,
            &full_design,
            &smoothing_params.lambda,
            &penalties,
            self.family,
            max_inner_iter,
            tolerance,
        )?;

        self.coefficients = Some(final_result.coefficients);
        self.fitted_values = Some(final_result.fitted_values);
        self.linear_predictor = Some(final_result.linear_predictor);
        self.weights = Some(final_result.weights);
        self.deviance = Some(final_result.deviance);
        self.smoothing_params = Some(smoothing_params);
        self.fitted = true;

        Ok(())
    }

    /// Predict response for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(GAMError::InvalidParameter(
                "Model has not been fitted yet".to_string()
            ));
        }

        let coefficients = self.coefficients.as_ref().unwrap();
        let n = x.nrows();

        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(
                format!("X has {} columns but model has {} smooth terms",
                    x.ncols(), self.smooth_terms.len())
            ));
        }

        // Construct design matrix
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in self.smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
        }

        let mut full_design = Array2::zeros((n, total_basis));
        let mut col_offset = 0;

        for design in &design_matrices {
            let num_cols = design.ncols();
            for i in 0..n {
                for j in 0..num_cols {
                    full_design[[i, col_offset + j]] = design[[i, j]];
                }
            }
            col_offset += num_cols;
        }

        // Compute linear predictor
        let eta = full_design.dot(coefficients);

        // Apply inverse link
        let predictions: Array1<f64> = eta.iter()
            .map(|&e| self.family.inverse_link(e))
            .collect();

        Ok(predictions)
    }

    /// Get effective degrees of freedom
    pub fn edf(&self) -> Option<f64> {
        if !self.fitted {
            return None;
        }

        // Simplified: count non-zero coefficients
        // A proper implementation would compute tr(influence matrix)
        self.coefficients.as_ref().map(|coef| {
            coef.iter().filter(|&&c| c.abs() > 1e-10).count() as f64
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gam_creation() {
        let gam = GAM::new(Family::Gaussian);
        assert_eq!(gam.smooth_terms.len(), 0);
        assert!(!gam.fitted);
    }

    #[test]
    fn test_gam_add_smooth() {
        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline(
            "x1".to_string(),
            10,
            0.0,
            1.0
        ).unwrap();

        gam.add_smooth(smooth);
        assert_eq!(gam.smooth_terms.len(), 1);
    }

    #[test]
    fn test_gam_fit() {
        let n = 50;

        // Generate test data: y = sin(2*pi*x) + noise
        let x_data: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let y_data: Array1<f64> = x_data.iter()
            .enumerate()
            .map(|(i, &xi)| {
                (2.0 * std::f64::consts::PI * xi).sin() + 0.1 * (i as f64 % 3.0 - 1.0)
            })
            .collect();

        let x_matrix = x_data.clone().into_shape((n, 1)).unwrap();

        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline(
            "x".to_string(),
            15,
            0.0,
            1.0
        ).unwrap();

        gam.add_smooth(smooth);

        let result = gam.fit(
            &x_matrix,
            &y_data,
            OptimizationMethod::GCV,
            5,  // outer iterations
            50, // inner iterations
            1e-4
        );

        assert!(result.is_ok());
        assert!(gam.fitted);
        assert!(gam.coefficients.is_some());
        assert!(gam.deviance.is_some());
    }
}
