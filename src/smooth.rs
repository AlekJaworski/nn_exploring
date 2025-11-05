//! Smoothing parameter selection using REML optimization

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::reml::{reml_criterion, gcv_criterion};

/// Smoothing parameter optimization method
#[derive(Debug, Clone, Copy)]
pub enum OptimizationMethod {
    REML,
    GCV,
}

/// Container for smoothing parameters
#[derive(Debug, Clone)]
pub struct SmoothingParameter {
    pub lambda: Vec<f64>,
    pub method: OptimizationMethod,
}

impl SmoothingParameter {
    /// Create new smoothing parameters with initial values
    pub fn new(num_smooths: usize, method: OptimizationMethod) -> Self {
        Self {
            lambda: vec![1.0; num_smooths],
            method,
        }
    }

    /// Optimize smoothing parameters using REML or GCV
    pub fn optimize(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        if penalties.len() != self.lambda.len() {
            return Err(GAMError::DimensionMismatch(
                "Number of penalties must match number of lambdas".to_string()
            ));
        }

        match self.method {
            OptimizationMethod::REML => {
                self.optimize_reml(y, x, w, penalties, max_iter, tolerance)
            },
            OptimizationMethod::GCV => {
                self.optimize_gcv(y, x, w, penalties, max_iter, tolerance)
            },
        }
    }

    /// Optimize using REML criterion
    fn optimize_reml(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        // Simple grid search followed by Newton-Raphson refinement
        // For production, would use more sophisticated optimization

        // Work in log space for numerical stability
        let mut log_lambda: Vec<f64> = self.lambda.iter()
            .map(|l| l.ln())
            .collect();

        for iter in 0..max_iter {
            let mut converged = true;

            // Optimize each lambda separately (coordinate descent)
            for i in 0..log_lambda.len() {
                let old_log_lambda = log_lambda[i];

                // Compute current lambdas
                let current_lambda: Vec<f64> = log_lambda.iter()
                    .map(|l| l.exp())
                    .collect();

                // Compute total penalty
                let mut total_penalty = Array2::zeros(penalties[0].dim());
                for (j, penalty) in penalties.iter().enumerate() {
                    total_penalty = total_penalty + &(penalty * current_lambda[j]);
                }

                // Evaluate REML at current point
                let reml_current = reml_criterion(
                    y, x, w,
                    1.0, // lambda is incorporated in total_penalty
                    &total_penalty,
                    None
                )?;

                // Numerical gradient using finite differences
                let delta = 0.01;
                log_lambda[i] += delta;

                let lambda_plus: Vec<f64> = log_lambda.iter()
                    .map(|l| l.exp())
                    .collect();

                let mut total_penalty_plus = Array2::zeros(penalties[0].dim());
                for (j, penalty) in penalties.iter().enumerate() {
                    total_penalty_plus = total_penalty_plus + &(penalty * lambda_plus[j]);
                }

                let reml_plus = reml_criterion(
                    y, x, w,
                    1.0,
                    &total_penalty_plus,
                    None
                )?;

                let gradient = (reml_plus - reml_current) / delta;

                // Simple gradient descent step
                let step_size = 0.1;
                let new_log_lambda = old_log_lambda - step_size * gradient;

                log_lambda[i] = new_log_lambda;

                // Check convergence for this parameter
                if (new_log_lambda - old_log_lambda).abs() > tolerance {
                    converged = false;
                }
            }

            if converged {
                break;
            }
        }

        // Update lambda values
        self.lambda = log_lambda.iter()
            .map(|l| l.exp())
            .collect();

        Ok(())
    }

    /// Optimize using GCV criterion
    fn optimize_gcv(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        // Similar to REML but using GCV criterion
        let mut log_lambda: Vec<f64> = self.lambda.iter()
            .map(|l| l.ln())
            .collect();

        for iter in 0..max_iter {
            let mut converged = true;

            for i in 0..log_lambda.len() {
                let old_log_lambda = log_lambda[i];

                let current_lambda: Vec<f64> = log_lambda.iter()
                    .map(|l| l.exp())
                    .collect();

                let mut total_penalty = Array2::zeros(penalties[0].dim());
                for (j, penalty) in penalties.iter().enumerate() {
                    total_penalty = total_penalty + &(penalty * current_lambda[j]);
                }

                let gcv_current = gcv_criterion(
                    y, x, w,
                    1.0,
                    &total_penalty,
                )?;

                // Numerical gradient
                let delta = 0.01;
                log_lambda[i] += delta;

                let lambda_plus: Vec<f64> = log_lambda.iter()
                    .map(|l| l.exp())
                    .collect();

                let mut total_penalty_plus = Array2::zeros(penalties[0].dim());
                for (j, penalty) in penalties.iter().enumerate() {
                    total_penalty_plus = total_penalty_plus + &(penalty * lambda_plus[j]);
                }

                let gcv_plus = gcv_criterion(
                    y, x, w,
                    1.0,
                    &total_penalty_plus,
                )?;

                let gradient = (gcv_plus - gcv_current) / delta;

                let step_size = 0.1;
                let new_log_lambda = old_log_lambda - step_size * gradient;

                log_lambda[i] = new_log_lambda;

                if (new_log_lambda - old_log_lambda).abs() > tolerance {
                    converged = false;
                }
            }

            if converged {
                break;
            }
        }

        self.lambda = log_lambda.iter()
            .map(|l| l.exp())
            .collect();

        Ok(())
    }

    /// Grid search over lambda values to find good starting point
    pub fn grid_search(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda_min: f64,
        lambda_max: f64,
        num_points: usize,
        method: OptimizationMethod,
    ) -> Result<f64> {
        let log_lambda_min = lambda_min.ln();
        let log_lambda_max = lambda_max.ln();
        let step = (log_lambda_max - log_lambda_min) / (num_points - 1) as f64;

        let mut best_lambda = lambda_min;
        let mut best_score = f64::INFINITY;

        for i in 0..num_points {
            let log_lambda = log_lambda_min + step * i as f64;
            let lambda = log_lambda.exp();

            let score = match method {
                OptimizationMethod::REML => {
                    reml_criterion(y, x, w, lambda, penalty, None)?
                },
                OptimizationMethod::GCV => {
                    gcv_criterion(y, x, w, lambda, penalty)?
                },
            };

            if score < best_score {
                best_score = score;
                best_lambda = lambda;
            }
        }

        Ok(best_lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing_parameter_creation() {
        let sp = SmoothingParameter::new(2, OptimizationMethod::REML);
        assert_eq!(sp.lambda.len(), 2);
        assert_eq!(sp.lambda[0], 1.0);
    }

    #[test]
    fn test_grid_search() {
        let n = 20;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| ((i as f64) * 0.1).powi(j as i32));
        let w = Array1::ones(n);
        let penalty = Array2::eye(p);

        let result = SmoothingParameter::grid_search(
            &y,
            &x,
            &w,
            &penalty,
            0.001,
            10.0,
            20,
            OptimizationMethod::GCV
        );

        assert!(result.is_ok());
        let lambda = result.unwrap();
        assert!(lambda > 0.0);
    }
}
