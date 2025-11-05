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
            lambda: vec![0.1; num_smooths],  // Better starting point than 1.0
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
        _max_iter: usize,
        _tolerance: f64,
    ) -> Result<()> {
        // For single smooth case
        if penalties.len() != 1 {
            panic!("Multiple smooths not yet properly implemented for REML");
        }

        // Use grid search to find optimal lambda
        // (Gradient descent has convergence issues with REML)
        let mut best_lambda = self.lambda[0];
        let mut best_reml = f64::INFINITY;

        // Fine grid search
        for i in 0..50 {
            let log_lambda = -4.0 + i as f64 * 0.12;  // -4 to 2 (0.0001 to 100)
            let lambda = 10.0_f64.powf(log_lambda);
            let reml = reml_criterion(y, x, w, lambda, &penalties[0], None)?;

            if reml < best_reml {
                best_reml = reml;
                best_lambda = lambda;
            }
        }

        self.lambda[0] = best_lambda;
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

        for _iter in 0..max_iter {
            let mut converged = true;

            for i in 0..log_lambda.len() {
                let old_log_lambda = log_lambda[i];

                // For single smooth case
                if penalties.len() != 1 {
                    panic!("Multiple smooths not yet properly implemented for GCV");
                }

                let lambda_current = log_lambda[i].exp();

                let gcv_current = gcv_criterion(
                    y, x, w,
                    lambda_current,
                    &penalties[0],
                )?;

                // Numerical gradient
                let delta = 0.01;
                log_lambda[i] += delta;
                let lambda_plus = log_lambda[i].exp();

                let gcv_plus = gcv_criterion(
                    y, x, w,
                    lambda_plus,
                    &penalties[0],
                )?;

                // Reset
                log_lambda[i] = old_log_lambda;

                let gradient = (gcv_plus - gcv_current) / delta;

                let step_size = 0.5;
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
