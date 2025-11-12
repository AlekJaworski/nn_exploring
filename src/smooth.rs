//! Smoothing parameter selection using REML optimization

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::reml::{reml_criterion, gcv_criterion, reml_criterion_multi, reml_gradient_multi, reml_hessian_multi};
use crate::linalg::solve;

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

    /// Optimize using REML criterion with Newton's method
    ///
    /// Implements Wood (2011) fast stable REML optimization using joint Newton method
    /// for multiple smoothing parameters
    fn optimize_reml(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        // Use Newton's method for all cases (single or multiple smooths)
        // This matches mgcv's fast-REML.fit approach
        self.optimize_reml_newton_multi(y, x, w, penalties, max_iter, tolerance)
    }

    /// Grid search for single smooth (kept for stability)
    fn optimize_reml_grid_single(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
    ) -> Result<()> {
        let mut best_lambda = self.lambda[0];
        let mut best_reml = f64::INFINITY;

        // Coarse grid search to find approximate optimum
        for i in 0..50 {
            let log_lambda = -4.0 + i as f64 * 0.12;  // -4 to 2 (0.0001 to 100)
            let lambda = 10.0_f64.powf(log_lambda);
            let reml = reml_criterion(y, x, w, lambda, penalty, None)?;

            if reml < best_reml {
                best_reml = reml;
                best_lambda = lambda;
            }
        }

        // Refine with finer grid search around best lambda
        let log_best = best_lambda.ln();
        let search_width = 0.15;  // Search ±0.15 in log space
        for i in 0..30 {
            let log_lambda = log_best - search_width + i as f64 * (2.0 * search_width / 29.0);
            let lambda = log_lambda.exp();
            if lambda > 0.0 {
                let reml = reml_criterion(y, x, w, lambda, penalty, None)?;

                if reml < best_reml {
                    best_reml = reml;
                    best_lambda = lambda;
                }
            }
        }

        self.lambda[0] = best_lambda;
        Ok(())
    }

    /// Newton optimization for multiple smoothing parameters
    ///
    /// Optimizes all λᵢ jointly using Newton's method on log(λᵢ)
    /// Following Wood (2011) JRSS-B algorithm
    fn optimize_reml_newton_multi(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        let m = penalties.len();

        // Work in log space for stability
        let mut log_lambda: Vec<f64> = self.lambda.iter()
            .map(|l| l.ln())
            .collect();

        let max_step = 4.0;  // Maximum step size in log space (Wood 2011)
        let max_half = 30;   // Maximum step halvings

        for iter in 0..max_iter {
            // Current lambdas
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // Compute gradient and Hessian
            let gradient = reml_gradient_multi(y, x, w, &lambdas, penalties)?;
            let hessian = reml_hessian_multi(y, x, w, &lambdas, penalties)?;

            // Check for convergence
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < tolerance {
                self.lambda = lambdas;
                return Ok(());
            }

            // Compute Newton step: step = -H^(-1) · g
            let mut step = solve(hessian.clone(), -gradient.clone())?;

            // Limit step size (Wood 2011: max step = 4-5 in log space)
            let step_size: f64 = step.iter().map(|s| s * s).sum::<f64>().sqrt();
            if step_size > max_step {
                let scale = max_step / step_size;
                for s in step.iter_mut() {
                    *s *= scale;
                }
            }

            // Line search with step halving
            let current_reml = reml_criterion_multi(y, x, w, &lambdas, penalties, None)?;
            let mut best_reml = current_reml;
            let mut best_step_scale = 0.0;

            for half in 0..=max_half {
                let step_scale = 0.5_f64.powi(half as i32);

                // Try new log_lambda values
                let new_log_lambda: Vec<f64> = log_lambda.iter()
                    .zip(step.iter())
                    .map(|(l, s)| l + s * step_scale)
                    .collect();

                let new_lambdas: Vec<f64> = new_log_lambda.iter()
                    .map(|l| l.exp())
                    .collect();

                // Evaluate REML
                match reml_criterion_multi(y, x, w, &new_lambdas, penalties, None) {
                    Ok(new_reml) => {
                        if new_reml < best_reml {
                            best_reml = new_reml;
                            best_step_scale = step_scale;
                        } else if best_step_scale > 0.0 {
                            // Found an improvement earlier, no further improvement now - stop
                            break;
                        }
                        // If no improvement yet (best_step_scale == 0), keep trying smaller steps
                    },
                    Err(_) => {
                        // Numerical issue - try smaller step
                        continue;
                    }
                }
            }

            // Update log_lambda
            if best_step_scale > 0.0 {
                for i in 0..m {
                    log_lambda[i] += step[i] * best_step_scale;
                }
            } else {
                // No improvement found - converged or stuck
                break;
            }

            // Print progress (optional)
            if iter % 5 == 0 {
                eprintln!("REML Newton iter {}: REML={:.6}, grad_norm={:.6}",
                         iter, best_reml, grad_norm);
            }
        }

        // Update final lambdas
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

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
