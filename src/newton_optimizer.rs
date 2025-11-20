//! Newton-PIRLS optimizer for REML smoothing parameter estimation
//!
//! Implements Newton's method with line search, matching mgcv's approach.

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::reml::{reml_gradient_multi_qr, reml_hessian_multi_qr};

#[cfg(feature = "blas")]
use ndarray_linalg::Solve;

/// Result of Newton-PIRLS optimization
#[derive(Debug, Clone)]
pub struct NewtonResult {
    /// Optimal log-smoothing parameters
    pub log_lambda: Array1<f64>,
    /// Optimal smoothing parameters (exp of log_lambda)
    pub lambda: Array1<f64>,
    /// Final REML criterion value
    pub reml_value: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Final gradient norm
    pub gradient_norm: f64,
    /// Convergence flag
    pub converged: bool,
    /// Convergence message
    pub message: String,
}

/// Newton-PIRLS optimizer for REML criterion
///
/// Uses Newton's method with line search to minimize REML criterion:
/// ρ_new = ρ_old - α·H^{-1}·g
///
/// where:
/// - ρ = log(λ) are log-smoothing parameters
/// - g = ∂REML/∂ρ is the gradient
/// - H = ∂²REML/∂ρ² is the Hessian
/// - α is the step size from line search
pub struct NewtonPIRLS {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Gradient convergence tolerance
    pub grad_tol: f64,
    /// Relative REML change tolerance
    pub reml_tol: f64,
    /// Minimum step size in line search
    pub min_step: f64,
    /// Line search backtracking factor
    pub backtrack_factor: f64,
    /// Maximum line search iterations
    pub max_line_search: usize,
    /// Print iteration details
    pub verbose: bool,
}

impl Default for NewtonPIRLS {
    fn default() -> Self {
        NewtonPIRLS {
            max_iter: 100,
            grad_tol: 1e-6,
            reml_tol: 1e-8,
            min_step: 1e-10,
            backtrack_factor: 0.5,
            max_line_search: 20,
            verbose: false,
        }
    }
}

impl NewtonPIRLS {
    /// Create a new Newton-PIRLS optimizer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize REML criterion using Newton's method
    ///
    /// # Arguments
    /// * `y` - Response vector (n,)
    /// * `x` - Design matrix (n, p)
    /// * `w` - Weights (n,)
    /// * `initial_log_lambda` - Starting log-smoothing parameters (m,)
    /// * `penalties` - Penalty matrices, one per smooth (m × [p, p])
    ///
    /// # Returns
    /// * `NewtonResult` with optimal parameters and convergence info
    pub fn optimize(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        initial_log_lambda: &Array1<f64>,
        penalties: &[Array2<f64>],
    ) -> Result<NewtonResult> {
        let m = initial_log_lambda.len();

        if penalties.len() != m {
            return Err(GAMError::DimensionMismatch(
                format!("Number of penalties ({}) must match log_lambda ({})", penalties.len(), m)
            ));
        }

        let mut log_lambda = initial_log_lambda.clone();
        let mut iteration = 0;
        let mut converged = false;
        let mut message = String::new();

        if self.verbose {
            eprintln!("Newton-PIRLS Optimization");
            eprintln!("========================");
            eprintln!("Initial ρ: {:?}", log_lambda);
        }

        loop {
            iteration += 1;

            // Convert to λ scale
            let lambda: Vec<f64> = log_lambda.iter().map(|x| x.exp()).collect();

            // Compute gradient
            let gradient = reml_gradient_multi_qr(y, x, w, &lambda, penalties)?;
            let grad_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

            if self.verbose {
                eprintln!("\nIteration {}: max|grad| = {:.6e}", iteration,
                         gradient.iter().map(|g| g.abs()).fold(0.0f64, f64::max));
            }

            // Check convergence
            if grad_norm < self.grad_tol {
                converged = true;
                message = format!("Converged: gradient norm {:.6e} < {:.6e}", grad_norm, self.grad_tol);
                break;
            }

            if iteration >= self.max_iter {
                message = format!("Max iterations ({}) reached", self.max_iter);
                break;
            }

            // Compute Hessian
            let hessian = reml_hessian_multi_qr(y, x, w, &lambda, penalties)?;

            // Solve Newton system: H·Δρ = -g
            let delta_rho = self.solve_newton_system(&hessian, &gradient)?;

            // Check if we have a descent direction: g'·Δρ should be negative
            let descent_check: f64 = gradient.iter().zip(delta_rho.iter())
                .map(|(g, d)| g * d)
                .sum();

            if self.verbose {
                eprintln!("  Descent check (g'·Δρ): {:.6e} (should be < 0)", descent_check);
            }

            // If not descent direction, try steepest descent instead
            let delta_rho = if descent_check > 0.0 {
                if self.verbose {
                    eprintln!("  WARNING: Not a descent direction, using steepest descent");
                }
                gradient.mapv(|g| -g)  // Steepest descent: -g
            } else {
                delta_rho
            };

            // Line search to find step size
            let step_size = self.line_search(
                y, x, w, &log_lambda, &delta_rho, penalties
            )?;

            if self.verbose {
                eprintln!("  Line search result: step = {:.6e}", step_size);
            }

            if step_size < self.min_step {
                message = format!("Line search failed: step size {:.6e} < {:.6e}", step_size, self.min_step);
                break;
            }

            // Update parameters
            log_lambda = &log_lambda + &(delta_rho.mapv(|d| step_size * d));

            if self.verbose {
                eprintln!("  Step size: {:.6e}", step_size);
                eprintln!("  New ρ: {:?}", log_lambda);
            }
        }

        // Final evaluation
        let lambda: Vec<f64> = log_lambda.iter().map(|x| x.exp()).collect();
        let final_gradient = reml_gradient_multi_qr(y, x, w, &lambda, penalties)?;
        let gradient_norm = final_gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        // Compute final REML value
        let reml_value = self.compute_reml(y, x, w, &lambda, penalties)?;

        if self.verbose {
            eprintln!("\n{}", message);
            eprintln!("Final ρ: {:?}", log_lambda);
            eprintln!("Final λ: {:?}", lambda);
            eprintln!("Final REML: {:.6}", reml_value);
            eprintln!("Iterations: {}", iteration);
        }

        Ok(NewtonResult {
            log_lambda: log_lambda.clone(),
            lambda: Array1::from_vec(lambda),
            reml_value,
            iterations: iteration,
            gradient_norm,
            converged,
            message,
        })
    }

    /// Solve Newton system H·Δρ = -g
    fn solve_newton_system(&self, hessian: &Array2<f64>, gradient: &Array1<f64>) -> Result<Array1<f64>> {
        let neg_gradient = gradient.mapv(|g| -g);

        #[cfg(feature = "blas")]
        {
            // Use BLAS solver
            match hessian.solve(&neg_gradient) {
                Ok(delta) => Ok(delta),
                Err(_) => {
                    // Hessian might be singular, add ridge
                    let mut h_ridge = hessian.clone();
                    let ridge = 1e-6 * hessian.diag().iter().map(|x| x.abs()).fold(0.0f64, f64::max);
                    for i in 0..h_ridge.nrows() {
                        h_ridge[[i, i]] += ridge;
                    }
                    h_ridge.solve(&neg_gradient)
                        .map_err(|_| GAMError::SingularMatrix)
                }
            }
        }

        #[cfg(not(feature = "blas"))]
        {
            // Fallback: use our linalg solver
            use crate::linalg::solve;
            solve(hessian.clone(), neg_gradient)
        }
    }

    /// Line search with backtracking to ensure REML decreases
    ///
    /// Finds α such that REML(ρ + α·Δρ) < REML(ρ)
    fn line_search(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        log_lambda: &Array1<f64>,
        delta_rho: &Array1<f64>,
        penalties: &[Array2<f64>],
    ) -> Result<f64> {
        // Current REML value
        let lambda_current: Vec<f64> = log_lambda.iter().map(|x| x.exp()).collect();
        let reml_current = self.compute_reml(y, x, w, &lambda_current, penalties)?;

        let mut step = 1.0;

        if self.verbose {
            eprintln!("  Line search: REML_current = {:.6}", reml_current);
        }

        for iter in 0..self.max_line_search {
            // Try step
            let log_lambda_new = log_lambda + &delta_rho.mapv(|d| step * d);
            let lambda_new: Vec<f64> = log_lambda_new.iter().map(|x| x.exp()).collect();

            match self.compute_reml(y, x, w, &lambda_new, penalties) {
                Ok(reml_new) => {
                    if self.verbose && iter < 5 {
                        eprintln!("    step={:.3e}: REML={:.6}, Δ={:.3e}",
                                 step, reml_new, reml_new - reml_current);
                    }

                    // Check if REML decreased
                    if reml_new < reml_current {
                        if self.verbose {
                            eprintln!("  Accepted step: {:.6e}", step);
                        }
                        return Ok(step);
                    }
                }
                Err(e) => {
                    if self.verbose && iter < 5 {
                        eprintln!("    step={:.3e}: FAILED ({:?})", step, e);
                    }
                }
            }

            // Backtrack
            step *= self.backtrack_factor;

            if step < self.min_step {
                break;
            }
        }

        if self.verbose {
            eprintln!("  Line search failed, returning step={:.6e}", step);
        }
        // Return best step found (might be very small)
        Ok(step)
    }

    /// Compute REML criterion value
    fn compute_reml(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        lambda: &[f64],
        penalties: &[Array2<f64>],
    ) -> Result<f64> {
        // Use the same REML formula as gradient/Hessian computation
        use crate::reml::reml_criterion_multi;
        reml_criterion_multi(y, x, w, lambda, penalties, None)
    }
}
