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
    /// Use trust region instead of line search
    pub use_trust_region: bool,
    /// Initial trust region radius
    pub initial_trust_radius: f64,
    /// Maximum trust region radius
    pub max_trust_radius: f64,
    /// Trust region acceptance threshold (typically 0.1-0.25)
    pub eta: f64,
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
            use_trust_region: false,  // Disable: gradient has ~30% error, needs refinement
            initial_trust_radius: 1.0,
            max_trust_radius: 10.0,
            eta: 0.15,  // Accept step if actual/predicted reduction > 0.15
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
        let mut trust_radius = self.initial_trust_radius;

        if self.verbose {
            eprintln!("Newton-PIRLS Optimization");
            eprintln!("========================");
            eprintln!("Initial ρ: {:?}", log_lambda);
            if self.use_trust_region {
                eprintln!("Using trust region method (Δ={:.3})", trust_radius);
            } else {
                eprintln!("Using line search");
            }
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

            // Choose between trust region and line search
            let step = if self.use_trust_region {
                // Trust region method
                let (step, new_radius) = self.trust_region_step(
                    y, x, w, &log_lambda, &gradient, &hessian, penalties, trust_radius
                )?;
                trust_radius = new_radius;

                if self.verbose {
                    eprintln!("  New trust radius: Δ={:.3e}", trust_radius);
                }

                // Check if step was rejected (all zeros)
                let step_norm = step.iter().map(|x| x * x).sum::<f64>().sqrt();
                if step_norm < 1e-12 {
                    // Step rejected, shrink trust region and continue
                    trust_radius *= 0.5;
                    if trust_radius < 1e-8 {
                        message = format!("Trust region too small: Δ={:.6e}", trust_radius);
                        break;
                    }
                    continue;
                }

                step
            } else {
                // Line search method
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

                delta_rho.mapv(|d| step_size * d)
            };

            // Update parameters
            log_lambda = &log_lambda + &step;

            if self.verbose {
                let step_norm = step.iter().map(|x| x * x).sum::<f64>().sqrt();
                eprintln!("  Step norm: {:.6e}", step_norm);
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

    /// Trust region step: solve subproblem and update trust radius
    ///
    /// Solves: min_{Δρ} g'·Δρ + (1/2)·Δρ'·H·Δρ subject to ||Δρ|| ≤ Δ
    ///
    /// Uses dogleg method for efficiency:
    /// - If Hessian is PD and Newton step within trust region: use Newton step
    /// - Otherwise: combine Cauchy point (steepest descent) and Newton step
    ///
    /// Returns: (step, new_trust_radius)
    fn trust_region_step(
        &self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        log_lambda: &Array1<f64>,
        gradient: &Array1<f64>,
        hessian: &Array2<f64>,
        penalties: &[Array2<f64>],
        trust_radius: f64,
    ) -> Result<(Array1<f64>, f64)> {
        let m = gradient.len();

        // Compute current REML
        let lambda: Vec<f64> = log_lambda.iter().map(|x| x.exp()).collect();
        let reml_current = self.compute_reml(y, x, w, &lambda, penalties)?;

        // Try to solve H·Δρ = -g with ridge regularization if needed
        let mut delta_rho_newton = Array1::zeros(m);
        let mut hessian_reg = hessian.clone();

        // Add ridge to ensure positive definiteness
        // Use stronger ridge since gradient has ~30% error
        let max_diag = hessian_reg.diag().iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let ridge = 1e-2 * max_diag.max(1.0);  // Increased from 1e-4 to 1e-2
        for i in 0..m {
            hessian_reg[[i, i]] += ridge;
        }

        // Solve using Cholesky (will fallback if needed)
        delta_rho_newton = self.solve_newton_system(&hessian_reg, gradient)?;

        // Check if Newton step is a descent direction: g'·Δρ < 0
        let descent_check: f64 = gradient.iter().zip(delta_rho_newton.iter())
            .map(|(g, d)| g * d)
            .sum();

        // If not descent, set Newton step to zero (will use Cauchy instead)
        if descent_check > 0.0 {
            if self.verbose {
                eprintln!("  Trust region: Newton step not descent (g'·Δρ={:.3e}), using Cauchy", descent_check);
            }
            delta_rho_newton = Array1::zeros(m);
        }

        // Compute Newton step norm
        let newton_norm = delta_rho_newton.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Compute Cauchy point (steepest descent direction)
        // Δρ_C = -τ·g where τ minimizes quadratic along -g direction
        let grad_norm_sq: f64 = gradient.iter().map(|g| g * g).sum();
        let hg = hessian.dot(gradient);
        let g_hg: f64 = gradient.iter().zip(hg.iter()).map(|(g, hg)| g * hg).sum();

        // τ for Cauchy point
        let tau = if g_hg > 0.0 {
            // Hessian is positive in this direction
            (grad_norm_sq / g_hg).min(trust_radius / grad_norm_sq.sqrt())
        } else {
            // Hessian is negative/zero in this direction, go to boundary
            trust_radius / grad_norm_sq.sqrt()
        };

        let delta_rho_cauchy = gradient.mapv(|g| -tau * g);
        let cauchy_norm = delta_rho_cauchy.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Dogleg path:
        // 1. If Newton step within trust region AND non-zero: use it
        // 2. If Cauchy point outside trust region: scale Cauchy to boundary
        // 3. Otherwise: interpolate between Cauchy and Newton
        let delta_rho = if newton_norm > 1e-10 && newton_norm <= trust_radius {
            // Newton step is within trust region
            if self.verbose {
                eprintln!("  Trust region: Using Newton step (norm={:.3e} ≤ Δ={:.3e})", newton_norm, trust_radius);
            }
            delta_rho_newton
        } else if cauchy_norm >= trust_radius {
            // Scale Cauchy point to trust region boundary
            if self.verbose {
                eprintln!("  Trust region: Using scaled Cauchy point");
            }
            delta_rho_cauchy.mapv(|x| x * trust_radius / cauchy_norm)
        } else {
            // Dogleg: find point on line from Cauchy to Newton that hits boundary
            // ||δ_C + β·(δ_N - δ_C))||² = Δ²
            let diff = &delta_rho_newton - &delta_rho_cauchy;
            let a: f64 = diff.iter().map(|x| x * x).sum::<f64>();
            let b: f64 = 2.0 * delta_rho_cauchy.iter().zip(diff.iter()).map(|(c, d)| c * d).sum::<f64>();
            let c: f64 = cauchy_norm * cauchy_norm - trust_radius * trust_radius;

            let beta = if a > 1e-10 {
                (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)
            } else {
                1.0
            };

            if self.verbose {
                eprintln!("  Trust region: Dogleg interpolation (β={:.3})", beta);
            }
            &delta_rho_cauchy + &diff.mapv(|x| beta * x)
        };

        // Evaluate REML at new point
        let log_lambda_new = log_lambda + &delta_rho;
        let lambda_new: Vec<f64> = log_lambda_new.iter().map(|x| x.exp()).collect();
        let reml_new = self.compute_reml(y, x, w, &lambda_new, penalties)?;

        // Compute actual vs predicted reduction
        let actual_reduction = reml_current - reml_new;

        // Predicted reduction: m(0) - m(δ) = -g'·δ - (1/2)·δ'·H·δ
        let h_delta = hessian.dot(&delta_rho);
        let predicted_reduction = -(
            gradient.iter().zip(delta_rho.iter()).map(|(g, d)| g * d).sum::<f64>()
            + 0.5 * delta_rho.iter().zip(h_delta.iter()).map(|(d, hd)| d * hd).sum::<f64>()
        );

        let rho = if predicted_reduction.abs() < 1e-10 {
            0.0
        } else {
            actual_reduction / predicted_reduction
        };

        if self.verbose {
            eprintln!("  Trust region: actual={:.3e}, predicted={:.3e}, ρ={:.3}",
                     actual_reduction, predicted_reduction, rho);
        }

        // Update trust region radius based on agreement
        let new_radius = if rho < 0.25 {
            // Poor agreement: shrink trust region
            0.25 * trust_radius
        } else if rho > 0.75 && (delta_rho.iter().map(|x| x * x).sum::<f64>().sqrt() - trust_radius).abs() < 1e-6 {
            // Good agreement and at boundary: expand trust region
            (2.0 * trust_radius).min(self.max_trust_radius)
        } else {
            // Acceptable: keep trust region
            trust_radius
        };

        // Accept step if rho > eta
        let step = if rho > self.eta {
            if self.verbose {
                eprintln!("  Trust region: ACCEPT (ρ={:.3} > η={:.3})", rho, self.eta);
            }
            delta_rho
        } else {
            if self.verbose {
                eprintln!("  Trust region: REJECT (ρ={:.3} ≤ η={:.3})", rho, self.eta);
            }
            Array1::zeros(m)  // Reject step
        };

        Ok((step, new_radius))
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
