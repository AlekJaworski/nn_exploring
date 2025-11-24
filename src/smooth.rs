//! Smoothing parameter selection using REML optimization

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::reml::{reml_criterion, gcv_criterion, reml_criterion_multi, reml_gradient_multi, reml_gradient_multi_qr, reml_gradient_multi_qr_adaptive, reml_hessian_multi};
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
            lambda: vec![0.1; num_smooths],  // Will be refined in optimize()
            method,
        }
    }

    /// Optimize smoothing parameters using REML or GCV with adaptive initialization
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

        // Adaptive initialization: lambda_i = 0.1 * trace(S_i) / trace(X'WX)
        // This scales initialization based on problem characteristics
        self.initialize_lambda_adaptive(x, w, penalties);

        match self.method {
            OptimizationMethod::REML => {
                self.optimize_reml(y, x, w, penalties, max_iter, tolerance)
            },
            OptimizationMethod::GCV => {
                self.optimize_gcv(y, x, w, penalties, max_iter, tolerance)
            },
        }
    }

    /// Initialize lambda values adaptively based on penalty and design matrix scales
    fn initialize_lambda_adaptive(
        &mut self,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
    ) {
        let n = x.nrows();
        let p = x.ncols();

        // Compute trace(X'WX) / n to get scale-invariant measure
        // This makes initialization independent of sample size
        let mut xtwx_trace_per_n = 0.0;
        for j in 0..p {
            let mut col_weighted_sq = 0.0;
            for i in 0..n {
                col_weighted_sq += x[[i, j]] * x[[i, j]] * w[i];
            }
            xtwx_trace_per_n += col_weighted_sq;
        }
        xtwx_trace_per_n /= n as f64;

        // Fallback if matrix is degenerate
        if xtwx_trace_per_n < 1e-10 {
            xtwx_trace_per_n = 1.0;
        }

        // Initialize each lambda based on its penalty matrix scale
        for (i, penalty) in penalties.iter().enumerate() {
            let mut penalty_trace = 0.0;
            let penalty_size = penalty.nrows().min(penalty.ncols());
            for j in 0..penalty_size {
                penalty_trace += penalty[[j, j]];
            }

            // FIXED: Scale-invariant initialization
            // lambda ~ 0.1 * trace(S) / (trace(X'WX)/n)
            // This makes starting lambda independent of n
            if penalty_trace > 1e-10 {
                self.lambda[i] = 0.1 * penalty_trace / xtwx_trace_per_n;
            } else {
                self.lambda[i] = 0.1;  // Fallback for near-zero penalty
            }

            // Clamp to reasonable range [1e-6, 1e6]
            self.lambda[i] = self.lambda[i].max(1e-6).min(1e6);
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

        // Maximum step size in log space (following Wood 2011 and mgcv)
        // This prevents overly aggressive Newton steps that require excessive backtracking
        // max_step=4 means we clamp λ_new/λ_old to [e^-4, e^4] = [0.018, 54.6]
        let max_step = 4.0;    // Conservative step size to match mgcv
        let max_half = 30;     // Maximum step halvings

        let mut prev_reml = f64::INFINITY;

        for iter in 0..max_iter {
            // Current lambdas
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // Compute current REML value for convergence check
            let current_reml = reml_criterion_multi(y, x, w, &lambdas, penalties, None)?;

            // Compute gradient and Hessian
            // Use QR-based gradient computation (adaptive: block-wise for large n >= 2000)
            let gradient = reml_gradient_multi_qr_adaptive(y, x, w, &lambdas, penalties)?;
            let mut hessian = reml_hessian_multi(y, x, w, &lambdas, penalties)?;

            // Debug output: show raw Hessian before conditioning
            if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
                eprintln!("\n[SMOOTH_DEBUG] Raw Hessian at λ={:?}:", lambdas);
                for i in 0..m {
                    for j in 0..m {
                        eprint!("  H[{},{}]={:.6e}", i, j, hessian[[i,j]]);
                    }
                    eprintln!();
                }
                eprintln!("[SMOOTH_DEBUG] Gradient: {:?}", gradient);
            }

            // ===================================================================
            // CRITICAL: Condition Hessian like mgcv to ensure stable convergence
            // ===================================================================
            // mgcv uses ridge regularization + diagonal preconditioning
            // This prevents ill-conditioning that causes tiny steps in late iterations

            // 1. Add adaptive ridge FIRST (before preconditioning)
            //    Ridge increases with iteration to handle increasing ill-conditioning
            let min_diag_orig = (0..m).map(|i| hessian[[i, i]]).fold(f64::INFINITY, f64::min);
            let max_diag_orig = (0..m).map(|i| hessian[[i, i]]).fold(0.0f64, f64::max);

            // CRITICAL: Diagonal preconditioning like mgcv (fast-REML.r)
            // This handles ill-conditioning from vastly different smoothing parameter scales
            // Transform: H_new = D^-1 * H * D^-1 where D = diag(sqrt(diag(H)))

            let mut diag_precond = Array1::<f64>::zeros(m);
            for i in 0..m {
                let d = hessian[[i, i]];
                // If diagonal is negative or tiny, use 1.0 (don't precondition that component)
                diag_precond[i] = if d > 1e-10 { d.sqrt() } else { 1.0 };
            }

            if std::env::var("MGCV_PROFILE").is_ok() {
                let cond_est = max_diag_orig / min_diag_orig.max(1e-10);
                eprintln!("[PROFILE]   Hessian diag range: [{:.6e}, {:.6e}], condition: {:.2e}",
                         min_diag_orig, max_diag_orig, cond_est);
                eprintln!("[PROFILE]   Preconditioner: {:?}", diag_precond.as_slice().unwrap_or(&[]));
            }

            // Apply preconditioning to Hessian: H_ij = H_ij / (d_i * d_j)
            for i in 0..m {
                for j in 0..m {
                    hessian[[i, j]] /= diag_precond[i] * diag_precond[j];
                }
            }

            // Add small ridge for numerical stability (after preconditioning)
            let ridge = 1e-7;
            for i in 0..m {
                hessian[[i, i]] += ridge;
            }

            // Check for convergence using multiple criteria
            // Use L-infinity norm (max absolute value) like mgcv, not L2 norm
            let grad_norm_l2: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            let grad_norm_linf: f64 = gradient.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
            let reml_change = if prev_reml.is_finite() {
                ((current_reml - prev_reml) / prev_reml.abs().max(1e-10)).abs()
            } else {
                f64::INFINITY
            };

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] Newton iter {}: grad_L2={:.6}, grad_Linf={:.6}, REML={:.6}, REML_change={:.6e}",
                         iter + 1, grad_norm_l2, grad_norm_linf, current_reml, reml_change);
                eprintln!("[PROFILE]   lambda={:?}", lambdas);
                eprintln!("[PROFILE]   log_lambda={:?}", log_lambda);
                eprintln!("[PROFILE]   gradient={:?}", gradient.as_slice().unwrap_or(&[]));
            }

            // Converged if EITHER:
            // 1. Gradient L-infinity norm is small (gradient convergence)
            // 2. REML value change is tiny (value convergence for asymptotic cases like λ→∞)
            // mgcv uses both criteria to handle different convergence scenarios
            //
            // NOTE: mgcv's default tolerance is 0.05-0.1 for the gradient Linf norm
            // Using 0.05 to match mgcv's convergence behavior
            if grad_norm_linf < 0.05 {
                self.lambda = lambdas;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Converged after {} iterations (gradient criterion: {:.6} < 0.05)", iter + 1, grad_norm_linf);
                }
                return Ok(());
            }

            // REML change convergence: DISABLED for now to test gradient convergence
            // The Hessian may be approximate, causing tiny REML steps even with large gradient
            // Better to rely on gradient criterion
            let relative_reml_change = reml_change / current_reml.abs().max(1.0);
            if false && iter > 5 && relative_reml_change < 1e-6 {
                self.lambda = lambdas;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Converged after {} iterations (REML relative change: {:.2e} < 1e-6)",
                             iter + 1, relative_reml_change);
                }
                return Ok(());
            }

            prev_reml = current_reml;

            // Compute Newton step: step = -H^(-1) · g
            // With preconditioning: solve (D^-1 H D^-1) step_precond = -(D^-1 g)
            // Then back-transform: step = D^-1 step_precond

            // Precondition gradient: g_precond = D^-1 * g
            let mut gradient_precond = Array1::<f64>::zeros(m);
            for i in 0..m {
                gradient_precond[i] = gradient[i] / diag_precond[i];
            }

            // Solve preconditioned system
            let step_precond = solve(hessian.clone(), -gradient_precond)?;

            // Back-transform: step = D^-1 * step_precond
            let mut step = Array1::<f64>::zeros(m);
            for i in 0..m {
                step[i] = step_precond[i] / diag_precond[i];
            }

            // Limit step size (Wood 2011: max step = 4-5 in log space)
            let step_size: f64 = step.iter().map(|s| s * s).sum::<f64>().sqrt();
            if step_size > max_step {
                let scale = max_step / step_size;
                for s in step.iter_mut() {
                    *s *= scale;
                }
            }

            // Line search with step halving (reuse current_reml from above)
            let mut best_reml = current_reml;
            let mut best_step_scale = 0.0;
            let step_size_clamped = if step_size > max_step { max_step } else { step_size };

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE]   Line search: step_norm={:.6}, current_REML={:.6}",
                         step_size_clamped, current_reml);
            }

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
                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!("[PROFILE]     half={}: scale={:.4}, REML={:.6}, improvement={}",
                                     half, step_scale, new_reml, new_reml < best_reml);
                        }
                        if new_reml < best_reml {
                            best_reml = new_reml;
                            best_step_scale = step_scale;
                        } else if best_step_scale > 0.0 {
                            // Found an improvement earlier, no further improvement now - stop
                            if std::env::var("MGCV_PROFILE").is_ok() {
                                eprintln!("[PROFILE]   Best step scale: {:.4}", best_step_scale);
                            }
                            break;
                        }
                        // If no improvement yet (best_step_scale == 0), keep trying smaller steps
                    },
                    Err(_) => {
                        // Numerical issue - try smaller step
                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!("[PROFILE]     half={}: ERROR (numerical issue)", half);
                        }
                        continue;
                    }
                }
            }

            // Update log_lambda
            if best_step_scale > 0.0 {
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE]   Accepted Newton step, scale={:.4}", best_step_scale);
                }
                for i in 0..m {
                    log_lambda[i] += step[i] * best_step_scale;
                }
            } else {
                // Newton failed - try steepest descent as fallback
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE]   Newton failed, trying steepest descent");
                }

                // Steepest descent: step = -gradient (scaled very small)
                // Recompute gradient since it was moved earlier
                let gradient_sd = reml_gradient_multi_qr_adaptive(y, x, w, &lambdas, penalties)?;

                // Try progressively smaller steepest descent steps
                let mut sd_worked = false;
                for scale in &[0.01, 0.001, 0.0001] {
                    let sd_step: Vec<f64> = gradient_sd.iter().map(|g| -g * scale).collect();

                    let new_log_lambda_sd: Vec<f64> = log_lambda.iter()
                        .zip(sd_step.iter())
                        .map(|(l, s)| l + s)
                        .collect();

                    let new_lambdas_sd: Vec<f64> = new_log_lambda_sd.iter().map(|l| l.exp()).collect();

                    if let Ok(new_reml_sd) = reml_criterion_multi(y, x, w, &new_lambdas_sd, penalties, None) {
                        if std::env::var("MGCV_PROFILE").is_ok() {
                            eprintln!("[PROFILE]     SD scale={}: REML={:.6} (current={:.6}, improvement={})",
                                     scale, new_reml_sd, current_reml, new_reml_sd < current_reml);
                        }
                        if new_reml_sd < current_reml {
                            for i in 0..m {
                                log_lambda[i] = new_log_lambda_sd[i];
                            }
                            if std::env::var("MGCV_PROFILE").is_ok() {
                                eprintln!("[PROFILE]   Steepest descent succeeded (scale={}): REML={:.6}", scale, new_reml_sd);
                            }
                            sd_worked = true;
                            break;
                        }
                    } else if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE]     SD scale={}: REML computation failed", scale);
                    }
                }

                if !sd_worked {
                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE]   Steepest descent failed at all scales, stopping");
                    }
                    break;
                }

            }
        }

        // Update final lambdas
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!("[PROFILE] Reached max iterations ({}) without convergence", max_iter);
        }

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
        assert_eq!(sp.lambda[0], 0.1);  // Updated to match current default
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
