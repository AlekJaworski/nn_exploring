//! Smoothing parameter selection using REML optimization

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::linalg::solve;
use crate::chunked_qr::IncrementalQR;
use std::time::Instant;
#[cfg(feature = "blas")]
use crate::reml::{reml_criterion, gcv_criterion, reml_criterion_multi, reml_gradient_multi_qr_adaptive, reml_gradient_multi_qr_adaptive_cached_edf, reml_hessian_multi_cached, penalty_sqrt, compute_xtwx_cholesky};
#[cfg(feature = "blas")]
pub use crate::reml::ScaleParameterMethod;
#[cfg(not(feature = "blas"))]
use crate::reml::{reml_criterion, gcv_criterion, reml_criterion_multi, reml_gradient_multi};

/// Smoothing parameter optimization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationMethod {
    REML,
    GCV,
}

/// REML optimization algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum REMLAlgorithm {
    Newton,         // Newton's method with Hessian (RECOMMENDED: fast, stable, matches bam())
    FellnerSchall,  // Fellner-Schall iteration (DEPRECATED: broken by penalty normalization)
}

/// Container for smoothing parameters
#[derive(Debug, Clone)]
pub struct SmoothingParameter {
    pub lambda: Vec<f64>,
    pub method: OptimizationMethod,
    pub reml_algorithm: REMLAlgorithm,
    /// Method for computing scale parameter φ (only used with BLAS feature)
    /// - Rank: Fast O(1) using penalty matrix ranks (default)
    /// - EDF: Exact O(p³/3) using effective degrees of freedom (matches mgcv)
    #[cfg(feature = "blas")]
    pub scale_method: ScaleParameterMethod,
}

impl SmoothingParameter {
    /// Create new smoothing parameters with initial values
    pub fn new(num_smooths: usize, method: OptimizationMethod) -> Self {
        Self {
            lambda: vec![0.1; num_smooths],  // Will be refined in optimize()
            method,
            reml_algorithm: REMLAlgorithm::Newton,  // Default to Newton (matches bam())
            #[cfg(feature = "blas")]
            scale_method: ScaleParameterMethod::EDF,  // Default to EDF (matches mgcv)
        }
    }

    /// Create with specific REML algorithm
    pub fn new_with_algorithm(num_smooths: usize, method: OptimizationMethod, algorithm: REMLAlgorithm) -> Self {
        Self {
            lambda: vec![0.1; num_smooths],
            method,
            reml_algorithm: algorithm,
            #[cfg(feature = "blas")]
            scale_method: ScaleParameterMethod::Rank,
        }
    }
    
    /// Create with EDF-based scale parameter (matches mgcv exactly)
    /// 
    /// This uses Effective Degrees of Freedom instead of penalty ranks
    /// for computing the scale parameter φ. More accurate for ill-conditioned
    /// problems (k >> n) but adds O(p³/3) cost per iteration.
    #[cfg(feature = "blas")]
    pub fn new_with_edf(num_smooths: usize, method: OptimizationMethod) -> Self {
        Self {
            lambda: vec![0.1; num_smooths],
            method,
            reml_algorithm: REMLAlgorithm::Newton,
            scale_method: ScaleParameterMethod::EDF,
        }
    }
    
    /// Set the scale parameter method
    #[cfg(feature = "blas")]
    pub fn with_scale_method(mut self, method: ScaleParameterMethod) -> Self {
        self.scale_method = method;
        self
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
        // DISABLED FOR DEBUGGING - using fixed init for testing
        // Try starting from a reasonable middle value
        for i in 0..self.lambda.len() {
            self.lambda[i] = 1.0; // Start from λ=1
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
        // Dispatch based on selected algorithm
        match self.reml_algorithm {
            REMLAlgorithm::Newton => {
                self.optimize_reml_newton_multi(y, x, w, penalties, max_iter, tolerance)
            },
            REMLAlgorithm::FellnerSchall => {
                self.optimize_reml_fellner_schall(y, x, w, penalties, max_iter, tolerance)
            },
        }
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
    #[cfg(feature = "blas")]
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

        // OPTIMIZATION: Pre-compute sqrt_penalties once (expensive eigendecomp)
        // Penalties don't change during Newton optimization, so cache them
        let sqrt_penalties_start = std::time::Instant::now();
        let mut sqrt_penalties = Vec::new();
        let mut penalty_ranks = Vec::new();
        for penalty in penalties.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            let rank = sqrt_pen.ncols();
            sqrt_penalties.push(sqrt_pen);
            penalty_ranks.push(rank);
        }
        if std::env::var("MGCV_PROFILE").is_ok() {
            let sqrt_pen_time = sqrt_penalties_start.elapsed();
            eprintln!("[PROFILE] Pre-computed sqrt_penalties: {:.2}ms", sqrt_pen_time.as_secs_f64() * 1000.0);
        }

        // OPTIMIZATION: Pre-compute X'WX and X'Wy (don't change during optimization)
        // This avoids O(np²) recomputation every iteration
        use crate::reml::compute_xtwx;
        let xtwx_start = Instant::now();
        let xtwx = compute_xtwx(x, w);

        // Compute X'Wy (also constant)
        let x_weighted = {
            let mut x_w = x.clone();
            for i in 0..x.nrows() {
                for j in 0..x.ncols() {
                    x_w[[i, j]] *= w[i].sqrt();
                }
            }
            x_w
        };
        let mut y_weighted = Array1::zeros(y.len());
        for i in 0..y.len() {
            y_weighted[i] = y[i] * w[i].sqrt();
        }
        let xtwy = x_weighted.t().dot(&y_weighted);

        if std::env::var("MGCV_PROFILE").is_ok() {
            let xtwx_time = xtwx_start.elapsed();
            eprintln!("[PROFILE] Pre-computed X'WX and X'Wy: {:.2}ms", xtwx_time.as_secs_f64() * 1000.0);
        }
        
        // Pre-compute Cholesky of X'WX for EDF computation (if using EDF method)
        let xtwx_chol: Option<Array2<f64>> = if self.scale_method == ScaleParameterMethod::EDF {
            let chol_start = Instant::now();
            let chol = compute_xtwx_cholesky(&xtwx)?;
            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] Pre-computed X'WX Cholesky for EDF: {:.2}ms", chol_start.elapsed().as_secs_f64() * 1000.0);
            }
            Some(chol)
        } else {
            None
        };

        // Work in log space for stability
        let mut log_lambda: Vec<f64> = self.lambda.iter()
            .map(|l| l.ln())
            .collect();

        // Maximum step size in log space (following Wood 2011 and mgcv)
        // This prevents overly aggressive Newton steps that require excessive backtracking
        // max_step=4 means we clamp λ_new/λ_old to [e^-4, e^4] = [0.018, 54.6]
        let max_step = 4.0;    // Conservative step size to match mgcv

        // OPTIMIZATION: Armijo constant for line search
        // Accepts steps with "sufficient decrease": f(x + αd) ≤ f(x) + c₁·α·∇f'·d
        // Standard value c₁ = 0.01 (very lenient, prefers larger steps)
        let armijo_c1 = 0.01;

        let mut prev_reml = f64::INFINITY;

        for iter in 0..max_iter {
            // Current lambdas
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // Compute current REML value for convergence check
            let current_reml = reml_criterion_multi(y, x, w, &lambdas, penalties, None)?;

            // Compute gradient and Hessian
            // Use QR-based gradient computation (adaptive: block-wise for large n >= 2000)
            // OPTIMIZATION: Pass cached sqrt_penalties, X'WX, X'Wy to avoid recomputation
            let t_grad = Instant::now();
            let gradient = reml_gradient_multi_qr_adaptive_cached_edf(
                y, x, w, &lambdas, penalties, 
                Some(&sqrt_penalties), Some(&xtwx), Some(&xtwy),
                xtwx_chol.as_ref(), self.scale_method
            )?;
            let grad_time = t_grad.elapsed().as_micros();

            let t_hess = Instant::now();
            // OPTIMIZATION: Use cached X'WX to avoid recomputation (~2-3ms savings for n=5000)
            let mut hessian = reml_hessian_multi_cached(y, x, w, &lambdas, penalties, &xtwx)?;
            let hess_time = t_hess.elapsed().as_micros();

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE]     Gradient: {:.2}ms, Hessian: {:.2}ms",
                         grad_time as f64 / 1000.0, hess_time as f64 / 1000.0);
            }

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

            // REML change convergence: Stop if making negligible progress
            // After a few iterations, if REML barely changes, we're done
            // Use relative change to be scale-invariant
            if iter >= 2 && reml_change < 1e-5 {
                self.lambda = lambdas;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Converged after {} iterations (REML change: {:.2e} < 1e-5)",
                             iter + 1, reml_change);
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

            // OPTIMIZATION: Adaptive line search with Armijo condition
            // Compute directional derivative: gradient · step (for Armijo condition)
            let grad_dot_step: f64 = gradient.iter().zip(step.iter())
                .map(|(g, s)| g * s)
                .sum();

            // OPTIMIZATION: Adaptive max_half based on convergence progress
            // Near convergence (small gradient), Newton step is likely good - use fewer halvings
            // Far from convergence, may need more exploration
            let max_half = if grad_norm_linf < 0.1 {
                10  // Near convergence - fewer line search iterations
            } else if grad_norm_linf < 1.0 {
                20  // Moderate - standard search
            } else {
                30  // Far from convergence - thorough search
            };

            let mut best_reml = current_reml;
            let mut best_step_scale = 0.0;
            let step_size_clamped = if step_size > max_step { max_step } else { step_size };

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE]   Line search: step_norm={:.6}, current_REML={:.6}, max_half={}",
                         step_size_clamped, current_reml, max_half);
                eprintln!("[PROFILE]     grad·step={:.6e} (expect decrease)", grad_dot_step);
            }

            let t_linesearch = Instant::now();
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
                        // OPTIMIZATION: Armijo condition for early stopping
                        // Accept if: new_reml ≤ current_reml + c₁ * step_scale * grad·step
                        // Since we're minimizing and grad·step should be negative, this is:
                        // new_reml ≤ current_reml - c₁ * step_scale * |grad·step|
                        let armijo_threshold = current_reml + armijo_c1 * step_scale * grad_dot_step;
                        let satisfies_armijo = new_reml <= armijo_threshold;

                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!("[PROFILE]     half={}: scale={:.4}, REML={:.6}, armijo={}",
                                     half, step_scale, new_reml, satisfies_armijo);
                        }

                        if new_reml < best_reml {
                            best_reml = new_reml;
                            best_step_scale = step_scale;

                            // OPTIMIZATION: Early stopping with Armijo condition
                            // If this step satisfies Armijo, accept it immediately
                            // This avoids over-precise line search that wastes time
                            if satisfies_armijo && half > 0 {
                                // Accept this step (but always try full step first, hence half > 0)
                                if std::env::var("MGCV_PROFILE").is_ok() {
                                    eprintln!("[PROFILE]   Armijo condition satisfied, accepting scale={:.4}", step_scale);
                                }
                                break;
                            }
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
            let linesearch_time = t_linesearch.elapsed().as_micros();

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE]     Line search: {:.2}ms", linesearch_time as f64 / 1000.0);
            }

            // Update log_lambda
            // Reject steps smaller than 1e-6 as they're effectively zero and waste time
            const MIN_STEP_SIZE: f64 = 1e-6;

            if best_step_scale > MIN_STEP_SIZE {
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE]   Accepted Newton step, scale={:.4}", best_step_scale);
                }
                for i in 0..m {
                    log_lambda[i] += step[i] * best_step_scale;
                }
            } else {
                // Newton line search found no meaningful improvement (step too small or zero)
                // If gradient is already small, accept convergence rather than waste time
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE]   Newton step too small (scale={:.3e}), checking gradient", best_step_scale);
                }

                if grad_norm_linf < 0.1 {
                    self.lambda = lambdas;
                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE] Converged after {} iterations (gradient {:.6} < 0.1, no further progress possible)",
                                 iter + 1, grad_norm_linf);
                    }
                    return Ok(());
                }

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
                        eprintln!("[PROFILE]   Steepest descent failed at all scales");
                    }
                    // Check if we're close enough to converged before giving up
                    // When at a minimum, no further progress is possible but gradient may still be small
                    let gradient_check = reml_gradient_multi_qr_adaptive(y, x, w, &lambdas, penalties)?;
                    let grad_norm_final = gradient_check.iter().map(|g| g.abs()).fold(0.0f64, f64::max);

                    // Use relaxed gradient tolerance (0.1) since we can't make further progress
                    // mgcv uses 0.05-0.1, so 0.1 is reasonable when at numerical limits
                    let relaxed_tol = 0.1;
                    if grad_norm_final < relaxed_tol {
                        self.lambda = lambdas;
                        if std::env::var("MGCV_PROFILE").is_ok() {
                            eprintln!("[PROFILE] Converged after {} iterations (gradient {:.6} < {:.6} at numerical limit)",
                                     iter + 1, grad_norm_final, relaxed_tol);
                        }
                        return Ok(());
                    }

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE]   Gradient {:.6} still too large (tolerance={:.6}), stopping",
                                 grad_norm_final, relaxed_tol);
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

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_newton_multi(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[Array2<f64>],
        _max_iter: usize,
        _tolerance: f64,
    ) -> Result<()> {
        Err(GAMError::InvalidParameter(
            "Newton REML optimization requires the 'blas' feature. Use Fellner-Schall or GCV instead.".to_string()
        ))
    }

    /// Optimize using REML with Fellner-Schall iteration (fREML)
    ///
    /// This is a simpler, faster alternative to Newton's method.
    /// Update formula: λ_new = λ_old × (tr(A^{-1}·S) / rank(S))
    ///
    /// Based on Wood & Fasiolo (2017) - "A generalized Fellner-Schall method"
    /// Typically converges in 3-5 iterations vs 7-10 for Newton
    ///
    /// OPTIMIZED: Uses Cholesky decomposition instead of full inverse (~3x faster)
    #[cfg(feature = "blas")]
    fn optimize_reml_fellner_schall(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        use crate::reml::compute_xtwx;
        use ndarray_linalg::{Cholesky, InverseInto, UPLO};

        let m = penalties.len();
        let p = x.ncols();
        let _n = x.nrows();

        // Pre-compute sqrt_penalties for rank computation
        let mut penalty_ranks = Vec::new();
        for penalty in penalties.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            penalty_ranks.push(sqrt_pen.ncols());
        }

        // Pre-compute X'WX (constant across iterations)
        let xtwx = compute_xtwx(x, w);

        // Work in log space for numerical stability
        let mut log_lambda: Vec<f64> = self.lambda.iter().map(|l| l.ln()).collect();

        // Pre-allocate array to reduce allocations
        let mut a = Array2::<f64>::zeros((p, p));

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!("[PROFILE] Starting Fellner-Schall optimization (Cholesky)");
        }

        for iter in 0..max_iter {
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // Compute A = X'WX + Σλᵢ·Sᵢ + ridge·I
            a.assign(&xtwx);
            for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
                a.scaled_add(*lambda, penalty);
            }

            // Add small ridge for numerical stability
            let mut max_diag: f64 = 1.0;
            for i in 0..p {
                max_diag = max_diag.max(a[[i, i]].abs());
            }
            let ridge_scale = 1e-5 * (1.0 + (m as f64).sqrt());
            let ridge = ridge_scale * max_diag;
            for i in 0..p {
                a[[i, i]] += ridge;
            }

            // Compute Cholesky decomposition: A = L·L'
            let cholesky = match a.cholesky(UPLO::Lower) {
                Ok(l) => l,
                Err(_) => {
                    // Fallback: increase ridge if Cholesky fails
                    for i in 0..p {
                        a[[i, i]] += ridge * 10.0;
                    }
                    a.cholesky(UPLO::Lower)
                        .map_err(|_| GAMError::SingularMatrix)?
                }
            };

            // Compute A^{-1} via Cholesky (faster than general inverse)
            let a_inv = match cholesky.inv_into() {
                Ok(inv) => inv,
                Err(_) => {
                    return Err(GAMError::SingularMatrix);
                }
            };

            // Fellner-Schall update for each smoothing parameter
            let mut new_log_lambda = log_lambda.clone();
            let mut max_change: f64 = 0.0;

            for i in 0..m {
                let penalty_i = &penalties[i];
                let rank_i = penalty_ranks[i] as f64;

                // Compute tr(A^{-1}·Sᵢ)
                // For block-diagonal penalties, only non-zero elements of S_i contribute
                // Use fast BLAS matrix multiply then extract diagonal
                let ainv_s = a_inv.dot(penalty_i);

                // Sum full trace (all diagonal elements)
                let mut trace = 0.0;
                for j in 0..p {
                    trace += ainv_s[[j, j]];
                }

                // Debug: check which elements are non-zero
                if std::env::var("MGCV_TRACE_DEBUG").is_ok() && iter == 0 {
                    let mut nonzero_count = 0;
                    let mut trace_nonzero = 0.0;
                    for j in 0..p {
                        if penalty_i.row(j).iter().any(|&x| x.abs() > 1e-10) {
                            trace_nonzero += ainv_s[[j, j]];
                            nonzero_count += 1;
                        }
                    }
                    eprintln!("[TRACE_DEBUG] Smooth {}: full_trace={:.6}, nonzero_trace={:.6}, nonzero_cols={}, p={}",
                             i, trace, trace_nonzero, nonzero_count, p);
                }

                // Fellner-Schall update: λ_new = λ_old × (trace / rank)
                // trace = tr(A^{-1}·S) represents effective degrees of freedom for this smooth
                // If trace < rank: EDF too low (over-smoothing) → λ too high → decrease λ
                // If trace > rank: EDF too high (under-smoothing) → λ too low → increase λ
                // In log space: log(λ_new) = log(λ_old) + log(trace / rank)

                // Use moderate damping
                let damping = 0.5;

                // Compute ratio - NO CLAMPING on trace itself!
                // trace can legitimately be < rank (over-smoothed) or > rank (under-smoothed)
                let ratio = if rank_i > 1e-10 { trace / rank_i } else { 1.0 };

                // Clamp ratio to prevent too-extreme steps per iteration
                let ratio_clamped = ratio.clamp(0.2, 5.0);  // 0.2x to 5x per iteration
                let log_ratio = ratio_clamped.ln();
                new_log_lambda[i] = log_lambda[i] + damping * log_ratio;

                // Enforce reasonable bounds on λ itself: [1e-7, 1e5]
                new_log_lambda[i] = new_log_lambda[i].max((1e-7f64).ln()).min((1e5f64).ln());

                // Track maximum change for convergence
                let change = (new_log_lambda[i] - log_lambda[i]).abs();
                max_change = max_change.max(change);

                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] FS iter {}: smooth {}: λ={:.6}, trace={:.6}, rank={}, ratio={:.3}, change={:.6}",
                             iter, i, lambdas[i], trace, rank_i, ratio, change);
                }
            }

            // Check convergence
            if max_change < tolerance {
                self.lambda = new_log_lambda.iter().map(|l| l.exp()).collect();
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Fellner-Schall converged in {} iterations", iter + 1);
                }
                return Ok(());
            }

            log_lambda = new_log_lambda;
        }

        // Update final lambda values even if not converged
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!("[PROFILE] Fellner-Schall reached max iterations ({}) without convergence", max_iter);
        }

        Ok(())
    }

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_fellner_schall(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[Array2<f64>],
        _max_iter: usize,
        _tolerance: f64,
    ) -> Result<()> {
        Err(GAMError::InvalidParameter(
            "Fellner-Schall REML optimization requires the 'blas' feature. Use GCV or coordinate descent instead.".to_string()
        ))
    }

    /// Optimize using REML with Fellner-Schall iteration in chunked mode
    ///
    /// This version processes data in chunks to avoid forming the full design matrix.
    /// Uses incremental QR decomposition and QR-based trace computation.
    ///
    /// # Arguments
    /// * `y` - Response vector (n)
    /// * `x` - Design matrix (n × p)
    /// * `w` - Weights vector (n)
    /// * `penalties` - Penalty matrices (p × p each)
    /// * `chunk_size` - Number of rows to process at a time
    /// * `max_iter` - Maximum Fellner-Schall iterations
    /// * `tolerance` - Convergence tolerance for log(λ)
    #[cfg(feature = "blas")]
    fn optimize_reml_fellner_schall_chunked(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[Array2<f64>],
        chunk_size: usize,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        let m = penalties.len();
        let p = x.ncols();
        let n = x.nrows();

        // Pre-compute penalty ranks
        let mut penalty_ranks = Vec::new();
        for penalty in penalties.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            penalty_ranks.push(sqrt_pen.ncols());
        }

        // Work in log space for numerical stability
        let mut log_lambda: Vec<f64> = self.lambda.iter().map(|l| l.ln()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!("[PROFILE] Starting chunked Fellner-Schall optimization (chunk_size={})", chunk_size);
        }

        for iter in 0..max_iter {
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // Build augmented system: [X; √λ₁·√S₁; √λ₂·√S₂; ...]
            // This gives us R such that R'R = X'WX + Σλᵢ·Sᵢ
            let mut qr = IncrementalQR::new(p);

            // Process X in chunks
            let num_chunks = (n + chunk_size - 1) / chunk_size;
            for chunk_idx in 0..num_chunks {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(n);

                let x_chunk = x.slice(ndarray::s![start..end, ..]).to_owned();
                let y_chunk = y.slice(ndarray::s![start..end]).to_owned();
                let w_chunk = w.slice(ndarray::s![start..end]).to_owned();

                qr.update_chunk(&x_chunk, &y_chunk, Some(&w_chunk))?;
            }

            // Augment with penalty terms: √λᵢ·√Sᵢ
            // penalty_sqrt returns L (p × rank) such that L·L' = S
            // We need to augment with L' scaled by √λ (rank × p rows)
            for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
                let sqrt_pen = penalty_sqrt(penalty)?;
                if sqrt_pen.ncols() > 0 {  // Only if penalty has non-zero rank
                    let scaled_sqrt_t = sqrt_pen.t().to_owned() * lambda.sqrt();

                    // Augment with zero response for penalty rows
                    let penalty_y = Array1::zeros(scaled_sqrt_t.nrows());
                    qr.update_chunk(&scaled_sqrt_t, &penalty_y, None)?;
                }
            }

            // Add small ridge for numerical stability
            let mut max_diag: f64 = 1.0;
            for i in 0..p {
                max_diag = max_diag.max(qr.r[[i, i]].abs());
            }
            let ridge_scale = 1e-5 * (1.0 + (m as f64).sqrt());
            let ridge = ridge_scale * max_diag;

            // Add ridge as diagonal augmentation
            let ridge_sqrt = Array2::from_diag(&Array1::from_elem(p, ridge.sqrt()));
            let ridge_y = Array1::zeros(p);
            qr.update_chunk(&ridge_sqrt, &ridge_y, None)?;

            // Fellner-Schall update for each smoothing parameter
            let mut new_log_lambda = log_lambda.clone();
            let mut max_change: f64 = 0.0;

            for i in 0..m {
                let penalty_i = &penalties[i];
                let rank_i = penalty_ranks[i] as f64;

                // Compute tr(A^{-1}·Sᵢ) using QR-based method
                let trace = qr.trace_ainv_s(penalty_i)?;

                // Fellner-Schall update
                let step_size = 0.5;
                let adjustment = step_size * (trace - rank_i) / rank_i;
                new_log_lambda[i] = log_lambda[i] - adjustment;

                // Track maximum change for convergence
                let change = (new_log_lambda[i] - log_lambda[i]).abs();
                max_change = max_change.max(change);

                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Chunked FS iter {}: smooth {}: λ={:.6}, trace={:.6}, rank={}, adj={:.6}, change={:.6}",
                             iter, i, lambdas[i], trace, rank_i, adjustment, change);
                }
            }

            // Check convergence
            if max_change < tolerance {
                self.lambda = new_log_lambda.iter().map(|l| l.exp()).collect();
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Chunked Fellner-Schall converged in {} iterations", iter + 1);
                }
                return Ok(());
            }

            log_lambda = new_log_lambda;
        }

        // Update final lambda values even if not converged
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!("[PROFILE] Chunked Fellner-Schall reached max iterations ({}) without convergence", max_iter);
        }

        Ok(())
    }

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_fellner_schall_chunked(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[Array2<f64>],
        _chunk_size: usize,
        _max_iter: usize,
        _tolerance: f64,
    ) -> Result<()> {
        Err(GAMError::InvalidParameter(
            "Chunked Fellner-Schall REML optimization requires the 'blas' feature. Use GCV or coordinate descent instead.".to_string()
        ))
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

    #[test]
    fn test_chunked_fellner_schall_basic() {
        use crate::penalty::compute_penalty;
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        // Small test case
        let n = 100;
        let k = 10;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Simple identity penalty
        let penalty = Array2::eye(k);
        let penalties = vec![penalty];

        let mut sp = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall
        );

        // Test with chunk size of 25
        let result = sp.optimize_reml_fellner_schall_chunked(
            &y, &x, &w, &penalties,
            25,  // chunk_size
            10,  // max_iter
            1e-4  // tolerance
        );

        assert!(result.is_ok());
        assert!(sp.lambda[0] > 0.0);
        assert!(sp.lambda[0].is_finite());
    }

    #[test]
    fn test_chunked_vs_batch_agreement() {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        // Create test data
        let n = 200;
        let k = 12;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Create a non-trivial penalty (second-order difference)
        let mut penalty = Array2::zeros((k, k));
        for i in 0..k {
            penalty[[i, i]] = 2.0;
            if i > 0 {
                penalty[[i, i-1]] = -1.0;
            }
            if i < k - 1 {
                penalty[[i, i+1]] = -1.0;
            }
        }
        let penalties = vec![penalty];

        // Optimize with batch method
        let mut sp_batch = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall
        );
        sp_batch.optimize_reml_fellner_schall(
            &y, &x, &w, &penalties, 30, 1e-6
        ).unwrap();

        // Optimize with chunked method (chunk_size = 50)
        let mut sp_chunked = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall
        );
        sp_chunked.optimize_reml_fellner_schall_chunked(
            &y, &x, &w, &penalties,
            50,   // chunk_size
            30,   // max_iter
            1e-6  // tolerance
        ).unwrap();

        // Results should be very similar (within 1% relative error)
        let relative_error = (sp_batch.lambda[0] - sp_chunked.lambda[0]).abs() / sp_batch.lambda[0];
        println!("Batch λ: {:.6}, Chunked λ: {:.6}, Relative error: {:.6}",
                 sp_batch.lambda[0], sp_chunked.lambda[0], relative_error);

        assert!(relative_error < 0.01,
                "Chunked and batch methods should agree within 1%: batch={:.6}, chunked={:.6}, error={:.6}",
                sp_batch.lambda[0], sp_chunked.lambda[0], relative_error);
    }

    #[test]
    fn test_chunked_multiple_smooths() {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        // Create test data with 2 smooths
        let n = 150;
        let k = 8;

        let x = Array2::random((n, 2*k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Two penalties - one for each smooth
        let mut penalty1 = Array2::zeros((2*k, 2*k));
        for i in 0..k {
            penalty1[[i, i]] = 1.0;
        }

        let mut penalty2 = Array2::zeros((2*k, 2*k));
        for i in k..(2*k) {
            penalty2[[i, i]] = 1.0;
        }

        let penalties = vec![penalty1, penalty2];

        let mut sp = SmoothingParameter::new_with_algorithm(
            2,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall
        );

        let result = sp.optimize_reml_fellner_schall_chunked(
            &y, &x, &w, &penalties,
            50,   // chunk_size
            20,   // max_iter
            1e-4  // tolerance
        );

        assert!(result.is_ok());
        assert_eq!(sp.lambda.len(), 2);
        assert!(sp.lambda[0] > 0.0 && sp.lambda[0].is_finite());
        assert!(sp.lambda[1] > 0.0 && sp.lambda[1].is_finite());
    }

    #[test]
    fn test_chunked_various_chunk_sizes() {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;

        let n = 100;
        let k = 10;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        let penalty = Array2::eye(k);
        let penalties = vec![penalty];

        // Test different chunk sizes
        let chunk_sizes = vec![10, 25, 50, 100, 200];  // Include sizes larger than n
        let mut results = Vec::new();

        for &chunk_size in &chunk_sizes {
            let mut sp = SmoothingParameter::new_with_algorithm(
                1,
                OptimizationMethod::REML,
                REMLAlgorithm::FellnerSchall
            );

            let result = sp.optimize_reml_fellner_schall_chunked(
                &y, &x, &w, &penalties,
                chunk_size,
                20,
                1e-6
            );

            assert!(result.is_ok(), "Failed with chunk_size={}", chunk_size);
            results.push(sp.lambda[0]);
        }

        // All results should be similar (within 5% of each other)
        let mean = results.iter().sum::<f64>() / results.len() as f64;
        for (i, &lambda) in results.iter().enumerate() {
            let relative_diff = (lambda - mean).abs() / mean;
            assert!(relative_diff < 0.05,
                    "Chunk size {} gave λ={:.6}, too far from mean={:.6} (diff={:.2}%)",
                    chunk_sizes[i], lambda, mean, relative_diff * 100.0);
        }
    }
}
