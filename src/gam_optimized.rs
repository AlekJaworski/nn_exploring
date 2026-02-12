//! Optimized GAM fitting with caching and improved matrix operations

#[cfg(feature = "blas")]
use crate::reml::ScaleParameterMethod;
use crate::{
    block_penalty::BlockPenalty,
    discrete::{DiscretizeConfig, DiscretizedDesign},
    gam::{SmoothTerm, GAM},
    pirls::{fit_pirls_cached, fit_pirls_discretized},
    smooth::{OptimizationMethod, SmoothingParameter},
    GAMError, Result,
};
use ndarray::{s, Array1, Array2};
use std::time::Instant;

/// Helper struct to cache computations during GAM fitting
struct FitCache {
    /// Full design matrix (n x p) — kept for REML gradient/Hessian which need per-row access
    design_matrix: Array2<f64>,
    /// Discretized design for fast X'WX, X'Wy, eta computation
    discretized: Option<DiscretizedDesign>,
    /// Penalty matrices (one per smooth, block-diagonal representation)
    penalties: Vec<BlockPenalty>,
    /// Penalty scale factors (one per smooth)
    penalty_scales: Vec<f64>,
    /// X'X matrix (cached for reuse)
    xtx: Option<Array2<f64>>,
}

impl FitCache {
    /// Build cache from data and smooth terms
    fn new(x: &Array2<f64>, smooth_terms: &[SmoothTerm]) -> Result<Self> {
        let cache_start = Instant::now();
        let n = x.nrows();

        // Evaluate all basis functions (this is expensive, so cache it)
        let basis_start = Instant::now();
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut covariates: Vec<Array1<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
            covariates.push(x_col);
        }
        let basis_time = basis_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Basis evaluation: {:.2}ms",
                basis_time.as_secs_f64() * 1000.0
            );
        }

        // Build discretized design for efficient scatter-gather operations.
        // This stores only unique/binned basis rows per term (m << n) plus
        // an index array, enabling O(n*k + m*k^2) X'WX instead of O(n*k^2).
        //
        // Skip discretization for small n where overhead isn't worth it.
        // At n=1000, compression is poor (ratio ~1.0x) and BLAS is faster.
        let disc_start = Instant::now();
        let config = DiscretizeConfig {
            max_unique_1d: 1000,
            min_n_for_discretize: 2000, // Only discretize for n >= 2000
        };
        let discretized = if n >= config.min_n_for_discretize {
            Some(DiscretizedDesign::new(
                &design_matrices,
                &covariates,
                &config,
            ))
        } else {
            None
        };
        let disc_time = disc_start.elapsed();

        if std::env::var("MGCV_PROFILE").is_ok() {
            if let Some(ref disc) = discretized {
                let total_compressed: usize = disc.terms.iter().map(|t| t.num_compressed()).sum();
                let total_full: usize = disc
                    .terms
                    .iter()
                    .map(|t| t.num_observations() * t.num_basis)
                    .sum();
                eprintln!(
                    "[PROFILE] Discretization: {:.2}ms (compressed {} -> {} entries, {:.1}x)",
                    disc_time.as_secs_f64() * 1000.0,
                    total_full,
                    total_compressed * disc.terms.iter().map(|t| t.num_basis).max().unwrap_or(1),
                    disc.terms
                        .iter()
                        .map(|t| t.compression_ratio())
                        .sum::<f64>()
                        / disc.terms.len() as f64,
                );
            } else {
                eprintln!(
                    "[PROFILE] Discretization: skipped (n={} < {})",
                    n, config.min_n_for_discretize
                );
            }
        }

        // Build full design matrix using efficient slicing (not loops!)
        // We still keep this for REML gradient/Hessian which need per-row X access
        let mut full_design = Array2::zeros((n, total_basis));
        let mut col_offset = 0;

        for design in &design_matrices {
            let num_cols = design.ncols();
            // Use ndarray slicing - much faster than element-by-element
            full_design
                .slice_mut(s![.., col_offset..col_offset + num_cols])
                .assign(design);
            col_offset += num_cols;
        }

        // Compute penalty normalizations (cache these!)
        let penalty_start = Instant::now();
        let mut penalties = Vec::new();
        let mut penalty_scales = Vec::new();
        col_offset = 0;

        for (idx, smooth) in smooth_terms.iter().enumerate() {
            let num_basis = smooth.num_basis();
            let design = &design_matrices[idx];

            // Compute infinity norms (for mgcv-style normalization)
            let inf_norm_x = design
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let ma_xx = inf_norm_x * inf_norm_x;

            let inf_norm_s = (0..num_basis)
                .map(|i| {
                    (0..num_basis)
                        .map(|j| smooth.penalty[[i, j]].abs())
                        .sum::<f64>()
                })
                .fold(0.0f64, f64::max);

            let scale_factor = if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0
            };

            penalty_scales.push(scale_factor);

            // Build block penalty (only stores the k×k non-zero block, not the full p×p matrix)
            let scaled_block = &smooth.penalty * scale_factor;
            let block_penalty = BlockPenalty::new(scaled_block, col_offset, total_basis);

            penalties.push(block_penalty);
            col_offset += num_basis;
        }
        let penalty_time = penalty_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Penalty computation: {:.2}ms",
                penalty_time.as_secs_f64() * 1000.0
            );
        }

        let cache_time = cache_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Total cache build: {:.2}ms",
                cache_time.as_secs_f64() * 1000.0
            );
        }

        Ok(FitCache {
            design_matrix: full_design,
            discretized,
            penalties,
            penalty_scales,
            xtx: None,
        })
    }

    /// Get or compute X'X (cached for efficiency).
    /// Uses scatter-gather via discretized design when available (much faster for large n).
    fn get_xtx(&mut self) -> &Array2<f64> {
        if self.xtx.is_none() {
            let xtx_start = Instant::now();
            let xtx = if let Some(ref disc) = self.discretized {
                disc.compute_xtx()
            } else {
                let xt = self.design_matrix.t().to_owned();
                xt.dot(&self.design_matrix)
            };
            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE] X'X computation: {:.2}ms (discretized={})",
                    xtx_start.elapsed().as_secs_f64() * 1000.0,
                    self.discretized.is_some(),
                );
            }
            self.xtx = Some(xtx);
        }
        self.xtx.as_ref().unwrap()
    }

    /// Run PiRLS using the discretized path when available, falling back to cached.
    fn run_pirls(
        &self,
        y: &Array1<f64>,
        lambda: &[f64],
        family: crate::pirls::Family,
        max_iter: usize,
        tolerance: f64,
        cached_xtx: Option<&Array2<f64>>,
    ) -> Result<crate::pirls::PiRLSResult> {
        if let Some(ref disc) = self.discretized {
            fit_pirls_discretized(
                y,
                &self.design_matrix,
                lambda,
                &self.penalties,
                family,
                max_iter,
                tolerance,
                disc,
                cached_xtx,
            )
        } else {
            fit_pirls_cached(
                y,
                &self.design_matrix,
                lambda,
                &self.penalties,
                family,
                max_iter,
                tolerance,
                cached_xtx,
            )
        }
    }
}

/// Initialize lambda with smart heuristic based on data
fn initialize_lambda_smart(y: &Array1<f64>, x: &Array2<f64>, penalty: &BlockPenalty) -> f64 {
    // Use a heuristic based on the ratio of signal variance to penalty norm
    let y_var = {
        let y_mean = y.sum() / y.len() as f64;
        y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / y.len() as f64
    };

    // Penalty norm (Frobenius) - only from the non-zero block
    let penalty_norm = penalty.block.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Design matrix norm
    let x_norm = x.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Heuristic: lambda ~ y_var / (x_norm^2 / n) * penalty_norm
    let n = y.len() as f64;
    let lambda_init = (y_var * penalty_norm * n) / (x_norm * x_norm + 1e-10);

    // Clamp to reasonable range
    lambda_init.max(1e-6).min(1e6)
}

impl GAM {
    /// Optimized GAM fitting with caching and improved convergence
    ///
    /// Improvements over standard fit():
    /// - Caches design matrix and penalty computations
    /// - Uses ndarray slicing instead of loops for matrix construction
    /// - Better lambda initialization
    /// - Adaptive tolerance for early stopping
    /// - Caches X'X computation
    #[cfg(feature = "blas")]
    pub fn fit_optimized(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        self.fit_optimized_with_scale_method(
            x,
            y,
            opt_method,
            max_outer_iter,
            max_inner_iter,
            tolerance,
            crate::reml::ScaleParameterMethod::EDF,
        )
    }

    #[cfg(feature = "blas")]
    pub fn fit_optimized_with_scale_method(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
        scale_method: crate::reml::ScaleParameterMethod,
    ) -> Result<()> {
        self.fit_optimized_full(
            x,
            y,
            opt_method,
            max_outer_iter,
            max_inner_iter,
            tolerance,
            scale_method,
            None,
        )
    }

    #[cfg(feature = "blas")]
    pub fn fit_optimized_full(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
        scale_method: crate::reml::ScaleParameterMethod,
        algorithm: Option<crate::smooth::REMLAlgorithm>,
    ) -> Result<()> {
        let n = y.len();

        if x.nrows() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                n
            )));
        }

        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(format!(
                "X has {} columns but model has {} smooth terms",
                x.ncols(),
                self.smooth_terms.len()
            )));
        }

        // Build cache (design matrix, penalties, normalizations)
        let mut cache = FitCache::new(x, &self.smooth_terms)?;

        // Initialize smoothing parameters with chosen algorithm
        // Auto-select Fellner-Schall for small n (< 2000) where Newton's expensive
        // gradient/Hessian computations don't pay off. FS is much faster per iteration.
        let selected_algorithm = algorithm.unwrap_or_else(|| {
            if n < 2000 {
                crate::smooth::REMLAlgorithm::FellnerSchall
            } else {
                crate::smooth::REMLAlgorithm::Newton
            }
        });

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Selected algorithm: {:?} (n={})",
                selected_algorithm, n
            );
        }

        let mut smoothing_params = SmoothingParameter::new_with_algorithm(
            self.smooth_terms.len(),
            opt_method,
            selected_algorithm,
        );
        smoothing_params.scale_method = scale_method;

        // Smart initialization for lambda
        if !cache.penalties.is_empty() {
            let init_lambda = initialize_lambda_smart(y, x, &cache.penalties[0]);
            for lambda in &mut smoothing_params.lambda {
                *lambda = init_lambda;
            }
        }

        let mut weights = Array1::ones(n);

        let mut total_pirls_time = 0.0;
        let mut total_reml_time = 0.0;

        // Adaptive iteration count for Newton: use more iterations for larger/more complex problems
        let num_basis: usize = cache.design_matrix.ncols();
        let newton_max_iter = if num_basis >= n {
            50 // More iterations for overparameterized case
        } else if num_basis >= n / 2 {
            30 // Moderate iterations for k close to n
        } else {
            10 // Standard for well-posed problems
        };

        let is_newton = smoothing_params.reml_algorithm == crate::smooth::REMLAlgorithm::Newton;

        // For Gaussian family, pre-compute X'X once and reuse for all PiRLS calls.
        // Gaussian has constant weights (w=1), so X'WX = X'X never changes.
        let is_gaussian = matches!(self.family, crate::pirls::Family::Gaussian);
        let cached_xtx = if is_gaussian {
            Some(cache.get_xtx().clone())
        } else {
            None
        };
        let xtx_ref = cached_xtx.as_ref();

        if is_newton {
            // Newton converges fully in a single call — no outer loop needed.
            // For Gaussian family the weights are constant, so PiRLS converges in 1 step
            // and Newton finds optimal lambda internally. Running a second outer iteration
            // would just re-derive the same lambda (wasting ~50% of REML time).
            //
            // Flow: PiRLS(init_lambda) → Newton(converge) → PiRLS(optimal_lambda) → done

            // Step 1: PiRLS with initial lambda (uses discretized scatter-gather)
            let pirls_start = Instant::now();
            let pirls_result = cache.run_pirls(
                y,
                &smoothing_params.lambda,
                self.family,
                max_inner_iter,
                tolerance,
                xtx_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;
            weights = pirls_result.weights.clone();

            // Step 2: Newton optimization to convergence
            // Pass cached X'WX from discretized design to skip O(n*p^2) in REML
            let reml_start = Instant::now();
            let reml_xtwx = if is_gaussian {
                // For Gaussian, X'WX = X'X (w=1), already cached
                cached_xtx.clone()
            } else if let Some(ref disc) = cache.discretized {
                Some(disc.compute_xtwx(&weights))
            } else {
                None
            };
            smoothing_params.optimize_with_beta_and_xtwx(
                y,
                &cache.design_matrix,
                &weights,
                &cache.penalties,
                newton_max_iter,
                tolerance,
                Some(&pirls_result.coefficients),
                reml_xtwx.as_ref(),
            )?;
            total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

            // Step 3: Final PiRLS with optimal lambda (uses discretized scatter-gather)
            let pirls_start = Instant::now();
            let final_result = cache.run_pirls(
                y,
                &smoothing_params.lambda,
                self.family,
                max_inner_iter,
                tolerance,
                xtx_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
            }

            self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
        } else {
            // Fellner-Schall: outer loop required — FS does one update step per call,
            // so we iterate PiRLS + FS until lambda converges.
            for outer_iter in 0..max_outer_iter {
                // PiRLS with current smoothing parameters (uses discretized scatter-gather)
                let pirls_start = Instant::now();
                let pirls_result = cache.run_pirls(
                    y,
                    &smoothing_params.lambda,
                    self.family,
                    max_inner_iter,
                    tolerance,
                    xtx_ref,
                )?;
                total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                weights = pirls_result.weights.clone();

                // Update smoothing parameters (one FS step)
                // Pass cached X'WX from discretized design
                let reml_start = Instant::now();
                let old_lambda = smoothing_params.lambda.clone();

                let fs_xtwx = if is_gaussian {
                    cached_xtx.clone()
                } else if let Some(ref disc) = cache.discretized {
                    Some(disc.compute_xtwx(&weights))
                } else {
                    None
                };
                smoothing_params.optimize_with_beta_and_xtwx(
                    y,
                    &cache.design_matrix,
                    &weights,
                    &cache.penalties,
                    newton_max_iter,
                    tolerance,
                    Some(&pirls_result.coefficients),
                    fs_xtwx.as_ref(),
                )?;
                total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

                // Check convergence
                let max_lambda_change = old_lambda
                    .iter()
                    .zip(smoothing_params.lambda.iter())
                    .map(|(old, new)| (old.ln() - new.ln()).abs())
                    .fold(0.0f64, f64::max);

                let adaptive_tol = if outer_iter > 3 {
                    tolerance * 2.0
                } else {
                    tolerance
                };

                if max_lambda_change < adaptive_tol {
                    // Do final fit
                    let pirls_start = Instant::now();
                    let final_result = cache.run_pirls(
                        y,
                        &smoothing_params.lambda,
                        self.family,
                        max_inner_iter,
                        tolerance,
                        xtx_ref,
                    )?;
                    total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                        eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
                    }

                    self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
                    return Ok(());
                }
            }

            // Reached max iterations - use current fit
            let pirls_start = Instant::now();
            let final_result = cache.run_pirls(
                y,
                &smoothing_params.lambda,
                self.family,
                max_inner_iter,
                tolerance,
                xtx_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
            }

            self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
        }

        Ok(())
    }
}
