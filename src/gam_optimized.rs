//! Optimized GAM fitting with caching and improved matrix operations

#[cfg(feature = "blas")]
use crate::reml::ScaleParameterMethod;
use crate::{
    block_penalty::BlockPenalty,
    discrete::{DiscretizeConfig, DiscretizedDesign},
    gam::{SmoothTerm, GAM},
    pirls::{fit_pirls_cached, fit_pirls_discretized, fit_pirls_tdist},
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
    ///
    /// On first construction we apply mgcv's sum-to-zero identifiability
    /// constraint to each smooth (`SmoothTerm::apply_sum_to_zero_centering`),
    /// shrinking each smooth's effective basis from k to k-1, and prepend
    /// an explicit intercept column so the full design has columns
    /// `[1 | Z_1' X_1 | Z_2' X_2 | ...]` matching mgcv's lpmatrix layout.
    /// The block penalties offset by 1 to account for the unpenalized
    /// intercept.
    ///
    /// The `mgcv_exact` flag controls whether per-smooth penalty
    /// normalisation (`ma_xx / inf_norm_s`) is applied. mgcv does NOT
    /// rescale the penalty matrix; with mgcv_exact=true we skip that
    /// step so our reported λ sits in raw-Z'SZ coordinates (closer to
    /// mgcv's but still not identical because mgcv additionally
    /// diagonalises the penalty via nat.param — that's a follow-up).
    fn new(
        x: &Array2<f64>,
        smooth_terms: &mut [SmoothTerm],
        mgcv_exact: bool,
    ) -> Result<Self> {
        let cache_start = Instant::now();
        let n = x.nrows();

        // Evaluate all basis functions (this is expensive, so cache it).
        // First time through, also computes the sum-to-zero constraint Z
        // and transforms the per-smooth penalty.
        let basis_start = Instant::now();
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut covariates: Vec<Array1<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in smooth_terms.iter_mut().enumerate() {
            let x_col = x.column(i).to_owned();
            if smooth.is_random_effect {
                // Random effects: identity penalty provides identifiability —
                // no sum-to-zero or pc-anchoring needed. Leave constraint_matrix = None
                // so the full k-column basis is used as-is.
            } else if smooth.pc_value.is_some() {
                // pc-anchoring replaces sum-to-zero: enforce f(pc) = 0.
                smooth.apply_pc_anchoring()?;
            } else if mgcv_exact {
                // mgcv-exact: normalise penalty using ||X_raw||_∞²/||S_raw||_∞
                // BEFORE applying the centring Z (matches smooth.r:3766
                // ordering). Our default does it after Z, which gives a
                // different scale factor.
                smooth.apply_mgcv_normalisation_then_centring(&x_col)?;
            } else {
                smooth.apply_sum_to_zero_centering(&x_col)?;
            }
            let basis_matrix = smooth.evaluate(&x_col)?; // applies Z internally
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
        // mgcv_exact mode disables the discretized path; the
        // scatter-gather X'WX has tiny binning errors (~1e-5) that
        // shift predictions enough to break the 1e-3 byte-for-byte
        // bar. Default mode keeps discretization for perf.
        let mgcv_exact_disable_disc =
            !design_matrices.is_empty() && std::env::var("MGCV_EXACT_FIT").is_ok();
        let discretized = if n >= config.min_n_for_discretize && !mgcv_exact_disable_disc {
            Some(DiscretizedDesign::new(
                &design_matrices,
                &covariates,
                &config,
                true,
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

        // Build full design matrix [1 | smooth_1 | smooth_2 | ...] —
        // matches mgcv's lpmatrix layout. The intercept column absorbs
        // the global mean now that each smooth is sum-to-zero centered.
        let total_cols = total_basis + 1;
        let mut full_design = Array2::zeros((n, total_cols));
        full_design.column_mut(0).fill(1.0);

        let mut col_offset = 1; // first col is the intercept
        for design in &design_matrices {
            let num_cols = design.ncols();
            full_design
                .slice_mut(s![.., col_offset..col_offset + num_cols])
                .assign(design);
            col_offset += num_cols;
        }

        // Compute penalty normalizations (cache these!)
        // Block penalties live in the smooth columns; the intercept column
        // (offset 0) is unpenalized, so first smooth starts at offset 1.
        let penalty_start = Instant::now();
        let mut penalties = Vec::new();
        let mut penalty_scales = Vec::new();
        col_offset = 1;

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

            // mgcv_exact mode: no penalty rescaling — λ multiplies raw
            // Z'SZ. Default mode: rescale per smooth (mgcv-style fast
            // numerics; not the same as mgcv's actual penalty).
            let scale_factor = if mgcv_exact {
                1.0
            } else if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0
            };

            penalty_scales.push(scale_factor);

            // Build block penalty. total_size = total_cols (= 1 + Σ(k_j-1))
            // because the penalty acts on the full beta including intercept.
            let scaled_block = &smooth.penalty * scale_factor;
            let block_penalty = BlockPenalty::new(scaled_block, col_offset, total_cols);

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
    /// For `Family::TDist`, dispatches to `fit_pirls_tdist` which manages the
    /// outer σ²/df profiling loop with proper per-observation t-weights.
    fn run_pirls(
        &self,
        y: &Array1<f64>,
        lambda: &[f64],
        family: crate::pirls::Family,
        max_iter: usize,
        tolerance: f64,
        cached_xtx: Option<&Array2<f64>>,
    ) -> Result<crate::pirls::PiRLSResult> {
        // t-dist family needs a specialised fitter with σ²/df outer loop
        if let crate::pirls::Family::TDist { df, .. } = family {
            let fixed_df = if df > 0.0 { Some(df) } else { None };
            return fit_pirls_tdist(
                y,
                &self.design_matrix,
                lambda,
                &self.penalties,
                fixed_df,
                max_iter,
                tolerance,
            );
        }

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

        // mgcv_exact mode: signal PiRLS to use a tiny ridge (1e-12)
        // instead of the default 1e-5 * (1+sqrt(m)) * max_diag, which
        // perturbs β enough to shift predictions by ~1e-3. We use an
        // env var so PiRLS can read it without threading a flag
        // through every call. Cleared in a guard below.
        struct MgcvExactGuard {
            was_set: bool,
        }
        impl Drop for MgcvExactGuard {
            fn drop(&mut self) {
                if !self.was_set {
                    std::env::remove_var("MGCV_EXACT_FIT");
                }
            }
        }
        let _ridge_guard = if self.mgcv_exact {
            let was_set = std::env::var("MGCV_EXACT_FIT").is_ok();
            std::env::set_var("MGCV_EXACT_FIT", "1");
            Some(MgcvExactGuard { was_set })
        } else {
            None
        };

        // Build cache (design matrix, penalties, normalizations)
        let mut cache = FitCache::new(x, &mut self.smooth_terms, self.mgcv_exact)?;

        // Initialize smoothing parameters with chosen algorithm.
        //
        // Family-conditional defaults, measured on the parity battery:
        //
        //   - Gaussian: Newton converges to ~7-22× tighter Bar A
        //     predictions than Fellner-Schall under mgcv-style penalty
        //     normalization (issue #4 in the status doc). The compute
        //     cost is bounded — newton_max_iter is already capped per
        //     problem size below.
        //   - Non-Gaussian (binomial / poisson / Gamma): Newton's outer
        //     update can step toward unstable lambdas before IRLS has
        //     stabilized, producing huge β (~1e7) on binomial. FS's
        //     conservative single-step update converges, even if
        //     suboptimally. Until non-Gaussian Newton has trust-region
        //     globalization, FS is the safer default.
        //
        // Pass `algorithm="newton"` / `algorithm="fellner-schall"` to
        // override.
        let _ = n;
        let is_gaussian_family = matches!(self.family, crate::pirls::Family::Gaussian);
        let selected_algorithm = algorithm.unwrap_or_else(|| {
            // In mgcv_exact mode the closed-form gradient/Hessian
            // (reml_gradient_mgcv_exact_closed_form) extends to canonical-link
            // exponential families via the envelope theorem — Newton at
            // PiRLS-converged β converges to mgcv's λ to ~10% relerr on
            // binomial/poisson, vs FS landing ~30× off.
            //
            // The historical "Newton destabilizes IRLS for binomial" issue
            // (commit 2c) was fixed by the eigenvalue-ABS Hessian (4d) and
            // mgcv_exact's tighter convergence criteria (3i+).
            if is_gaussian_family || self.mgcv_exact {
                crate::smooth::REMLAlgorithm::Newton
            } else {
                crate::smooth::REMLAlgorithm::FellnerSchall
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
        // mgcv_exact: optimizer wiring DISABLED in 3h after probing.
        // FD gradient/Hessian (smooth.rs::reml_gradient_finite_diff +
        // reml_hessian_finite_diff) makes the optimizer self-consistent
        // but converges to OUR mgcv-exact-formula's minimum, which
        // still differs from mgcv's true minimum by a small λ-dependent
        // amount (residual REML formula gap, see 3e). Stage 4 went
        // 5/12 → 2/12 because the slightly-off optimum differs from
        // mgcv's by enough to break the 1e-3 prediction threshold.
        // Re-enable once the REML formula gap is closed (likely needs
        // mgcv's QR-based log|H| in nat.param-style reparameterised
        // space). Until then mgcv_exact stays at basis-only.
        let _ = (&cache.penalties, &mut smoothing_params); // silence unused
        if self.mgcv_exact {
            let mut mp: usize = 1;
            for (idx, smooth) in self.smooth_terms.iter().enumerate() {
                let nb = smooth.num_basis();
                let rank_s = crate::reml::estimate_rank_eigen(&cache.penalties[idx]);
                mp += nb.saturating_sub(rank_s);
            }
            smoothing_params.mgcv_exact_score = true;
            smoothing_params.mp = mp;
        }
        // Family-specific scale parameter:
        //   Binomial / Poisson: φ = 1 (known by construction)
        //   Gaussian / Gamma:   φ profiled from Pearson chi-squared
        // The Fellner-Schall update needs the right φ; estimating it from
        // (y - Xβ)² is wrong for non-Gaussian since Xβ=η not μ.
        smoothing_params.phi_fixed = match self.family {
            crate::pirls::Family::Binomial | crate::pirls::Family::Poisson => Some(1.0),
            crate::pirls::Family::Gaussian
            | crate::pirls::Family::Gamma
            | crate::pirls::Family::GammaLog
            | crate::pirls::Family::TDist { .. } => None,
        };
        smoothing_params.family = self.family;
        // Stash the ORIGINAL response for the IFT gradient/Hessian path. For
        // non-Gaussian, the optimizer is called with the working response z;
        // y_original carries the true y so IFT can evaluate
        // ∂D_GLM/∂β = -2 X'(y - μ)/(V·g'). Skip for Gaussian (z = y).
        if !matches!(self.family, crate::pirls::Family::Gaussian) {
            smoothing_params.y_original = Some(y.clone());
        }

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

        // Newton iteration cap. mgcv defaults to 200 with proper
        // score-scaled convergence; in our parity battery mgcv typically
        // converges in 3-10 iters but takes the full count when a
        // smoothing parameter is asymptoting to infinity. We mirror
        // mgcv: 200 outer iterations, but the user-passed max_iter is
        // also respected (so callers can shorten for snappy fits).
        let newton_max_iter = max_outer_iter.max(200);

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
            // For Gaussian (W=I, z=y), Newton converges fully in one call —
            // each score evaluation gives exact β̂(λ).
            //
            // For non-Gaussian, mgcv (gam.fit3.r:1444, 1500-1504, 1571-1576)
            // calls full inner-PIRLS at every line-search trial λ', so the
            // score is evaluated at IRLS-converged β̂(λ') rather than a
            // one-step approximation with stale (β, w, z). We thread a
            // callback through the Newton optimizer that does exactly
            // that — see `PirlsCallback` in smooth.rs and the
            // line-search refresh logic in
            // `optimize_reml_newton_multi_with_xtwx`.
            //
            // With per-trial PIRLS the previous "10-pass outer Newton"
            // workaround is folded into the line search and one Newton
            // call suffices.

            // Initial PIRLS at the starting λ (unchanged from Gaussian path)
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

            let working_response: Array1<f64> = if is_gaussian {
                y.clone()
            } else {
                let mut z = pirls_result.linear_predictor.clone();
                for i in 0..z.len() {
                    let dmu_deta =
                        self.family.d_inverse_link(pirls_result.linear_predictor[i]);
                    if dmu_deta.abs() > 1e-10 {
                        z[i] += (y[i] - pirls_result.fitted_values[i]) / dmu_deta;
                    }
                }
                z
            };

            let reml_start = Instant::now();
            let reml_xtwx = if is_gaussian {
                cached_xtx.clone()
            } else if let Some(ref disc) = cache.discretized {
                Some(disc.compute_xtwx(&weights))
            } else {
                None
            };

            // Build the per-trial-λ PIRLS-refresh callback for non-Gaussian.
            // Borrows `cache`, `y`, `family`, `max_inner_iter`, `tolerance`,
            // `xtx_ref` — all `Copy` or shared references that don't conflict
            // with the parallel `&cache.design_matrix` / `&cache.penalties`
            // borrows passed into `optimize_with_beta_xtwx_and_pirls_callback`.
            let family = self.family;
            // For trial-λ refresh inside Newton's line search, cap PIRLS at
            // a fraction of the outer cap. The trial λ is close to the
            // current λ so β_trial converges in 5-20 iters with warm start;
            // the prior code used the full max_inner_iter (typically 100)
            // and ran ~36 iters per call, hammering O(n*p²) for n≥5000
            // saturating-λ cases (#59). mgcv's gam.fit3.r:1500 uses an
            // analogous cap on the PIRLS-in-line-search step. With this
            // cap the 2d_binomial_n5000 case drops from 1358 ms to ~250 ms
            // without changing the converged λ.
            let max_inner = (max_inner_iter / 5).max(15);
            let tol = tolerance;

            // Track inner PIRLS calls + iterations for profile output
            let mut callback_pirls_calls: usize = 0;
            let mut callback_pirls_iters: usize = 0;

            let callback_result = if is_gaussian {
                // Gaussian: skip callback (frozen W=I, z=y is exact).
                let prev_lambda = smoothing_params.lambda.clone();
                let _ = prev_lambda;
                smoothing_params.optimize_with_beta_and_xtwx(
                    &working_response,
                    &cache.design_matrix,
                    &weights,
                    &cache.penalties,
                    newton_max_iter,
                    tolerance,
                    Some(&pirls_result.coefficients),
                    reml_xtwx.as_ref(),
                )
            } else {
                // Non-Gaussian: refresh PIRLS at every trial λ.
                let cache_ref = &cache;
                let y_ref = y;
                let mut callback = |trial_lambdas: &[f64]| -> Result<crate::smooth::PirlsRefresh> {
                    callback_pirls_calls += 1;
                    let res = cache_ref.run_pirls(
                        y_ref,
                        trial_lambdas,
                        family,
                        max_inner,
                        tol,
                        xtx_ref,
                    )?;
                    callback_pirls_iters += res.iterations;
                    // z = η + (y - μ)/(dμ/dη)
                    let mut z = res.linear_predictor.clone();
                    for i in 0..z.len() {
                        let dmu_deta = family.d_inverse_link(res.linear_predictor[i]);
                        if dmu_deta.abs() > 1e-10 {
                            z[i] += (y_ref[i] - res.fitted_values[i]) / dmu_deta;
                        }
                    }
                    let xtwx = if let Some(ref disc) = cache_ref.discretized {
                        disc.compute_xtwx(&res.weights)
                    } else {
                        crate::reml::compute_xtwx(&cache_ref.design_matrix, &res.weights)
                    };
                    Ok(crate::smooth::PirlsRefresh {
                        beta: res.coefficients,
                        weights: res.weights,
                        working_response: z,
                        xtwx,
                    })
                };
                smoothing_params.optimize_with_beta_xtwx_and_pirls_callback(
                    &working_response,
                    &cache.design_matrix,
                    &weights,
                    &cache.penalties,
                    newton_max_iter,
                    tolerance,
                    Some(&pirls_result.coefficients),
                    reml_xtwx.as_ref(),
                    Some(&mut callback),
                )
            };
            callback_result?;
            total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE] Per-trial PIRLS refresh: {} calls, {} total inner IRLS iters",
                    callback_pirls_calls, callback_pirls_iters
                );
            }

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
