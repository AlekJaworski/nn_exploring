//! Optimized GAM fitting with caching and improved matrix operations

#[cfg(feature = "blas")]
use crate::reml::ScaleParameterMethod;
use crate::{
    block_penalty::BlockPenalty,
    discrete::{compute_xtwx_discrete, DiscreteConfig, DiscreteDesign},
    gam::{SmoothTerm, GAM},
    pirls::{fit_pirls_cached, fit_pirls_discretized, fit_pirls_tdist},
    reml::compute_xtwx_dispatch,
    smooth::{OptimizationMethod, SmoothingParameter},
    GAMError, Result,
};
use ndarray::{s, Array1, Array2};
use std::time::Instant;

/// Threshold on n above which discretization is built when the caller
/// has opted in via `discrete_enabled=true`. Matches mgcv's discussion
/// in `?bam`: discrete=TRUE shines on "very large data sets". Smaller
/// n: the un-binned GEMM is already fast enough that scatter-gather
/// overhead doesn't pay off.
const DISCRETE_MIN_N: usize = 2000;

/// Helper struct to cache computations during GAM fitting
struct FitCache {
    /// Full design matrix (n x p) — kept for REML gradient/Hessian which need per-row access
    design_matrix: Array2<f64>,
    /// Discretized design for fast X'WX, X'Wy, eta computation.
    /// Built only when the caller opted into the discrete fast path
    /// AND `n >= DISCRETE_MIN_N`. Otherwise `None`, which makes every
    /// `compute_xtwx_dispatch(disc.as_ref(), ...)` fall through to the
    /// un-binned BLAS path (byte-identical to the pre-D4 master).
    discrete: Option<DiscreteDesign>,
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
        discrete_enabled: bool,
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

        // D4: build the discretized design when the caller opted in.
        // Gate threshold lives at this call site (not on `DiscreteConfig`),
        // per the design note: `DiscreteConfig` only carries `max_bins_1d`;
        // the "discretize only when n is big enough" decision belongs here.
        //
        // The centring mut-borrow on `smooth_terms` ended above (line 82).
        // We can now take an immutable borrow safely. `SmoothTerm` is
        // *not* Clone (boxed dyn), so we don't move it — we pass a
        // borrowed slice into `DiscreteDesign::new` which evaluates each
        // smooth's basis on its compressed bin centres internally.
        let disc_start = Instant::now();
        let _ = &covariates;
        let discrete: Option<DiscreteDesign> = if discrete_enabled && n >= DISCRETE_MIN_N {
            let cfg = DiscreteConfig::default();
            Some(DiscreteDesign::new(
                smooth_terms,
                x,
                /* has_intercept = */ true,
                &cfg,
            ))
        } else {
            None
        };
        let disc_time = disc_start.elapsed();

        if std::env::var("MGCV_PROFILE").is_ok() {
            let status = match (discrete_enabled, discrete.is_some()) {
                (false, _) => "disabled".to_string(),
                (true, false) => format!("skipped (n={} < {})", n, DISCRETE_MIN_N),
                (true, true) => {
                    let nrs: Vec<usize> = discrete
                        .as_ref()
                        .unwrap()
                        .marginals
                        .iter()
                        .map(|m| m.nr)
                        .collect();
                    format!("built (nr={:?}, n={})", nrs, n)
                }
            };
            eprintln!(
                "[PROFILE] Discretization: {} ({:.2}ms)",
                status,
                disc_time.as_secs_f64() * 1000.0,
            );
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
            discrete,
            penalties,
            penalty_scales,
            xtx: None,
        })
    }

    /// Get or compute X'X (cached for efficiency).
    /// Uses scatter-gather via discrete design when available (much faster for large n).
    fn get_xtx(&mut self) -> &Array2<f64> {
        if self.xtx.is_none() {
            let xtx_start = Instant::now();
            let xtx = if let Some(ref disc) = self.discrete {
                // X'X is the W=I (all-ones weight) case of X'WX. The
                // scatter-gather kernel handles this branch naturally.
                let ones = Array1::ones(self.design_matrix.nrows());
                compute_xtwx_discrete(disc, &ones)
            } else {
                let xt = self.design_matrix.t().to_owned();
                xt.dot(&self.design_matrix)
            };
            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE] X'X computation: {:.2}ms (discrete={})",
                    xtx_start.elapsed().as_secs_f64() * 1000.0,
                    self.discrete.is_some(),
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
        prior_weights: Option<&Array1<f64>>,
    ) -> Result<crate::pirls::PiRLSResult> {
        self.run_pirls_with_options(y, lambda, family, max_iter, tolerance, cached_xtx, false, prior_weights)
    }

    /// Variant that lets the caller signal "the outer Newton drives σ²" for
    /// TDist (`tdist_outer_sigma2 = true`). When true, the inner PIRLS uses
    /// the family's `sigma2` as a fixed input instead of profiling it via
    /// MLE iteration. This is the gam.fit5 path — required for the outer
    /// LAML Newton on log σ² to actually move the σ² value (otherwise the
    /// inner MLE would overwrite each trial value).
    fn run_pirls_with_options(
        &self,
        y: &Array1<f64>,
        lambda: &[f64],
        family: crate::pirls::Family,
        max_iter: usize,
        tolerance: f64,
        cached_xtx: Option<&Array2<f64>>,
        tdist_outer_sigma2: bool,
        prior_weights: Option<&Array1<f64>>,
    ) -> Result<crate::pirls::PiRLSResult> {
        // t-dist family needs a specialised fitter. df sentinel: 0.0 ⟹
        // PIRLS profiles df via internal 1D Brent (auto-mode, the default
        // for `family="t-dist"` with no df arg), > 0 ⟹ user-fixed (or set
        // by the outer Newton on log df when `tdist_profile = true`).
        // When `tdist_outer_sigma2` is true, σ² is also treated as fixed
        // (driven by the outer log σ² Newton block).
        if let crate::pirls::Family::TDist { df, sigma2 } = family {
            let fixed_df = if df > 0.0 { Some(df) } else { None };
            let fixed_sigma2 = if tdist_outer_sigma2 {
                Some(sigma2)
            } else {
                None
            };
            return fit_pirls_tdist(
                y,
                &self.design_matrix,
                lambda,
                &self.penalties,
                fixed_df,
                fixed_sigma2,
                max_iter,
                tolerance,
                prior_weights,
            );
        }

        // Quantile (qgam-style) needs a specialised IRLS using ELF weights.
        if let crate::pirls::Family::Quantile { tau, sigma } = family {
            if prior_weights.is_some() {
                return Err(GAMError::OptimizationFailed(
                    "weights= not yet supported for family='quantile'; \
                     supported families are: gaussian, binomial, poisson, \
                     gamma, quasibinomial, quasipoisson, t-dist (scat)"
                        .to_string(),
                ));
            }
            return crate::pirls::fit_pirls_quantile(
                y,
                &self.design_matrix,
                lambda,
                &self.penalties,
                tau,
                sigma,
                max_iter,
                tolerance,
            );
        }

        if let Some(ref disc) = self.discrete {
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
                prior_weights,
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
                prior_weights,
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

        // Snapshot per-row prior weights for this fit (mgcv's `weights=`
        // arg). Cloned out once so the (possibly None) reference can be
        // re-passed into every PIRLS callsite below — including the
        // closures that borrow `cache` mutably, where re-reading
        // `self.prior_weights` would conflict.
        let prior_weights_owned: Option<Array1<f64>> = self.prior_weights.clone();
        if let Some(ref pw) = prior_weights_owned {
            if pw.len() != y.len() {
                return Err(GAMError::DimensionMismatch(format!(
                    "prior weights length ({}) must match y length ({})",
                    pw.len(),
                    y.len()
                )));
            }
        }
        let prior_weights_ref: Option<&Array1<f64>> = prior_weights_owned.as_ref();

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
        let mut cache = FitCache::new(
            x,
            &mut self.smooth_terms,
            self.mgcv_exact,
            self.discrete_enabled,
        )?;

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
        // Quantile (qgam-style ELF) doesn't fit the closed-form REML gradient
        // — that gradient assumes Gaussian-like deviance/dispersion that
        // ELF doesn't have. Without the LAML coupling (mgcv's extended.family
        // Dd/ls hooks), Newton stalls at the initial λ. Use Fellner-Schall
        // until the LAML port lands.
        let is_quantile_family = matches!(self.family, crate::pirls::Family::Quantile { .. });
        let selected_algorithm = algorithm.unwrap_or_else(|| {
            // In mgcv_exact mode the closed-form gradient/Hessian
            // (reml_gradient_mgcv_exact_closed_form) extends to canonical-link
            // exponential families via the envelope theorem — Newton at
            // PiRLS-converged β converges to mgcv's λ to ~10% relerr on
            // binomial/poisson, vs FS landing ~30× off.
            if is_quantile_family {
                crate::smooth::REMLAlgorithm::FellnerSchall
            } else if is_gaussian_family || self.mgcv_exact {
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
            crate::pirls::Family::Binomial | crate::pirls::Family::Poisson
            | crate::pirls::Family::NegBin { .. }
            // Quantile/ELF: σ is the family parameter; φ stays at 1 by convention.
            | crate::pirls::Family::Quantile { .. } => Some(1.0),
            // QuasiPoisson/QuasiBinomial: dispersion is profiled, not fixed at 1.
            crate::pirls::Family::Gaussian
            | crate::pirls::Family::QuasiPoisson
            | crate::pirls::Family::QuasiBinomial
            | crate::pirls::Family::Gamma
            | crate::pirls::Family::GammaLog
            | crate::pirls::Family::TDist { .. }
            | crate::pirls::Family::Tweedie { .. }
            | crate::pirls::Family::InverseGaussian => None,
        };
        smoothing_params.family = self.family;
        // Profile-p for Tweedie: enable the θ Newton step in the outer loop.
        // Initial θ=0 ⟹ p=1.5 (midpoint of [1.001, 1.999]).
        smoothing_params.tweedie_profile = self.tweedie_profile;
        smoothing_params.tweedie_theta = 0.0; // start at p=1.5
                                              // Profile-θ for NegBin: enable the log(θ) Newton step in the outer loop.
                                              // Initial log(θ) = log(2.0) (θ=2 is a reasonable starting point).
        smoothing_params.negbin_profile = self.negbin_profile;
        if let crate::pirls::Family::NegBin { theta } = self.family {
            smoothing_params.negbin_log_theta = theta.ln();
        } else {
            smoothing_params.negbin_log_theta = 2.0_f64.ln();
        }
        // Profile-df for scat (TDist): outer Newton on mgcv's theta1 =
        // log(df - 2) at the ρ-loop level when caller passed no fixed df.
        // Internal Brent in `fit_pirls_tdist` is disabled — PIRLS treats
        // family.df as fixed within an inner fit; only the outer Newton
        // moves it. Initial theta1 seeded from family.df so user-fixed
        // mode (tdist_profile=false) and auto mode share state cleanly.
        smoothing_params.tdist_profile = self.tdist_profile;
        if let crate::pirls::Family::TDist { df, .. } = self.family {
            smoothing_params.tdist_log_df = (df.max(2.0_f64 + 1e-8) - 2.0).ln();
            // Seed σ² from sample variance — the gam.fit5 outer Newton on
            // log σ² then refines from there. Also overwrite the family
            // enum's σ² so the FIRST PIRLS call uses this seed (otherwise
            // the init `sigma2: 1.0` would feed PIRLS).
            let y_mean: f64 = y.iter().sum::<f64>() / (n as f64).max(1.0);
            let y_var: f64 =
                y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
            let sigma2_seed = y_var.max(1e-6);
            let log_seed = sigma2_seed.ln();
            smoothing_params.tdist_log_sigma2 = log_seed;
            smoothing_params.tdist_log_sigma2_lo = log_seed - 4.605_f64; // ÷100
            smoothing_params.tdist_log_sigma2_hi = log_seed + 4.605_f64; // ×100
            self.family = crate::pirls::Family::TDist {
                df,
                sigma2: sigma2_seed,
            };
            smoothing_params.family = self.family;
        } else {
            smoothing_params.tdist_log_df = (5.0_f64 - 2.0).ln();
            smoothing_params.tdist_log_sigma2 = 0.0_f64;
        }
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
        // With prior weights, X'WX ≠ X'X (W = diag(prior)), so the cache
        // doesn't apply — PIRLS recomputes X'WX on the fly per fit.
        let is_gaussian = matches!(self.family, crate::pirls::Family::Gaussian);
        let cached_xtx = if is_gaussian && prior_weights_ref.is_none() {
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
            let pirls_result = cache.run_pirls_with_options(
                y,
                &smoothing_params.lambda,
                self.family,
                max_inner_iter,
                tolerance,
                xtx_ref,
                self.tdist_profile,
                prior_weights_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;
            weights = pirls_result.weights.clone();

            let working_response = pirls_result.working_response.clone();

            let reml_start = Instant::now();
            let reml_xtwx = if is_gaussian {
                cached_xtx.clone()
            } else if let Some(ref disc) = cache.discrete {
                Some(compute_xtwx_discrete(disc, &weights))
            } else {
                None
            };

            // Build the per-trial-λ PIRLS-refresh callback for non-Gaussian.
            // Borrows `cache`, `y`, `max_inner_iter`, `tolerance`,
            // `xtx_ref` — all `Copy` or shared references that don't conflict
            // with the parallel `&cache.design_matrix` / `&cache.penalties`
            // borrows passed into `optimize_with_beta_xtwx_and_pirls_callback`.
            //
            // Family is shared via Arc<Mutex<Family>> so the outer Newton loop
            // can publish freshly profiled family-shape params (TDist df/σ²,
            // Tweedie p, NegBin θ) into the closure between iterations. A
            // plain `let family = self.family;` Copy capture would freeze
            // the closure's family at the starting value, defeating the
            // outer-loop df / θ / p Newton steps. Mutex (vs. Cell) only
            // because `SmoothingParameter` lives inside a `#[pyclass]` that
            // requires Send + Sync; the lock is uncontended (single-threaded
            // Newton).
            let family_cell = std::sync::Arc::new(std::sync::Mutex::new(self.family));
            smoothing_params.family_cell = Some(std::sync::Arc::clone(&family_cell));
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
                let family_cell_ref = std::sync::Arc::clone(&family_cell);
                let outer_sigma2_profile = self.tdist_profile;
                let mut callback = |trial_lambdas: &[f64]| -> Result<crate::smooth::PirlsRefresh> {
                    callback_pirls_calls += 1;
                    // Read the latest family from the shared cell — the outer
                    // Newton loop publishes fresh df/σ²/p/θ here between iters.
                    let family = *family_cell_ref.lock().expect("family_cell mutex poisoned");
                    let res = cache_ref.run_pirls_with_options(
                        y_ref,
                        trial_lambdas,
                        family,
                        max_inner,
                        tol,
                        xtx_ref,
                        outer_sigma2_profile,
                        prior_weights_ref,
                    )?;
                    callback_pirls_iters += res.iterations;
                    let xtwx = compute_xtwx_dispatch(
                        cache_ref.discrete.as_ref(),
                        &cache_ref.design_matrix,
                        &res.weights,
                    );
                    let sigma2 = res.sigma2;
                    let df = res.df;
                    Ok(crate::smooth::PirlsRefresh {
                        beta: res.coefficients,
                        weights: res.weights,
                        working_response: res.working_response,
                        xtwx,
                        sigma2,
                        df,
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
            // Drop the shared family cell now that the Newton call returned —
            // future re-uses of `smoothing_params` would otherwise carry a
            // stale Rc pointing at this fit's cell.
            smoothing_params.family_cell = None;
            total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE] Per-trial PIRLS refresh: {} calls, {} total inner IRLS iters",
                    callback_pirls_calls, callback_pirls_iters
                );
            }

            // Propagate the converged family params (Tweedie p, NegBin θ,
            // scat df) back to self.family so the final PIRLS uses them.
            // For TDist always sync (σ² is profiled inside fit_pirls_tdist
            // even when df is user-fixed); the smoothing_params.family
            // refresh path at smooth.rs:912 keeps it current.
            if self.tweedie_profile
                || self.negbin_profile
                || smoothing_params.tdist_profile
                || matches!(self.family, crate::pirls::Family::TDist { .. })
            {
                self.family = smoothing_params.family;
            }

            // Step 3: Final PiRLS with optimal lambda (uses discretized scatter-gather)
            let pirls_start = Instant::now();
            let final_result = cache.run_pirls_with_options(
                y,
                &smoothing_params.lambda,
                self.family,
                max_inner_iter,
                tolerance,
                xtx_ref,
                self.tdist_profile,
                prior_weights_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            // Compute REML/LAML score at converged Newton state — used by
            // wrapper-level θ-profilers / parity probes for true-likelihood
            // extended families (scat, Tweedie, NegBin).
            let final_xtwx = compute_xtwx_dispatch(
                cache.discrete.as_ref(),
                &cache.design_matrix,
                &final_result.weights,
            );
            // Working response for the final REML score evaluation:
            //   - Gaussian: z = y (exact).
            //   - TDist gam.fit5 profile path: z = η − dmu/dmu2 (Newton working
            //     response). The optimizer used this throughout via the PIRLS
            //     callback; using it here ensures the reported REML matches the
            //     optimizer's converged value rather than an inconsistent re-
            //     evaluation with z = y (which would add ~8 units of spurious gap).
            //   - Other non-Gaussian: z = η + (y − μ)/(dμ/dη).
            let z_score: Array1<f64> = if matches!(self.family, crate::pirls::Family::Gaussian) {
                y.clone()
            } else if self.tdist_profile {
                if let crate::pirls::Family::TDist { df, sigma2 } = self.family {
                    final_result
                        .linear_predictor
                        .iter()
                        .enumerate()
                        .map(|(i, &eta)| {
                            let r = y[i] - eta;
                            let s2 = sigma2.max(1e-300);
                            let denom = df * s2 + r * r;
                            let dmu = -2.0 * (df + 1.0) * r / denom;
                            let obs_dmu2 = 2.0 * (df + 1.0) * (df * s2 - r * r) / (denom * denom);
                            let exp_dmu2 = 2.0 * (df + 1.0) / (s2 * (df + 3.0));
                            let dmu2 = if obs_dmu2.is_finite() && obs_dmu2 > 1e-12 {
                                obs_dmu2
                            } else {
                                exp_dmu2.max(1e-12)
                            };
                            eta - dmu / dmu2
                        })
                        .collect()
                } else {
                    y.clone()
                }
            } else {
                final_result.working_response.clone()
            };
            smoothing_params.last_score = crate::reml::reml_criterion_multi_cached_mgcv_exact(
                &z_score,
                &cache.design_matrix,
                &final_result.weights,
                &smoothing_params.lambda,
                &cache.penalties,
                Some(&final_xtwx),
                None,
                smoothing_params.mp,
                self.family,
                smoothing_params.y_original.as_ref(),
            )
            .ok();

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
                eprintln!("[PROFILE] REML score: {:?}", smoothing_params.last_score);
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
                    prior_weights_ref,
                )?;
                total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                weights = pirls_result.weights.clone();

                // Update smoothing parameters (one FS step)
                // Pass cached X'WX from discretized design
                let reml_start = Instant::now();
                let old_lambda = smoothing_params.lambda.clone();

                let fs_xtwx = if is_gaussian {
                    cached_xtx.clone()
                } else if let Some(ref disc) = cache.discrete {
                    Some(compute_xtwx_discrete(disc, &weights))
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
                        prior_weights_ref,
                    )?;
                    total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                    // Compute REML/LAML score at converged state — read by
                    // wrapper-level σ profilers to drive Brent on σ.
                    let final_xtwx = compute_xtwx_dispatch(
                        cache.discrete.as_ref(),
                        &cache.design_matrix,
                        &final_result.weights,
                    );
                    smoothing_params.last_score =
                        crate::reml::reml_criterion_multi_cached_mgcv_exact(
                            y,
                            &cache.design_matrix,
                            &final_result.weights,
                            &smoothing_params.lambda,
                            &cache.penalties,
                            Some(&final_xtwx),
                            None,
                            smoothing_params.mp,
                            self.family,
                            smoothing_params.y_original.as_ref(),
                        )
                        .ok();

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                        eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
                        eprintln!("[PROFILE] REML score: {:?}", smoothing_params.last_score);
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
                prior_weights_ref,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            // Score even on the no-converge fall-through so wrapper-level
            // σ profilers (Brent on σ) always have a number to compare.
            let final_xtwx = compute_xtwx_dispatch(
                cache.discrete.as_ref(),
                &cache.design_matrix,
                &final_result.weights,
            );
            smoothing_params.last_score = crate::reml::reml_criterion_multi_cached_mgcv_exact(
                y,
                &cache.design_matrix,
                &final_result.weights,
                &smoothing_params.lambda,
                &cache.penalties,
                Some(&final_xtwx),
                None,
                smoothing_params.mp,
                self.family,
                smoothing_params.y_original.as_ref(),
            )
            .ok();

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
                eprintln!("[PROFILE] REML score: {:?}", smoothing_params.last_score);
            }

            self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
        }

        Ok(())
    }
}
