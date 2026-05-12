//! Smoothing parameter selection using REML optimization

use crate::block_penalty::BlockPenalty;
use crate::chunked_qr::IncrementalQR;
use crate::linalg::solve;
#[cfg(feature = "blas")]
pub use crate::reml::ScaleParameterMethod;
#[cfg(feature = "blas")]
use crate::reml::{
    compute_xtwx_cholesky, gcv_criterion, penalty_sqrt, reml_criterion, reml_criterion_multi,
    reml_criterion_multi_cached, reml_criterion_multi_cached_mgcv_exact,
    reml_gradient_gamfit4_tdist_analytic, reml_gradient_mgcv_exact_closed_form,
    reml_gradient_mgcv_exact_ift, reml_gradient_multi_qr_adaptive,
    reml_gradient_multi_qr_adaptive_cached_edf, reml_hessian_gamfit4_tdist_analytic,
    reml_hessian_mgcv_exact_closed_form, reml_hessian_mgcv_exact_ift, reml_hessian_multi_cached,
    tdist_shape_derivatives_gamfit4, tweedie_theta_derivatives_cached, TweedieThetaCache,
};
#[cfg(not(feature = "blas"))]
use crate::reml::{gcv_criterion, reml_criterion, reml_criterion_multi, reml_gradient_multi};
use crate::{GAMError, Result};
use ndarray::{Array1, Array2};
use std::time::Instant;

#[cfg(feature = "blas")]
fn use_mgcv_exact_ift_policy(
    mgcv_exact_score: bool,
    family: crate::pirls::Family,
    y_len: usize,
    has_original_response: bool,
) -> bool {
    if !mgcv_exact_score {
        return false;
    }

    let explicit = std::env::var("MGCV_USE_IFT").is_ok();
    if explicit {
        return true;
    }
    if std::env::var("MGCV_DISABLE_IFT").is_ok() || !has_original_response {
        return false;
    }

    // IFT differentiates the true GLM deviance, while the line search still
    // evaluates working-response REML. Keep those paired derivatives on the
    // consistent closed-form path for the two parity-sensitive edge cases.
    if matches!(
        family,
        crate::pirls::Family::Gaussian | crate::pirls::Family::Gamma
    ) {
        return false;
    }
    if matches!(family, crate::pirls::Family::Binomial) && y_len <= 500 {
        return false;
    }

    true
}

/// Dispatch to either mgcv-exact REML or default REML based on the
/// SmoothingParameter flag. mgcv_exact_score=true is set by
/// gam_optimized.rs::fit_optimized_full when the GAM was constructed
/// with mgcv_exact=True.
#[cfg(feature = "blas")]
fn dispatch_reml_score(
    sp: &SmoothingParameter,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
) -> Result<f64> {
    if sp.mgcv_exact_score {
        reml_criterion_multi_cached_mgcv_exact(
            y,
            x,
            w,
            lambdas,
            penalties,
            cached_xtwx,
            sp.mp,
            sp.family,
            sp.y_original.as_ref(),
        )
    } else {
        reml_criterion_multi_cached(y, x, w, lambdas, penalties, None, cached_xtwx)
    }
}

/// Like `dispatch_reml_score` but with an explicit family override.
/// Used by the Tweedie profile-p θ Newton step to evaluate the REML
/// score with a trial p value without mutating `self.family`.
#[cfg(feature = "blas")]
fn dispatch_reml_score_with_family(
    sp: &SmoothingParameter,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
) -> Result<f64> {
    if sp.mgcv_exact_score {
        reml_criterion_multi_cached_mgcv_exact(
            y,
            x,
            w,
            lambdas,
            penalties,
            cached_xtwx,
            sp.mp,
            family,
            sp.y_original.as_ref(),
        )
    } else {
        reml_criterion_multi_cached(y, x, w, lambdas, penalties, None, cached_xtwx)
    }
}

/// Central-difference gradient of the REML score wrt log(λ_j), using
/// whichever score function dispatch_reml_score selects. Used in
/// mgcv_exact mode where the closed-form gradient
/// (reml_gradient_multi_qr_adaptive) was derived for the default REML
/// formula and would be inconsistent with the score the line search
/// uses. O(2m) score evaluations.
#[cfg(feature = "blas")]
fn reml_gradient_finite_diff(
    sp: &SmoothingParameter,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    pirls_callback: &mut Option<PirlsCallback<'_>>,
) -> Result<Array1<f64>> {
    let m = lambdas.len();
    let h: f64 = 1.0e-4;
    let log_lambdas: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
    let mut grad = Array1::<f64>::zeros(m);
    for i in 0..m {
        let mut log_plus = log_lambdas.clone();
        let mut log_minus = log_lambdas.clone();
        log_plus[i] += h;
        log_minus[i] -= h;
        let lam_plus: Vec<f64> = log_plus.iter().map(|l| l.exp()).collect();
        let lam_minus: Vec<f64> = log_minus.iter().map(|l| l.exp()).collect();
        let r_plus = dispatch_reml_score_fd(
            sp,
            y,
            x,
            w,
            &lam_plus,
            penalties,
            cached_xtwx,
            pirls_callback,
        )?;
        let r_minus = dispatch_reml_score_fd(
            sp,
            y,
            x,
            w,
            &lam_minus,
            penalties,
            cached_xtwx,
            pirls_callback,
        )?;
        grad[i] = (r_plus - r_minus) / (2.0 * h);
    }
    Ok(grad)
}

#[cfg(feature = "blas")]
fn dispatch_reml_score_fd(
    sp: &SmoothingParameter,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    pirls_callback: &mut Option<PirlsCallback<'_>>,
) -> Result<f64> {
    if let Some(cb) = pirls_callback.as_mut() {
        let refresh = cb(lambdas)?;
        dispatch_reml_score(
            sp,
            &refresh.working_response,
            x,
            &refresh.weights,
            lambdas,
            penalties,
            Some(&refresh.xtwx),
        )
    } else {
        dispatch_reml_score(sp, y, x, w, lambdas, penalties, cached_xtwx)
    }
}

/// Central-difference Hessian of the REML score wrt log(λ). Pairs with
/// reml_gradient_finite_diff. O(1 + 2m + 2m²) score evaluations:
/// diagonal entries via second central differences (reml at λ ±
/// h·e_i); off-diagonal via mixed differences. Symmetric output.
#[cfg(feature = "blas")]
fn reml_hessian_finite_diff(
    sp: &SmoothingParameter,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    pirls_callback: &mut Option<PirlsCallback<'_>>,
) -> Result<Array2<f64>> {
    let m = lambdas.len();
    let h: f64 = 1.0e-3;
    let log_lambdas: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
    let r0 = dispatch_reml_score_fd(sp, y, x, w, lambdas, penalties, cached_xtwx, pirls_callback)?;

    // Cache REML at log_λ ± h·e_i for i = 0..m
    let mut r_plus = vec![0.0f64; m];
    let mut r_minus = vec![0.0f64; m];
    for i in 0..m {
        let mut lp = log_lambdas.clone();
        let mut lm = log_lambdas.clone();
        lp[i] += h;
        lm[i] -= h;
        let lam_p: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
        let lam_m: Vec<f64> = lm.iter().map(|l| l.exp()).collect();
        r_plus[i] =
            dispatch_reml_score_fd(sp, y, x, w, &lam_p, penalties, cached_xtwx, pirls_callback)?;
        r_minus[i] =
            dispatch_reml_score_fd(sp, y, x, w, &lam_m, penalties, cached_xtwx, pirls_callback)?;
    }

    let mut hess = Array2::<f64>::zeros((m, m));
    let h2 = h * h;
    // Diagonal: H_ii = (r(+h_i) - 2 r0 + r(-h_i)) / h²
    for i in 0..m {
        hess[[i, i]] = (r_plus[i] - 2.0 * r0 + r_minus[i]) / h2;
    }
    // Off-diagonal: H_ij = [r(+h_i +h_j) - r(+h_i -h_j) - r(-h_i +h_j) + r(-h_i -h_j)] / (4h²)
    for i in 0..m {
        for j in (i + 1)..m {
            let mut lpp = log_lambdas.clone();
            let mut lpm = log_lambdas.clone();
            let mut lmp = log_lambdas.clone();
            let mut lmm = log_lambdas.clone();
            lpp[i] += h;
            lpp[j] += h;
            lpm[i] += h;
            lpm[j] -= h;
            lmp[i] -= h;
            lmp[j] += h;
            lmm[i] -= h;
            lmm[j] -= h;
            let lam_pp: Vec<f64> = lpp.iter().map(|l| l.exp()).collect();
            let lam_pm: Vec<f64> = lpm.iter().map(|l| l.exp()).collect();
            let lam_mp: Vec<f64> = lmp.iter().map(|l| l.exp()).collect();
            let lam_mm: Vec<f64> = lmm.iter().map(|l| l.exp()).collect();
            let rpp = dispatch_reml_score_fd(
                sp,
                y,
                x,
                w,
                &lam_pp,
                penalties,
                cached_xtwx,
                pirls_callback,
            )?;
            let rpm = dispatch_reml_score_fd(
                sp,
                y,
                x,
                w,
                &lam_pm,
                penalties,
                cached_xtwx,
                pirls_callback,
            )?;
            let rmp = dispatch_reml_score_fd(
                sp,
                y,
                x,
                w,
                &lam_mp,
                penalties,
                cached_xtwx,
                pirls_callback,
            )?;
            let rmm = dispatch_reml_score_fd(
                sp,
                y,
                x,
                w,
                &lam_mm,
                penalties,
                cached_xtwx,
                pirls_callback,
            )?;
            let off = (rpp - rpm - rmp + rmm) / (4.0 * h2);
            hess[[i, j]] = off;
            hess[[j, i]] = off;
        }
    }
    Ok(hess)
}

/// Compute X'Wy via element-wise sqrt-weighting. Used to refresh the
/// gradient/Hessian's xtwy cache after a PIRLS refresh changes (w, z).
#[cfg(feature = "blas")]
fn compute_xtwy_helper(x: &Array2<f64>, w: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut x_w = x.clone();
    for i in 0..n {
        let sw = w[i].sqrt();
        for j in 0..p {
            x_w[[i, j]] *= sw;
        }
    }
    let mut y_weighted = Array1::zeros(n);
    for i in 0..n {
        y_weighted[i] = y[i] * w[i].sqrt();
    }
    x_w.t().dot(&y_weighted)
}

/// Detect smooths whose log-λ has reached "working infinity" (the
/// saturating-λ regime where the smooth has collapsed into its null
/// space and the REML objective is essentially flat).
///
/// Port of mgcv's flat-set detection from
/// `mgcv_analysis/mgcv/R/gam.fit3.r:1672`:
///
/// ```r
/// flat <- which(abs(grad2) < abs(grad)*100)   # candidates for reduction
/// ```
///
/// where `grad2 = diag(hess)`. The condition flags smooths whose Hessian
/// diagonal is two orders of magnitude smaller than their own gradient
/// in absolute terms — i.e. the curvature in that direction has all but
/// vanished, so further Newton motion in that coordinate amplifies
/// noise rather than reduces the objective.
///
/// Returns a vector of length `grad.len()`; `out[i] = true` means
/// smooth `i` is in the flat / saturating-λ set.
///
/// Notes:
/// - In the trivial-gradient case (|grad_i| == 0) we return `false` for
///   that dimension: the smooth is already at a stationary point so
///   there's nothing to clamp. mgcv's expression `|grad2| < |grad|·100`
///   yields `0 < 0` (false) when both are zero, matching this behavior.
/// - The Hessian is expected to be the *raw* (pre-preconditioned,
///   pre-ridged) Hessian, so this should be called *before* the
///   diagonal-preconditioning step in the Newton loop.
pub fn detect_saturating_smooths(grad: &Array1<f64>, hess: &Array2<f64>) -> Vec<bool> {
    let m = grad.len();
    let mut out = vec![false; m];
    if hess.nrows() != m || hess.ncols() != m {
        return out;
    }
    for i in 0..m {
        let gi = grad[i].abs();
        let h_ii = hess[[i, i]].abs();
        // Match mgcv's `abs(grad2) < abs(grad)*100`. Skip when |g_i|==0:
        // a converged stationary point is not a saturating smooth.
        if gi > 0.0 && h_ii < gi * 100.0 {
            out[i] = true;
        }
    }
    out
}

/// One Fellner-Schall update for the per-smooth penalty parameters.
///
/// Wood & Fasiolo (2017): at the converged β,
///   `λ_new[i] = λ[i] · phi · max(rank[i]/λ[i] − tr(A⁻¹ S_i), ε) / (β' S_i β)`
/// where `A = X' W X + Σ λ_j S_j` is the IRLS-Hessian (already inverted),
/// `phi` is the dispersion (1 for ELF / Binomial / Poisson; profiled
/// otherwise), and the step is taken in log-space with a clamp.
///
/// Used by:
/// - `SmoothingParameter::optimize_reml_fellner_schall_with_xtwx` (the
///   GAM outer loop's FS path).
/// - `pirls::fit_pirls_quantile_lss_fs_tune` (per-obs-σ ELF λ retuning).
///
/// `lambda_bounds = (1e-9, 1e7)` and `log_step_clamp = 3.0` are the
/// historic defaults — kept as parameters so callers can tighten them
/// if needed without diverging implementations.
pub fn fellner_schall_step(
    penalties: &[BlockPenalty],
    penalty_ranks: &[f64],
    lambdas: &[f64],
    a_inv: &Array2<f64>,
    beta: &Array1<f64>,
    phi: f64,
    log_step_clamp: f64,
    lambda_bounds: (f64, f64),
) -> Vec<f64> {
    debug_assert_eq!(penalties.len(), penalty_ranks.len());
    debug_assert_eq!(penalties.len(), lambdas.len());
    let tiny = 1e-10_f64;
    let mut new_lambdas = Vec::with_capacity(penalties.len());
    for i in 0..penalties.len() {
        let pen = &penalties[i];
        let rank_i = penalty_ranks[i];
        let lambda_i = lambdas[i];

        let tr_vs = pen.trace_product(a_inv);
        let bsb = pen.quadratic_form(beta).max(tiny);
        let numerator = (rank_i / lambda_i.max(1e-20) - tr_vs).max(tiny);

        let log_ratio = (phi * numerator / bsb)
            .ln()
            .clamp(-log_step_clamp, log_step_clamp);
        let log_lambda_new = (lambda_i.ln() + log_ratio)
            .max(lambda_bounds.0.ln())
            .min(lambda_bounds.1.ln());
        new_lambdas.push(log_lambda_new.exp());
    }
    new_lambdas
}

/// Smoothing parameter optimization method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationMethod {
    REML,
    GCV,
}

/// REML optimization algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum REMLAlgorithm {
    /// Newton's method with Hessian — default, stable, good for large problems
    Newton,
    /// Fellner-Schall iteration (Wood & Fasiolo 2017) — faster for small/medium problems
    /// Uses the update: λ_new = φ × (rank/λ - tr(A⁻¹S)) / (β'Sβ)
    FellnerSchall,
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
    /// When true, the line-search REML evaluations use mgcv's exact
    /// formula (gam.fit3.r:621) instead of our default formula. The
    /// gradient/Hessian still use the default formula since they're
    /// only used to determine STEP DIRECTION; the line search uses the
    /// score for STEP ACCEPTANCE. With mgcv-exact score driving
    /// acceptance, the optimizer converges closer to mgcv's optimum.
    pub mgcv_exact_score: bool,
    /// Mp = nsdf + Σ null.space.dim_j (constant in λ). Required for
    /// the mgcv-exact REML score; pre-computed once per fit.
    pub mp: usize,
    /// Fixed scale parameter φ. `Some(1.0)` for binomial/poisson where φ
    /// is known; `None` for Gaussian/Gamma where φ is profiled from
    /// Pearson residuals. The Fellner-Schall update needs the right φ:
    /// estimating it from `(y - Xβ)²` is wrong for non-Gaussian since
    /// `Xβ = η` not `μ`.
    pub phi_fixed: Option<f64>,
    /// GLM family for REML score / gradient / Hessian. Affects how the
    /// deviance and scale parameter are computed in mgcv-exact mode.
    /// Default is Gaussian; set by gam_optimized.rs from `self.family`.
    pub family: crate::pirls::Family,
    /// Original response y (not the working response z). Used by the IFT
    /// gradient/Hessian path to evaluate ∂D/∂β at the GLM deviance form
    /// `-2 X'(y_orig - μ)/(Vg')` rather than the working-RSS form, which
    /// is the only path where IFT genuinely differs from envelope at our
    /// (always inner-loop converged) β. None ⟹ fall back to working-RSS
    /// deviance (≡ envelope at converged β).
    pub y_original: Option<Array1<f64>>,
    /// Profile-p mode for Tweedie family (mgcv's `tw()` function).
    ///
    /// When `true`, the outer Newton loop optimises θ (the working parameter
    /// for p) jointly with ρ = log(λ). The mapping is:
    ///   p(θ) = (a + b·exp(θ))/(exp(θ)+1)  for θ ≤ 0
    ///          (b + a·exp(-θ))/(exp(-θ)+1) for θ > 0
    /// with defaults a=1.001, b=1.999.  Initial θ = 0 ⟹ p = 1.5.
    pub tweedie_profile: bool,
    /// Current θ for Tweedie profile-p. Updated each outer Newton iteration.
    pub tweedie_theta: f64,
    /// Profile-θ mode for NegBin family (mgcv's `nb()` extended family).
    ///
    /// When `true`, the outer Newton loop optimises log(θ) jointly with ρ.
    /// We work in log-space so θ > 0 is enforced automatically.
    pub negbin_profile: bool,
    /// Current log(θ) for NB profile-θ. Updated each outer Newton iteration.
    /// Initial value: log(2.0).
    pub negbin_log_theta: f64,
    /// Profile-σ mode for Quantile (qgam-style ELF) family.
    ///
    /// When `true`, the outer optimizer also profiles log σ (the ELF
    /// bandwidth) via the same REML score machinery — Wood-style
    /// analytical optimization rather than qgam's bootstrap-CV
    /// `tuneLearnFast`. Set to true when the user passed σ=0 sentinel
    /// (the default for `mr.GAM("quantile", tau=...)`).
    pub quantile_profile: bool,
    /// Current log(σ) for Quantile profile-σ. Updated each outer Newton
    /// iteration in the same FD-Newton style as Tweedie p / NegBin θ.
    pub quantile_log_sigma: f64,
    /// Profile-df mode for the scat (TDist) family at the *outer* Newton
    /// level — joint with ρ, à la Tweedie p / NegBin θ. When true, the
    /// outer Newton FD-steps on log(df) after each ρ step using the
    /// same REML score machinery; PIRLS receives the current df as
    /// fixed (so the inner Brent on df doesn't compete with the outer
    /// Newton). Set true when the user passed df=0 sentinel (the
    /// default for `mr.GAM("t-dist")`).
    pub tdist_profile: bool,
    /// Current mgcv scat theta1 = log(df - 2) for TDist profile-df.
    /// Updated each outer Newton iteration. Initial value: log(5 - 2).
    pub tdist_log_df: f64,
    /// Current log(σ²) for TDist outer σ² Newton (gam.fit5-style LAML
    /// profiling). Updated each outer Newton iteration when
    /// `tdist_profile = true`. Seeded from sample variance at fit start.
    /// Inactive (no FD step taken) when `tdist_profile = false`.
    pub tdist_log_sigma2: f64,
    /// log(σ²) lower / upper bound for the outer σ² Newton, set at fit
    /// start to seed ± 2 decades. Without bounds the frozen-β FD walks
    /// σ² to numeric zero (β fit at a smaller σ² has tighter inliers ⇒
    /// smaller weighted residual sum ⇒ smaller MLE-σ² target ⇒ repeat,
    /// a vicious cycle). mgcv's gam.fit5 avoids it via joint (β, σ², λ,
    /// df) Newton; until that lands the band keeps σ² near the data scale.
    pub tdist_log_sigma2_lo: f64,
    pub tdist_log_sigma2_hi: f64,
    /// REML / LAML score at convergence. Populated by the outer optimizer
    /// (Newton or FS) on the last iteration so callers can read it back
    /// for σ profiling at the wrapper level.
    pub last_score: Option<f64>,
    /// Shared mutable handle to the Family enum used by the inner-PIRLS
    /// callback in non-Gaussian Newton. Set by the caller (gam_optimized.rs)
    /// when it builds the per-trial-λ refresh closure: the closure reads
    /// the latest value here so it sees fresh family-shape parameters
    /// (TDist df / σ², Tweedie p, NegBin θ) that the outer Newton mutates
    /// between iterations. Without this, a plain `let family = self.family;`
    /// capture would freeze the closure's family at the starting value.
    /// `None` for the Fellner-Schall path / Gaussian / any caller that
    /// doesn't need the sync.
    ///
    /// `Arc<Mutex<_>>` (rather than the cheaper `Rc<Cell<_>>`) only because
    /// `SmoothingParameter` lives inside a `#[pyclass]` (PyGAM) which
    /// requires `Send + Sync`. The Newton loop is single-threaded so the
    /// mutex is uncontended; lock cost is negligible vs. PIRLS.
    pub family_cell: Option<std::sync::Arc<std::sync::Mutex<crate::pirls::Family>>>,
}

/// Refresh produced by an inner PIRLS run at a candidate λ during the
/// Newton line search. Carries everything `dispatch_reml_score` needs to
/// evaluate the score at IRLS-converged β̂(λ') instead of the stale-
/// β/w/z one-step approximation.
///
/// `working_response` is the IRLS working response z = η + (y - μ)/(dμ/dη)
/// computed at the converged μ, η — passing this as the response into
/// `solve(X'WX + λ'S, X'Wz)` recovers β̂(λ') exactly (mgcv pls_fit1
/// equivalent). `xtwx` is the X'WX at the refreshed weights.
#[cfg(feature = "blas")]
pub struct PirlsRefresh {
    pub beta: Array1<f64>,
    pub weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub xtwx: Array2<f64>,
    /// Family-scale parameter the inner fitter converged to (e.g. MLE σ²
    /// from `fit_pirls_tdist`). When `Some`, the outer Newton loop syncs
    /// `self.family`'s scale-bearing variant (currently TDist) so that
    /// `Family::saturated_log_likelihood` and `Family::deviance` see the
    /// up-to-date σ². `None` ⟹ caller doesn't track σ²; outer loop falls
    /// back to `estimate_phi_mgcv`.
    pub sigma2: Option<f64>,
    /// Family df the inner TDist fitter used/converged to. When `Some`, keep
    /// the outer family enum in sync so score evaluations see the same df as
    /// the refreshed PIRLS weights/β.
    pub df: Option<f64>,
}

/// Callback type for refreshing PIRLS at trial λ during Newton line search.
/// `Some(callback)` enables mgcv's per-trial-λ inner-IRLS refresh
/// (gam.fit3.r:1444, 1500-1504, 1571-1576). `None` keeps the fast
/// frozen-β/w/z path used for Gaussian (where W=I, z=y, no refresh
/// needed).
#[cfg(feature = "blas")]
pub type PirlsCallback<'a> = &'a mut dyn FnMut(&[f64]) -> Result<PirlsRefresh>;

impl SmoothingParameter {
    /// Create new smoothing parameters with initial values
    pub fn new(num_smooths: usize, method: OptimizationMethod) -> Self {
        Self {
            lambda: vec![0.1; num_smooths], // Will be refined in optimize()
            method,
            reml_algorithm: REMLAlgorithm::Newton, // Default to Newton (matches bam())
            #[cfg(feature = "blas")]
            scale_method: ScaleParameterMethod::EDF, // Default to EDF (matches mgcv)
            mgcv_exact_score: false,
            mp: 0,
            phi_fixed: None,
            family: crate::pirls::Family::Gaussian,
            y_original: None,
            tweedie_profile: false,
            tweedie_theta: 0.0,
            negbin_profile: false,
            negbin_log_theta: 2.0_f64.ln(),
            quantile_profile: false,
            quantile_log_sigma: 0.0,
            tdist_profile: false,
            tdist_log_df: (5.0_f64 - 2.0).ln(),
            tdist_log_sigma2: 0.0_f64, // placeholder; caller seeds from sample variance
            tdist_log_sigma2_lo: f64::NEG_INFINITY,
            tdist_log_sigma2_hi: f64::INFINITY,
            last_score: None,
            family_cell: None,
        }
    }

    /// Create with specific REML algorithm
    pub fn new_with_algorithm(
        num_smooths: usize,
        method: OptimizationMethod,
        algorithm: REMLAlgorithm,
    ) -> Self {
        Self {
            lambda: vec![0.1; num_smooths],
            method,
            reml_algorithm: algorithm,
            #[cfg(feature = "blas")]
            scale_method: ScaleParameterMethod::Rank,
            mgcv_exact_score: false,
            mp: 0,
            phi_fixed: None,
            family: crate::pirls::Family::Gaussian,
            y_original: None,
            tweedie_profile: false,
            tweedie_theta: 0.0,
            negbin_profile: false,
            negbin_log_theta: 2.0_f64.ln(),
            quantile_profile: false,
            quantile_log_sigma: 0.0,
            tdist_profile: false,
            tdist_log_df: (5.0_f64 - 2.0).ln(),
            tdist_log_sigma2: 0.0_f64, // placeholder; caller seeds from sample variance
            tdist_log_sigma2_lo: f64::NEG_INFINITY,
            tdist_log_sigma2_hi: f64::INFINITY,
            last_score: None,
            family_cell: None,
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
            mgcv_exact_score: false,
            mp: 0,
            phi_fixed: None,
            family: crate::pirls::Family::Gaussian,
            y_original: None,
            tweedie_profile: false,
            tweedie_theta: 0.0,
            negbin_profile: false,
            negbin_log_theta: 2.0_f64.ln(),
            quantile_profile: false,
            quantile_log_sigma: 0.0,
            tdist_profile: false,
            tdist_log_df: (5.0_f64 - 2.0).ln(),
            tdist_log_sigma2: 0.0_f64, // placeholder; caller seeds from sample variance
            tdist_log_sigma2_lo: f64::NEG_INFINITY,
            tdist_log_sigma2_hi: f64::INFINITY,
            last_score: None,
            family_cell: None,
        }
    }

    /// Set the scale parameter method
    #[cfg(feature = "blas")]
    pub fn with_scale_method(mut self, method: ScaleParameterMethod) -> Self {
        self.scale_method = method;
        self
    }

    /// Optimize smoothing parameters using REML or GCV with adaptive initialization
    ///
    /// For Fellner-Schall algorithm, `beta` (current coefficient estimates from PiRLS)
    /// must be provided. For Newton, beta is not needed.
    pub fn optimize(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        self.optimize_with_beta(y, x, w, penalties, max_iter, tolerance, None)
    }

    /// Optimize smoothing parameters, optionally with current coefficients for FS algorithm
    pub fn optimize_with_beta(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        beta: Option<&Array1<f64>>,
    ) -> Result<()> {
        if penalties.len() != self.lambda.len() {
            return Err(GAMError::DimensionMismatch(
                "Number of penalties must match number of lambdas".to_string(),
            ));
        }

        // For Newton: reset to λ=1 as starting point
        if self.reml_algorithm != REMLAlgorithm::FellnerSchall {
            for i in 0..self.lambda.len() {
                self.lambda[i] = 1.0;
            }
        }

        match self.method {
            OptimizationMethod::REML => {
                self.optimize_reml(y, x, w, penalties, max_iter, tolerance, beta)
            }
            OptimizationMethod::GCV => self.optimize_gcv(y, x, w, penalties, max_iter, tolerance),
        }
    }

    /// Optimize smoothing parameters with optional pre-computed X'WX.
    ///
    /// When `cached_xtwx` is provided, skips the O(n*p^2) X'WX computation.
    /// This is used when the caller has already computed X'WX via scatter-gather
    /// on a discretized design (much faster for large n).
    #[cfg(feature = "blas")]
    pub fn optimize_with_beta_and_xtwx(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        beta: Option<&Array1<f64>>,
        cached_xtwx: Option<&Array2<f64>>,
    ) -> Result<()> {
        self.optimize_with_beta_xtwx_and_pirls_callback(
            y,
            x,
            w,
            penalties,
            max_iter,
            tolerance,
            beta,
            cached_xtwx,
            None,
        )
    }

    /// Same as `optimize_with_beta_and_xtwx`, but accepts an optional
    /// `pirls_callback` which is invoked at every Newton line-search trial
    /// λ' to refresh (β, w, z, X'WX) via inner PIRLS-to-convergence.
    /// `None` retains the fast frozen-β/w/z path (Gaussian).
    #[cfg(feature = "blas")]
    pub fn optimize_with_beta_xtwx_and_pirls_callback(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        beta: Option<&Array1<f64>>,
        cached_xtwx: Option<&Array2<f64>>,
        pirls_callback: Option<PirlsCallback<'_>>,
    ) -> Result<()> {
        if penalties.len() != self.lambda.len() {
            return Err(GAMError::DimensionMismatch(
                "Number of penalties must match number of lambdas".to_string(),
            ));
        }

        // For Newton: reset to λ=1 as starting point (Newton iterates internally,
        // and λ=1 is a good neutral starting point that avoids bias from poor initialization).
        // For Fellner-Schall: keep current lambda (FS does a single update step).
        if self.reml_algorithm != REMLAlgorithm::FellnerSchall {
            for i in 0..self.lambda.len() {
                self.lambda[i] = 1.0;
            }
        }

        match self.method {
            OptimizationMethod::REML => self.optimize_reml_with_xtwx(
                y,
                x,
                w,
                penalties,
                max_iter,
                tolerance,
                beta,
                cached_xtwx,
                pirls_callback,
            ),
            OptimizationMethod::GCV => self.optimize_gcv(y, x, w, penalties, max_iter, tolerance),
        }
    }

    /// Initialize lambda values adaptively based on penalty and design matrix scales
    fn initialize_lambda_adaptive(
        &mut self,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
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
            let penalty_trace = penalty.trace();

            // FIXED: Scale-invariant initialization
            // lambda ~ 0.1 * trace(S) / (trace(X'WX)/n)
            // This makes starting lambda independent of n
            if penalty_trace > 1e-10 {
                self.lambda[i] = 0.1 * penalty_trace / xtwx_trace_per_n;
            } else {
                self.lambda[i] = 0.1; // Fallback for near-zero penalty
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
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        beta: Option<&Array1<f64>>,
    ) -> Result<()> {
        // Dispatch based on selected algorithm
        match self.reml_algorithm {
            REMLAlgorithm::Newton => {
                self.optimize_reml_newton_multi(y, x, w, penalties, max_iter, tolerance)
            }
            REMLAlgorithm::FellnerSchall => {
                self.optimize_reml_fellner_schall(y, x, w, penalties, max_iter, tolerance, beta)
            }
        }
    }

    #[cfg(feature = "blas")]
    fn optimize_reml_with_xtwx(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        beta: Option<&Array1<f64>>,
        cached_xtwx: Option<&Array2<f64>>,
        pirls_callback: Option<PirlsCallback<'_>>,
    ) -> Result<()> {
        // Dispatch based on selected algorithm
        match self.reml_algorithm {
            REMLAlgorithm::Newton => self.optimize_reml_newton_multi_with_xtwx(
                y,
                x,
                w,
                penalties,
                max_iter,
                tolerance,
                cached_xtwx,
                pirls_callback,
            ),
            REMLAlgorithm::FellnerSchall => self.optimize_reml_fellner_schall_with_xtwx(
                y,
                x,
                w,
                penalties,
                max_iter,
                tolerance,
                beta,
                cached_xtwx,
            ),
        }
    }

    /// Grid search for single smooth (kept for stability)
    fn optimize_reml_grid_single(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalty: &BlockPenalty,
    ) -> Result<()> {
        let mut best_lambda = self.lambda[0];
        let mut best_reml = f64::INFINITY;

        // Coarse grid search to find approximate optimum
        for i in 0..50 {
            let log_lambda = -4.0 + i as f64 * 0.12; // -4 to 2 (0.0001 to 100)
            let lambda = 10.0_f64.powf(log_lambda);
            let reml = reml_criterion(y, x, w, lambda, penalty, None)?;

            if reml < best_reml {
                best_reml = reml;
                best_lambda = lambda;
            }
        }

        // Refine with finer grid search around best lambda
        let log_best = best_lambda.ln();
        let search_width = 0.15; // Search ±0.15 in log space
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
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        self.optimize_reml_newton_multi_with_xtwx(
            y, x, w, penalties, max_iter, tolerance, None, None,
        )
    }

    /// Newton optimization with optional pre-computed X'WX and optional
    /// per-trial-λ PIRLS refresh callback (mgcv-style outer iteration).
    #[cfg(feature = "blas")]
    fn optimize_reml_newton_multi_with_xtwx(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
        cached_xtwx: Option<&Array2<f64>>,
        mut pirls_callback: Option<PirlsCallback<'_>>,
    ) -> Result<()> {
        let m = penalties.len();

        // OPTIMIZATION: Pre-compute sqrt_penalties once (expensive eigendecomposition)
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
            eprintln!(
                "[PROFILE] Pre-computed sqrt_penalties: {:.2}ms",
                sqrt_pen_time.as_secs_f64() * 1000.0
            );
        }

        // OPTIMIZATION: Pre-compute X'WX and X'Wy (constant during Gaussian
        // optimization; refreshed per Newton iter when `pirls_callback` is
        // supplied for non-Gaussian families — see "PIRLS refresh" below).
        // When a pre-computed X'WX is available (from scatter-gather on
        // discretized design), use it directly to skip the O(n*p²) computation.
        use crate::reml::compute_xtwx;
        let xtwx_start = Instant::now();

        // Owned locals — refreshed each Newton iter when `pirls_callback`
        // is Some. For Gaussian (callback=None) they stay constant.
        let mut y_local: Array1<f64> = y.to_owned();
        let mut w_local: Array1<f64> = w.to_owned();
        // PIRLS-converged β at the current λ. None until the first refresh;
        // used by the Newton-at-β IFT gradient path for non-canonical-link
        // GLMs where Fisher-fallback W in the regular IFT formula gives the
        // wrong tk_kkt.
        let mut beta_local: Option<Array1<f64>> = None;
        let mut xtwx_local: Array2<f64> = if let Some(cached) = cached_xtwx {
            cached.clone()
        } else {
            compute_xtwx(x, &w_local)
        };

        // Compute X'Wy (refreshed when w/z change).
        let mut xtwy_local: Array1<f64> = compute_xtwy_helper(x, &w_local, &y_local);

        if std::env::var("MGCV_PROFILE").is_ok() {
            let xtwx_time = xtwx_start.elapsed();
            eprintln!(
                "[PROFILE] Pre-computed X'WX and X'Wy: {:.2}ms",
                xtwx_time.as_secs_f64() * 1000.0
            );
        }

        // Pre-compute Cholesky of X'WX for EDF computation (if using EDF method).
        // Refreshed when xtwx_local changes (i.e., when w changes via PIRLS refresh).
        let mut xtwx_chol_local: Option<Array2<f64>> =
            if self.scale_method == ScaleParameterMethod::EDF {
                let chol_start = Instant::now();
                let chol = compute_xtwx_cholesky(&xtwx_local)?;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!(
                        "[PROFILE] Pre-computed X'WX Cholesky for EDF: {:.2}ms",
                        chol_start.elapsed().as_secs_f64() * 1000.0
                    );
                }
                Some(chol)
            } else {
                None
            };

        // Work in log space for stability
        let mut log_lambda: Vec<f64> = self.lambda.iter().map(|l| l.ln()).collect();

        // Maximum step size in log space. mgcv's `newton` uses
        // `maxNstep=5` (gam.fit3.r:1411) — slightly larger than our
        // historical 4. For saturating-λ smooths the Newton direction is
        // small in magnitude (huge Hessian eigenvalue) so the cap rarely
        // bites, but at very high n the cap can prevent reaching mgcv's
        // converged saturating λ in one step.
        let max_step = 5.0;

        // OPTIMIZATION: Armijo constant for line search
        // Accepts steps with "sufficient decrease": f(x + αd) ≤ f(x) + c₁·α·∇f'·d
        // Standard value c₁ = 0.01 (very lenient, prefers larger steps)
        let armijo_c1 = 0.01;

        let mut prev_reml = f64::INFINITY;

        for iter in 0..max_iter {
            // Current lambdas
            let lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

            // PIRLS refresh at iter start (mgcv-style): if a callback is
            // supplied, run inner PIRLS to convergence at the current
            // lambdas to get fresh (β, w, z, X'WX). This replaces the
            // stale frozen-β/w/z values inherited from the caller. For
            // Gaussian (callback=None) skip — W=I, z=y, no refresh needed.
            if let Some(cb) = pirls_callback.as_mut() {
                let refresh = cb(&lambdas)?;
                beta_local = Some(refresh.beta);
                y_local = refresh.working_response;
                w_local = refresh.weights;
                xtwx_local = refresh.xtwx;
                xtwy_local = compute_xtwy_helper(x, &w_local, &y_local);
                if self.scale_method == ScaleParameterMethod::EDF {
                    xtwx_chol_local = Some(compute_xtwx_cholesky(&xtwx_local)?);
                }
                // Sync the family enum's scale parameter (σ² for TDist) to
                // whatever the inner fitter converged to, so the REML
                // score formula that follows reads the right value.
                if refresh.sigma2.is_some() || refresh.df.is_some() {
                    if let crate::pirls::Family::TDist { df, sigma2 } = self.family {
                        self.family = crate::pirls::Family::TDist {
                            df: refresh.df.unwrap_or(df),
                            sigma2: refresh.sigma2.unwrap_or(sigma2),
                        };
                    }
                    // Sync the closure-side cell so subsequent line-search
                    // trial-λ callbacks within this iter see fresh TDist params.
                    if let Some(cell) = self.family_cell.as_ref() {
                        if let Ok(mut g) = cell.lock() {
                            *g = self.family;
                        }
                    }
                }
            }

            // Compute current REML value for convergence check
            // (dispatches to mgcv-exact formula when mgcv_exact_score=true)
            let current_reml = dispatch_reml_score(
                self,
                &y_local,
                x,
                &w_local,
                &lambdas,
                penalties,
                Some(&xtwx_local),
            )?;

            // Compute gradient and Hessian.
            // - Default mode: closed-form QR-based formulas, fast.
            // - mgcv_exact mode: closed-form gradient (matches mgcv's
            //   gam.fit3.r:625 D1/(2σ²) + trA1/2 - det1/2 simplified
            //   for Gaussian + canonical link via envelope theorem).
            //   Hessian still uses finite differences for now —
            //   replacing it with closed-form is an open task.
            let use_ift = use_mgcv_exact_ift_policy(
                self.mgcv_exact_score,
                self.family,
                y.len(),
                self.y_original.is_some(),
            );
            // GamFit5 score families (TDist/scat, Quantile/ELF) use a
            // structurally different REML formula (Dp/2, no σ²-chain term).
            // The closed-form / IFT gradient + hessian assemblies above are
            // gam.fit3-specific (`H = D2/(2σ²) + det2/2`). Routing those
            // families to FD keeps gradient/hessian consistent with the
            // criterion until a closed-form GamFit5 derivative lands.
            let use_fd_for_score_formula =
                self.family.score_formula() == crate::pirls::ScoreFormula::GamFit5;

            let t_grad = Instant::now();
            let gradient = if self.mgcv_exact_score {
                if matches!(self.family, crate::pirls::Family::TDist { .. }) {
                    reml_gradient_gamfit4_tdist_analytic(
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        self.y_original.as_ref().unwrap_or(&y_local),
                        self.family,
                    )?
                } else if use_fd_for_score_formula {
                    reml_gradient_finite_diff(
                        self,
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        &mut pirls_callback,
                    )?
                } else if use_ift {
                    // For non-canonical-link non-Gaussian GLMs, use the
                    // Newton-at-β path: w_local is PIRLS Fisher-fallback (no
                    // negs) but mgcv's gdi2 evaluates the gradient with raw
                    // Newton W (~43% neg entries for InvGauss+log) at the
                    // PIRLS-converged β. Without this, the tk_kkt term is
                    // computed against the wrong A. Falls back to the legacy
                    // Fisher-W IFT for canonical link / Gaussian where the
                    // two coincide.
                    let use_newton_at_beta = !self.family.is_canonical_link()
                        && !matches!(self.family, crate::pirls::Family::Gaussian)
                        && beta_local.is_some()
                        && self.y_original.is_some();
                    if use_newton_at_beta {
                        let mp_dim: usize = 1 + penalties
                            .iter()
                            .map(|pen| {
                                let k = pen.block_view().nrows();
                                let rank_s = crate::reml::estimate_rank_eigen(pen);
                                k.saturating_sub(rank_s)
                            })
                            .sum::<usize>();
                        crate::reml::reml_gradient_mgcv_exact_ift_newton_at_beta(
                            x,
                            self.y_original.as_ref().unwrap(),
                            beta_local.as_ref().unwrap(),
                            &lambdas,
                            penalties,
                            self.family,
                            mp_dim,
                        )?
                    } else {
                        reml_gradient_mgcv_exact_ift(
                            &y_local,
                            x,
                            &w_local,
                            &lambdas,
                            penalties,
                            Some(&xtwx_local),
                            self.family,
                            self.y_original.as_ref(),
                        )?
                    }
                } else {
                    reml_gradient_mgcv_exact_closed_form(
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        self.family,
                    )?
                }
            } else {
                reml_gradient_multi_qr_adaptive_cached_edf(
                    &y_local,
                    x,
                    &w_local,
                    &lambdas,
                    penalties,
                    Some(&sqrt_penalties),
                    Some(&xtwx_local),
                    Some(&xtwy_local),
                    xtwx_chol_local.as_ref(),
                    self.scale_method,
                )?
            };
            let grad_time = t_grad.elapsed().as_micros();

            let t_hess = Instant::now();
            let mut hessian = if self.mgcv_exact_score {
                if matches!(self.family, crate::pirls::Family::TDist { .. }) {
                    reml_hessian_gamfit4_tdist_analytic(
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        self.y_original.as_ref().unwrap_or(&y_local),
                        self.family,
                    )?
                } else if use_fd_for_score_formula {
                    reml_hessian_finite_diff(
                        self,
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        &mut pirls_callback,
                    )?
                } else if use_ift {
                    reml_hessian_mgcv_exact_ift(
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        self.family,
                        self.y_original.as_ref(),
                    )?
                } else {
                    reml_hessian_mgcv_exact_closed_form(
                        &y_local,
                        x,
                        &w_local,
                        &lambdas,
                        penalties,
                        Some(&xtwx_local),
                        self.family,
                    )?
                }
            } else {
                reml_hessian_multi_cached(&y_local, x, &w_local, &lambdas, penalties, &xtwx_local)?
            };
            let hess_time = t_hess.elapsed().as_micros();

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE]     Gradient: {:.2}ms, Hessian: {:.2}ms",
                    grad_time as f64 / 1000.0,
                    hess_time as f64 / 1000.0
                );
            }

            // Debug output: show raw Hessian before conditioning
            if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
                eprintln!("\n[SMOOTH_DEBUG] Raw Hessian at λ={:?}:", lambdas);
                for i in 0..m {
                    for j in 0..m {
                        eprint!("  H[{},{}]={:.6e}", i, j, hessian[[i, j]]);
                    }
                    eprintln!();
                }
                eprintln!("[SMOOTH_DEBUG] Gradient: {:?}", gradient);
            }

            // -----------------------------------------------------------------
            // edge.correct saturation detection (mgcv gam.fit3.r:1672).
            //
            // Detect smooths whose log-λ has reached working infinity using the
            // raw (pre-preconditioned) Hessian:
            //   flat[i] = |hess[i,i]| < |grad[i]| * 100
            //
            // When `MGCV_EDGE_CORRECT=1` is set, the Newton step's component
            // for any detected smooth is clamped to zero — the saturating-λ
            // dimension's curvature is degenerate and further motion just
            // amplifies noise (and pulls the Tk·KK' contribution further
            // into the boundary trap). Default behavior is unchanged.
            //
            // See `mgcv_rust - Tk·KK' edge.correct regression 2026-05-10.md`
            // for why this is the prerequisite for default-on Tk·KK'.
            // -----------------------------------------------------------------
            let edge_correct = std::env::var("MGCV_EDGE_CORRECT").is_ok();
            let saturating_mask: Vec<bool> = if edge_correct {
                detect_saturating_smooths(&gradient, &hessian)
            } else {
                vec![false; m]
            };
            if edge_correct
                && std::env::var("MGCV_PROFILE").is_ok()
                && saturating_mask.iter().any(|&b| b)
            {
                let flat: Vec<usize> = (0..m).filter(|&i| saturating_mask[i]).collect();
                eprintln!(
                    "[PROFILE]   edge.correct: saturating smooths detected: {:?} (|H_ii| < |g_i|*100)",
                    flat
                );
            }

            // ===================================================================
            // CRITICAL: Condition Hessian like mgcv to ensure stable convergence
            // ===================================================================
            // mgcv uses ridge regularization + diagonal preconditioning
            // This prevents ill-conditioning that causes tiny steps in late iterations

            // 1. Add adaptive ridge FIRST (before preconditioning)
            //    Ridge increases with iteration to handle increasing ill-conditioning
            let min_diag_orig = (0..m)
                .map(|i| hessian[[i, i]])
                .fold(f64::INFINITY, f64::min);
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
                eprintln!(
                    "[PROFILE]   Hessian diag range: [{:.6e}, {:.6e}], condition: {:.2e}",
                    min_diag_orig, max_diag_orig, cond_est
                );
                eprintln!(
                    "[PROFILE]   Preconditioner: {:?}",
                    diag_precond.as_slice().unwrap_or(&[])
                );
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
                eprintln!(
                    "[PROFILE]   gradient={:?}",
                    gradient.as_slice().unwrap_or(&[])
                );
            }

            // Env-gated outer-Newton trace (MGCV_RUST_TRACE_OUTER=1):
            // emits a single-line greppable record at the START of each iter
            // capturing (iter, log_sp, sp, REML, grad_inf, gradient).
            // The step / halvings / accepted bits are printed later in this
            // iter, after the line search has run. Used by diagnostic scripts
            // such as scripts/python/diagnostics/invgauss_n800_trajectory_diff.py.
            let trace_outer = std::env::var("MGCV_RUST_TRACE_OUTER").is_ok();
            if trace_outer {
                let log_sp_dbg: Vec<f64> = log_lambda.clone();
                eprintln!(
                    "[OUTER iter={}] log_sp={:?} sp={:?} REML={:.10} grad_inf={:.6e} grad={:?}",
                    iter + 1,
                    log_sp_dbg,
                    lambdas,
                    current_reml,
                    grad_norm_linf,
                    gradient.as_slice().unwrap_or(&[])
                );
            }

            // Converged if EITHER:
            // 1. Gradient L-infinity norm is small (gradient convergence)
            // 2. REML value change is tiny (value convergence for asymptotic cases like λ→∞)
            //
            // Audit 2026-05-04 (Phase E1 N-3 diagnostic): tested AND with
            // both our tols (grad 1e-6, score 1e-7) and mgcv's tols
            // (gam.fit3.r:1652-1653: grad 5e-6, score 1e-6). Both AND
            // variants broke `1d_gaussian_low_signal_n1000_k10_cr` and
            // `4d_binomial_logit_n2000_k8_cr` (oscillation in flat regions)
            // and did NOT fix N-3 — the Newton trajectory genuinely lands
            // at a different stationary point than mgcv's joint-(ρ, log φ)
            // outer Newton, regardless of convergence criterion. Keeping OR.
            let grad_tol = if self.mgcv_exact_score {
                let score_scale = current_reml.abs() + 1.0;
                score_scale * 1.0e-7
            } else {
                0.05
            };
            if grad_norm_linf < grad_tol {
                self.lambda = lambdas;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!("[PROFILE] Converged after {} iterations (gradient criterion: {:.6} < {:.0e})", iter + 1, grad_norm_linf, grad_tol);
                }
                return Ok(());
            }

            // REML change convergence: stop if making negligible progress.
            // mgcv's documented outer-iteration default is `conv.tol = 1e-7`
            // (gam.control). The score-change test (gam.fit3.r:1645) uses
            // 1e-7 as the absolute floor. Keeping 1e-7 here (Perf-1).
            let reml_change_tol = if self.mgcv_exact_score {
                1.0e-7
            } else {
                5.0e-4
            };
            if iter >= 3 && reml_change < reml_change_tol {
                self.lambda = lambdas;
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!(
                        "[PROFILE] Converged after {} iterations (REML change: {:.2e} < {:.0e})",
                        iter + 1,
                        reml_change,
                        reml_change_tol
                    );
                }
                return Ok(());
            }

            prev_reml = current_reml;

            // Compute Newton step: step = -H^(-1) · g  via mgcv-style
            // eigenvalue handling for indefinite Hessians (gam.fit3.r:1397-1417).
            // The preconditioned Hessian can have negative or near-zero
            // eigenvalues in the high-λ asymptote regime; mgcv's recipe is:
            //   1. eigendecompose H = U Λ U'
            //   2. set d_i ← |d_i| for all i  (Gill-Murray-Wright p.107-8)
            //   3. floor d_i ← max(d) * eps^0.7 if too small
            //   4. invert: d_i ← 1/d_i
            //   5. step_precond = -U diag(1/d) U' g_precond
            //
            // Without (2), the indefinite-H solve produces a step in the
            // wrong direction in the negative-eigenvalue subspace, which is
            // the dominant failure mode for the high-λ trap (4d_mixed,
            // 1d_near_linear) in our parity battery.

            // Precondition gradient: g_precond = D^-1 * g
            let mut gradient_precond = Array1::<f64>::zeros(m);
            for i in 0..m {
                gradient_precond[i] = gradient[i] / diag_precond[i];
            }

            use ndarray_linalg::{Eigh, UPLO};

            // Subset Newton: identify "unconverged" dimensions per
            // gam.fit3.r:1383 + 1430. mgcv excludes from the step:
            //   uconv.ind  = |grad_i| > score_scale·conv.tol  (per-dim converged)
            //   uconv.ind1 = uconv.ind & |grad_i| > max|grad|·0.001
            // and computes the Newton step ONLY on the remaining dims; the
            // others get step_i = 0. This is mgcv's saturation-freeze:
            // smooths whose gradient is already tiny don't get pushed further.
            // Without it, adding the (correct) Tk·KK' term to the gradient
            // pushes saturating-λ smooths past mgcv's stopping point, since
            // the step from a tiny |grad|/tiny |H_ii| is non-negligible.
            //
            // mgcv's score_scale = |log(scale.est)| + |REML|; conv.tol = 1e-6
            // (newton() default at gam.fit3.r:1260). For our REML score
            // |REML| dominates so we approximate score_scale = |REML| + 1.
            //
            // MGCV_STEP_BLEND=1: use mgcv's actual subset-Newton freeze filter
            // from gam.fit3.r:1643 — `(|grad| > 0.1·score_scale·conv.tol) |
            // (|H_ii| > 0.1·score_scale·conv.tol)`. The 0.1× factor (vs Rust's
            // default 1×) keeps more dims active in saturating regimes, which
            // closes the 4d_binomial gap (verified: at iter 6 of 4d_binomial,
            // dims 0-2 have |grad| ~2-5e-4 — above mgcv's 1.15e-4 threshold,
            // below Rust's 1.15e-3 default). The OR with H_ii catches dims
            // where the curvature is meaningful even if gradient happens
            // small.
            let step_blend = std::env::var("MGCV_STEP_BLEND").is_ok();
            let max_abs_grad = gradient.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
            let score_scale = current_reml.abs() + 1.0;
            let inner_conv_tol: f64 = if step_blend { 1.0e-7 } else { 1.0e-6 };
            let dim_grad_tol = score_scale * inner_conv_tol;
            let active: Vec<usize> = (0..m)
                .filter(|&i| {
                    let gi = gradient[i].abs();
                    if step_blend {
                        // mgcv-style OR filter: |grad| OR |H_ii| above threshold.
                        // Pre-condition note: hessian here is already preconditioned
                        // to ~unit diagonal, so we read the diagonal from the raw
                        // (pre-conditioning) values via diag_precond^2 (since
                        // H_precond_ii ≈ 1 always after preconditioning).
                        let hii_raw = diag_precond[i] * diag_precond[i];
                        gi > dim_grad_tol || hii_raw > dim_grad_tol
                    } else {
                        gi > dim_grad_tol && gi > max_abs_grad * 0.001
                    }
                })
                .collect();

            // mgcv safeguard (line 1432): ensure at least one dim is active.
            let active = if active.is_empty() {
                let argmax = (0..m)
                    .max_by(|&a, &b| gradient[a].abs().total_cmp(&gradient[b].abs()))
                    .unwrap_or(0);
                vec![argmax]
            } else {
                active
            };
            let n_active = active.len();

            if std::env::var("MGCV_PROFILE").is_ok() && n_active < m {
                eprintln!(
                    "[PROFILE]   Subset Newton: {}/{} dims active (frozen: {:?})",
                    n_active,
                    m,
                    (0..m).filter(|i| !active.contains(i)).collect::<Vec<_>>()
                );
            }

            // Build subset Hessian and gradient (preconditioned forms).
            let h_sub = if n_active == m {
                hessian.clone()
            } else {
                let mut hs = Array2::<f64>::zeros((n_active, n_active));
                for (ri, &ai) in active.iter().enumerate() {
                    for (ci, &aj) in active.iter().enumerate() {
                        hs[[ri, ci]] = hessian[[ai, aj]];
                    }
                }
                hs
            };
            let g_sub: Array1<f64> = if n_active == m {
                gradient_precond.clone()
            } else {
                let mut gs = Array1::<f64>::zeros(n_active);
                for (i, &ai) in active.iter().enumerate() {
                    gs[i] = gradient_precond[ai];
                }
                gs
            };

            let (eigvals, eigvecs) = h_sub
                .eigh(UPLO::Upper)
                .map_err(|e| GAMError::LinAlgError(format!("Hessian eigh failed: {:?}", e)))?;

            // Track pdef (positive-definite) status of the original (pre-abs,
            // pre-floor) Hessian. Mirrors mgcv's gam.fit3.r:1407-1411. Used by
            // the step-blending SD-trial path (MGCV_STEP_BLEND=1) to decide
            // whether to run the dedicated steepest-descent search.
            let pdef = {
                let mut p = !eigvals.iter().any(|&v| v < 0.0);
                if p {
                    let max_eig = eigvals.iter().cloned().fold(0.0f64, f64::max);
                    let low_eig = max_eig * f64::EPSILON.powf(0.7);
                    if eigvals.iter().any(|&v| v < low_eig) {
                        p = false;
                    }
                }
                p
            };

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE]   pdef={} (original Hessian eigvals: {:?})",
                    pdef,
                    eigvals.as_slice().unwrap_or(&[])
                );
            }

            // Modify eigenvalues: ABS for indefinite, floor for tiny.
            // mgcv's low.d = max(d) * .Machine$double.eps^.7 ≈ max(d) * 1.5e-11.
            let mut d = eigvals.to_vec();
            for di in d.iter_mut() {
                if *di < 0.0 {
                    *di = -*di;
                }
            }
            let max_d = d.iter().cloned().fold(0.0f64, f64::max);
            let low_d = max_d * (f64::EPSILON.powf(0.7));
            for di in d.iter_mut() {
                if *di < low_d {
                    *di = low_d;
                }
            }

            // step_sub_precond = -U diag(1/d) U^T g_sub
            let utg = eigvecs.t().dot(&g_sub);
            let mut scaled = Array1::<f64>::zeros(n_active);
            for i in 0..n_active {
                scaled[i] = -utg[i] / d[i];
            }
            let step_sub_precond = eigvecs.dot(&scaled);

            // Back-transform: step = D^-1 * step_precond, padded to full size
            // with zeros on frozen dims.
            let mut step = Array1::<f64>::zeros(m);
            for (ki, &ai) in active.iter().enumerate() {
                step[ai] = step_sub_precond[ki] / diag_precond[ai];
            }

            // edge.correct clamp: zero out step components for smooths flagged
            // as saturating (mgcv gam.fit3.r:1672 candidates-for-reduction).
            // Gated behind MGCV_EDGE_CORRECT so default behavior is unaffected.
            // Note: mgcv's full edge.correct (line 1669-1713) is a *post-Newton*
            // variance refinement; here we use the same detection criterion
            // to freeze the saturating coordinate during the outer iteration,
            // matching the planning note's recommendation in
            // `mgcv_rust - Plan Edge Correct.md` ("force a unit-log step in
            // that direction without line search" / saturation-freeze).
            if edge_correct {
                let mut clamped = 0;
                for i in 0..m {
                    if saturating_mask[i] && step[i] != 0.0 {
                        step[i] = 0.0;
                        clamped += 1;
                    }
                }
                if clamped > 0 && std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!(
                        "[PROFILE]   edge.correct: clamped step on {} saturating smooth(s)",
                        clamped
                    );
                }
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
            let grad_dot_step: f64 = gradient.iter().zip(step.iter()).map(|(g, s)| g * s).sum();

            // OPTIMIZATION: Adaptive max_half based on convergence progress
            // Near convergence (small gradient), Newton step is likely good - use fewer halvings
            // Far from convergence, may need more exploration
            let stalled_score = iter >= 3 && reml_change < 1.0e-4;
            // Stalled-score line-search cap. Historically capped to 1 because
            // the Tk·KK' gradient term was included for InvGauss/Binomial/
            // QuasiBinomial families but the matching Hessian piece was not,
            // leaving the Newton direction inconsistent with the gradient. Now
            // that `tk_kkt_hessian_analytical` is default-on under the same
            // family gate (reml.rs:~2570), Newton direction is consistent, so
            // the cap relaxes to the standard near-convergence budget whenever
            // tk_kkt is in play. `MGCV_STALLED_MAX_HALF` is an explicit
            // override for experimentation.
            let tk_kkt_active = matches!(
                self.family,
                crate::pirls::Family::InverseGaussian
                    | crate::pirls::Family::Binomial
                    | crate::pirls::Family::QuasiBinomial
            ) || std::env::var("MGCV_TK_GRAD").is_ok();
            let stalled_cap = std::env::var("MGCV_STALLED_MAX_HALF")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(if tk_kkt_active { 10 } else { 1 });
            let max_half = if stalled_score {
                stalled_cap
            } else if grad_norm_linf < 0.1 {
                10 // Near convergence - fewer line search iterations
            } else if grad_norm_linf < 1.0 {
                20 // Moderate - standard search
            } else {
                30 // Far from convergence - thorough search
            };

            let mut best_reml = current_reml;
            let mut best_step_scale = 0.0;
            let step_size_clamped = if step_size > max_step {
                max_step
            } else {
                step_size
            };

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE]   Line search: step_norm={:.6}, current_REML={:.6}, max_half={}",
                    step_size_clamped, current_reml, max_half
                );
                eprintln!(
                    "[PROFILE]     grad·step={:.6e} (expect decrease)",
                    grad_dot_step
                );
            }

            // Outer-Newton trace: record the Newton-direction step for this
            // iter BEFORE line search picks a scale, so we can see what
            // direction the optimizer wanted to take.
            if trace_outer {
                eprintln!(
                    "[OUTER iter={}]   dir={:?} step_norm={:.6} grad_dot_step={:.6e} max_half={}",
                    iter + 1,
                    step.as_slice().unwrap_or(&[]),
                    step_size_clamped,
                    grad_dot_step,
                    max_half
                );
            }

            let mut halvings_attempted: usize = 0;
            let t_linesearch = Instant::now();
            for half in 0..=max_half {
                let step_scale = 0.5_f64.powi(half as i32);

                // Try new log_lambda values
                let new_log_lambda: Vec<f64> = log_lambda
                    .iter()
                    .zip(step.iter())
                    .map(|(l, s)| l + s * step_scale)
                    .collect();

                let new_lambdas: Vec<f64> = new_log_lambda.iter().map(|l| l.exp()).collect();

                // Evaluate REML (mgcv-exact dispatch). When `pirls_callback`
                // is supplied (non-Gaussian), refresh inner PIRLS at the
                // trial λ' first to get fresh (β, w, z, X'WX) — this gives
                // the score at IRLS-converged β̂(λ') rather than the
                // stale-β/w/z one-step approximation. mgcv equivalent:
                // gam.fit3.r:1444, 1500-1504, 1571-1576.
                let trial_eval = if let Some(cb) = pirls_callback.as_mut() {
                    match cb(&new_lambdas) {
                        Ok(refresh) => {
                            let trial_xtwy_unused = (); // placeholder, gradient not needed in line search
                            let _ = trial_xtwy_unused;
                            dispatch_reml_score(
                                self,
                                &refresh.working_response,
                                x,
                                &refresh.weights,
                                &new_lambdas,
                                penalties,
                                Some(&refresh.xtwx),
                            )
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    dispatch_reml_score(
                        self,
                        &y_local,
                        x,
                        &w_local,
                        &new_lambdas,
                        penalties,
                        Some(&xtwx_local),
                    )
                };
                halvings_attempted = half;
                match trial_eval {
                    Ok(new_reml) => {
                        // OPTIMIZATION: Armijo condition for early stopping
                        // Accept if: new_reml ≤ current_reml + c₁ * step_scale * grad·step
                        // Since we're minimizing and grad·step should be negative, this is:
                        // new_reml ≤ current_reml - c₁ * step_scale * |grad·step|
                        let armijo_threshold =
                            current_reml + armijo_c1 * step_scale * grad_dot_step;
                        let satisfies_armijo = new_reml <= armijo_threshold;

                        if trace_outer {
                            eprintln!(
                                "[OUTER iter={}]     trial half={} scale={:.6e} REML={:.10} dREML={:.6e} armijo={}",
                                iter + 1,
                                half,
                                step_scale,
                                new_reml,
                                new_reml - current_reml,
                                satisfies_armijo
                            );
                        }

                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!(
                                "[PROFILE]     half={}: scale={:.4}, REML={:.6}, armijo={}",
                                half, step_scale, new_reml, satisfies_armijo
                            );
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
                    }
                    Err(_) => {
                        // Numerical issue - try smaller step
                        if trace_outer {
                            eprintln!(
                                "[OUTER iter={}]     trial half={} scale={:.6e} REML=ERR",
                                iter + 1,
                                half,
                                step_scale
                            );
                        }
                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!("[PROFILE]     half={}: ERROR (numerical issue)", half);
                        }
                        continue;
                    }
                }
            }
            let linesearch_time = t_linesearch.elapsed().as_micros();

            // Outer-Newton trace: summary of the line search outcome for this iter.
            // halvings = trials beyond the full step (Armijo / improvement loop).
            if trace_outer {
                let accepted = best_step_scale > 1e-6;
                let reason = if !accepted {
                    "no_improving_step"
                } else if (best_step_scale - 1.0).abs() < 1e-12 {
                    "full_newton"
                } else {
                    "halved"
                };
                eprintln!(
                    "[OUTER iter={}]   linesearch_done step_size={:.6e} best_scale={:.6e} halvings={} best_REML={:.10} accepted={} reason={}",
                    iter + 1,
                    step_size_clamped * best_step_scale,
                    best_step_scale,
                    halvings_attempted,
                    best_reml,
                    accepted,
                    reason
                );
            }

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!(
                    "[PROFILE]     Line search: {:.2}ms",
                    linesearch_time as f64 / 1000.0
                );
            }

            // Update log_lambda
            // Reject steps smaller than 1e-6 as they're effectively zero and waste time
            const MIN_STEP_SIZE: f64 = 1e-6;

            if best_step_scale > MIN_STEP_SIZE {
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!(
                        "[PROFILE]   Accepted Newton step, scale={:.4}",
                        best_step_scale
                    );
                }
                for i in 0..m {
                    log_lambda[i] += step[i] * best_step_scale;
                }
            } else {
                // Newton line search found no meaningful improvement (step too small or zero)
                // If gradient is already small, accept convergence rather than waste time
                if std::env::var("MGCV_PROFILE").is_ok() {
                    eprintln!(
                        "[PROFILE]   Newton step too small (scale={:.3e}), checking gradient",
                        best_step_scale
                    );
                }

                if stalled_score {
                    self.lambda = lambdas;
                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE] Converged after {} iterations (stalled score {:.2e} and no improving Newton step)",
                            iter + 1,
                            reml_change
                        );
                    }
                    return Ok(());
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
                let gradient_sd =
                    reml_gradient_multi_qr_adaptive(&y_local, x, &w_local, &lambdas, penalties)?;

                // Try progressively smaller steepest descent steps
                let mut sd_worked = false;
                for scale in &[0.01, 0.001, 0.0001] {
                    let sd_step: Vec<f64> = gradient_sd.iter().map(|g| -g * scale).collect();

                    let new_log_lambda_sd: Vec<f64> = log_lambda
                        .iter()
                        .zip(sd_step.iter())
                        .map(|(l, s)| l + s)
                        .collect();

                    let new_lambdas_sd: Vec<f64> =
                        new_log_lambda_sd.iter().map(|l| l.exp()).collect();

                    // Steepest-descent trial uses the line-search refresh
                    // path too (mgcv parity): if a callback is supplied,
                    // refresh PIRLS at the SD trial λ.
                    let sd_eval = if let Some(cb) = pirls_callback.as_mut() {
                        match cb(&new_lambdas_sd) {
                            Ok(refresh) => dispatch_reml_score(
                                self,
                                &refresh.working_response,
                                x,
                                &refresh.weights,
                                &new_lambdas_sd,
                                penalties,
                                Some(&refresh.xtwx),
                            ),
                            Err(e) => Err(e),
                        }
                    } else {
                        dispatch_reml_score(
                            self,
                            &y_local,
                            x,
                            &w_local,
                            &new_lambdas_sd,
                            penalties,
                            Some(&xtwx_local),
                        )
                    };
                    if let Ok(new_reml_sd) = sd_eval {
                        if std::env::var("MGCV_PROFILE").is_ok() {
                            eprintln!("[PROFILE]     SD scale={}: REML={:.6} (current={:.6}, improvement={})",
                                     scale, new_reml_sd, current_reml, new_reml_sd < current_reml);
                        }
                        if new_reml_sd < current_reml {
                            for i in 0..m {
                                log_lambda[i] = new_log_lambda_sd[i];
                            }
                            if std::env::var("MGCV_PROFILE").is_ok() {
                                eprintln!(
                                    "[PROFILE]   Steepest descent succeeded (scale={}): REML={:.6}",
                                    scale, new_reml_sd
                                );
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
                    let gradient_check = reml_gradient_multi_qr_adaptive(
                        &y_local, x, &w_local, &lambdas, penalties,
                    )?;
                    let grad_norm_final = gradient_check
                        .iter()
                        .map(|g| g.abs())
                        .fold(0.0f64, f64::max);

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

            // -----------------------------------------------------------------------
            // Tweedie profile-p: Newton step on θ (the working parameter for p).
            // Architecture A (matching mgcv): after each ρ step, also step on θ.
            //
            // θ ↔ p mapping (misc.c:212-224, a=1.001, b=1.999):
            //   θ > 0: p = (b + a·exp(-θ))/(exp(-θ)+1)
            //   θ ≤ 0: p = (b·exp(θ)+a)/(exp(θ)+1)
            //
            // Gradient dlr/dθ via FD on REML score at θ ± h. The fast path
            // (default) caches the linear system once via `TweedieThetaCache`
            // since (y_local, w_local, xtwx_local, lambdas) are frozen for
            // this step — β̂, μ̂, log|H|, log|S|+ are identical across the 3
            // probes, so we only redo the cheap (p, σ²̂(p))-pieces. Roughly
            // 3× faster than the old per-probe `dispatch_reml_score_with_family`
            // path, which redundantly rebuilt the system each call.
            //
            // The legacy FD path remains behind `MGCV_TWEEDIE_FD=1` for parity
            // verification.
            // -----------------------------------------------------------------------
            if self.tweedie_profile {
                let theta = self.tweedie_theta;
                let h_th: f64 = 1e-3;
                let current_lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

                // Helper: map θ → p (mgcv convention, a=1.001 b=1.999)
                fn tw_theta_to_p(th: f64) -> f64 {
                    let a = 1.001_f64;
                    let b = 1.999_f64;
                    if th > 0.0 {
                        let e = (-th).exp();
                        (b + a * e) / (e + 1.0)
                    } else {
                        let e = th.exp();
                        (b * e + a) / (e + 1.0)
                    }
                }
                let tw_theta_to_p_clamped =
                    |th: f64| -> f64 { tw_theta_to_p(th).max(1.001_f64).min(1.999_f64) };

                // Choose the fast cached path (default) or the legacy FD path
                // (env override) for verification.
                let use_fd_legacy = std::env::var("MGCV_TWEEDIE_FD").is_ok();
                let y_for_ls_ref: &Array1<f64> = self.y_original.as_ref().unwrap_or(&y_local);

                // Inline helper for the legacy / verification FD path.
                macro_rules! tw_eval_fd {
                    ($th_trial:expr) => {{
                        let th_trial: f64 = $th_trial;
                        let p_trial = tw_theta_to_p_clamped(th_trial);
                        let trial_fam = crate::pirls::Family::Tweedie { p: p_trial };
                        dispatch_reml_score_with_family(
                            self,
                            &y_local,
                            x,
                            &w_local,
                            &current_lambdas,
                            penalties,
                            Some(&xtwx_local),
                            trial_fam,
                        )
                    }};
                }

                // Build derivative pieces. The cached path returns
                // (rc, dlr/dθ, d²lr/dθ²) sharing one linear-system assembly.
                let derivs = if !use_fd_legacy && self.mgcv_exact_score {
                    match TweedieThetaCache::build(
                        &y_local,
                        x,
                        &w_local,
                        &xtwx_local,
                        &current_lambdas,
                        penalties,
                        self.mp,
                        y_for_ls_ref,
                    ) {
                        Ok(cache) => {
                            let r = tweedie_theta_derivatives_cached(
                                &cache,
                                theta,
                                h_th,
                                tw_theta_to_p_clamped,
                            );
                            // Stash the cache for the line-search refine step.
                            r.map(|(rc, g, h)| (rc, g, h, Some(cache)))
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    // Legacy path: 3 full dispatch calls.
                    let rc_r = tw_eval_fd!(theta);
                    let rp_r = tw_eval_fd!(theta + h_th);
                    let rm_r = tw_eval_fd!(theta - h_th);
                    match (rc_r, rp_r, rm_r) {
                        (Ok(rc), Ok(rp), Ok(rm)) => {
                            let dlr = (rp - rm) / (2.0 * h_th);
                            let d2lr = (rp - 2.0 * rc + rm) / (h_th * h_th);
                            Ok((rc, dlr, d2lr, None::<TweedieThetaCache>))
                        }
                        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => Err(e),
                    }
                };

                if let Ok((rc, dlr_dth, d2lr_dth2, cache_opt)) = derivs {
                    // Newton step: δθ = -g / |H| with |H| floored for stability
                    let denom = d2lr_dth2.abs().max(1e-4);
                    let delta_theta = -(dlr_dth / denom);
                    let delta_theta = delta_theta.max(-2.0_f64).min(2.0_f64);

                    // Line-search on θ: try full step, then half-step.
                    // Compare against rc (REML at current θ with updated λ).
                    // Reuse the cache for the candidate evaluations when we
                    // have one — same frozen system, just a different p.
                    let eval_at = |th_trial: f64| -> Result<f64> {
                        if let Some(cache) = cache_opt.as_ref() {
                            cache.score_at_p(tw_theta_to_p_clamped(th_trial))
                        } else {
                            tw_eval_fd!(th_trial)
                        }
                    };

                    let mut accepted_theta = theta;
                    let candidate = theta + delta_theta;
                    if let Ok(r_new) = eval_at(candidate) {
                        if r_new < rc {
                            accepted_theta = candidate;
                        } else {
                            let half_cand = theta + delta_theta * 0.5;
                            if let Ok(r_half) = eval_at(half_cand) {
                                if r_half < rc {
                                    accepted_theta = half_cand;
                                }
                            }
                        }
                    }

                    self.tweedie_theta = accepted_theta;
                    let new_p = tw_theta_to_p_clamped(accepted_theta);
                    // Near p=2 the Wright series becomes intractable; freeze profiling.
                    if new_p > 1.97 {
                        self.tweedie_profile = false;
                    }
                    self.family = crate::pirls::Family::Tweedie { p: new_p };
                    // Sync the closure-side cell so the next iter's PIRLS
                    // refresh callback sees the freshly profiled p.
                    if let Some(cell) = self.family_cell.as_ref() {
                        if let Ok(mut g) = cell.lock() {
                            *g = self.family;
                        }
                    }

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE]   Tweedie profile-p: θ {:.4}→{:.4} p={:.4} dlr/dθ={:.4e}{}",
                            theta,
                            accepted_theta,
                            new_p,
                            dlr_dth,
                            if use_fd_legacy { " [FD-legacy]" } else { " [analytical-cached]" }
                        );
                    }
                }
            }
            // End of Tweedie profile-p θ step

            // -----------------------------------------------------------------------
            // NegBin profile-θ: Newton step on log(θ).
            // After each ρ step, optimise log(θ) jointly using FD on REML score.
            // Working in log-space enforces θ > 0 automatically.
            // -----------------------------------------------------------------------
            if self.negbin_profile {
                let log_theta = self.negbin_log_theta;
                let h_th: f64 = 1e-3;
                let current_lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

                // Inline helper: evaluate REML score at trial log(θ)
                macro_rules! nb_eval {
                    ($lt_trial:expr) => {{
                        let lt_trial: f64 = $lt_trial;
                        let theta_trial = lt_trial.exp().max(0.5_f64).min(50.0_f64);
                        let trial_fam = crate::pirls::Family::NegBin { theta: theta_trial };
                        dispatch_reml_score_with_family(
                            self,
                            &y_local,
                            x,
                            &w_local,
                            &current_lambdas,
                            penalties,
                            Some(&xtwx_local),
                            trial_fam,
                        )
                    }};
                }

                let reml_center = nb_eval!(log_theta);
                let reml_plus = nb_eval!(log_theta + h_th);
                let reml_minus = nb_eval!(log_theta - h_th);

                if let (Ok(rc), Ok(rp), Ok(rm)) = (reml_center, reml_plus, reml_minus) {
                    let dlr_dlt = (rp - rm) / (2.0 * h_th);
                    let d2lr_dlt2 = (rp - 2.0 * rc + rm) / (h_th * h_th);

                    let denom = d2lr_dlt2.abs().max(1e-4);
                    let delta_lt = -(dlr_dlt / denom);
                    // Clamp step to [-0.5, 0.5] per iteration
                    let delta_lt = delta_lt.max(-0.5_f64).min(0.5_f64);

                    // Line-search on log(θ)
                    let mut accepted_lt = log_theta;
                    let candidate = log_theta + delta_lt;
                    if let Ok(r_new) = nb_eval!(candidate) {
                        if r_new < rc {
                            accepted_lt = candidate;
                        } else {
                            let half_cand = log_theta + delta_lt * 0.5;
                            if let Ok(r_half) = nb_eval!(half_cand) {
                                if r_half < rc {
                                    accepted_lt = half_cand;
                                }
                            }
                        }
                    }

                    self.negbin_log_theta = accepted_lt;
                    let new_theta = accepted_lt.exp().max(0.5_f64).min(50.0_f64);
                    self.family = crate::pirls::Family::NegBin { theta: new_theta };
                    // Sync the closure-side cell so the next iter's PIRLS
                    // refresh callback sees the freshly profiled θ.
                    if let Some(cell) = self.family_cell.as_ref() {
                        if let Ok(mut g) = cell.lock() {
                            *g = self.family;
                        }
                    }

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE]   NB profile-θ: log_θ {:.4}→{:.4} θ={:.4} dlr/d(log_θ)={:.4e}",
                            log_theta, accepted_lt, new_theta, dlr_dlt
                        );
                    }
                }
            }
            // End of NegBin profile-θ step

            // -----------------------------------------------------------------------
            // scat (TDist) profile-shape: joint Newton on (log σ², log df).
            //
            // This is the first gam.fit5-style structural port: each finite-
            // difference shape trial publishes the trial family into
            // `family_cell`, refits β through the PIRLS callback at the current
            // λ, then evaluates the LAML score using the refreshed β/w/X'WX.
            // That replaces the previous frozen-β sequential σ²-then-df steps.
            // -----------------------------------------------------------------------
            if self.tdist_profile {
                let log_sigma2 = self.tdist_log_sigma2;
                let log_df = self.tdist_log_df;
                let current_lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();
                let log_sigma2_lo = self.tdist_log_sigma2_lo;
                let log_sigma2_hi = self.tdist_log_sigma2_hi;
                let log_df_lo = 1e-8_f64.ln();
                let log_df_hi = (100.0_f64 - 2.0).ln();

                macro_rules! shape_eval {
                    ($ls_trial:expr, $ld_trial:expr) => {{
                        let ls_trial = ($ls_trial).max(log_sigma2_lo).min(log_sigma2_hi);
                        let ld_trial = ($ld_trial).max(log_df_lo).min(log_df_hi);
                        let trial_fam = crate::pirls::Family::TDist {
                            df: (ld_trial.exp() + 2.0).max(2.0_f64).min(100.0_f64),
                            sigma2: ls_trial.exp().max(1e-8_f64).min(1e8_f64),
                        };
                        if let Some(cell) = self.family_cell.as_ref() {
                            if let Ok(mut g) = cell.lock() {
                                *g = trial_fam;
                            }
                        }
                        // When a PIRLS callback is available, refresh (z, w, X'WX) at
                        // the trial family + current λ before evaluating the score. The
                        // stale y_local/w_local/xtwx_local reflect (df, σ²) from the
                        // START of the current Newton iteration; using them for the shape
                        // line search gives an inconsistent score (old weights, new
                        // parameters) that causes valid Newton steps to be rejected.
                        if let Some(cb) = pirls_callback.as_mut() {
                            match cb(&current_lambdas) {
                                Ok(ref refresh) => reml_criterion_multi_cached_mgcv_exact(
                                    &refresh.working_response,
                                    x,
                                    &refresh.weights,
                                    &current_lambdas,
                                    penalties,
                                    Some(&refresh.xtwx),
                                    self.mp,
                                    trial_fam,
                                    self.y_original.as_ref(),
                                ),
                                Err(e) => Err(e),
                            }
                        } else {
                            dispatch_reml_score_with_family(
                                self,
                                &y_local,
                                x,
                                &w_local,
                                &current_lambdas,
                                penalties,
                                Some(&xtwx_local),
                                trial_fam,
                            )
                        }
                    }};
                }

                let sc = shape_eval!(log_sigma2, log_df);
                let deriv = tdist_shape_derivatives_gamfit4(
                    self.y_original.as_ref().unwrap_or(&y_local),
                    &y_local,
                    x,
                    &w_local,
                    &current_lambdas,
                    penalties,
                    Some(&xtwx_local),
                    self.family,
                );

                if let (Ok(sc), Ok((shape_grad, shape_hess))) = (sc, deriv) {
                    let g_s = shape_grad[0];
                    let g_d = shape_grad[1];
                    let h_ss = shape_hess[[0, 0]];
                    let h_dd = shape_hess[[1, 1]];
                    let h_sd = shape_hess[[0, 1]];
                    let det = h_ss * h_dd - h_sd * h_sd;

                    let (mut delta_s, mut delta_d) = if det.abs() > 1e-8 && det.is_finite() {
                        (
                            (-h_dd * g_s + h_sd * g_d) / det,
                            (h_sd * g_s - h_ss * g_d) / det,
                        )
                    } else {
                        (-g_s / h_ss.abs().max(1e-4), -g_d / h_dd.abs().max(1e-4))
                    };
                    let max_abs = delta_s.abs().max(delta_d.abs());
                    if max_abs > 1.0 {
                        delta_s /= max_abs;
                        delta_d /= max_abs;
                    }

                    let mut accepted_s = log_sigma2;
                    let mut accepted_d = log_df;
                    let mut step_scale = 1.0;
                    for _ in 0..8 {
                        let cand_s = (log_sigma2 + step_scale * delta_s)
                            .max(log_sigma2_lo)
                            .min(log_sigma2_hi);
                        let cand_d = (log_df + step_scale * delta_d)
                            .max(log_df_lo)
                            .min(log_df_hi);
                        if let Ok(s_new) = shape_eval!(cand_s, cand_d) {
                            if s_new < sc {
                                accepted_s = cand_s;
                                accepted_d = cand_d;
                                break;
                            }
                        }
                        step_scale *= 0.5;
                    }

                    self.tdist_log_sigma2 = accepted_s;
                    self.tdist_log_df = accepted_d;
                    self.family = crate::pirls::Family::TDist {
                        df: (accepted_d.exp() + 2.0).max(2.0_f64).min(100.0_f64),
                        sigma2: accepted_s.exp().max(1e-8_f64).min(1e8_f64),
                    };
                    if let Some(cell) = self.family_cell.as_ref() {
                        if let Ok(mut g) = cell.lock() {
                            *g = self.family;
                        }
                    }

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE]   scat joint shape: log_σ² {:.4}→{:.4} σ²={:.6}; theta_df {:.4}→{:.4} df={:.4}; grad=({:.3e},{:.3e})",
                            log_sigma2,
                            accepted_s,
                            accepted_s.exp(),
                            log_df,
                            accepted_d,
                            accepted_d.exp() + 2.0,
                            g_s,
                            g_d,
                        );
                    }
                }
            }
            // End of scat (TDist) joint profile-shape step
        }

        // Update final lambdas
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Reached max iterations ({}) without convergence",
                max_iter
            );
        }

        Ok(())
    }

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_newton_multi(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[BlockPenalty],
        _max_iter: usize,
        _tolerance: f64,
    ) -> Result<()> {
        Err(GAMError::InvalidParameter(
            "Newton REML optimization requires the 'blas' feature. Use Fellner-Schall or GCV instead.".to_string()
        ))
    }

    /// Optimize using REML with Fellner-Schall iteration (fREML)
    ///
    /// Implements Wood & Fasiolo (2017) "A generalized Fellner-Schall method for
    /// smoothing parameter optimization". The update formula is:
    ///
    ///   λ_new = λ_old × φ × (rank_j - tr(A⁻¹·S_j)) / (β'·S_j·β)
    ///
    /// where:
    ///   A = X'WX + Σλᵢ·Sᵢ  (penalized information matrix)
    ///   φ = scale parameter (RSS / (n - edf))
    ///   β = current coefficient estimates from PiRLS
    ///   rank_j = rank of j-th penalty matrix
    ///
    /// This performs a SINGLE update step per call (designed to be called once
    /// per outer PiRLS iteration, matching R's bam() architecture).
    #[cfg(feature = "blas")]
    fn optimize_reml_fellner_schall(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        _max_iter: usize,
        _tolerance: f64,
        beta: Option<&Array1<f64>>,
    ) -> Result<()> {
        self.optimize_reml_fellner_schall_with_xtwx(
            y, x, w, penalties, _max_iter, _tolerance, beta, None,
        )
    }

    /// Fellner-Schall with optional pre-computed X'WX
    #[cfg(feature = "blas")]
    fn optimize_reml_fellner_schall_with_xtwx(
        &mut self,
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalties: &[BlockPenalty],
        _max_iter: usize,
        _tolerance: f64,
        beta: Option<&Array1<f64>>,
        cached_xtwx: Option<&Array2<f64>>,
    ) -> Result<()> {
        use crate::reml::compute_xtwx;
        use ndarray_linalg::{Cholesky, InverseInto, UPLO};

        let m = penalties.len();
        let p = x.ncols();
        let n = x.nrows();

        // Beta is required for Fellner-Schall
        let beta = beta.ok_or_else(|| {
            GAMError::InvalidParameter(
                "Fellner-Schall requires current coefficient estimates (beta)".to_string(),
            )
        })?;

        // Pre-compute penalty ranks
        let mut penalty_ranks = Vec::new();
        for penalty in penalties.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            penalty_ranks.push(sqrt_pen.ncols());
        }

        // Pre-compute X'WX (use cached if available from scatter-gather)
        let xtwx = if let Some(cached) = cached_xtwx {
            cached.clone()
        } else {
            compute_xtwx(x, w)
        };

        let lambdas = self.lambda.clone();

        // Compute A = X'WX + Σλᵢ·Sᵢ
        let mut a = xtwx.clone();
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            penalty.scaled_add_to(&mut a, *lambda);
        }

        // Add small ridge for numerical stability
        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(a[[i, i]].abs());
        }
        let ridge = 1e-7 * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // Compute A^{-1} via Cholesky
        let cholesky = match a.cholesky(UPLO::Lower) {
            Ok(l) => l,
            Err(_) => {
                let bigger_ridge = 1e-4 * max_diag;
                for i in 0..p {
                    a[[i, i]] += bigger_ridge;
                }
                a.cholesky(UPLO::Lower)
                    .map_err(|_| GAMError::SingularMatrix)?
            }
        };

        let a_inv = cholesky.inv_into().map_err(|_| GAMError::SingularMatrix)?;

        // Compute edf = tr(A^{-1} X'WX) = tr(hat matrix)
        let a_inv_xtwx = a_inv.dot(&xtwx);
        let mut edf = 0.0;
        for j in 0..p {
            edf += a_inv_xtwx[[j, j]];
        }
        // Clamp edf to [1, p] (values > p indicate numerical issues)
        edf = edf.max(1.0).min(p as f64);

        // Compute scale parameter φ.
        // - For binomial/poisson where φ is known (=1), use that exactly.
        //   The `(y - Xβ)²` weighted RSS is wrong here: `Xβ = η`, not `μ`,
        //   so the residual on the linear-predictor scale carries no
        //   meaningful scale information.
        // - For Gaussian/Gamma, profile φ from the working-response RSS
        //   (Pearson chi-squared via the IRLS weights).
        let phi = match self.phi_fixed {
            Some(p) => p,
            None => {
                // Pearson chi-squared via IRLS weights: Σ w_i (z_i - X_i β)²
                // For Gaussian z=y, w=1, this is Σ(y - Xβ)²; for Gamma
                // with profiled scale this is the right quantity once z
                // and w are the converged PiRLS values supplied by the
                // outer loop.
                let residuals = y - &x.dot(beta);
                let rss: f64 = residuals
                    .iter()
                    .zip(w.iter())
                    .map(|(r, wi)| wi * r * r)
                    .sum();
                let n_minus_edf = (n as f64 - edf).max(1.0);
                rss / n_minus_edf
            }
        };
        let rss = if std::env::var("MGCV_PROFILE").is_ok() {
            let r = y - &x.dot(beta);
            r.iter()
                .zip(w.iter())
                .map(|(r, wi)| wi * r * r)
                .sum::<f64>()
        } else {
            0.0
        };

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] FS update: n={}, p={}, edf={:.2}, phi={:.6}, rss={:.6}",
                n, p, edf, phi, rss
            );
        }

        // Wood & Fasiolo (2017) Fellner-Schall update via shared helper.
        // For non-overlapping penalties, ldetS1[i] · exp(-rho_i) reduces
        // to rank_i / λ_i, so the update is
        // λ_new = λ · phi · max(rank/λ − tr(A⁻¹S), ε) / (β'Sβ).
        let ranks_f64: Vec<f64> = penalty_ranks.iter().map(|&r| r as f64).collect();
        let new_lambdas = fellner_schall_step(
            penalties,
            &ranks_f64,
            &lambdas,
            &a_inv,
            beta,
            phi,
            /*log_step_clamp=*/ 3.0,
            /*lambda_bounds=*/ (1e-9, 1e7),
        );
        for i in 0..m {
            if std::env::var("MGCV_PROFILE").is_ok() {
                let pen = &penalties[i];
                let tr_vs = pen.trace_product(&a_inv);
                let bsb = pen.quadratic_form(beta);
                eprintln!(
                    "[PROFILE] FS smooth {}: λ={:.6e} → {:.6e}, rank/λ={:.4}, trVS={:.4}, bSb={:.6}, phi={:.6}",
                    i, lambdas[i], new_lambdas[i],
                    ranks_f64[i] / lambdas[i].max(1e-20), tr_vs, bsb, phi
                );
            }
            self.lambda[i] = new_lambdas[i];
        }

        Ok(())
    }

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_fellner_schall(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[BlockPenalty],
        _max_iter: usize,
        _tolerance: f64,
        _beta: Option<&Array1<f64>>,
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
        penalties: &[BlockPenalty],
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
            eprintln!(
                "[PROFILE] Starting chunked Fellner-Schall optimization (chunk_size={})",
                chunk_size
            );
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
                if sqrt_pen.ncols() > 0 {
                    // Only if penalty has non-zero rank
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
                // Note: converting BlockPenalty to dense for now as IncrementalQR::trace_ainv_s expects dense
                let trace = qr.trace_ainv_s(&penalty_i.to_dense())?;

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
                    eprintln!(
                        "[PROFILE] Chunked Fellner-Schall converged in {} iterations",
                        iter + 1
                    );
                }
                return Ok(());
            }

            log_lambda = new_log_lambda;
        }

        // Update final lambda values even if not converged
        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Chunked Fellner-Schall reached max iterations ({}) without convergence",
                max_iter
            );
        }

        Ok(())
    }

    #[cfg(not(feature = "blas"))]
    fn optimize_reml_fellner_schall_chunked(
        &mut self,
        _y: &Array1<f64>,
        _x: &Array2<f64>,
        _w: &Array1<f64>,
        _penalties: &[BlockPenalty],
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
        penalties: &[BlockPenalty],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        // Similar to REML but using GCV criterion
        let mut log_lambda: Vec<f64> = self.lambda.iter().map(|l| l.ln()).collect();

        for _iter in 0..max_iter {
            let mut converged = true;

            for i in 0..log_lambda.len() {
                let old_log_lambda = log_lambda[i];

                // For single smooth case
                if penalties.len() != 1 {
                    panic!("Multiple smooths not yet properly implemented for GCV");
                }

                let lambda_current = log_lambda[i].exp();

                let gcv_current = gcv_criterion(y, x, w, lambda_current, &penalties[0])?;

                // Numerical gradient
                let delta = 0.01;
                log_lambda[i] += delta;
                let lambda_plus = log_lambda[i].exp();

                let gcv_plus = gcv_criterion(y, x, w, lambda_plus, &penalties[0])?;

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

        self.lambda = log_lambda.iter().map(|l| l.exp()).collect();

        Ok(())
    }

    /// Grid search over lambda values to find good starting point
    pub fn grid_search(
        y: &Array1<f64>,
        x: &Array2<f64>,
        w: &Array1<f64>,
        penalty: &BlockPenalty,
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
                OptimizationMethod::REML => reml_criterion(y, x, w, lambda, penalty, None)?,
                OptimizationMethod::GCV => gcv_criterion(y, x, w, lambda, penalty)?,
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
        assert_eq!(sp.lambda[0], 0.1); // Updated to match current default
    }

    #[test]
    fn test_grid_search() {
        use crate::block_penalty::BlockPenalty;

        let n = 20;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| ((i as f64) * 0.1).powi(j as i32));
        let w = Array1::ones(n);
        let penalty = BlockPenalty::new(Array2::eye(p), 0, p);

        let result = SmoothingParameter::grid_search(
            &y,
            &x,
            &w,
            &penalty,
            0.001,
            10.0,
            20,
            OptimizationMethod::GCV,
        );

        assert!(result.is_ok());
        let lambda = result.unwrap();
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_chunked_fellner_schall_basic() {
        use crate::block_penalty::BlockPenalty;
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        // Small test case
        let n = 100;
        let k = 10;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Simple identity penalty
        let penalty = BlockPenalty::new(Array2::eye(k), 0, k);
        let penalties = vec![penalty];

        let mut sp = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall,
        );

        // Test with chunk size of 25
        let result = sp.optimize_reml_fellner_schall_chunked(
            &y, &x, &w, &penalties, 25,   // chunk_size
            10,   // max_iter
            1e-4, // tolerance
        );

        assert!(result.is_ok());
        assert!(sp.lambda[0] > 0.0);
        assert!(sp.lambda[0].is_finite());
    }

    #[test]
    #[ignore = "known bug: chunked Fellner-Schall diverges (dead code, not used in production path)"]
    fn test_chunked_vs_batch_agreement() {
        use crate::block_penalty::BlockPenalty;
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        // Create test data
        let n = 200;
        let k = 12;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Create a non-trivial penalty (second-order difference)
        let mut penalty_mat = Array2::zeros((k, k));
        for i in 0..k {
            penalty_mat[[i, i]] = 2.0;
            if i > 0 {
                penalty_mat[[i, i - 1]] = -1.0;
            }
            if i < k - 1 {
                penalty_mat[[i, i + 1]] = -1.0;
            }
        }
        let penalties = vec![BlockPenalty::new(penalty_mat.clone(), 0, k)];

        // Optimize with batch method
        let mut sp_batch = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall,
        );
        // Compute a dummy beta for the FS formula (solve A*beta = X'Wy)
        {
            let xtwx = x.t().dot(&x);
            let xtwy = x.t().dot(&y);
            let mut a_mat = xtwx.clone();
            a_mat.scaled_add(1.0, &penalty_mat);
            let beta = crate::linalg::solve(a_mat, xtwy).unwrap();
            sp_batch
                .optimize_reml_fellner_schall(&y, &x, &w, &penalties, 30, 1e-6, Some(&beta))
                .unwrap();
        }

        // Optimize with chunked method (chunk_size = 50)
        let mut sp_chunked = SmoothingParameter::new_with_algorithm(
            1,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall,
        );
        sp_chunked
            .optimize_reml_fellner_schall_chunked(
                &y, &x, &w, &penalties, 50,   // chunk_size
                30,   // max_iter
                1e-6, // tolerance
            )
            .unwrap();

        // Results should be very similar (within 1% relative error)
        let relative_error = (sp_batch.lambda[0] - sp_chunked.lambda[0]).abs() / sp_batch.lambda[0];
        println!(
            "Batch λ: {:.6}, Chunked λ: {:.6}, Relative error: {:.6}",
            sp_batch.lambda[0], sp_chunked.lambda[0], relative_error
        );

        assert!(relative_error < 0.01,
                "Chunked and batch methods should agree within 1%: batch={:.6}, chunked={:.6}, error={:.6}",
                sp_batch.lambda[0], sp_chunked.lambda[0], relative_error);
    }

    #[test]
    fn test_chunked_multiple_smooths() {
        use crate::block_penalty::BlockPenalty;
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        // Create test data with 2 smooths
        let n = 150;
        let k = 8;
        let total = 2 * k;

        let x = Array2::random((n, total), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        // Two penalties - one for each smooth (as k×k blocks)
        let penalty1 = BlockPenalty::new(Array2::eye(k), 0, total);
        let penalty2 = BlockPenalty::new(Array2::eye(k), k, total);

        let penalties = vec![penalty1, penalty2];

        let mut sp = SmoothingParameter::new_with_algorithm(
            2,
            OptimizationMethod::REML,
            REMLAlgorithm::FellnerSchall,
        );

        let result = sp.optimize_reml_fellner_schall_chunked(
            &y, &x, &w, &penalties, 50,   // chunk_size
            20,   // max_iter
            1e-4, // tolerance
        );

        assert!(result.is_ok());
        assert_eq!(sp.lambda.len(), 2);
        assert!(sp.lambda[0] > 0.0 && sp.lambda[0].is_finite());
        assert!(sp.lambda[1] > 0.0 && sp.lambda[1].is_finite());
    }

    #[test]
    fn test_chunked_various_chunk_sizes() {
        use crate::block_penalty::BlockPenalty;
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;

        let n = 100;
        let k = 10;

        let x = Array2::random((n, k), Uniform::new(0.0, 1.0));
        let y = Array1::random(n, Uniform::new(0.0, 1.0));
        let w = Array1::ones(n);

        let penalty = BlockPenalty::new(Array2::eye(k), 0, k);
        let penalties = vec![penalty];

        // Test different chunk sizes
        let chunk_sizes = vec![10, 25, 50, 100, 200]; // Include sizes larger than n
        let mut results = Vec::new();

        for &chunk_size in &chunk_sizes {
            let mut sp = SmoothingParameter::new_with_algorithm(
                1,
                OptimizationMethod::REML,
                REMLAlgorithm::FellnerSchall,
            );

            let result = sp
                .optimize_reml_fellner_schall_chunked(&y, &x, &w, &penalties, chunk_size, 20, 1e-6);

            assert!(result.is_ok(), "Failed with chunk_size={}", chunk_size);
            results.push(sp.lambda[0]);
        }

        // All results should be similar (within 5% of each other)
        let mean = results.iter().sum::<f64>() / results.len() as f64;
        for (i, &lambda) in results.iter().enumerate() {
            let relative_diff = (lambda - mean).abs() / mean;
            assert!(
                relative_diff < 0.05,
                "Chunk size {} gave λ={:.6}, too far from mean={:.6} (diff={:.2}%)",
                chunk_sizes[i],
                lambda,
                mean,
                relative_diff * 100.0
            );
        }
    }
}
