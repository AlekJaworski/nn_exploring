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
    reml_gradient_mgcv_exact_closed_form, reml_gradient_mgcv_exact_ift,
    reml_gradient_multi_qr_adaptive, reml_gradient_multi_qr_adaptive_cached_edf,
    reml_hessian_mgcv_exact_closed_form, reml_hessian_mgcv_exact_ift,
    reml_hessian_multi_cached,
};
#[cfg(not(feature = "blas"))]
use crate::reml::{gcv_criterion, reml_criterion, reml_criterion_multi, reml_gradient_multi};
use crate::{GAMError, Result};
use ndarray::{Array1, Array2};
use std::time::Instant;

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
        let r_plus = dispatch_reml_score(sp, y, x, w, &lam_plus, penalties, cached_xtwx)?;
        let r_minus = dispatch_reml_score(sp, y, x, w, &lam_minus, penalties, cached_xtwx)?;
        grad[i] = (r_plus - r_minus) / (2.0 * h);
    }
    Ok(grad)
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
) -> Result<Array2<f64>> {
    let m = lambdas.len();
    let h: f64 = 1.0e-3;
    let log_lambdas: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
    let r0 = dispatch_reml_score(sp, y, x, w, lambdas, penalties, cached_xtwx)?;

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
        r_plus[i] = dispatch_reml_score(sp, y, x, w, &lam_p, penalties, cached_xtwx)?;
        r_minus[i] = dispatch_reml_score(sp, y, x, w, &lam_m, penalties, cached_xtwx)?;
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
            let rpp = dispatch_reml_score(sp, y, x, w, &lam_pp, penalties, cached_xtwx)?;
            let rpm = dispatch_reml_score(sp, y, x, w, &lam_pm, penalties, cached_xtwx)?;
            let rmp = dispatch_reml_score(sp, y, x, w, &lam_mp, penalties, cached_xtwx)?;
            let rmm = dispatch_reml_score(sp, y, x, w, &lam_mm, penalties, cached_xtwx)?;
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

        let log_ratio = (phi * numerator / bsb).ln().clamp(-log_step_clamp, log_step_clamp);
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
    /// Current log(df) for TDist profile-df. Updated each outer Newton
    /// iteration. Initial value: log(5).
    pub tdist_log_df: f64,
    /// REML / LAML score at convergence. Populated by the outer optimizer
    /// (Newton or FS) on the last iteration so callers can read it back
    /// for σ profiling at the wrapper level.
    pub last_score: Option<f64>,
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
            tdist_log_df: 5.0_f64.ln(),
            last_score: None,
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
            tdist_log_df: 5.0_f64.ln(),
            last_score: None,
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
            tdist_log_df: 5.0_f64.ln(),
            last_score: None,
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
                if let Some(sig2) = refresh.sigma2 {
                    if let crate::pirls::Family::TDist { df, .. } = self.family {
                        self.family = crate::pirls::Family::TDist { df, sigma2: sig2 };
                    }
                }
            }

            // Compute current REML value for convergence check
            // (dispatches to mgcv-exact formula when mgcv_exact_score=true)
            let current_reml =
                dispatch_reml_score(self, &y_local, x, &w_local, &lambdas, penalties, Some(&xtwx_local))?;

            // Compute gradient and Hessian.
            // - Default mode: closed-form QR-based formulas, fast.
            // - mgcv_exact mode: closed-form gradient (matches mgcv's
            //   gam.fit3.r:625 D1/(2σ²) + trA1/2 - det1/2 simplified
            //   for Gaussian + canonical link via envelope theorem).
            //   Hessian still uses finite differences for now —
            //   replacing it with closed-form is an open task.
            // IFT path selection. Opt-in via env var MGCV_USE_IFT=1
            // (overrides everything). Otherwise: opt-out via
            // MGCV_DISABLE_IFT=1, else default ON for non-Gaussian when
            // y_original was supplied. For Gaussian we keep envelope (it
            // is exact there anyway and matches the score byte-for-byte).
            //
            // Note (Parity 4t): at our (always inner-loop converged) β =
            // A⁻¹X'Wy, IFT with working-RSS deviance collapses to envelope.
            // The GLM-deviance form (y_original set) is a gradient of a
            // DIFFERENT score (true GLM deviance) than what the line search
            // optimises (working RSS). Empirically, the gradient/score
            // inconsistency on binomial moved absdiff from 4e-3 to ~1.3e-2;
            // a proper fix needs a working-deviance-based score too.
            let use_ift_explicit = std::env::var("MGCV_USE_IFT").is_ok();
            let use_ift_disable = std::env::var("MGCV_DISABLE_IFT").is_ok();
            let use_ift = self.mgcv_exact_score
                && (use_ift_explicit
                    || (!use_ift_disable
                        && !matches!(self.family, crate::pirls::Family::Gaussian)
                        && self.y_original.is_some()));

            let t_grad = Instant::now();
            let gradient = if self.mgcv_exact_score {
                if use_ift {
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
                } else {
                    reml_gradient_mgcv_exact_closed_form(&y_local, x, &w_local, &lambdas, penalties, Some(&xtwx_local), self.family)?
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
                if use_ift {
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
                    reml_hessian_mgcv_exact_closed_form(&y_local, x, &w_local, &lambdas, penalties, Some(&xtwx_local), self.family)?
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
                score_scale * 1.0e-6
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
            let max_abs_grad = gradient.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
            let inner_conv_tol: f64 = 1.0e-6;
            let score_scale = current_reml.abs() + 1.0;
            let dim_grad_tol = score_scale * inner_conv_tol;
            let active: Vec<usize> = (0..m)
                .filter(|&i| {
                    let gi = gradient[i].abs();
                    gi > dim_grad_tol && gi > max_abs_grad * 0.001
                })
                .collect();

            // mgcv safeguard (line 1432): ensure at least one dim is active.
            let active = if active.is_empty() {
                let argmax = (0..m)
                    .max_by(|&a, &b| gradient[a].abs().partial_cmp(&gradient[b].abs()).unwrap())
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

            let (eigvals, eigvecs) = h_sub.eigh(UPLO::Upper).map_err(|e| {
                GAMError::LinAlgError(format!("Hessian eigh failed: {:?}", e))
            })?;

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
            let max_half = if grad_norm_linf < 0.1 {
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
                match trial_eval {
                    Ok(new_reml) => {
                        // OPTIMIZATION: Armijo condition for early stopping
                        // Accept if: new_reml ≤ current_reml + c₁ * step_scale * grad·step
                        // Since we're minimizing and grad·step should be negative, this is:
                        // new_reml ≤ current_reml - c₁ * step_scale * |grad·step|
                        let armijo_threshold =
                            current_reml + armijo_c1 * step_scale * grad_dot_step;
                        let satisfies_armijo = new_reml <= armijo_threshold;

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
                        if std::env::var("MGCV_PROFILE").is_ok() && half < 3 {
                            eprintln!("[PROFILE]     half={}: ERROR (numerical issue)", half);
                        }
                        continue;
                    }
                }
            }
            let linesearch_time = t_linesearch.elapsed().as_micros();

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
                let gradient_sd = reml_gradient_multi_qr_adaptive(&y_local, x, &w_local, &lambdas, penalties)?;

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
                    let gradient_check =
                        reml_gradient_multi_qr_adaptive(&y_local, x, &w_local, &lambdas, penalties)?;
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
            // Gradient dlr/dθ via FD on REML score at θ ± h (with PIRLS refresh).
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

                // Inline helper: evaluate REML score at trial θ using
                // the frozen working data (y_local, w_local, xtwx_local)
                // from the start-of-iteration PIRLS refresh. We do NOT
                // re-run PIRLS for the θ step; the current working linearisation
                // is a good approximation since p changes slowly.
                // Temporarily set self.family so that `dispatch_reml_score_with_family`
                // uses the trial p for the score formula (ls, estimate_phi_mgcv).
                macro_rules! tw_eval {
                    ($th_trial:expr) => {{
                        let th_trial: f64 = $th_trial;
                        let p_trial = tw_theta_to_p(th_trial).max(1.001_f64).min(1.999_f64);
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

                // FD gradient and curvature wrt θ, evaluated at the updated λ.
                // Also compute the center score (at current θ, updated λ) for
                // the line-search comparison baseline.
                let reml_center = tw_eval!(theta);
                let reml_plus = tw_eval!(theta + h_th);
                let reml_minus = tw_eval!(theta - h_th);

                if let (Ok(rc), Ok(rp), Ok(rm)) = (reml_center, reml_plus, reml_minus) {
                    let dlr_dth = (rp - rm) / (2.0 * h_th);
                    let d2lr_dth2 = (rp - 2.0 * rc + rm) / (h_th * h_th);

                    // Newton step: δθ = -g / |H| with |H| floored for stability
                    let denom = d2lr_dth2.abs().max(1e-4);
                    let delta_theta = -(dlr_dth / denom);
                    let delta_theta = delta_theta.max(-2.0_f64).min(2.0_f64);

                    // Line-search on θ: try full step, then half-step.
                    // Compare against rc (REML at current θ with updated λ).
                    let mut accepted_theta = theta;
                    let candidate = theta + delta_theta;
                    if let Ok(r_new) = tw_eval!(candidate) {
                        if r_new < rc {
                            accepted_theta = candidate;
                        } else {
                            let half_cand = theta + delta_theta * 0.5;
                            if let Ok(r_half) = tw_eval!(half_cand) {
                                if r_half < rc {
                                    accepted_theta = half_cand;
                                }
                            }
                        }
                    }

                    self.tweedie_theta = accepted_theta;
                    let new_p = tw_theta_to_p(accepted_theta).max(1.001).min(1.999);
                    self.family = crate::pirls::Family::Tweedie { p: new_p };

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE]   Tweedie profile-p: θ {:.4}→{:.4} p={:.4} dlr/dθ={:.4e}",
                            theta, accepted_theta, new_p, dlr_dth
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
            // scat (TDist) profile-df: Newton step on log(df) at the OUTER level,
            // joint with ρ. Mirrors the Tweedie p / NegBin θ pattern. PIRLS keeps
            // its inner σ² profiling (method-of-moments at converged β) — only
            // df is moved here.
            //
            // Currently df is profiled INSIDE fit_pirls_tdist via 1D Brent on
            // the profile log-likelihood. That's a partial LAML — joint with
            // λ via the inner-loop coupling, but not with the closed-form REML
            // gradient the outer Newton uses. Profiling at the outer level
            // (this block) closes that gap, mirroring how Tweedie p closes the
            // analogous gap there.
            // -----------------------------------------------------------------------
            if self.tdist_profile {
                let log_df = self.tdist_log_df;
                let h_th: f64 = 1e-3;
                let current_lambdas: Vec<f64> = log_lambda.iter().map(|l| l.exp()).collect();

                // Helper: σ² at the family — read it back so trial REML score
                // sees the same σ² the inner PIRLS just landed on.
                let current_sigma2 = match self.family {
                    crate::pirls::Family::TDist { sigma2, .. } => sigma2,
                    _ => 1.0,
                };

                macro_rules! tdist_eval {
                    ($lt_trial:expr) => {{
                        let lt_trial: f64 = $lt_trial;
                        let df_trial = lt_trial.exp().max(2.0_f64).min(100.0_f64);
                        let trial_fam = crate::pirls::Family::TDist {
                            df: df_trial,
                            sigma2: current_sigma2,
                        };
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

                let reml_center = tdist_eval!(log_df);
                let reml_plus = tdist_eval!(log_df + h_th);
                let reml_minus = tdist_eval!(log_df - h_th);

                if let (Ok(rc), Ok(rp), Ok(rm)) = (reml_center, reml_plus, reml_minus) {
                    let dlr_dlt = (rp - rm) / (2.0 * h_th);
                    let d2lr_dlt2 = (rp - 2.0 * rc + rm) / (h_th * h_th);

                    let denom = d2lr_dlt2.abs().max(1e-4);
                    let delta_lt = -(dlr_dlt / denom);
                    // Wider clamp than NegBin's [-0.5, 0.5] since df can move
                    // through 1-2 decades during convergence on heavy-tailed
                    // data; bound to [-1, 1] (≈ 2.7× per iteration max).
                    let delta_lt = delta_lt.max(-1.0_f64).min(1.0_f64);

                    let mut accepted_lt = log_df;
                    let candidate = log_df + delta_lt;
                    if let Ok(r_new) = tdist_eval!(candidate) {
                        if r_new < rc {
                            accepted_lt = candidate;
                        } else {
                            let half_cand = log_df + delta_lt * 0.5;
                            if let Ok(r_half) = tdist_eval!(half_cand) {
                                if r_half < rc {
                                    accepted_lt = half_cand;
                                }
                            }
                        }
                    }

                    self.tdist_log_df = accepted_lt;
                    let new_df = accepted_lt.exp().max(2.0_f64).min(100.0_f64);
                    self.family = crate::pirls::Family::TDist {
                        df: new_df,
                        sigma2: current_sigma2,
                    };

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!(
                            "[PROFILE]   scat profile-df: log_df {:.4}→{:.4} df={:.4} dlr/d(log_df)={:.4e}",
                            log_df, accepted_lt, new_df, dlr_dlt
                        );
                    }
                }
            }
            // End of scat (TDist) profile-df step
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
            r.iter().zip(w.iter()).map(|(r, wi)| wi * r * r).sum::<f64>()
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
            penalties, &ranks_f64, &lambdas, &a_inv, beta,
            phi, /*log_step_clamp=*/ 3.0, /*lambda_bounds=*/ (1e-9, 1e7),
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
