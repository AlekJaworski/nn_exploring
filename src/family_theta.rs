//! Outer Newton on family-shape parameters at fixed β (Path B helper, task B5).
//!
//! Mirrors mgcv `estimate.theta` (efam.r:5) for the `scat` extended family —
//! given (β̂, X, y, prior_w) and prior estimates of `(log σ², log(df - 2))`,
//! perform a 2-D Newton step on the negative log-likelihood of the t-density
//! wrt the working coords `θ = (log σ², log(ν - 2))`. Step-halving (Armijo)
//! mirrors mgcv's loop body exactly.
//!
//! This is the **outer θ** helper used by Path B's `fit_pirls_fastreml`
//! driver between fREML iterations (mgcv `bgam.fitd` body, bam.r:614-630).
//! Independent of the inner λ Newton — that's what makes the fREML approach
//! fast for the binned hot path.
//!
//! ### Math contract
//!
//! Working coords (user-facing): `θ = (log σ², log(ν - min_df))`. We default
//! `min_df = 3.0` to match mgcv `scat`'s `min.df = 3` (efam.r:3552), which the
//! fREML outer-Newton requires for parity: the step clamp, PSD Hessian
//! thresholding, and convergence tolerance all act on the transformed
//! variable, so the floor matters even though the (β, ν, σ²) optimum is
//! basis-invariant in theory. Kept configurable via
//! [`estimate_scat_theta_outer_with_min_df`]. Note the t-density's variance
//! floor is ν > 2; mgcv's `min.df = 3` is a stricter bound for numerical
//! stability of the IRLS weights.
//!
//! Negative log-likelihood (per `nlogl` in efam.r:14):
//!     nll(σ², ν) = Σ_i w_i · [
//!         -lgamma((ν+1)/2) + lgamma(ν/2) + ½·log(π·ν·σ²)
//!         + ½·(ν+1)·log(1 + (y_i - μ_i)² / (ν·σ²))
//!     ]
//! where μ_i = η_i (identity link is assumed for scat per mgcv default).
//! `w_i` is the prior weight (`prior_weights[i]`; defaults to 1).
//!
//! Gradient and Hessian wrt `(log σ², log(ν - min_df))` are derived by
//! literal port of mgcv `scat$Dd` (efam.r:3616-3687) and `scat$ls`
//! (efam.r:3699-3722). Both use mgcv's native coords `(log(ν - min_df), log σ)`
//! per-row; we apply a permutation + scale-by-½ on the σ row to land in
//! `(log σ², log(ν - min_df))` user coords.
//!
//! ### Convergence
//!
//! Mirrors mgcv `estimate.theta`:
//!   - Per-coord active-set: a coord is "active" iff |g_i| > tol·(|nll|+1).
//!   - At each iter solve the eigen-PSD-corrected linear system
//!     `H[active,active] · δ = -g[active]`.
//!   - Limit `max |δ_i| ≤ 4` (mgcv `ms` clamp at efam.r:69).
//!   - Halve step until deviance decreases (Armijo); abort after 25 halvings.
//!   - Stop when all coords inactive or step length below tol.

use ndarray::Array1;

use crate::pirls::{digamma, log_gamma, trigamma};

/// Result of one outer-Newton run on `(log σ², log(df - min_df))`.
#[derive(Debug, Clone)]
pub struct ThetaNewtonStep {
    /// Converged `log σ²`.
    pub log_sigma2: f64,
    /// Converged `log(ν - min_df)`.
    pub log_df_minus2: f64,
    /// Negative log-likelihood at the final iterate.
    pub deviance: f64,
    /// True iff Newton terminated by the gradient-norm test (not by `max_iters`).
    pub converged: bool,
    /// Newton iterations consumed.
    pub iters: usize,
    /// Final gradient norm `‖g‖_∞` (in working coords). Useful for callers
    /// that want to gate on Path B convergence on its own scale.
    pub grad_inf_norm: f64,
}

/// Default value for `min_df`. Matches mgcv's `scat(min.df = 3)` default
/// (efam.r:3552). The fREML outer Newton's step clamp / PSD threshold /
/// convergence test all act on `log(ν - min_df)`, so this floor must match
/// mgcv to land on the same fREML optimum. Override via
/// [`estimate_scat_theta_outer_with_min_df`] when needed.
pub const DEFAULT_MIN_DF: f64 = 3.0;

/// 2-D outer Newton on `(log σ², log(df - 2))` at fixed β. See module-level
/// docs for the math contract.
///
/// `eta` is `X·β̂` (identity link assumed). `prior_weights` is the `weights=`
/// argument — pass `None` for unit weights. `max_iters` caps the outer
/// Newton loop. `tol` is the relative gradient tolerance:
/// converged ⇔ `max_i |g_i| ≤ tol · (|nll| + 1)` (mgcv efam.r:56).
pub fn estimate_scat_theta_outer(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_sigma2_init: f64,
    log_df_minus2_init: f64,
    max_iters: usize,
    tol: f64,
) -> ThetaNewtonStep {
    estimate_scat_theta_outer_with_scale(
        y,
        eta,
        prior_weights,
        log_sigma2_init,
        log_df_minus2_init,
        1.0,
        max_iters,
        tol,
    )
}

/// 2-D outer Newton with explicit mgcv scale. `scale` enters mgcv's
/// `estimate.theta` as `dev/(2*scale) - ls`; fREML passes `exp(log.phi)`.
pub fn estimate_scat_theta_outer_with_scale(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_sigma2_init: f64,
    log_df_minus2_init: f64,
    scale: f64,
    max_iters: usize,
    tol: f64,
) -> ThetaNewtonStep {
    estimate_scat_theta_outer_with_min_df(
        y,
        eta,
        prior_weights,
        log_sigma2_init,
        log_df_minus2_init,
        DEFAULT_MIN_DF,
        scale,
        max_iters,
        tol,
    )
}

/// 1-D wrapper: hold ν fixed (user passes `log_df_minus2_fixed`) and Newton
/// only on `log σ²`. Used when the caller fixes `df=` upfront. Same Armijo
/// halving semantics as the 2-D path; converged when `|g_σ| ≤ tol·(|nll|+1)`.
pub fn estimate_scat_log_sigma2_outer(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_sigma2_init: f64,
    log_df_minus2_fixed: f64,
    max_iters: usize,
    tol: f64,
) -> ThetaNewtonStep {
    estimate_scat_log_sigma2_outer_with_scale(
        y,
        eta,
        prior_weights,
        log_sigma2_init,
        log_df_minus2_fixed,
        1.0,
        max_iters,
        tol,
    )
}

/// 1-D log-σ² Newton with explicit mgcv scale.
pub fn estimate_scat_log_sigma2_outer_with_scale(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_sigma2_init: f64,
    log_df_minus2_fixed: f64,
    scale: f64,
    max_iters: usize,
    tol: f64,
) -> ThetaNewtonStep {
    let min_df = DEFAULT_MIN_DF;
    let scale = scale.max(1e-300);
    let mut log_s2 = log_sigma2_init;
    let log_d = log_df_minus2_fixed;

    let mut state = nlogl_full(y, eta, prior_weights, log_s2, log_d, min_df, scale);
    let mut iters_used = 0usize;
    let mut converged_flag = false;
    let mut grad_inf = state.g[0].abs();

    for it in 0..max_iters {
        iters_used = it;
        // Active-set test in the 1-D slot (mgcv `uconv`).
        if state.g[0].abs() <= tol * (state.nll.abs() + 1.0) {
            converged_flag = true;
            grad_inf = state.g[0].abs();
            break;
        }
        // PSD-corrected scalar Hessian (mgcv `eh$values <- abs(eh$values)`).
        let h_abs = state.h[(0, 0)].abs();
        let thresh = (h_abs * 1e-5).max(1e-12);
        let h_eff = h_abs.max(thresh);
        let mut step = -state.g[0] / h_eff;
        // mgcv `ms` clamp (max |step| ≤ 4).
        if step.abs() > 4.0 {
            step = step.signum() * 4.0;
        }

        // Step halving (Armijo) — mgcv efam.r:73-82.
        let mut s = step;
        let mut trial_state = nlogl_full(y, eta, prior_weights, log_s2 + s, log_d, min_df, scale);
        let mut step_failed = false;
        let mut inner = 0usize;
        while trial_state.nll - state.nll > f64::EPSILON.powf(0.75) * state.nll.abs() {
            s *= 0.5;
            inner += 1;
            if (log_s2 + s - log_s2).abs() == 0.0 || inner > 25 {
                step_failed = true;
                break;
            }
            // Only nll is needed for the Armijo test; derivs recomputed on accept.
            let nll_only =
                nlogl_value_only(y, eta, prior_weights, log_s2 + s, log_d, min_df, scale);
            trial_state.nll = nll_only;
        }
        if step_failed {
            grad_inf = state.g[0].abs();
            iters_used = it + 1;
            break;
        }
        log_s2 += s;
        // Refresh gradient/Hessian at accepted step.
        state = nlogl_full(y, eta, prior_weights, log_s2, log_d, min_df, scale);
        grad_inf = state.g[0].abs();
        iters_used = it + 1;
        if state.g[0].abs() <= tol * (state.nll.abs() + 1.0) {
            converged_flag = true;
            break;
        }
    }

    ThetaNewtonStep {
        log_sigma2: log_s2,
        log_df_minus2: log_d,
        deviance: state.nll,
        converged: converged_flag,
        iters: iters_used,
        grad_inf_norm: grad_inf,
    }
}

/// 2-D outer Newton with explicit `min_df`. The general workhorse — see
/// [`estimate_scat_theta_outer`] for the default-`min_df=2` shortcut.
pub fn estimate_scat_theta_outer_with_min_df(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_sigma2_init: f64,
    log_df_minus2_init: f64,
    min_df: f64,
    scale: f64,
    max_iters: usize,
    tol: f64,
) -> ThetaNewtonStep {
    let scale = scale.max(1e-300);
    let mut log_s2 = log_sigma2_init;
    let mut log_d = log_df_minus2_init;

    let mut state = nlogl_full(y, eta, prior_weights, log_s2, log_d, min_df, scale);
    let mut iters_used = 0usize;
    let mut converged_flag = false;
    let mut grad_inf = state.g.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

    for it in 0..max_iters {
        iters_used = it;
        // mgcv per-coord active set: uconv[i] = |g_i| > tol·(|nll|+1).
        let active = [
            state.g[0].abs() > tol * (state.nll.abs() + 1.0),
            state.g[1].abs() > tol * (state.nll.abs() + 1.0),
        ];
        if !active[0] && !active[1] {
            converged_flag = true;
            break;
        }

        // Build the PSD-corrected Newton step on the active block.
        let (mut step0, mut step1) = (0.0_f64, 0.0_f64);
        match (active[0], active[1]) {
            (true, true) => {
                // 2×2 block: eigendecompose, lift non-positive eigenvalues
                // to `max(|λ|) · 1e-5`, solve δ = -V Λ⁻¹ Vᵀ g.
                let (h11, h12, h22) = (state.h[(0, 0)], state.h[(0, 1)], state.h[(1, 1)]);
                let (lam1, lam2, c, s) = sym_eig_2x2(h11, h12, h22);
                let max_abs = lam1.abs().max(lam2.abs()).max(1e-300);
                let thresh = max_abs * 1e-5;
                let l1 = lam1.abs().max(thresh);
                let l2 = lam2.abs().max(thresh);
                // V = [[c, -s], [s, c]], V^T g
                let g0 = state.g[0];
                let g1 = state.g[1];
                let vt_g0 = c * g0 + s * g1;
                let vt_g1 = -s * g0 + c * g1;
                let inv0 = vt_g0 / l1;
                let inv1 = vt_g1 / l2;
                // δ = -V · diag(1/λ) · Vᵀ g
                step0 = -(c * inv0 - s * inv1);
                step1 = -(s * inv0 + c * inv1);
            }
            (true, false) => {
                let h_abs = state.h[(0, 0)].abs();
                let thresh = (h_abs * 1e-5).max(1e-12);
                let h_eff = h_abs.max(thresh);
                step0 = -state.g[0] / h_eff;
            }
            (false, true) => {
                let h_abs = state.h[(1, 1)].abs();
                let thresh = (h_abs * 1e-5).max(1e-12);
                let h_eff = h_abs.max(thresh);
                step1 = -state.g[1] / h_eff;
            }
            (false, false) => unreachable!(),
        }

        // mgcv `ms` clamp (max-abs step ≤ 4).
        let ms = step0.abs().max(step1.abs());
        if ms > 4.0 {
            let scale = 4.0 / ms;
            step0 *= scale;
            step1 *= scale;
        }

        // Step halving (Armijo) — mgcv efam.r:73-82.
        let mut s = 1.0_f64;
        let mut step_failed = false;
        let mut inner = 0usize;
        let mut cand_s2 = log_s2 + step0;
        let mut cand_d = log_d + step1;
        let mut trial_nll = nlogl_value_only(y, eta, prior_weights, cand_s2, cand_d, min_df, scale);
        // mgcv tests `nll1 - nll > eps^.75 · |nll|`, halving until it
        // doesn't (i.e. until the step is non-degrading within tolerance).
        while trial_nll - state.nll > f64::EPSILON.powf(0.75) * state.nll.abs() {
            s *= 0.5;
            inner += 1;
            cand_s2 = log_s2 + s * step0;
            cand_d = log_d + s * step1;
            if (cand_s2 == log_s2 && cand_d == log_d) || inner > 25 {
                step_failed = true;
                break;
            }
            trial_nll = nlogl_value_only(y, eta, prior_weights, cand_s2, cand_d, min_df, scale);
        }
        if step_failed {
            iters_used = it + 1;
            break;
        }
        log_s2 = cand_s2;
        log_d = cand_d;

        // Refresh gradient/Hessian at the accepted point.
        state = nlogl_full(y, eta, prior_weights, log_s2, log_d, min_df, scale);
        grad_inf = state.g.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        iters_used = it + 1;
        if state.g[0].abs() <= tol * (state.nll.abs() + 1.0)
            && state.g[1].abs() <= tol * (state.nll.abs() + 1.0)
        {
            converged_flag = true;
            break;
        }
    }

    ThetaNewtonStep {
        log_sigma2: log_s2,
        log_df_minus2: log_d,
        deviance: state.nll,
        converged: converged_flag,
        iters: iters_used,
        grad_inf_norm: grad_inf,
    }
}

/// Cache holding `nll(θ)`, its gradient and Hessian wrt the user-coord
/// vector `(log σ², log(ν - min_df))`.
#[derive(Debug, Clone)]
struct NlogLState {
    nll: f64,
    g: [f64; 2],
    /// Symmetric 2×2 Hessian indexed by `(0,0) = ∂²/∂(log σ²)²`,
    /// `(1,1) = ∂²/∂(log(ν-min_df))²`, off-diag = mixed.
    h: SymHess2,
}

#[derive(Debug, Clone, Copy)]
struct SymHess2 {
    h00: f64,
    h01: f64,
    h11: f64,
}

impl std::ops::Index<(usize, usize)> for SymHess2 {
    type Output = f64;
    fn index(&self, idx: (usize, usize)) -> &f64 {
        match idx {
            (0, 0) => &self.h00,
            (0, 1) | (1, 0) => &self.h01,
            (1, 1) => &self.h11,
            _ => panic!("SymHess2 index out of range"),
        }
    }
}

/// Compute nll + gradient + Hessian at `(log_s2, log_d)`.
fn nlogl_full(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_s2: f64,
    log_d: f64,
    min_df: f64,
    scale: f64,
) -> NlogLState {
    let n = y.len();
    debug_assert_eq!(eta.len(), n);
    if let Some(pw) = prior_weights.as_ref() {
        debug_assert_eq!(pw.len(), n);
    }

    let sigma2 = log_s2.exp();
    let nu2 = log_d.exp(); // = ν - min_df
    let nu = nu2 + min_df;
    let sigma = sigma2.sqrt();

    // ── Accumulators for dev = Σ wt·(ν+1)·log(a) and Dth, Dth2 cols
    //    in mgcv coords θ_mgcv = (log(ν-min_df), log σ). ──
    let mut dev = 0.0_f64;
    // Dth col sums (length 2): [Σ Dth[,1], Σ Dth[,2]].
    let mut sum_dth = [0.0_f64; 2];
    // Dth2 col sums (length 3): [Σ Dth2[,1] (th1,th1), Σ Dth2[,2] (th1,th2), Σ Dth2[,3] (th2,th2)].
    let mut sum_dth2 = [0.0_f64; 3];
    // sum of weights (for ls).
    let mut sum_w = 0.0_f64;

    let nu1 = nu + 1.0;
    let inv_sigma2 = 1.0 / sigma2.max(1e-300);

    for i in 0..n {
        let w_i = prior_weights.as_ref().map(|pw| pw[i]).unwrap_or(1.0);
        sum_w += w_i;
        let ym = y[i] - eta[i];
        let a = 1.0 + (ym * ym) * inv_sigma2 / nu;
        let log_a = a.ln();

        // Dev contribution (per dev.resids in scat).
        dev += w_i * nu1 * log_a;

        // Per-row Dd ingredients — literal port of mgcv scat$Dd (efam.r:3616-3687)
        // with `theta = (log(ν-min_df), log σ)` (mgcv coords).
        let nu_sig2_a = nu * sigma2 * a;
        let sig2_a = sigma2 * a;
        let f = nu1 * ym / nu_sig2_a;
        let f1 = ym / nu_sig2_a;
        let fym = f * ym;
        let f1ym = f1 * ym;
        let ymsig2a = ym / sig2_a;
        let fymf1 = fym * f1;
        let fymf1ym = fym * f1ym;

        let nu2nu = nu2 / nu; // (ν - min_df)/ν

        // Dth[,1] (mgcv θ1 = log(ν - min_df)).
        let dth1 = w_i * nu2 * (log_a - fym / nu);
        // Dth[,2] (mgcv θ2 = log σ).
        let dth2 = -2.0 * w_i * fym;
        sum_dth[0] += dth1;
        sum_dth[1] += dth2;

        // Dth2[,1] = ∂²D/∂θ1² (efam.r:3665).
        let dth2_11 = w_i
            * (nu2 * log_a
                + nu2nu * ym * ym * (-2.0 * nu2 - nu1 + 2.0 * nu1 * nu2nu - nu1 * nu2nu * f1ym)
                    / nu_sig2_a);
        // Dth2[,2] = ∂²D/∂θ1∂θ2 (efam.r:3667).
        let dth2_12 = 2.0 * w_i * (fym - ym * ymsig2a - fymf1ym) * nu2nu;
        // Dth2[,3] = ∂²D/∂θ2² (efam.r:3668).
        let dth2_22 = 4.0 * w_i * fym * (1.0 - f1ym);
        sum_dth2[0] += dth2_11;
        sum_dth2[1] += dth2_12;
        sum_dth2[2] += dth2_22;
    }

    // ── ls and its derivs (mgcv scat$ls, efam.r:3699-3722) ──
    // term_per = lgamma((ν+1)/2) - lgamma(ν/2) - log(σ·(π·ν)^.5).
    let half_nu1 = (nu + 1.0) * 0.5;
    let half_nu = nu * 0.5;
    let term_ls = log_gamma(half_nu1)
        - log_gamma(half_nu)
        - (sigma * (std::f64::consts::PI * nu).sqrt()).ln();
    let ls = sum_w * term_ls;

    // First deriv of per-row term wrt θ_mgcv:
    //   lsth_per_1 = nu2 · ψ((ν+1)/2)/2 - nu2 · ψ(ν/2)/2 - 0.5·(nu2/ν)
    //   lsth_per_2 = -1
    let psi_half_nu1 = digamma(half_nu1);
    let psi_half_nu = digamma(half_nu);
    let nu2nu = nu2 / nu;
    let lsth_per_1 = nu2 * psi_half_nu1 * 0.5 - nu2 * psi_half_nu * 0.5 - 0.5 * nu2nu;
    let lsth1 = [sum_w * lsth_per_1, -sum_w];

    // Second deriv per-row (lsth2[1,1] only; others zero):
    //   = nu2² · ψ'((ν+1)/2)/4 + nu2 · ψ((ν+1)/2)/2
    //     - nu2² · ψ'(ν/2)/4 - nu2 · ψ(ν/2)/2
    //     + 0.5·(nu2/ν)² - 0.5·(nu2/ν)
    let trig_half_nu1 = trigamma(half_nu1);
    let trig_half_nu = trigamma(half_nu);
    let lsth2_per_11 = nu2 * nu2 * trig_half_nu1 * 0.25 + nu2 * psi_half_nu1 * 0.5
        - nu2 * nu2 * trig_half_nu * 0.25
        - nu2 * psi_half_nu * 0.5
        + 0.5 * nu2nu * nu2nu
        - 0.5 * nu2nu;
    let lsth2_11 = sum_w * lsth2_per_11;

    // ── Assemble nll, g, H in mgcv coords θ_mgcv = (log(ν-min_df), log σ). ──
    let dev_scale = 0.5 / scale.max(1e-300);
    let nll = dev_scale * dev - ls;
    let g_mgcv = [
        dev_scale * sum_dth[0] - lsth1[0], // wrt log(ν - min_df)
        dev_scale * sum_dth[1] - lsth1[1], // wrt log σ
    ];
    let h_mgcv_11 = dev_scale * sum_dth2[0] - lsth2_11; // ∂²/∂θ1²
    let h_mgcv_12 = dev_scale * sum_dth2[1]; // ∂²/∂θ1∂θ2 (ls cross is 0)
    let h_mgcv_22 = dev_scale * sum_dth2[2]; // ∂²/∂θ2²

    // ── Chain-rule to user coords θ_user = (log σ², log(ν - min_df)). ──
    // θ_user_0 = log σ² = 2·θ_mgcv_2  →  ∂/∂θ_user_0 = (1/2)·∂/∂θ_mgcv_2
    // θ_user_1 = log(ν - min_df) = θ_mgcv_1  →  ∂/∂θ_user_1 = ∂/∂θ_mgcv_1
    let g_user = [0.5 * g_mgcv[1], g_mgcv[0]];
    let h_user_00 = 0.25 * h_mgcv_22; // (1/2)²
    let h_user_01 = 0.5 * h_mgcv_12; // (1/2) · 1
    let h_user_11 = h_mgcv_11;

    NlogLState {
        nll,
        g: g_user,
        h: SymHess2 {
            h00: h_user_00,
            h01: h_user_01,
            h11: h_user_11,
        },
    }
}

/// Value-only nll (skips derivative work). Used inside the Armijo halving
/// loop; mgcv does the same (`deriv=0` flag in efam.r:81).
fn nlogl_value_only(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    log_s2: f64,
    log_d: f64,
    min_df: f64,
    scale: f64,
) -> f64 {
    let n = y.len();
    debug_assert_eq!(eta.len(), n);
    if let Some(pw) = prior_weights.as_ref() {
        debug_assert_eq!(pw.len(), n);
    }
    let sigma2 = log_s2.exp();
    let nu2 = log_d.exp();
    let nu = nu2 + min_df;
    let sigma = sigma2.sqrt();
    let nu1 = nu + 1.0;
    let inv_sigma2 = 1.0 / sigma2.max(1e-300);

    let mut dev = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for i in 0..n {
        let w_i = prior_weights.as_ref().map(|pw| pw[i]).unwrap_or(1.0);
        sum_w += w_i;
        let ym = y[i] - eta[i];
        let a = 1.0 + (ym * ym) * inv_sigma2 / nu;
        dev += w_i * nu1 * a.ln();
    }
    let half_nu1 = (nu + 1.0) * 0.5;
    let half_nu = nu * 0.5;
    let term_ls = log_gamma(half_nu1)
        - log_gamma(half_nu)
        - (sigma * (std::f64::consts::PI * nu).sqrt()).ln();
    let ls = sum_w * term_ls;
    0.5 * dev / scale.max(1e-300) - ls
}

/// Symmetric 2×2 eigendecomposition. Returns `(λ1, λ2, c, s)` where
/// `V = [[c, -s], [s, c]]` is the orthogonal matrix of eigenvectors and
/// `H = V · diag(λ1, λ2) · Vᵀ`. Closed-form (Jacobi rotation).
#[inline]
fn sym_eig_2x2(h11: f64, h12: f64, h22: f64) -> (f64, f64, f64, f64) {
    if h12.abs() < 1e-300 {
        return (h11, h22, 1.0, 0.0);
    }
    let tr = h11 + h22;
    let det = h11 * h22 - h12 * h12;
    let disc = ((tr * tr - 4.0 * det).max(0.0)).sqrt();
    let lam1 = 0.5 * (tr + disc);
    let lam2 = 0.5 * (tr - disc);
    // Eigenvector for lam1: solve (H - lam1·I) v = 0 → v ∝ (h12, lam1 - h11).
    let vx = h12;
    let vy = lam1 - h11;
    let nrm = (vx * vx + vy * vy).sqrt().max(1e-300);
    let c = vx / nrm;
    let s = vy / nrm;
    (lam1, lam2, c, s)
}

/// Convenience helper: convert mgcv-style finite y data to the (y - μ) vector
/// used everywhere here. Public for tests that want to construct residuals
/// from a known β̂ + design without re-implementing the gather.
pub fn residuals_from_fit(y: &Array1<f64>, eta: &Array1<f64>) -> Array1<f64> {
    debug_assert_eq!(y.len(), eta.len());
    y.iter().zip(eta.iter()).map(|(&yi, &ei)| yi - ei).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// FD-check the analytical gradient + Hessian at a hand-picked point.
    /// Synthetic 30-row fixture; central differences at h=1e-5; tolerances
    /// 5e-6 on gradient, 1e-3 on Hessian (worst-case central-diff truncation
    /// for a function with O(1) curvature).
    #[test]
    fn analytical_derivs_match_finite_difference() {
        let n = 30usize;
        let y: Array1<f64> = (0..n).map(|i| 0.5 + 0.1 * (i as f64).sin()).collect();
        let eta: Array1<f64> = (0..n).map(|i| 0.4 + 0.08 * (i as f64).cos()).collect();
        let log_s2 = 0.2_f64;
        let log_d = 0.5_f64;
        let min_df = DEFAULT_MIN_DF;

        let state = nlogl_full(&y, &eta, None, log_s2, log_d, min_df, 1.0);

        let h = 1e-5_f64;
        // g[0]
        let f_plus = nlogl_value_only(&y, &eta, None, log_s2 + h, log_d, min_df, 1.0);
        let f_minus = nlogl_value_only(&y, &eta, None, log_s2 - h, log_d, min_df, 1.0);
        let fd_g0 = (f_plus - f_minus) / (2.0 * h);
        // g[1]
        let f_plus = nlogl_value_only(&y, &eta, None, log_s2, log_d + h, min_df, 1.0);
        let f_minus = nlogl_value_only(&y, &eta, None, log_s2, log_d - h, min_df, 1.0);
        let fd_g1 = (f_plus - f_minus) / (2.0 * h);
        assert!(
            (state.g[0] - fd_g0).abs() < 5e-6,
            "g[0]: ana={}, fd={}",
            state.g[0],
            fd_g0
        );
        assert!(
            (state.g[1] - fd_g1).abs() < 5e-6,
            "g[1]: ana={}, fd={}",
            state.g[1],
            fd_g1
        );

        // Hessian via central differences on gradient components.
        let hh = 1e-4_f64;
        let gp_s2 = nlogl_full(&y, &eta, None, log_s2 + hh, log_d, min_df, 1.0).g;
        let gm_s2 = nlogl_full(&y, &eta, None, log_s2 - hh, log_d, min_df, 1.0).g;
        let gp_d = nlogl_full(&y, &eta, None, log_s2, log_d + hh, min_df, 1.0).g;
        let gm_d = nlogl_full(&y, &eta, None, log_s2, log_d - hh, min_df, 1.0).g;

        let fd_h00 = (gp_s2[0] - gm_s2[0]) / (2.0 * hh);
        let fd_h11 = (gp_d[1] - gm_d[1]) / (2.0 * hh);
        let fd_h01 = 0.5 * ((gp_s2[1] - gm_s2[1]) / (2.0 * hh) + (gp_d[0] - gm_d[0]) / (2.0 * hh));

        assert!(
            (state.h[(0, 0)] - fd_h00).abs() < 1e-3,
            "H[0,0]: ana={}, fd={}",
            state.h[(0, 0)],
            fd_h00
        );
        assert!(
            (state.h[(1, 1)] - fd_h11).abs() < 1e-3,
            "H[1,1]: ana={}, fd={}",
            state.h[(1, 1)],
            fd_h11
        );
        assert!(
            (state.h[(0, 1)] - fd_h01).abs() < 1e-3,
            "H[0,1]: ana={}, fd={}",
            state.h[(0, 1)],
            fd_h01
        );
    }
}
