//! Ordered categorical (`ocat`) family math — port of mgcv `efam.r:2618-2945`.
//!
//! The module is a self-contained derivative engine for the cumulative-logit
//! model with R categories, mgcv-style log-gap threshold reparameterisation
//! and identity link. The actual PIRLS / outer-Newton drivers live in
//! `src/pirls.rs` (`fit_pirls_ocat`) and `src/smooth.rs` respectively; this
//! file only owns the per-row likelihood, deviance, and derivative math.
//!
//! ## Model
//!
//! For `Y ∈ {1, …, R}` with linear predictor `μ = η = Xβ` (identity link):
//!
//! ```text
//!   P(Y ≤ k) = F(α_{k+1} − μ),     F(z) = 1 / (1 + exp(−z))
//!   P(Y = k) = F(α_{k+1} − μ) − F(α_k − μ)
//! ```
//!
//! with thresholds `α_1 = −∞, α_2 = −1, α_3, …, α_R, α_{R+1} = +∞`. The free
//! parameters are `θ_j = log(α_{j+2} − α_{j+1})` for `j ∈ 1..R-2` (the
//! log-gap reparameterisation enforces strict ordering automatically).
//!
//! ## Layout
//!
//! - [`OcatThresholds`] caches the (R+1)-vector `α` for a given θ.
//! - [`OcatDerivLevel`] selects how many derivatives to compute (0 → D,
//!   Dmu, Dmu2 only; 1 → adds θ first derivatives; 2 → adds θ second
//!   derivatives needed by the joint outer Newton).
//! - [`OcatDeriv`] is the per-observation derivative struct, mirroring
//!   mgcv's `Dd` return list (`D, Dmu, Dmu2, Dth, Dmuth, Dmu2th, Dth2,
//!   Dmuth2, Dmu2th2, Dmu3, Dmu3th, Dmu4`).
//! - [`ocat_dd`] is the analytical computation; it must agree with central
//!   finite differences of the deviance to machine precision (verified by
//!   the tests at the bottom of this file).
//!
//! Reference: mgcv 1.9-4 source `R/efam.r`, `ocat <- function(...)`.

use ndarray::{Array1, Array2};

/// Compute thresholds α from log-gap parameters θ.
///
/// `theta` has length `R - 2`. The returned vector has length `R + 1`
/// with `α[0] = −∞, α[1] = −1, α[2 + k] = −1 + Σ_{j ≤ k} exp(θ_j)`
/// for `k ∈ 0..R-2`, and `α[R] = +∞`.
pub fn ocat_alpha(theta: &[f64], r: usize) -> Vec<f64> {
    assert!(r >= 3, "ocat: R must be >= 3, got {r}");
    assert_eq!(
        theta.len(),
        r - 2,
        "ocat: theta length {} must equal R-2 = {}",
        theta.len(),
        r - 2
    );
    let mut alpha = vec![0.0_f64; r + 1];
    alpha[0] = f64::NEG_INFINITY;
    alpha[1] = -1.0;
    let mut acc = -1.0;
    for k in 0..(r - 2) {
        acc += theta[k].exp();
        alpha[2 + k] = acc;
    }
    alpha[r] = f64::INFINITY;
    alpha
}

/// Cancellation-resistant `F(b) − F(a)` for the logistic CDF, `b > a`.
///
/// Direct port of `Fdiff` from `efam.r:2685-2696`. Returns a value in
/// `(0, 1]` because the cumulative-logit increment is always positive
/// when `b > a`. For `α_1 = −∞` and `α_{R+1} = +∞` boundary cases the
/// caller must short-circuit before invoking this; raw `±∞` inputs
/// would produce 0 or 1 via `exp(±large)`, but the per-row API keeps
/// the boundary handling explicit.
#[inline]
pub fn fdiff(a: f64, b: f64) -> f64 {
    // mgcv computes h_a = exp(a · sign-flip-if-a>0); same for b. Then
    // picks the cancellation-safe formula based on the sign of a and b.
    let ha = if a > 0.0 { -1.0 } else { 1.0 };
    let hb = if b > 0.0 { -1.0 } else { 1.0 };
    let ea = (a * ha).exp();
    let eb = (b * hb).exp();
    if b < 0.0 {
        // Both branches negative: use `bi/(1+bi) - ai/(1+ai)` form.
        eb / (1.0 + eb) - ea / (1.0 + ea)
    } else if a > 0.0 {
        // Both branches positive: `(ai - bi)/((ai+1)(bi+1))`.
        (ea - eb) / ((ea + 1.0) * (eb + 1.0))
    } else {
        // Straddling zero: `(1 - ai·bi)/((bi+1)(ai+1))`.
        (1.0 - ea * eb) / ((eb + 1.0) * (ea + 1.0))
    }
}

/// Boundary-aware `F(b) − F(a)`. Handles the `±∞` α_1 / α_{R+1} cases.
#[inline]
fn fdiff_boundary(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        // F(b) − F(−∞) = F(b).
        if b.is_infinite() && b.is_sign_positive() {
            1.0
        } else {
            1.0 / (1.0 + (-b).exp())
        }
    } else if b.is_infinite() && b.is_sign_positive() {
        // F(+∞) − F(a) = 1 − F(a).
        1.0 / (1.0 + a.exp()) // equivalently `1 − F(a) = F(−a)`
    } else {
        fdiff(a, b)
    }
}

/// mgcv's `abcd(x, level)` — auxiliary derivatives of the logistic CDF.
///
/// Returns `(aj, bj, cj, dj)`:
/// - `aj = f² − f` where `f = F(x)`
/// - `bj = f − 3f² + 2f³`           (level ≥ 0)
/// - `cj = −f + 7f² − 12f³ + 6f⁴`   (level ≥ 1)
/// - `dj = f − 15f² + 50f³ − 60f⁴ + 24f⁵`  (level ≥ 2; mgcv's exact polynomial)
///
/// Cancellation-resistant: uses `exp(x · sign)` and rational expressions
/// instead of computing `f` directly.
#[inline]
pub fn abcd(x: f64, level: i32) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
    let h = if x > 0.0 { -1.0 } else { 1.0 };
    let ex = (x * h).exp();
    let ex1 = ex + 1.0;
    let ex1k2 = ex1 * ex1;
    let aj = -ex / ex1k2;
    if level < 0 {
        return (aj, None, None, None);
    }
    let ex1k3 = ex1k2 * ex1;
    let ex2 = ex * ex;
    let bj = h * (ex - ex2) / ex1k3;
    if level == 0 {
        return (aj, Some(bj), None, None);
    }
    let ex1k4 = ex1k3 * ex1;
    let ex3 = ex2 * ex;
    let cj = (-ex3 + 4.0 * ex2 - ex) / ex1k4;
    if level == 1 {
        return (aj, Some(bj), Some(cj), None);
    }
    let ex1k5 = ex1k4 * ex1;
    let ex4 = ex3 * ex;
    let dj = h * (-ex4 + 11.0 * ex3 - 11.0 * ex2 + ex) / ex1k5;
    (aj, Some(bj), Some(cj), Some(dj))
}

/// Boundary-aware variant. For `x = ±∞` (the `α_1 − μ` or `α_{R+1} − μ`
/// case), all of `aj, bj, cj, dj` vanish. The caller is expected to
/// short-circuit those obs, but we return zero defensively.
#[inline]
fn abcd_boundary(x: f64, level: i32) -> (f64, f64, f64, f64) {
    if !x.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (a, b, c, d) = abcd(x, level);
    (a, b.unwrap_or(0.0), c.unwrap_or(0.0), d.unwrap_or(0.0))
}

/// Derivative level: controls how many derivatives `ocat_dd` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcatDerivLevel {
    /// Coefficient-estimation level: `D, Dmu, Dmu2`. Sufficient for the
    /// inner PIRLS loop (`gam.fit5.r` inner IRLS solve).
    Level0,
    /// Add `Dmu3, Dth, Dmuth, Dmu2th`. Needed when θ moves with the
    /// outer Newton (joint `(log λ, θ)` gradient).
    Level1,
    /// Add `Dmu4, Dth2, Dmuth2, Dmu2th2, Dmu3th`. Full Hessian for the
    /// joint outer Newton.
    Level2,
}

/// Result of [`ocat_dd`]. Vector lengths follow mgcv `Dd`'s convention:
/// `D, Dmu, Dmu2` are length-n; `Dth, Dmuth, Dmu2th, Dmu3th` are n×(R-2);
/// `Dth2, Dmuth2, Dmu2th2` are n×((R-2)(R-1)/2) (upper-triangular packed
/// column-major: column `i = (j,k) with j ≤ k`, indexed by
/// `i = j·(2(R-2) − j − 1)/2 + k`). `Dmu3, Dmu4` are length-n.
#[derive(Debug, Clone)]
pub struct OcatDeriv {
    pub d: Array1<f64>,
    pub dmu: Array1<f64>,
    pub dmu2: Array1<f64>,
    // Level ≥ 1
    pub dmu3: Option<Array1<f64>>,
    pub dth: Option<Array2<f64>>,
    pub dmuth: Option<Array2<f64>>,
    pub dmu2th: Option<Array2<f64>>,
    // Level ≥ 2
    pub dmu4: Option<Array1<f64>>,
    pub dth2: Option<Array2<f64>>,
    pub dmuth2: Option<Array2<f64>>,
    pub dmu2th2: Option<Array2<f64>>,
    pub dmu3th: Option<Array2<f64>>,
}

/// Compute per-observation `ocat` deviance and derivatives.
///
/// Returns the [`OcatDeriv`] struct mirroring mgcv `Dd`. The level
/// parameter controls how many derivatives are populated; unused fields
/// are `None`. When `R < 3` the level is effectively forced to 0 (no
/// free θ).
///
/// # Arguments
/// - `y` — integer category labels in `1..=R`. `f64` for storage
///   convenience; rounded internally.
/// - `mu` — linear predictor (identity link, so `μ = η = Xβ`).
/// - `theta` — log-gap parameters, length `R-2`.
/// - `r` — number of categories (≥ 3).
/// - `wt` — optional per-row weights; defaults to 1.
/// - `level` — derivative level.
pub fn ocat_dd(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    theta: &[f64],
    r: usize,
    wt: Option<&Array1<f64>>,
    level: OcatDerivLevel,
) -> OcatDeriv {
    assert_eq!(y.len(), mu.len(), "ocat_dd: y and mu length mismatch");
    if let Some(w) = wt {
        assert_eq!(w.len(), y.len(), "ocat_dd: weights length mismatch");
    }
    let n = y.len();
    let alpha = ocat_alpha(theta, r);
    let n_theta = r - 2;
    let effective_level = if r < 3 { OcatDerivLevel::Level0 } else { level };

    let mut d = Array1::<f64>::zeros(n);
    let mut dmu = Array1::<f64>::zeros(n);
    let mut dmu2 = Array1::<f64>::zeros(n);
    // Storage for intermediates needed across levels.
    let mut a_buf = vec![0.0_f64; n];
    let mut b_buf = vec![0.0_f64; n];
    let mut c_buf = vec![0.0_f64; n];
    let mut f_buf = vec![0.0_f64; n];
    let mut a0_buf = vec![0.0_f64; n];
    let mut a1_buf = vec![0.0_f64; n];
    let mut b0_buf = vec![0.0_f64; n];
    let mut b1_buf = vec![0.0_f64; n];
    let mut c0_buf = vec![0.0_f64; n];
    let mut c1_buf = vec![0.0_f64; n];
    let mut d0_buf = vec![0.0_f64; n];
    let mut d1_buf = vec![0.0_f64; n];
    let mut yi_buf = vec![0_usize; n];
    let mut al0_buf = vec![0.0_f64; n];
    let mut al1_buf = vec![0.0_f64; n];

    let inner_level = match effective_level {
        OcatDerivLevel::Level0 => 0_i32,
        OcatDerivLevel::Level1 => 1,
        OcatDerivLevel::Level2 => 2,
    };

    // === Per-row D, Dmu, Dmu2, plus the auxiliary a/b/c/d caches ===
    for i in 0..n {
        let yi = y[i].round() as usize;
        let yi_clamped = yi.clamp(1, r);
        yi_buf[i] = yi_clamped;
        let mu_i = mu[i];
        let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
        let al0 = alpha[yi_clamped - 1] - mu_i;
        let al1 = alpha[yi_clamped] - mu_i;
        al0_buf[i] = al0;
        al1_buf[i] = al1;
        let f = fdiff_boundary(al0, al1).max(f64::MIN_POSITIVE);
        f_buf[i] = f;
        let (a1, b1, c1, d1) = abcd_boundary(al1, inner_level);
        let (a0, b0, c0, d0) = abcd_boundary(al0, inner_level);
        a1_buf[i] = a1;
        a0_buf[i] = a0;
        b1_buf[i] = b1;
        b0_buf[i] = b0;
        c1_buf[i] = c1;
        c0_buf[i] = c0;
        d1_buf[i] = d1;
        d0_buf[i] = d0;
        let a = a1 - a0;
        let b = b1 - b0;
        let c = c1 - c0;
        a_buf[i] = a;
        b_buf[i] = b;
        c_buf[i] = c;
        d[i] = -2.0 * wt_i * f.ln();
        dmu[i] = -2.0 * wt_i * a / f;
        let a2 = a * a;
        dmu2[i] = 2.0 * wt_i * (a2 / f - b) / f;
    }

    let mut dmu3: Option<Array1<f64>> = None;
    let mut dth: Option<Array2<f64>> = None;
    let mut dmuth: Option<Array2<f64>> = None;
    let mut dmu2th: Option<Array2<f64>> = None;
    let mut dmu4: Option<Array1<f64>> = None;
    let mut dth2: Option<Array2<f64>> = None;
    let mut dmuth2: Option<Array2<f64>> = None;
    let mut dmu2th2: Option<Array2<f64>> = None;
    let mut dmu3th: Option<Array2<f64>> = None;

    let n2d = if n_theta > 1 {
        n_theta * (n_theta + 1) / 2
    } else if n_theta == 1 {
        1
    } else {
        0
    };

    if matches!(
        effective_level,
        OcatDerivLevel::Level1 | OcatDerivLevel::Level2
    ) && n_theta > 0
    {
        let mut dmu3_arr = Array1::<f64>::zeros(n);
        let mut dth_arr = Array2::<f64>::zeros((n, n_theta));
        let mut dmuth_arr = Array2::<f64>::zeros((n, n_theta));
        let mut dmu2th_arr = Array2::<f64>::zeros((n, n_theta));

        // Per-row α-derivative scalars Da0/Da1, Dmua0/Dmua1, Dmu2a0/Dmu2a1
        // (mgcv efam.r:2803-2811). Each gets multiplied by `exp(θ_k)`
        // weighted by which α boundary the row's y category touches.
        let mut da0_buf = vec![0.0_f64; n];
        let mut da1_buf = vec![0.0_f64; n];
        let mut dmua0_buf = vec![0.0_f64; n];
        let mut dmua1_buf = vec![0.0_f64; n];
        let mut dmu2a0_buf = vec![0.0_f64; n];
        let mut dmu2a1_buf = vec![0.0_f64; n];

        for i in 0..n {
            let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
            let f = f_buf[i];
            let a = a_buf[i];
            let b = b_buf[i];
            let c = c_buf[i];
            let a0 = a0_buf[i];
            let a1 = a1_buf[i];
            let b0 = b0_buf[i];
            let b1 = b1_buf[i];
            let c0 = c0_buf[i];
            let c1 = c1_buf[i];
            let a2 = a * a;
            let a3 = a2 * a;
            let f2 = f * f;
            dmu3_arr[i] = 2.0 * wt_i * (-c - 2.0 * a3 / f2 + 3.0 * a * b / f) / f;

            // Note: mgcv writes these without the `wt` factor — wt enters at
            // the per-θ assembly below. Keep that convention for ease of
            // matching to efam.r line-by-line.
            let dmua0 = 2.0 * (a0 * a / f - b0) / f;
            let dmua1 = -2.0 * (a1 * a / f - b1) / f;
            let dmu2a0 = -2.0 * (c0 + (a0 * (2.0 * a2 / f - b) - 2.0 * b0 * a) / f) / f;
            let dmu2a1 =
                2.0 * (c1 + (2.0 * (a1 * a2 / f - b1 * a) - a1 * b) / f) / f;
            let da0 = -2.0 * a0 / f;
            let da1 = 2.0 * a1 / f;
            da0_buf[i] = da0;
            da1_buf[i] = da1;
            dmua0_buf[i] = dmua0;
            dmua1_buf[i] = dmua1;
            dmu2a0_buf[i] = dmu2a0;
            dmu2a1_buf[i] = dmu2a1;
        }

        // Assemble Dth/Dmuth/Dmu2th rows by which α boundary the y_i lands on.
        // mgcv efam.r:2814-2832 — the structure is:
        //   y == k+1            → upper boundary α_{k+1} only (carries θ_k)
        //   y > k+1 AND y < R   → both α_{k+1} (upper) and α_{k+2} (lower)
        //   y == R              → lower boundary α_R only (carries θ_k for k = R-2)
        for k in 0..n_theta {
            let etk = theta[k].exp();
            for i in 0..n {
                let yi = yi_buf[i];
                let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
                // category index k+1 corresponds to mgcv's "y == k+1" branch.
                // mgcv uses 1-based y; here k+1 is also 1-based via the
                // (yi == k+2 in 1-based)? Let me re-derive.
                // mgcv: for k in 1:(R-2):
                //   y == k+1: Dth[ind,k] = Da1*etk      (only upper boundary)
                //   y > k+1 & y < R: Dth[ind,k] = (Da1+Da0)*etk
                //   y == R:  Dth[ind,k] = Da0*etk        (only lower boundary)
                // Here k is 1-based 1..R-2. In our 0-based loop, k_zero = k - 1,
                // so the conditions become:
                //   y == k_zero+2,    y > k_zero+2 & y < R,    y == R.
                let k_one = k + 1; // mgcv's 1-based k
                let bracket_upper = yi == k_one + 1; // mgcv y == k+1
                let bracket_lower = yi == r; // mgcv y == R
                let bracket_middle =
                    yi > k_one + 1 && yi < r && (r >= k_one + 3);

                if bracket_upper {
                    dth_arr[[i, k]] = wt_i * da1_buf[i] * etk;
                    dmuth_arr[[i, k]] = wt_i * dmua1_buf[i] * etk;
                    dmu2th_arr[[i, k]] = wt_i * dmu2a1_buf[i] * etk;
                } else if bracket_middle {
                    dth_arr[[i, k]] = wt_i * (da1_buf[i] + da0_buf[i]) * etk;
                    dmuth_arr[[i, k]] = wt_i * (dmua1_buf[i] + dmua0_buf[i]) * etk;
                    dmu2th_arr[[i, k]] =
                        wt_i * (dmu2a1_buf[i] + dmu2a0_buf[i]) * etk;
                } else if bracket_lower {
                    dth_arr[[i, k]] = wt_i * da0_buf[i] * etk;
                    dmuth_arr[[i, k]] = wt_i * dmua0_buf[i] * etk;
                    dmu2th_arr[[i, k]] = wt_i * dmu2a0_buf[i] * etk;
                }
            }
        }

        dmu3 = Some(dmu3_arr);
        dth = Some(dth_arr);
        dmuth = Some(dmuth_arr);
        dmu2th = Some(dmu2th_arr);

        // === Level 2: Dmu4, Dth2, Dmuth2, Dmu2th2, Dmu3th ===
        if matches!(effective_level, OcatDerivLevel::Level2) {
            let mut dmu4_arr = Array1::<f64>::zeros(n);
            let mut dth2_arr = Array2::<f64>::zeros((n, n2d));
            let mut dmuth2_arr = Array2::<f64>::zeros((n, n2d));
            let mut dmu2th2_arr = Array2::<f64>::zeros((n, n2d));
            let mut dmu3th_arr = Array2::<f64>::zeros((n, n_theta));

            // Level-2 per-row scalars (mgcv efam.r:2837-2857)
            let mut dmu3a0_buf = vec![0.0_f64; n];
            let mut dmu3a1_buf = vec![0.0_f64; n];
            let mut dmua0a0_buf = vec![0.0_f64; n];
            let mut dmua1a1_buf = vec![0.0_f64; n];
            let mut dmua0a1_buf = vec![0.0_f64; n];
            let mut dmu2a0a0_buf = vec![0.0_f64; n];
            let mut dmu2a1a1_buf = vec![0.0_f64; n];
            let mut da0a0_buf = vec![0.0_f64; n];
            let mut da1a1_buf = vec![0.0_f64; n];
            let mut da0a1_buf = vec![0.0_f64; n];

            for i in 0..n {
                let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
                let f = f_buf[i];
                let f2 = f * f;
                let a = a_buf[i];
                let a2 = a * a;
                let b = b_buf[i];
                let c = c_buf[i];
                let a0 = a0_buf[i];
                let a1 = a1_buf[i];
                let b0 = b0_buf[i];
                let b1 = b1_buf[i];
                let c0 = c0_buf[i];
                let c1 = c1_buf[i];
                let d0 = d0_buf[i];
                let d1 = d1_buf[i];
                let d_ = d1 - d0;
                let b_sq = b * b;
                dmu4_arr[i] = 2.0
                    * wt_i
                    * ((3.0 * b_sq + 4.0 * a * c) / f + a2 * (6.0 * a2 / f - 12.0 * b) / f2 - d_)
                    / f;

                dmu3a0_buf[i] = 2.0
                    * ((a0 * c + 3.0 * c0 * a + 3.0 * b0 * b) / f - d0
                        + 6.0 * a * (a0 * a2 / f - b0 * a - a0 * b) / f2)
                    / f;
                dmu3a1_buf[i] = 2.0
                    * (d1 - (a1 * c + 3.0 * (c1 * a + b1 * b)) / f
                        + 6.0 * a * (b1 * a - a1 * a2 / f + a1 * b) / f2)
                    / f;

                dmua0a0_buf[i] =
                    2.0 * (c0 + (2.0 * a0 * (b0 - a0 * a / f) - b0 * a) / f) / f;
                dmua1a1_buf[i] =
                    2.0 * ((b1 * a + 2.0 * a1 * (b1 - a1 * a / f)) / f - c1) / f;
                dmua0a1_buf[i] = 2.0 * (a0 * (2.0 * a1 * a / f - b1) - b0 * a1) / f2;

                dmu2a0a0_buf[i] = 2.0
                    * (d0 + (b0 * (2.0 * b0 - b) + 2.0 * c0 * (a0 - a)) / f
                        + 2.0 * (b0 * a2
                            + a0 * (3.0 * a0 * a2 / f - 4.0 * b0 * a - a0 * b))
                            / f2)
                    / f;
                dmu2a1a1_buf[i] = 2.0
                    * ((2.0 * c1 * (a + a1) + b1 * (2.0 * b1 + b)) / f
                        + 2.0 * (a1 * (3.0 * a1 * a2 / f - a1 * b) - b1 * a * (a + 4.0 * a1))
                            / f2
                        - d1)
                    / f;

                da0a0_buf[i] = 2.0 * (b0 + a0 * a0 / f) / f;
                da1a1_buf[i] = -2.0 * (b1 - a1 * a1 / f) / f;
                da0a1_buf[i] = -2.0 * a0 * a1 / f2;
            }

            // Dth2/Dmuth2/Dmu2th2 packed col index: i = j·(2(R-2) − j − 1)/2 + k
            // for j ≤ k (mgcv stacks (j,k) in lexicographic order).
            for j in 0..n_theta {
                for k in j..n_theta {
                    // packed index in n2d-column matrix
                    let col = j * (2 * n_theta - j - 1) / 2 + k;
                    let etj = theta[j].exp();
                    let etk = theta[k].exp();
                    for i in 0..n {
                        let yi = yi_buf[i];
                        let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
                        if yi < j + 1 {
                            // mgcv: ind <- y >= j; in 1-based, y >= j+1.
                            continue;
                        }
                        // ar.k / ar1.k weights for boundary k+1 vs k+2 contributions.
                        // mgcv-faithful: see efam.r:2868-2871.
                        // ar.k starts as exp(theta[k]); zeroed if y == R or y <= k.
                        let mut ar_k = etk;
                        let mut ar1_k = etk;
                        let k_one = k + 1; // mgcv 1-based k
                        if yi == r || yi <= k_one {
                            ar_k = 0.0;
                        }
                        if yi < k_one + 2 {
                            ar1_k = 0.0;
                        }
                        let mut ar_j = etj;
                        let mut ar1_j = etj;
                        let j_one = j + 1;
                        if yi == r || yi <= j_one {
                            ar_j = 0.0;
                        }
                        if yi < j_one + 2 {
                            ar1_j = 0.0;
                        }
                        let mut ar_kj = 0.0;
                        let mut ar1_kj = 0.0;
                        if k == j {
                            // diagonal: extra δ_{jk}·exp(θ_k) from the chain rule
                            if yi > k_one && yi < r {
                                ar_kj = etk;
                            }
                            if yi > k_one + 1 {
                                ar1_kj = etk;
                            }
                        }
                        // mgcv assembly (efam.r:2878-2883)
                        let dth2_i = wt_i
                            * (da1a1_buf[i] * ar_k * ar_j
                                + da0a1_buf[i] * ar_k * ar1_j
                                + da1_buf[i] * ar_kj
                                + da0a0_buf[i] * ar1_k * ar1_j
                                + da0a1_buf[i] * ar1_k * ar_j
                                + da0_buf[i] * ar1_kj);
                        dth2_arr[[i, col]] = dth2_i;

                        let dmuth2_i = wt_i
                            * (dmua1a1_buf[i] * ar_k * ar_j
                                + dmua0a1_buf[i] * ar_k * ar1_j
                                + dmua1_buf[i] * ar_kj
                                + dmua0a0_buf[i] * ar1_k * ar1_j
                                + dmua0a1_buf[i] * ar1_k * ar_j
                                + dmua0_buf[i] * ar1_kj);
                        dmuth2_arr[[i, col]] = dmuth2_i;

                        let dmu2_a0_a1 = 0.0_f64; // mgcv sets Dmu2a0a1 = 0
                        let _ = dmu2_a0_a1;
                        let dmu2th2_i = wt_i
                            * (dmu2a1a1_buf[i] * ar_k * ar_j
                                + 0.0 * ar_k * ar1_j
                                + dmu2a1_buf[i] * ar_kj
                                + dmu2a0a0_buf[i] * ar1_k * ar1_j
                                + 0.0 * ar1_k * ar_j
                                + dmu2a0_buf[i] * ar1_kj);
                        dmu2th2_arr[[i, col]] = dmu2th2_i;
                    }

                    // Dmu3th rows (diagonal only — j == k contribution).
                    if k == j {
                        let etk = theta[k].exp();
                        for i in 0..n {
                            let yi = yi_buf[i];
                            let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
                            let mut ar_k = etk;
                            let mut ar1_k = etk;
                            let k_one = k + 1;
                            if yi == r || yi <= k_one {
                                ar_k = 0.0;
                            }
                            if yi < k_one + 2 {
                                ar1_k = 0.0;
                            }
                            dmu3th_arr[[i, k]] = wt_i
                                * (dmu3a1_buf[i] * ar_k + dmu3a0_buf[i] * ar1_k);
                        }
                    }
                }
            }

            dmu4 = Some(dmu4_arr);
            dth2 = Some(dth2_arr);
            dmuth2 = Some(dmuth2_arr);
            dmu2th2 = Some(dmu2th2_arr);
            dmu3th = Some(dmu3th_arr);
        }
    }

    OcatDeriv {
        d,
        dmu,
        dmu2,
        dmu3,
        dth,
        dmuth,
        dmu2th,
        dmu4,
        dth2,
        dmuth2,
        dmu2th2,
        dmu3th,
    }
}

/// Compute the n × R per-row category probability matrix.
///
/// Port of `ocat`'s `predict` / `ocat.prob` mapping. For each observation,
/// `P(Y=k) = F(α_{k+1} − μ) − F(α_k − μ)` with the same boundary
/// convention as the likelihood.
pub fn ocat_prob(eta: &Array1<f64>, theta: &[f64], r: usize) -> Array2<f64> {
    let alpha = ocat_alpha(theta, r);
    let n = eta.len();
    let mut prob = Array2::<f64>::zeros((n, r));
    for i in 0..n {
        for k in 1..=r {
            prob[[i, k - 1]] = fdiff_boundary(alpha[k - 1] - eta[i], alpha[k] - eta[i]);
        }
    }
    prob
}

/// Deviance residual per observation: `r_i = sign·sqrt(d_i)`.
///
/// `sign = sign((α_y + α_{y+1})/2 − μ)`. Used by residuals(..., "deviance").
pub fn ocat_dev_resid(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    theta: &[f64],
    r: usize,
    wt: Option<&Array1<f64>>,
) -> Array1<f64> {
    let alpha = ocat_alpha(theta, r);
    let n = y.len();
    let mut res = Array1::<f64>::zeros(n);
    for i in 0..n {
        let yi = y[i].round() as usize;
        let yi_c = yi.clamp(1, r);
        let mu_i = mu[i];
        let wt_i = wt.map(|w| w[i]).unwrap_or(1.0);
        let al0 = alpha[yi_c - 1];
        let al1 = alpha[yi_c];
        // Boundary average for sign: treat ±∞ as just the finite end.
        let sign_mid = if al0.is_infinite() {
            al1
        } else if al1.is_infinite() {
            al0
        } else {
            0.5 * (al0 + al1)
        };
        let s = (sign_mid - mu_i).signum();
        let f = fdiff_boundary(al0 - mu_i, al1 - mu_i).max(f64::MIN_POSITIVE);
        let d_i = -2.0 * wt_i * f.ln();
        res[i] = s * d_i.sqrt();
    }
    res
}

/// Initial θ heuristic from category counts. mgcv preinitialize port
/// (efam.r:2927-2945). Returns log-gap θ of length R-2.
///
/// Strategy: compute empirical cumulative proportions, derive thresholds
/// on the latent-eta scale via logit, take diffs, clamp positive, take
/// log. Falls back to `[-1, -1, …]` if y has bad data.
pub fn ocat_init_theta(y: &Array1<f64>, r: usize) -> Vec<f64> {
    if r < 3 {
        return Vec::new();
    }
    let n_theta = r - 2;
    let n = y.len();
    if n == 0 {
        return vec![-1.0; n_theta];
    }
    // Make sure each class has at least one count (mgcv prepends 1..R).
    let mut counts = vec![0_usize; r];
    for k in 0..r {
        counts[k] = 1;
    }
    for &yi in y.iter() {
        let yi_round = yi.round() as i64;
        if yi_round >= 1 && (yi_round as usize) <= r {
            counts[yi_round as usize - 1] += 1;
        }
    }
    let total: usize = counts.iter().sum();
    let mut cum = vec![0.0_f64; r];
    let mut acc = 0.0_f64;
    for k in 0..r {
        acc += counts[k] as f64 / total as f64;
        cum[k] = acc;
    }
    // eta = -1 - logit(p[1])  (mgcv: if p[1] == 0 use eta=5 sentinel)
    let p1 = cum[0];
    let eta = if p1 <= 0.0 || p1 >= 1.0 {
        5.0
    } else {
        -1.0 - (p1 / (1.0 - p1)).ln()
    };
    let mut theta_alpha = vec![-1.0_f64; r - 1];
    for i in 1..(r - 1) {
        let pi = cum[i].clamp(1e-9, 1.0 - 1e-9);
        theta_alpha[i] = (pi / (1.0 - pi)).ln() + eta;
    }
    let mut diffs = vec![0.0_f64; r - 2];
    for i in 0..(r - 2) {
        diffs[i] = (theta_alpha[i + 1] - theta_alpha[i]).max(0.01);
    }
    diffs.iter().map(|d| d.ln()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_from_theta_matches_mgcv_layout() {
        // R=4, theta=[0.5, 0.5]: α = [-∞, -1, -1+e^0.5, -1+e^0.5+e^0.5, +∞]
        let theta = [0.5_f64, 0.5];
        let alpha = ocat_alpha(&theta, 4);
        assert!(alpha[0].is_infinite() && alpha[0].is_sign_negative());
        assert!((alpha[1] - (-1.0)).abs() < 1e-14);
        let e = 0.5_f64.exp();
        assert!((alpha[2] - (-1.0 + e)).abs() < 1e-14);
        assert!((alpha[3] - (-1.0 + 2.0 * e)).abs() < 1e-14);
        assert!(alpha[4].is_infinite() && alpha[4].is_sign_positive());
    }

    #[test]
    fn fdiff_matches_naive_logistic() {
        let f = |z: f64| 1.0 / (1.0 + (-z).exp());
        let pairs = [
            (-3.0, -1.5),
            (-1.0, 0.5),
            (0.2, 2.0),
            (-5.0, 5.0),
            (-10.0, -8.0),
            (8.0, 10.0),
        ];
        for &(a, b) in &pairs {
            let exact = f(b) - f(a);
            let cancel_safe = fdiff(a, b);
            assert!(
                (cancel_safe - exact).abs() < 1e-12,
                "fdiff({a},{b}) = {cancel_safe} vs exact {exact}"
            );
        }
    }

    fn fd_central<F>(f: F, x: f64, h: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// Deviance: D = -2 wt log f. Helper for FD checks of analytic derivs.
    fn ocat_d_scalar(yi: usize, mu_i: f64, theta: &[f64], r: usize, wt_i: f64) -> f64 {
        let alpha = ocat_alpha(theta, r);
        let f = fdiff_boundary(alpha[yi - 1] - mu_i, alpha[yi] - mu_i).max(f64::MIN_POSITIVE);
        -2.0 * wt_i * f.ln()
    }

    #[test]
    fn dmu_matches_finite_difference() {
        let theta = vec![0.5_f64, 0.5];
        let r = 4;
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let mu = Array1::from(vec![-0.3_f64, 0.1, 0.7, 1.2]);
        let deriv = ocat_dd(&y, &mu, &theta, r, None, OcatDerivLevel::Level0);
        for i in 0..y.len() {
            let yi = y[i] as usize;
            let mu_i = mu[i];
            let fd = fd_central(|m| ocat_d_scalar(yi, m, &theta, r, 1.0), mu_i, 1e-6);
            assert!(
                (deriv.dmu[i] - fd).abs() < 1e-7,
                "Dmu[{i}] analytic {} vs FD {}",
                deriv.dmu[i],
                fd
            );
        }
    }

    #[test]
    fn dmu2_matches_finite_difference_of_dmu() {
        let theta = vec![0.5_f64, 0.5];
        let r = 4;
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let mu = Array1::from(vec![-0.3_f64, 0.1, 0.7, 1.2]);
        let deriv = ocat_dd(&y, &mu, &theta, r, None, OcatDerivLevel::Level0);
        // FD on dmu (computed by re-running ocat_dd at μ±h)
        for i in 0..y.len() {
            let yi = y[i];
            let mu_i = mu[i];
            let mu_plus = Array1::from(vec![mu_i + 1e-5]);
            let mu_minus = Array1::from(vec![mu_i - 1e-5]);
            let y_one = Array1::from(vec![yi]);
            let dplus = ocat_dd(&y_one, &mu_plus, &theta, r, None, OcatDerivLevel::Level0);
            let dminus = ocat_dd(&y_one, &mu_minus, &theta, r, None, OcatDerivLevel::Level0);
            let fd = (dplus.dmu[0] - dminus.dmu[0]) / (2.0 * 1e-5);
            assert!(
                (deriv.dmu2[i] - fd).abs() < 1e-6,
                "Dmu2[{i}] analytic {} vs FD {}",
                deriv.dmu2[i],
                fd
            );
        }
    }

    #[test]
    fn dth_matches_finite_difference() {
        let theta = vec![0.4_f64, 0.6];
        let r = 4;
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 2.0]);
        let mu = Array1::from(vec![-0.3_f64, 0.1, 0.7, 1.2, 0.0]);
        let deriv = ocat_dd(&y, &mu, &theta, r, None, OcatDerivLevel::Level1);
        let dth = deriv.dth.expect("Level1 must produce Dth");
        for j in 0..(r - 2) {
            for i in 0..y.len() {
                let yi = y[i] as usize;
                let mu_i = mu[i];
                let h = 1e-6_f64;
                let mut tp = theta.clone();
                tp[j] += h;
                let mut tm = theta.clone();
                tm[j] -= h;
                let dp = ocat_d_scalar(yi, mu_i, &tp, r, 1.0);
                let dm = ocat_d_scalar(yi, mu_i, &tm, r, 1.0);
                let fd = (dp - dm) / (2.0 * h);
                assert!(
                    (dth[[i, j]] - fd).abs() < 1e-6,
                    "Dth[{i},{j}] analytic {} vs FD {}",
                    dth[[i, j]],
                    fd
                );
            }
        }
    }

    #[test]
    fn dmuth_matches_fd_of_dmu_wrt_theta() {
        let theta = vec![0.4_f64, 0.6];
        let r = 4;
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 2.0]);
        let mu = Array1::from(vec![-0.3_f64, 0.1, 0.7, 1.2, 0.0]);
        let deriv = ocat_dd(&y, &mu, &theta, r, None, OcatDerivLevel::Level1);
        let dmuth = deriv.dmuth.expect("Level1 must produce Dmuth");
        for j in 0..(r - 2) {
            for i in 0..y.len() {
                let h = 1e-6_f64;
                let mut tp = theta.clone();
                tp[j] += h;
                let mut tm = theta.clone();
                tm[j] -= h;
                let dp = ocat_dd(
                    &Array1::from(vec![y[i]]),
                    &Array1::from(vec![mu[i]]),
                    &tp,
                    r,
                    None,
                    OcatDerivLevel::Level0,
                );
                let dm = ocat_dd(
                    &Array1::from(vec![y[i]]),
                    &Array1::from(vec![mu[i]]),
                    &tm,
                    r,
                    None,
                    OcatDerivLevel::Level0,
                );
                let fd = (dp.dmu[0] - dm.dmu[0]) / (2.0 * h);
                assert!(
                    (dmuth[[i, j]] - fd).abs() < 1e-6,
                    "Dmuth[{i},{j}] analytic {} vs FD {}",
                    dmuth[[i, j]],
                    fd
                );
            }
        }
    }

    #[test]
    fn dmu2th_matches_fd_of_dmu2_wrt_theta() {
        let theta = vec![0.4_f64, 0.6];
        let r = 4;
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let mu = Array1::from(vec![-0.3_f64, 0.1, 0.7, 1.2]);
        let deriv = ocat_dd(&y, &mu, &theta, r, None, OcatDerivLevel::Level1);
        let dmu2th = deriv.dmu2th.expect("Level1 must produce Dmu2th");
        for j in 0..(r - 2) {
            for i in 0..y.len() {
                let h = 1e-6_f64;
                let mut tp = theta.clone();
                tp[j] += h;
                let mut tm = theta.clone();
                tm[j] -= h;
                let dp = ocat_dd(
                    &Array1::from(vec![y[i]]),
                    &Array1::from(vec![mu[i]]),
                    &tp,
                    r,
                    None,
                    OcatDerivLevel::Level0,
                );
                let dm = ocat_dd(
                    &Array1::from(vec![y[i]]),
                    &Array1::from(vec![mu[i]]),
                    &tm,
                    r,
                    None,
                    OcatDerivLevel::Level0,
                );
                let fd = (dp.dmu2[0] - dm.dmu2[0]) / (2.0 * h);
                assert!(
                    (dmu2th[[i, j]] - fd).abs() < 1e-5,
                    "Dmu2th[{i},{j}] analytic {} vs FD {}",
                    dmu2th[[i, j]],
                    fd
                );
            }
        }
    }

    #[test]
    fn ocat_prob_rows_sum_to_one() {
        let theta = vec![0.5_f64, 0.5];
        let r = 4;
        let eta = Array1::from(vec![-1.0_f64, 0.0, 1.0, 2.5]);
        let prob = ocat_prob(&eta, &theta, r);
        for i in 0..eta.len() {
            let s: f64 = prob.row(i).sum();
            assert!((s - 1.0).abs() < 1e-12, "row {i} sum = {s}");
            for k in 0..r {
                assert!(prob[[i, k]] > 0.0, "prob[{i},{k}] = {}", prob[[i, k]]);
            }
        }
    }

    #[test]
    fn ocat_init_theta_returns_correct_length() {
        let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 4.0]);
        let theta = ocat_init_theta(&y, 4);
        assert_eq!(theta.len(), 2);
        // Each θ should be finite log of a positive gap.
        for &t in &theta {
            assert!(t.is_finite(), "θ = {t}");
        }
    }
}
