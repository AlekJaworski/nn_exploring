//! Joint outer-Newton search vector abstraction (Step 1+2 of
//! `docs/JOINT_OUTER_NEWTON_DESIGN.md`). Carries `log λ` + family-shape
//! extras (Tweedie θ, NegBin log θ, TDist `(log σ², log(df-2))`) in one
//! struct so the outer-Newton harness shares Newton+line-search machinery
//! across families. Pure structural refactor — no behaviour change.

/// Family-shape parameter kind. Drives per-family dispatch (derivative
/// path + Family enum rebuild on commit). Future variants (Gaussian
/// `log φ`, Quantile `log σ`, ocat thresholds) plug into the same harness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtraKind {
    /// Tweedie working parameter θ (mgcv convention, a=1.001 b=1.999).
    TweedieTheta,
    /// NegBin `log θ` — log-space enforces θ > 0.
    NegBinLogTheta,
    /// TDist `log σ²` — paired with `TDistLogDfM2` for 2-D joint Newton.
    TDistLogSigma2,
    /// TDist `log(df - min.df)` (mgcv `scat$theta1`).
    TDistLogDfM2,
}

/// One family-shape parameter in working coords. Bounds and step cap are
/// working-coord values (e.g. for `NegBinLogTheta` these constrain `log θ`,
/// not `θ`).
#[derive(Debug, Clone, Copy)]
pub struct ExtraParam {
    pub kind: ExtraKind,
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub step_cap: f64,
}

/// Outer-Newton search vector: M `log λ` followed by K family extras.
/// Layout matches mgcv's `gam.outer` (λ first, extras at tail).
#[derive(Debug, Clone)]
pub struct OuterSearchVector {
    pub log_lambda: Vec<f64>,
    pub extras: Vec<ExtraParam>,
}

impl OuterSearchVector {
    #[inline]
    pub fn dim(&self) -> usize {
        self.log_lambda.len() + self.extras.len()
    }

    /// Index of the first extra of the given kind in `self.extras`.
    pub fn find_kind(&self, kind: ExtraKind) -> Option<usize> {
        self.extras.iter().position(|e| e.kind == kind)
    }
}

/// 1-D Newton + 2-step line-search halving used by Tweedie-θ and
/// NegBin-log-θ blocks. Preserves the previous inline implementation:
/// `δ = -g / max(|H|, 1e-4)`, clamped to `[-step_cap, +step_cap]`; line
/// search tries full step then half step; accepts on strict improvement.
/// Bounds clamping happens inside the caller's `eval_at` closure.
#[cfg(feature = "blas")]
pub(crate) fn newton_1d_with_halving<F>(
    base: f64,
    g: f64,
    h: f64,
    rc: f64,
    step_cap: f64,
    mut eval_at: F,
) -> f64
where
    F: FnMut(f64) -> crate::Result<f64>,
{
    let denom = h.abs().max(1e-4);
    let delta = (-(g / denom)).max(-step_cap).min(step_cap);
    let candidate = base + delta;
    if let Ok(r_new) = eval_at(candidate) {
        if r_new < rc {
            return candidate;
        }
        let half = base + delta * 0.5;
        if let Ok(r_half) = eval_at(half) {
            if r_half < rc {
                return half;
            }
        }
    }
    base
}

/// 2-D Newton + repeated line-search halving used by the TDist
/// (`log σ²`, `log(df-2)`) joint shape block. Det-floored 2×2 inverse,
/// per-coord descent fallback on singularity, `max(|δ_s|,|δ_d|) ≤ 1.0`
/// renormalisation, `max_halvings` step-scale halvings stopping on the
/// first improvement.
#[cfg(feature = "blas")]
pub(crate) fn newton_2d_with_halving<F>(
    base_s: f64,
    base_d: f64,
    g_s: f64,
    g_d: f64,
    h_ss: f64,
    h_dd: f64,
    h_sd: f64,
    rc: f64,
    bounds_s: (f64, f64),
    bounds_d: (f64, f64),
    max_halvings: usize,
    mut eval_at: F,
) -> (f64, f64)
where
    F: FnMut(f64, f64) -> crate::Result<f64>,
{
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
    let mut accepted_s = base_s;
    let mut accepted_d = base_d;
    let mut step_scale = 1.0_f64;
    for _ in 0..max_halvings {
        let cand_s = (base_s + step_scale * delta_s)
            .max(bounds_s.0)
            .min(bounds_s.1);
        let cand_d = (base_d + step_scale * delta_d)
            .max(bounds_d.0)
            .min(bounds_d.1);
        if let Ok(s_new) = eval_at(cand_s, cand_d) {
            if s_new < rc {
                accepted_s = cand_s;
                accepted_d = cand_d;
                break;
            }
        }
        step_scale *= 0.5;
    }
    (accepted_s, accepted_d)
}
