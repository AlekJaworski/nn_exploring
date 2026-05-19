//! GLM link functions with coordinated forward / inverse / derivative paths.
//!
//! ## Conjugation invariant
//!
//! Every method (link, inverse_link, d_inverse_link, d2_link, d3_link,
//! d4_link) is **defined** as a composition of one of two single-source
//! safety transforms — [`Link::safe_eta`] (for eta-domain inputs) or
//! [`Link::safe_mu`] (for mu-domain inputs) — followed by the unguarded
//! math.
//!
//! Formally, for every eta:
//!
//! ```text
//! inverse_link(eta)   ≡ inverse_link(safe_eta(eta))
//! d_inverse_link(eta) ≡ d_inverse_link(safe_eta(eta))
//! ```
//!
//! and likewise for the mu-domain methods. This guarantees the chain
//! rule stays self-consistent under any safety clamp: μ and dμ/dη are
//! always evaluated at exactly the same eta. The IRLS working pair
//!
//! ```text
//! w = (dμ/dη)² / V(μ)        z = η + (y − μ)/(dμ/dη)
//! ```
//!
//! therefore stays finite even when an over-shooting iterate pushes
//! eta past the saturation boundary — without this invariant, a
//! one-sided clamp on `inverse_link` (mu clamped to exp(20)) while
//! `d_inverse_link = exp(eta)` overflows produces NaN/inf weights
//! that propagate into the fREML gradient and fail the eigen-clamped
//! Newton step.
//!
//! Conceptually this mirrors a conjugation `g · f · g⁻¹` in group theory:
//! the safety transform `g` is applied identically before every
//! evaluation, so the action of `g` on the function family is uniform.
//!
//! Adding or tightening a safety clamp is therefore a one-line change
//! to `safe_eta` or `safe_mu`; the forward, inverse, and every
//! derivative pick it up automatically.

/// Link function kind. Each variant carries the parameters of its
/// safety transform so different links can be tuned independently
/// (different eta_max for log vs logit, etc.).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Link {
    /// Identity: μ = η. No safety clamp.
    Identity,
    /// Log link: μ = exp(η). η upper-bounded at `eta_max` so μ cannot
    /// overflow (`exp(20) ≈ 4.85e8`, finite and well within f64). The
    /// μ-domain transform mirrors this — log(μ) is undefined at μ = 0
    /// and overflows past μ = exp(eta_max).
    Log { eta_max: f64 },
    /// Logit link: μ = 1/(1+exp(−η)). η symmetrically clamped to
    /// `[-eta_max, eta_max]` so μ stays away from {0, 1}; without
    /// this, `log(μ/(1-μ))` overflows and the working weight
    /// `μ·(1-μ)` underflows to zero.
    Logit { eta_max: f64 },
    /// Reciprocal link: μ = 1/η. η floored at `|η| ≥ eta_eps` so
    /// 1/η stays bounded; μ is similarly floored away from 0 by the
    /// mu-domain transform.
    Reciprocal { eta_eps: f64 },
}

/// Default eta-saturation threshold for log and logit links. Mirrors
/// the pre-Link inline clamp (`eta.min(20)` / `eta.clamp(-20, 20)`)
/// and is consistent with mgcv's `etamax` defaults.
pub const DEFAULT_ETA_MAX: f64 = 20.0;
/// Default eta-floor for the reciprocal link.
pub const DEFAULT_ETA_EPS: f64 = 1e-10;

impl Link {
    // -------- safety transforms (single source per direction) --------

    /// Idempotent η-safety transform. Every η-domain method below
    /// passes its input through this first — that is what enforces
    /// the conjugation invariant.
    #[inline]
    pub fn safe_eta(&self, eta: f64) -> f64 {
        match *self {
            Link::Identity => eta,
            Link::Log { eta_max } => eta.min(eta_max),
            Link::Logit { eta_max } => eta.clamp(-eta_max, eta_max),
            Link::Reciprocal { eta_eps } => {
                if eta.abs() < eta_eps {
                    if eta >= 0.0 {
                        eta_eps
                    } else {
                        -eta_eps
                    }
                } else {
                    eta
                }
            }
        }
    }

    /// Idempotent μ-safety transform. Every μ-domain method below
    /// passes its input through this first.
    #[inline]
    pub fn safe_mu(&self, mu: f64) -> f64 {
        match *self {
            Link::Identity => mu,
            Link::Log { eta_max } => {
                let hi = eta_max.exp();
                let lo = (-eta_max).exp();
                mu.clamp(lo, hi)
            }
            Link::Logit { eta_max } => {
                let lo = 1.0 / (1.0 + eta_max.exp());
                let hi = 1.0 - lo;
                mu.clamp(lo, hi)
            }
            Link::Reciprocal { eta_eps } => {
                // For reciprocal, μ and η swap roles. The forward path
                // η = 1/μ needs |μ| ≥ eta_eps to stay bounded.
                if mu.abs() < eta_eps {
                    if mu >= 0.0 {
                        eta_eps
                    } else {
                        -eta_eps
                    }
                } else {
                    mu
                }
            }
        }
    }

    // -------- forward / inverse / derivatives ------------------------

    /// Forward link g(μ) = η.
    #[inline]
    pub fn link(&self, mu: f64) -> f64 {
        let m = self.safe_mu(mu);
        match *self {
            Link::Identity => m,
            Link::Log { .. } => m.ln(),
            Link::Logit { .. } => (m / (1.0 - m)).ln(),
            Link::Reciprocal { .. } => 1.0 / m,
        }
    }

    /// Inverse link g⁻¹(η) = μ.
    #[inline]
    pub fn inverse_link(&self, eta: f64) -> f64 {
        let e = self.safe_eta(eta);
        match *self {
            Link::Identity => e,
            Link::Log { .. } => e.exp(),
            Link::Logit { .. } => 1.0 / (1.0 + (-e).exp()),
            Link::Reciprocal { .. } => 1.0 / e,
        }
    }

    /// dμ/dη evaluated at `safe_eta(eta)` — the conjugation invariant
    /// in action: any clamp applied in `inverse_link` is applied here too.
    #[inline]
    pub fn d_inverse_link(&self, eta: f64) -> f64 {
        let e = self.safe_eta(eta);
        match *self {
            Link::Identity => 1.0,
            Link::Log { .. } => e.exp(),
            Link::Logit { .. } => {
                let mu = 1.0 / (1.0 + (-e).exp());
                mu * (1.0 - mu)
            }
            Link::Reciprocal { .. } => -1.0 / (e * e),
        }
    }

    /// d²g/dμ² evaluated at `safe_mu(mu)`.
    #[inline]
    pub fn d2_link(&self, mu: f64) -> f64 {
        let m = self.safe_mu(mu);
        match *self {
            Link::Identity => 0.0,
            Link::Log { .. } => -1.0 / (m * m),
            Link::Logit { .. } => {
                let one_m = 1.0 - m;
                -1.0 / (m * m) + 1.0 / (one_m * one_m)
            }
            Link::Reciprocal { .. } => 2.0 / (m * m * m),
        }
    }

    /// d³g/dμ³ evaluated at `safe_mu(mu)`.
    #[inline]
    pub fn d3_link(&self, mu: f64) -> f64 {
        let m = self.safe_mu(mu);
        match *self {
            Link::Identity => 0.0,
            Link::Log { .. } => 2.0 / (m * m * m),
            Link::Logit { .. } => {
                let one_m = 1.0 - m;
                2.0 / (m * m * m) + 2.0 / (one_m * one_m * one_m)
            }
            Link::Reciprocal { .. } => -6.0 / (m * m * m * m),
        }
    }

    /// d⁴g/dμ⁴ evaluated at `safe_mu(mu)`.
    #[inline]
    pub fn d4_link(&self, mu: f64) -> f64 {
        let m = self.safe_mu(mu);
        let m2 = m * m;
        match *self {
            Link::Identity => 0.0,
            Link::Log { .. } => -6.0 / (m2 * m2),
            Link::Logit { .. } => {
                let one_m = 1.0 - m;
                let om2 = one_m * one_m;
                -6.0 / (m2 * m2) + 6.0 / (om2 * om2)
            }
            Link::Reciprocal { .. } => 24.0 / (m2 * m2 * m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conjugation invariant: applying `safe_eta` twice is the
    /// same as applying it once (idempotence), and every eta-domain
    /// method is invariant under `safe_eta`.
    #[test]
    fn conjugation_invariant_eta() {
        let links = [
            Link::Identity,
            Link::Log { eta_max: 20.0 },
            Link::Logit { eta_max: 20.0 },
            Link::Reciprocal { eta_eps: 1e-10 },
        ];
        for link in &links {
            for &eta in &[-50.0, -25.0, -1.0, 0.0, 1.0, 25.0, 50.0, 100.0] {
                let e_safe = link.safe_eta(eta);
                // Idempotence: safe_eta(safe_eta(x)) == safe_eta(x)
                assert_eq!(
                    link.safe_eta(e_safe),
                    e_safe,
                    "{:?} safe_eta non-idempotent at {}",
                    link,
                    eta
                );
                // Conjugation: f(eta) == f(safe_eta(eta))
                assert_eq!(
                    link.inverse_link(eta),
                    link.inverse_link(e_safe),
                    "{:?} inverse_link not invariant under safe_eta at eta={}",
                    link,
                    eta
                );
                assert_eq!(
                    link.d_inverse_link(eta),
                    link.d_inverse_link(e_safe),
                    "{:?} d_inverse_link not invariant under safe_eta at eta={}",
                    link,
                    eta
                );
            }
        }
    }

    /// The conjugation invariant on the μ side.
    #[test]
    fn conjugation_invariant_mu() {
        let links = [
            Link::Identity,
            Link::Log { eta_max: 20.0 },
            Link::Logit { eta_max: 20.0 },
            Link::Reciprocal { eta_eps: 1e-10 },
        ];
        for link in &links {
            // Pick μ values that are valid for each link's natural domain;
            // identity/log/reciprocal accept any, logit needs μ ∈ (0,1).
            let test_mus: &[f64] = match link {
                Link::Logit { .. } => &[0.01, 0.25, 0.5, 0.75, 0.99],
                _ => &[1e-12, 1e-6, 0.1, 1.0, 10.0, 1e6, 1e12],
            };
            for &mu in test_mus {
                let m_safe = link.safe_mu(mu);
                assert_eq!(
                    link.safe_mu(m_safe),
                    m_safe,
                    "{:?} safe_mu non-idempotent at {}",
                    link,
                    mu
                );
                assert_eq!(
                    link.link(mu),
                    link.link(m_safe),
                    "{:?} link not invariant under safe_mu at mu={}",
                    link,
                    mu
                );
                assert_eq!(
                    link.d2_link(mu),
                    link.d2_link(m_safe),
                    "{:?} d2_link not invariant under safe_mu at mu={}",
                    link,
                    mu
                );
                assert_eq!(
                    link.d3_link(mu),
                    link.d3_link(m_safe),
                    "{:?} d3_link not invariant under safe_mu at mu={}",
                    link,
                    mu
                );
                assert_eq!(
                    link.d4_link(mu),
                    link.d4_link(m_safe),
                    "{:?} d4_link not invariant under safe_mu at mu={}",
                    link,
                    mu
                );
            }
        }
    }

    /// IRLS working weight stays finite past the saturation boundary —
    /// the regression that motivated this module.
    #[test]
    fn log_working_weight_finite_past_saturation() {
        let link = Link::Log { eta_max: 20.0 };
        // Negative-binomial-like variance: V(μ) = μ + μ²/θ.
        let theta = 2.0;
        let variance = |mu: f64| mu + mu * mu / theta;
        for &eta in &[10.0, 19.999, 20.0, 25.0, 100.0, 700.0] {
            let dmu = link.d_inverse_link(eta);
            let mu = link.inverse_link(eta);
            let v = variance(mu);
            let w = dmu * dmu / v;
            assert!(
                w.is_finite(),
                "w non-finite at eta={}: dmu={} mu={} v={}",
                eta,
                dmu,
                mu,
                v
            );
        }
    }

    /// In-range numeric agreement with the pre-Link implementation,
    /// so swapping in `Link` doesn't shift any parity fixture.
    #[test]
    fn matches_pre_link_in_range() {
        let log = Link::Log { eta_max: 20.0 };
        // η well below the clamp: should be exactly exp(η).
        for &eta in &[-5.0, -1.0, 0.0, 1.0, 5.0, 15.0, 19.0] {
            assert_eq!(log.inverse_link(eta), eta.exp());
            assert_eq!(log.d_inverse_link(eta), eta.exp());
        }
        let logit = Link::Logit { eta_max: 20.0 };
        for &eta in &[-15.0_f64, -1.0, 0.0, 1.0, 15.0] {
            let expected = 1.0 / (1.0 + (-eta).exp());
            assert_eq!(logit.inverse_link(eta), expected);
            assert_eq!(logit.d_inverse_link(eta), expected * (1.0 - expected));
        }
    }
}
