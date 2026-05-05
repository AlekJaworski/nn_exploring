//! PiRLS (Penalized Iteratively Reweighted Least Squares) algorithm for GAM fitting

use crate::block_penalty::BlockPenalty;
use crate::discrete::DiscretizedDesign;
use crate::linalg::solve;
use crate::reml::compute_xtwx;
use crate::{GAMError, Result};
use ndarray::{Array1, Array2};

/// Family and link function for GLM. Each variant uses its canonical link
/// EXCEPT `GammaLog`, which is the gamma family paired with log link
/// (a common production choice for positive responses where multiplicative
/// effects are wanted but the inverse link is awkward at small μ).
#[derive(Debug, Clone, Copy)]
pub enum Family {
    Gaussian,
    Binomial,
    Poisson,
    /// Gamma with canonical inverse link.
    Gamma,
    /// Gamma with log link — non-canonical, but the most common
    /// production choice for positive multiplicative responses.
    GammaLog,
    /// Scaled t-distribution (mgcv's `scat` family).
    ///
    /// Identity link: μ = η. This is a location-scale model, NOT a standard
    /// GLM exponential-family member. The IRLS weights are the t-likelihood
    /// weights `w_i = (df+1)/(df + r_i²/σ²)`, NOT the Fisher information
    /// of an exponential family. The special-purpose `fit_pirls_tdist`
    /// function implements the outer σ²/df profiling loop.
    ///
    /// `df` holds the *current* degrees-of-freedom value. When profiling,
    /// the outer loop updates it each iteration and passes a fresh variant.
    /// `sigma2` is stored here for use in `variance` so that the standard
    /// IRLS weight path (`dμ/dη)² / V(μ)`) yields the correct t-weight.
    ///
    /// **Layout**: IRLS weight `w_i = (df+1) / (df + r_i²/σ²)`.
    /// For identity link `dμ/dη = 1`, so setting `V(μ) = σ² * df / (df+1)`
    /// (constant, not residual-dependent) gives the *initial* Fisher weight.
    /// The actual per-observation weights are computed directly in
    /// `fit_pirls_tdist` — the `variance` method here is used only by code
    /// paths that do not call the tdist-specific fitter.
    TDist { df: f64, sigma2: f64 },
    /// Tweedie distribution with log link (1 < p < 2).
    ///
    /// Log link: η = log(μ), μ = exp(η). Variance function V(μ) = μ^p.
    /// The dispersion φ is profiled (not fixed at 1), similar to Gamma.
    ///
    /// The saturated log-likelihood is computed via the Dunn-Smyth (2005)
    /// series summation for 1 < p < 2 — a port of mgcv's `tweedious` C
    /// function (misc.c:170). For y=0 the density simplifies; for y>0 the
    /// series `W = Σ_j W_j` is summed using log-sum-exp to avoid overflow.
    Tweedie { p: f64 },
    /// Quasi-Poisson: same variance V(μ)=μ and log link as Poisson, but
    /// the dispersion φ is **estimated from data** rather than fixed at 1.
    /// mgcv estimates φ̂ = Dp / (n − trA) (Pearson-style).
    QuasiPoisson,
    /// Quasi-Binomial: same variance V(μ)=μ(1-μ) and logit link as
    /// Binomial, but the dispersion φ is **estimated from data** rather
    /// than fixed at 1. mgcv estimates φ̂ = Dp / (n − trA).
    QuasiBinomial,
    /// Inverse Gaussian distribution with log link.
    ///
    /// Log link: η = log(μ), μ = exp(η). Variance function V(μ) = μ³.
    /// The canonical link for Inverse Gaussian is 1/μ², but log link
    /// is the most common production choice. The dispersion φ is profiled,
    /// using the same closed-form φ̂ = Dp/(n-Mp) as Gaussian.
    InverseGaussian,
    /// Negative Binomial distribution with log link.
    ///
    /// Log link: η = log(μ), μ = exp(η). Variance function V(μ) = μ + μ²/θ.
    /// The dispersion θ > 0 is either fixed (mgcv's `negbin(theta=...)`) or
    /// profiled jointly with λ (mgcv's `nb()` extended family). The standard
    /// scale parameter φ is fixed at 1 for NB; θ carries all overdispersion.
    NegBin { theta: f64 },
}

impl Family {
    /// Variance function V(μ). Same regardless of link choice (link
    /// affects how η maps to μ, not V(μ) itself).
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial | Family::QuasiBinomial => mu * (1.0 - mu),
            Family::Poisson | Family::QuasiPoisson => mu,
            Family::Gamma | Family::GammaLog => mu * mu,
            // For TDist: variance is not μ-dependent (identity link, location-scale
            // model). We return σ² here so that the standard Fisher weight formula
            // (dμ/dη)²/V(μ) = 1/σ² gives the baseline Fisher weight.
            // The actual per-obs t-weights are computed in fit_pirls_tdist.
            Family::TDist { sigma2, .. } => *sigma2,
            Family::Tweedie { p } => mu.powf(*p),
            // Inverse Gaussian variance: V(μ) = μ³
            Family::InverseGaussian => mu * mu * mu,
            // NB variance: V(μ) = μ + μ²/θ
            Family::NegBin { theta } => mu + mu * mu / theta,
        }
    }

    /// Link function g(μ).
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => mu,
            Family::Binomial | Family::QuasiBinomial => (mu / (1.0 - mu)).ln(),
            Family::Poisson | Family::QuasiPoisson | Family::GammaLog | Family::Tweedie { .. }
            | Family::InverseGaussian | Family::NegBin { .. } => mu.ln(),
            Family::Gamma => 1.0 / mu,
        }
    }

    /// Inverse link function g^(-1)(η).
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => eta,
            Family::Binomial | Family::QuasiBinomial => {
                let eta_safe = eta.max(-20.0).min(20.0);
                1.0 / (1.0 + (-eta_safe).exp())
            }
            Family::Poisson | Family::QuasiPoisson | Family::GammaLog | Family::Tweedie { .. }
            | Family::InverseGaussian | Family::NegBin { .. } => {
                let eta_safe = eta.min(20.0);
                eta_safe.exp()
            }
            Family::Gamma => {
                let eta_safe = if eta.abs() < 1e-10 { 1e-10 } else { eta };
                1.0 / eta_safe
            }
        }
    }

    /// Derivative of inverse link function dμ/dη.
    pub fn d_inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => 1.0,
            Family::Binomial | Family::QuasiBinomial => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            }
            Family::Poisson | Family::QuasiPoisson | Family::GammaLog | Family::Tweedie { .. }
            | Family::InverseGaussian | Family::NegBin { .. } => eta.exp(),
            Family::Gamma => -1.0 / (eta * eta),
        }
    }

    /// First derivative of variance function: dV/dμ.
    /// Used by mgcv's full-Newton PIRLS (`gam.fit3.r:507`) for non-canonical
    /// links to compute the α correction `α = 1 + (y−μ)·(V'/V + g''·dμ/dη)`.
    pub fn dvar(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => 1.0 - 2.0 * mu,
            Family::Poisson | Family::QuasiPoisson => 1.0,
            Family::Gamma | Family::GammaLog => 2.0 * mu,
            Family::Tweedie { p } => p * mu.powf(p - 1.0),
            // dV/dμ = 3μ²
            Family::InverseGaussian => 3.0 * mu * mu,
            // NB dV/dμ = 1 + 2μ/θ
            Family::NegBin { theta } => 1.0 + 2.0 * mu / theta,
        }
    }

    /// Second derivative of variance function: d²V/dμ².
    /// Used by mgcv's α₁ derivative (`gdi.c:2548`) needed for the Tk
    /// weight-derivative term in the REML gradient.
    pub fn d2var(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => -2.0,
            Family::Poisson | Family::QuasiPoisson => 0.0,
            Family::Gamma | Family::GammaLog => 2.0,
            Family::Tweedie { p } => p * (p - 1.0) * mu.powf(p - 2.0),
            // d²V/dμ² = 6μ
            Family::InverseGaussian => 6.0 * mu,
            // NB d²V/dμ² = 2/θ
            Family::NegBin { theta } => 2.0 / theta,
        }
    }

    /// Second derivative of link function: d²g/dμ².
    pub fn d2link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => {
                let one_minus = 1.0 - mu;
                -1.0 / (mu * mu) + 1.0 / (one_minus * one_minus)
            }
            Family::Poisson | Family::QuasiPoisson | Family::GammaLog | Family::Tweedie { .. }
            | Family::InverseGaussian | Family::NegBin { .. } => -1.0 / (mu * mu),
            Family::Gamma => 2.0 / (mu * mu * mu),
        }
    }

    /// Third derivative of link function: d³g/dμ³.
    pub fn d3link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => {
                let one_minus = 1.0 - mu;
                2.0 / (mu * mu * mu) + 2.0 / (one_minus * one_minus * one_minus)
            }
            Family::Poisson | Family::QuasiPoisson | Family::GammaLog | Family::Tweedie { .. }
            | Family::InverseGaussian | Family::NegBin { .. } => 2.0 / (mu * mu * mu),
            Family::Gamma => -6.0 / (mu * mu * mu * mu),
        }
    }

    /// True iff the link is canonical for this family. Determines whether
    /// PIRLS uses Fisher scoring (canonical) or full Newton (non-canonical),
    /// per `gam.fit3.r:118`.
    pub fn is_canonical_link(&self) -> bool {
        match self {
            Family::Gaussian
            | Family::Binomial
            | Family::Poisson
            | Family::QuasiPoisson  // log link is canonical for Poisson/QuasiPoisson
            | Family::QuasiBinomial // logit link is canonical for Binomial/QuasiBinomial
            | Family::Gamma
            | Family::TDist { .. } => true,
            // Tweedie canonical link is μ^(1-p); log link is non-canonical → full Newton.
            // InverseGaussian canonical link is 1/μ²; log link is non-canonical → full Newton.
            // NB canonical link is log(μ/(μ+θ)); log link is non-canonical → full Newton.
            Family::GammaLog | Family::Tweedie { .. } | Family::InverseGaussian
            | Family::NegBin { .. } => false,
        }
    }

    /// Saturated log-likelihood `ls[1]` per `gam.fit3.r:2497-2548`
    /// (`fix.family.ls`). Used by mgcv's REML formula
    /// `REML = Dp/(2σ²) - ls[1] + log|H|/2 - log|S|+/2 - Mp/2·log(2π·σ²)`
    /// (`gam.fit3.r:616`). For Gaussian this term collapses with the
    /// `Mp/2 log(2πσ²)` term into `(n-Mp)/2 log(2πσ²)`; for the other
    /// families the saturated likelihood is family-specific and must be
    /// included explicitly. `weights` defaults to 1 for every observation.
    pub fn saturated_log_likelihood(&self, y: &Array1<f64>, scale: f64) -> f64 {
        let n = y.len() as f64;
        match self {
            Family::Gaussian => -0.5 * n * (2.0 * std::f64::consts::PI * scale).ln(),
            // Quasi-likelihood: no true probability density, so the saturated
            // log-likelihood is undefined. We approximate with the Gaussian
            // form -n/2·log(2π·φ), which matches what mgcv uses for the
            // profiled-dispersion path. The absolute REML score differs from
            // mgcv's by a constant, but the λ-optimum is unaffected.
            Family::QuasiPoisson | Family::QuasiBinomial => {
                -0.5 * n * (2.0 * std::f64::consts::PI * scale).ln()
            }
            Family::Poisson => y
                .iter()
                .map(|&yi| {
                    if yi <= 0.0 {
                        0.0
                    } else {
                        // dpois(y, y, log=T) = y·log(y) - y - lgamma(y+1)
                        yi * yi.ln() - yi - log_gamma(yi + 1.0)
                    }
                })
                .sum(),
            Family::Binomial => {
                // mgcv: -aic(y, n, y, w, 0)/2 with weights=1, n_trials=1.
                // Saturated dbinom(y, 1, y) = y·log(y) + (1-y)·log(1-y),
                // which is 0 for y ∈ {0, 1} and the entropy term otherwise.
                y.iter()
                    .map(|&yi| {
                        if yi <= 0.0 || yi >= 1.0 {
                            0.0
                        } else {
                            yi * yi.ln() + (1.0 - yi) * (1.0 - yi).ln()
                        }
                    })
                    .sum()
            }
            Family::Gamma | Family::GammaLog => {
                let inv_phi = 1.0 / scale;
                let k = -log_gamma(inv_phi) - scale.ln() * inv_phi - inv_phi;
                let sum_log_y: f64 = y.iter().map(|&yi| yi.max(1e-300).ln()).sum();
                n * k - sum_log_y
            }
            // For TDist we approximate with the Gaussian saturated log-likelihood.
            // The saturated t-log-likelihood at y=μ is constant across observations
            // (doesn't depend on β), so its effect on the REML optimum is a constant
            // shift — acceptable for v1 λ-selection. The REML score absolute value
            // will differ from mgcv's but the λ-optimum is unaffected.
            Family::TDist { .. } => {
                -0.5 * n * (2.0 * std::f64::consts::PI * scale).ln()
            }
            // Inverse Gaussian saturated log-likelihood:
            // ls = -n/2 · log(2π·φ) - 3/2 · Σ log(y_i)
            // (from dinvgauss at saturation μ=y; the -3/2·Σlog(y) is a
            // constant in β but depends on φ via the -n/2·log(2πφ) term).
            Family::InverseGaussian => {
                let sum_log_y: f64 = y
                    .iter()
                    .map(|&yi| if yi > 0.0 { yi.ln() } else { 0.0 })
                    .sum();
                -0.5 * n * (2.0 * std::f64::consts::PI * scale).ln() - 1.5 * sum_log_y
            }
            // NB saturated log-likelihood (at y=μ):
            //   ls = Σ_i [ lgamma(y_i + θ) - lgamma(θ) - lgamma(y_i + 1)
            //              + θ·log(θ/(θ+y_i)) + y_i·log(y_i/(θ+y_i)) ]
            // where y_i·log(y_i/(θ+y_i)) is taken as 0 when y_i = 0.
            // scale (φ) is unused — NB's overdispersion lives entirely in θ.
            Family::NegBin { theta } => {
                let theta = *theta;
                y.iter()
                    .map(|&yi| {
                        let yi_p_theta = yi + theta;
                        let lgam_term = log_gamma(yi_p_theta) - log_gamma(theta)
                            - log_gamma(yi + 1.0);
                        let log_ratio_theta = theta.ln() - yi_p_theta.ln();
                        let y_log_term = if yi > 0.0 {
                            yi * (yi.ln() - yi_p_theta.ln())
                        } else {
                            0.0
                        };
                        lgam_term + theta * log_ratio_theta + y_log_term
                    })
                    .sum()
            }
            // Tweedie saturated log-likelihood for 1 < p < 2, log link.
            // At y=μ: log f(y; y, φ, p) = l_base(y, φ, p) - log(y) + log W(y, φ, p)
            // where l_base = μ^(1-p) * (y/(1-p) - μ/(2-p)) / φ  evaluated at μ=y.
            //   = y^(1-p) * y * (1/(1-p) - 1/(2-p)) / φ
            //   = y^(2-p) * (1/((1-p)(2-p))) * (2-p-(1-p)) / φ
            //   = y^(2-p) / (φ * (2-p)) * (-1)  ... simplified:
            // At μ=y: l_base = y^(1-p) * (y/(1-p) - y/(2-p)) / φ
            //                = y^(2-p) * (1/(1-p) - 1/(2-p)) / φ
            // And log W is the Dunn-Smyth series log-sum.
            // For y=0: log f = -μ^(2-p)/(φ*(2-p)) → at μ=y=0 this is 0.
            Family::Tweedie { p } => {
                let phi = scale;
                let (log_w, _, _) = tweedie_series(y, phi, *p);
                let mut ls = 0.0f64;
                for (i, &yi) in y.iter().enumerate() {
                    if yi <= 0.0 {
                        // log f(0; 0, φ, p) = 0 at y=μ=0
                        continue;
                    }
                    let onep = 1.0 - p;
                    let twop = 2.0 - p;
                    // l_base at μ=y: y^(1-p) * (y/(1-p) - y/(2-p)) / φ
                    let l_base = yi.powf(onep) * yi * (1.0 / onep - 1.0 / twop) / phi;
                    ls += l_base - yi.ln() + log_w[i];
                }
                ls
            }
        }
    }

    /// Derivative of saturated log-likelihood w.r.t. σ² (dispersion/scale).
    ///
    /// Used to compute the σ²-chain correction term:
    ///   (∂REML/∂σ²) = -Dp/(2σ⁴) - dls/dσ² - Mp/(2σ²)
    ///
    /// For Gaussian and TDist: `ls = -n/2·log(2πσ²)`, so `dls/dσ² = -n/(2σ²)`.
    /// For Poisson/Binomial: σ² is fixed at 1 (not a free parameter), so the
    /// chain term is identically zero — this method is never called for those.
    /// For Gamma/GammaLog: `ls = n·[-lgamma(1/φ) - log(φ)/φ - 1/φ] - Σlog y`
    /// with φ = σ², so `dls/dφ = n·[digamma(1/φ) + log φ] / φ²`.
    pub fn dls_dsigma2(&self, y: &Array1<f64>, scale: f64) -> f64 {
        let n = y.len() as f64;
        match self {
            Family::Gaussian => -n / (2.0 * scale),
            // QuasiPoisson/QuasiBinomial: Gaussian-approximation ls → dls/dσ² = -n/(2σ²).
            // σ² is profiled (free parameter), not fixed at 1.
            Family::QuasiPoisson | Family::QuasiBinomial => -n / (2.0 * scale),
            // InverseGaussian: ls = -n/2·log(2πφ) - 3/2·Σlog(y), so dls/dφ = -n/(2φ).
            Family::InverseGaussian => -n / (2.0 * scale),
            Family::Poisson | Family::Binomial => 0.0, // scale = 1, not a free parameter
            // NB: scale φ is fixed at 1; θ carries all overdispersion.
            // dls/dσ² = 0 (ls doesn't depend on σ²).
            Family::NegBin { .. } => 0.0,
            Family::Gamma | Family::GammaLog => {
                // ls = n·[-lgamma(1/φ) - log(φ)/φ - 1/φ] - Σlog y
                // dls/dφ = n·[digamma(1/φ) + log(φ)] / φ²
                let inv_phi = 1.0 / scale;
                n * (digamma(inv_phi) + scale.ln()) / (scale * scale)
            }
            Family::TDist { .. } => -n / (2.0 * scale),
            // Tweedie dls/dφ: from ldTweedie0 R code (gam.fit3.r:2799):
            //   ld[,2] = -l_base/φ + dlogW/dφ
            // where l_base is the analytic density term at μ=y, and
            // dlogW/dφ = (dlogW/drho) * (1/φ)  [since rho = log φ].
            // We get dlogW/drho from tweedie_series as the second output.
            Family::Tweedie { p } => {
                let phi = scale;
                let (log_w, dlog_w_drho, _) = tweedie_series(y, phi, *p);
                let _ = log_w; // log_w already used for ls; here we only need derivs
                let mut dls = 0.0f64;
                for (i, &yi) in y.iter().enumerate() {
                    if yi <= 0.0 {
                        // For y=0: ls_i = -μ^(2-p)/(φ*(2-p)) at μ=y=0 is 0 → dls=0
                        continue;
                    }
                    let onep = 1.0 - p;
                    let twop = 2.0 - p;
                    let l_base = yi.powf(onep) * yi * (1.0 / onep - 1.0 / twop) / phi;
                    // dls_i/dphi = -l_base/phi + dlogW_i/drho / phi
                    dls += -l_base / phi + dlog_w_drho[i] / phi;
                }
                dls
            }
        }
    }
}

/// Digamma function ψ(x) = d/dx ln Γ(x).
///
/// Uses the recurrence ψ(x) = ψ(x+1) - 1/x to push x ≥ 6, then the
/// asymptotic expansion ψ(x) ≈ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k}).
/// Accurate to ~13 significant figures for x > 0.
pub(crate) fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence: push x ≥ 6 so the asymptotic series converges well.
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result += x.ln() - 0.5 * xinv;
    let mut t = xinv2;
    result -= t / 12.0;   // B2/2 = (1/6)/2
    t *= xinv2;
    result += t / 120.0;  // B4/4 = (-1/30)/4  → +1/120
    t *= xinv2;
    result -= t / 252.0;  // B6/6 = (1/42)/6   → -1/252
    t *= xinv2;
    result += t / 240.0;  // B8/8 = (-1/30)/8  → +1/240
    result
}

/// Trigamma function ψ'(x) = d²/dx² ln Γ(x) = d/dx ψ(x).
///
/// Uses the recurrence ψ'(x) = ψ'(x+1) + 1/x² to push x ≥ 6, then the
/// asymptotic series. Accurate to ~13 significant figures for x > 0.
pub(crate) fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    // Recurrence: push x ≥ 6 so the asymptotic series converges well.
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic: ψ'(x) ≈ 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1}
    let xinv = 1.0 / x;
    let xinv2 = xinv * xinv;
    result += xinv + 0.5 * xinv2;
    let mut t = xinv2 * xinv; // 1/x^3
    result += t / 6.0; // B2 = 1/6
    t *= xinv2; // 1/x^5
    result -= t / 30.0; // B4 = -1/30
    t *= xinv2; // 1/x^7
    result += t / 42.0; // B6 = 1/42
    t *= xinv2; // 1/x^9
    result -= t / 30.0; // B8 = -1/30
    result
}

impl Family {
    /// Solve for mgcv's φ̂ by family.
    ///
    /// For Gamma/GammaLog, solves the joint equation `dlr.dlphi = 0` from
    /// `gam.fit3.r:630` via Newton-Raphson in F(φ) = dp + 2n[ψ(1/φ)+log φ] + mp·φ.
    /// For Gaussian, uses the closed form φ = dp/(n-mp).
    /// For Poisson/Binomial, returns 1.0 (fixed dispersion).
    /// For TDist, returns phi_init unchanged (Gaussian-approximation σ̂²; no
    /// digamma-style fix for t-dist in v1).
    ///
    /// # Arguments
    /// * `y` — response vector (used only for `n = y.len()`)
    /// * `dp` — penalised deviance Dp = D + β'Sβ
    /// * `mp` — null-space dimension (intercept + per-smooth null spaces)
    /// * `gamma` — smoothing parameter inflation factor (typically 1.0)
    /// * `phi_init` — initial guess (caller passes D/(n-trA))
    pub fn estimate_phi_mgcv(
        &self,
        y: &Array1<f64>,
        dp: f64,
        mp: usize,
        _gamma: f64,
        phi_init: f64,
    ) -> f64 {
        let n = y.len() as f64;
        let mp_f = mp as f64;
        match self {
            Family::Gaussian => dp / (n - mp_f).max(1.0),
            // QuasiPoisson/QuasiBinomial: same closed-form φ̂ = Dp/(n-Mp) as Gaussian.
            // Dispersion is profiled, not fixed at 1.
            Family::QuasiPoisson | Family::QuasiBinomial => dp / (n - mp_f).max(1.0),
            // InverseGaussian: ls = -n/2·log(2πφ) form → same closed-form φ̂ as Gaussian.
            Family::InverseGaussian => dp / (n - mp_f).max(1.0),
            Family::Binomial | Family::Poisson | Family::NegBin { .. } => 1.0,
            Family::TDist { .. } => phi_init,
            Family::Gamma | Family::GammaLog => {
                // Newton-Raphson on
                //   F(φ) = dp + 2n[ψ(1/φ) + log φ] + mp·φ
                // F'(φ) = (2n/φ)[1 - ψ'(1/φ)/φ] + mp
                let mut phi = phi_init.max(1e-8);
                let tol_abs = 1e-10 * (dp.abs() + mp_f + 1.0);
                for _ in 0..30 {
                    let inv_phi = 1.0 / phi;
                    let f = dp + 2.0 * n * (digamma(inv_phi) + phi.ln()) + mp_f * phi;
                    if f.abs() < tol_abs {
                        break;
                    }
                    let fp = (2.0 * n / phi) * (1.0 - trigamma(inv_phi) * inv_phi) + mp_f;
                    // Guard against zero / near-zero derivative
                    if fp.abs() < 1e-15 {
                        break;
                    }
                    let delta = -f / fp;
                    // Damp to prevent runaway
                    let phi_new = (phi + delta).max(phi * 0.1).min(phi * 10.0);
                    let converged = (phi_new - phi).abs() < 1e-12 * phi;
                    phi = phi_new;
                    if converged {
                        break;
                    }
                }
                phi
            }
            // Tweedie: profile φ via Newton-Raphson on the REML score derivative
            //   dlr/dφ = -dp/(2φ²) - dls/dφ - mp/(2φ) = 0
            // We solve this numerically using FD for the second derivative of ls,
            // since the Tweedie series ls has no closed-form second derivative here.
            Family::Tweedie { p } => {
                let p = *p;
                let mut phi = phi_init.max(1e-8);
                for _ in 0..30 {
                    let dls = self.dls_dsigma2(y, phi);
                    // dlr/dphi = -dp/(2 phi^2) - dls/dphi - mp/(2 phi)
                    let f = -dp / (2.0 * phi * phi) - dls - mp_f / (2.0 * phi);
                    if f.abs() < 1e-10 * (dp.abs() / (phi * phi) + 1.0) {
                        break;
                    }
                    // FD for df/dphi
                    let h = phi * 1e-5;
                    let dls_plus = Family::Tweedie { p }.dls_dsigma2(y, phi + h);
                    let dls_minus = Family::Tweedie { p }.dls_dsigma2(y, phi - h);
                    let fp = dp / (phi * phi * phi)
                        - (dls_plus - dls_minus) / (2.0 * h)
                        + mp_f / (2.0 * phi * phi);
                    if fp.abs() < 1e-20 {
                        break;
                    }
                    let delta = -f / fp;
                    let phi_new = (phi + delta).max(phi * 0.1).min(phi * 10.0);
                    let converged = (phi_new - phi).abs() < 1e-12 * phi;
                    phi = phi_new;
                    if converged {
                        break;
                    }
                }
                phi
            }
        }
    }
}

/// Lanczos approximation of the log-Gamma function. Accurate to ~14 digits
/// for x > 0. Pulled in directly to avoid a dependency on `statrs` for this
/// single use.
fn log_gamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - log_gamma(1.0 - x);
    }
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut a = coef[0];
    for (i, &c) in coef.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Dunn-Smyth (2005) series for the Tweedie log-density (1 < p < 2).
///
/// Returns `(log_W, dlog_W_drho)` per observation, where:
/// - `log_W[i]` = log of the series sum W = Σ_j W_j for y[i]
/// - `dlog_W_drho[i]` = d(log W)/d(rho) where rho = log(phi)
///
/// The series term is (Dunn & Smyth 2005, eq after eq.4):
///   log W_j = j*(alpha*log(p-1) + rho/onep - log(2-p)) - lgamma(j+1) - lgamma(-j*alpha)
///             - j*alpha*log(y)
/// where alpha = (2-p)/(1-p), onep = 1-p.
///
/// Summation is done via log-sum-exp (wmax trick) to avoid overflow,
/// mirroring `tweedious` in mgcv/src/misc.c:170.
///
/// For y=0 entries: returns (0.0, 0.0) — the density for y=0 doesn't need the series.
/// Dunn-Smyth (2005) series for the Tweedie log-density (1 < p < 2).
///
/// Returns `(log_W, dlog_W_drho, dlog_W_dp)` per observation, where:
/// - `log_W[i]` = log of the series sum W = Σ_j W_j for y[i]
/// - `dlog_W_drho[i]` = d(log W)/d(rho) where rho = log(phi)
/// - `dlog_W_dp[i]` = d(log W)/dp (used for profile-p Newton step on theta)
///
/// The wp_base term for dlogW/dp comes from misc.c:231:
///   wp_base = (log(-(1-p)) + rho) / (1-p)^2 - alpha/(1-p) + 1/(2-p)
/// Per j: wp1_j_base = j * wp_base + (j/(1-p)^2) * digamma(-j*alpha)
/// Per observation: subtract j * log(y) / (1-p)^2 to get the full wp1_j
/// Then dlogW/dp = sum(wj_scaled * wp1j) / wi  (C misc.c:503: w1p[i] = wdlogwdp/wi)
fn tweedie_series(y: &Array1<f64>, phi: f64, p: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut log_w_out = vec![0.0f64; n];
    let mut dlog_w_drho_out = vec![0.0f64; n];
    let mut dlog_w_dp_out = vec![0.0f64; n];

    let onep = 1.0 - p;
    let onep2 = onep * onep; // (1-p)^2
    let twop = 2.0 - p;
    let alpha = twop / onep; // (2-p)/(1-p), negative for 1<p<2
    let rho = phi.ln();

    // w_base = j * (alpha*log(p-1) + rho/onep - log(2-p))
    // Note: log(-onep) = log(p-1) since onep = 1-p < 0 for 1<p<2.
    let log_pm1 = (p - 1.0).ln(); // = log(-onep) = log(p-1)
    let w_base = alpha * log_pm1 + rho / onep - twop.ln();
    // wb1_base: for obs i, wb1_j = -j/onep (= j/|onep| > 0 since onep < 0)
    let wb1_base = -1.0 / onep; // multiply by j to get wb1_j
    // wp_base for dlogW/dp: misc.c:231
    //   wp_base = (log(-onep) + rho)/onep^2 - alpha/onep + 1/(2-p)
    let wp_base = (log_pm1 + rho) / onep2 - alpha / onep + 1.0 / twop;

    let log_eps = f64::EPSILON * f64::EPSILON; // tolerance for convergence (~5e-32)

    for i in 0..n {
        let yi = y[i];
        if yi <= 0.0 {
            // y=0: no series needed
            log_w_out[i] = 0.0;
            dlog_w_drho_out[i] = 0.0;
            dlog_w_dp_out[i] = 0.0;
            continue;
        }

        let logy_i = yi.ln();
        let alogy_i = alpha * logy_i; // alpha * log(y_i)
        let logy1p2_i = logy_i / onep2; // log(y_i) / (1-p)^2  (for wp1 adjustment per obs)

        // j_max: the series mode near y^(2-p) / (phi*(2-p))
        let x = yi.powf(twop) / (phi * twop);
        let j_max = (x.floor() as i64).max(1);

        // First pass: find wmax by evaluating at j_max
        let wmax_j = j_max;
        let wmax_val = {
            let j = wmax_j as f64;
            j * w_base - log_gamma(j + 1.0) - log_gamma(-j * alpha) - j * alogy_i
        };

        let wmin = wmax_val + log_eps.ln();

        // Accumulate sums (scaled by exp(-wmax)):
        // wi = Σ exp(wj - wmax) = W / exp(wmax)
        // w1i = Σ exp(wj - wmax) * wb1_j  (for dlogW/drho numerator)
        // wdlogwdp = Σ exp(wj - wmax) * wp1_j  (for dlogW/dp numerator)
        let mut wi = 0.0f64;
        let mut w1i = 0.0f64;
        let mut wdlogwdp = 0.0f64;

        // Helper: compute wp1_j = j * wp_base + (j/onep2) * digamma(-j*alpha) - j * logy1p2_i
        // This is the per-j derivative of log W_j w.r.t. p (misc.c:333).
        let wp1_j = |jf: f64| -> f64 {
            let dig_term = (jf / onep2) * digamma(-jf * alpha);
            jf * wp_base + dig_term - jf * logy1p2_i
        };

        // Upsweep from j_max upward until wj < wmin
        let mut j = wmax_j;
        loop {
            let jf = j as f64;
            let wj = jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha) - jf * alogy_i;
            let wj_scaled = (wj - wmax_val).exp();
            wi += wj_scaled;
            w1i += wj_scaled * jf * wb1_base; // wb1_j = j * (-1/onep)
            wdlogwdp += wj_scaled * wp1_j(jf);
            if wj < wmin {
                break;
            }
            j += 1;
            if j > 10_000_000 {
                // Safety cap — shouldn't be reached for reasonable data
                break;
            }
        }

        // Downsweep from j_max-1 down to 1
        j = wmax_j - 1;
        while j >= 1 {
            let jf = j as f64;
            let wj = jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha) - jf * alogy_i;
            let wj_scaled = (wj - wmax_val).exp();
            wi += wj_scaled;
            w1i += wj_scaled * jf * wb1_base;
            wdlogwdp += wj_scaled * wp1_j(jf);
            if wj < wmin {
                break;
            }
            j -= 1;
        }

        // log W = wmax + log(wi)
        log_w_out[i] = wmax_val + wi.ln();
        // d(log W)/d(rho): w1[i] = -w1i/wi  (misc.c:502)
        dlog_w_drho_out[i] = -w1i / wi;
        // d(log W)/dp: w1p[i] = wdlogwdp / wi  (misc.c:503)
        dlog_w_dp_out[i] = wdlogwdp / wi;
    }

    (log_w_out, dlog_w_drho_out, dlog_w_dp_out)
}

/// Compute dls/dp for the Tweedie family — the derivative of the saturated
/// log-likelihood w.r.t. p. Used by the profile-p outer Newton step.
///
/// From ldTweedie (gam.fit3.r:2924): for y>0,
///   dls_i/dp = l_base_i * (log(mu) - 1/(2-p))  [at mu=y, log(mu)=log(y)]
///            + dlogW_i/dp
/// For y=0: from gam.fit3.r:2924:
///   dls_i/dp = -ls_i * (log(y) - 1/(2-p))  ... but ls_i = 0 at y=0 so dls=0.
pub(crate) fn tweedie_dls_dp(y: &Array1<f64>, phi: f64, p: f64) -> f64 {
    let (_log_w, _dlog_w_drho, dlog_w_dp) = tweedie_series(y, phi, p);
    let twop = 2.0 - p;
    let onep = 1.0 - p;
    let mut dls = 0.0f64;
    for (i, &yi) in y.iter().enumerate() {
        if yi <= 0.0 {
            continue;
        }
        let logy = yi.ln();
        // l_base at mu=y: y^(1-p) * y * (1/onep - 1/twop) / phi
        let l_base = yi.powf(onep) * yi * (1.0 / onep - 1.0 / twop) / phi;
        // dls_i/dp = l_base_i * (log(y_i) - 1/(2-p)) + dlogW_i/dp
        dls += l_base * (logy - 1.0 / twop) + dlog_w_dp[i];
    }
    dls
}

/// PiRLS fitting result
pub struct PiRLSResult {
    pub coefficients: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub linear_predictor: Array1<f64>,
    pub weights: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Fit a GAM using PiRLS algorithm
///
/// # Arguments
/// * `y` - Response vector
/// * `x` - Design matrix (basis functions evaluated at data points)
/// * `lambda` - Smoothing parameters (one per smooth term)
/// * `penalties` - Penalty matrices (one per smooth term)
/// * `family` - Distribution family
/// * `max_iter` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
pub fn fit_pirls(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    max_iter: usize,
    tolerance: f64,
) -> Result<PiRLSResult> {
    fit_pirls_cached(y, x, lambda, penalties, family, max_iter, tolerance, None)
}

/// Fit a GAM using PiRLS algorithm with optional cached X'X matrix
///
/// For Gaussian family, X'WX = X'X (constant weights = 1), so we can accept
/// a pre-computed X'X to avoid the O(n*p²) computation entirely.
///
/// # Arguments
/// * `cached_xtx` - Optional pre-computed X'X matrix (only valid for Gaussian family)
pub fn fit_pirls_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    max_iter: usize,
    tolerance: f64,
    cached_xtx: Option<&Array2<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }

    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string(),
        ));
    }

    // Fast path for Gaussian family: weights are constant (w=1), z=y on first iteration,
    // and PiRLS converges in exactly 1 step. Skip all the IRLS machinery.
    if matches!(family, Family::Gaussian) {
        return fit_pirls_gaussian_fast(y, x, lambda, penalties, p, cached_xtx);
    }

    // General IRLS path for non-Gaussian families
    // Initialize coefficients and linear predictor
    let mut beta = Array1::zeros(p);
    let mut eta = x.dot(&beta);

    // Initialize eta based on family
    for i in 0..n {
        let safe_y = match family {
            Family::Binomial | Family::QuasiBinomial => y[i].max(0.01).min(0.99),
            Family::Poisson | Family::QuasiPoisson | Family::Gamma | Family::GammaLog
            | Family::Tweedie { .. } | Family::InverseGaussian | Family::NegBin { .. } => y[i].max(0.1),
            // TDist and Gaussian use identity link; initialize to y directly
            Family::Gaussian | Family::TDist { .. } => y[i],
        };
        eta[i] = family.link(safe_y);
    }

    let mut converged = false;
    let mut iter = 0;

    // Pre-compute penalty total (doesn't change between iterations)
    let mut penalty_total = Array2::<f64>::zeros((p, p));
    for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
        penalty_j.scaled_add_to(&mut penalty_total, *lambda_j);
    }
    let num_penalties = lambda.len();
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + (num_penalties as f64).sqrt())
    };

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Compute fitted values μ = g^(-1)(η)
        let mu: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();

        // Compute working response z and IRLS weights w in a single pass.
        // Canonical link → Fisher scoring (E[w] = wf). Non-canonical → full
        // Newton (per gam.fit3.r:505-515): w = wf·α, z = η + (y−μ)/(dμ/dη·α),
        // where α = 1 + (y−μ)·(V'/V + g''·dμ/dη). The `c·` factor in mgcv
        // becomes 0 at convergence (y ≈ μ), giving α → 1 and Newton → Fisher.
        let use_fisher = family.is_canonical_link();
        let mut z = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        for i in 0..n {
            let dmu_deta = family.d_inverse_link(eta[i]);
            let variance = family.variance(mu[i]);
            let var_safe = variance.max(1e-10);
            if dmu_deta.abs() < 1e-10 {
                z[i] = eta[i];
                w[i] = 1e-10;
                continue;
            }
            let wf = (dmu_deta * dmu_deta) / var_safe;
            if use_fisher {
                z[i] = eta[i] + (y[i] - mu[i]) / dmu_deta;
                w[i] = wf.max(1e-10);
            } else {
                let c_resid = y[i] - mu[i];
                let dvar = family.dvar(mu[i]);
                let d2link = family.d2link(mu[i]);
                let mut alpha = 1.0 + c_resid * (dvar / var_safe + d2link * dmu_deta);
                // Clamp α positive so X'WX + λS stays PSD. For InverseGaussian+log,
                // α = 2y/μ − 1 which goes negative when y < μ/2 (common far from
                // optimum). Negative α makes W indefinite → Cholesky → NaN.
                // mgcv handles this via QR with column pivoting; we use a direct
                // solve, so clamp instead. The convergence trajectory differs but
                // the final point is the same.
                if alpha <= 0.0 {
                    alpha = f64::EPSILON;
                }
                z[i] = eta[i] + (y[i] - mu[i]) / (dmu_deta * alpha);
                w[i] = wf * alpha;
            }
        }

        // X'WX using BLAS (instead of manual triple-nested loop)
        let xtwx = compute_xtwx(x, &w);

        // Compute max diagonal for ridge scaling
        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;

        // Add adaptive ridge for numerical stability
        let ridge: f64 = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // X'Wz using BLAS: X' * (w .* z)
        let wz: Array1<f64> = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi).collect();
        let xtwz = x.t().dot(&wz);

        // Solve for new coefficients
        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // Update linear predictor
        eta = x.dot(&beta);

        // Check convergence
        let max_change = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(b, b_old)| (b - b_old).abs())
            .fold(0.0f64, f64::max);

        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    // Compute final fitted values
    let fitted_values: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();

    // Compute deviance
    let deviance = compute_deviance(y, &fitted_values, family);

    // Compute final weights
    let weights: Array1<f64> = eta
        .iter()
        .map(|&e| {
            let mu = family.inverse_link(e);
            let dmu_deta = family.d_inverse_link(e);
            let variance = family.variance(mu);
            ((dmu_deta * dmu_deta) / variance.max(1e-10)).max(1e-10)
        })
        .collect();

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        deviance,
        iterations: iter,
        converged,
    })
}

/// Fast path for Gaussian PiRLS: no iteration needed.
///
/// For Gaussian family with identity link:
/// - Weights w = 1 (constant)
/// - Working response z = y
/// - PiRLS converges in 1 step: β = (X'X + Σλ_jS_j)^{-1} X'y
///
/// This avoids all IRLS overhead and can reuse a cached X'X matrix.
fn fit_pirls_gaussian_fast(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    p: usize,
    cached_xtx: Option<&Array2<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();

    // X'X: use cached version or compute via BLAS
    let xtx = if let Some(cached) = cached_xtx {
        cached.clone()
    } else {
        // For Gaussian, w=1, so X'WX = X'X. Use BLAS: X' * X
        x.t().dot(x)
    };

    // Compute max diagonal for ridge scaling
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(xtx[[i, i]].abs());
    }

    // Build (X'X + Σλ_jS_j + ridge*I)
    let mut a = xtx;
    for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
        // a += lambda_j * penalty_j (only touches k×k block)
        penalty_j.scaled_add_to(&mut a, *lambda_j);
    }

    let num_penalties = lambda.len();
    // mgcv-exact mode uses a much smaller ridge so β is essentially
    // (X'X + λS)^{-1} X'y unperturbed; the default ridge of 1e-5 *
    // (1+sqrt(m)) * max_diag was causing predictions to shift by
    // ~1e-3 even at machine-precision-matched X / S / λ.
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + (num_penalties as f64).sqrt())
    };
    let ridge: f64 = ridge_scale * max_diag;
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    // X'y using BLAS
    let xty = x.t().dot(y);

    // Solve for coefficients: β = A^{-1} X'y
    let beta = solve(a, xty)?;

    // eta = X*β
    let eta = x.dot(&beta);

    // For Gaussian, fitted_values = eta (identity link)
    let fitted_values = eta.clone();

    // Deviance = Σ(y_i - μ_i)²
    let deviance: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(yi, fi)| (yi - fi).powi(2))
        .sum();

    // Weights are all 1.0 for Gaussian
    let weights = Array1::ones(n);

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        deviance,
        iterations: 1,
        converged: true,
    })
}

/// Compute deviance for a given family
pub fn compute_deviance(y: &Array1<f64>, mu: &Array1<f64>, family: Family) -> f64 {
    let mut deviance = 0.0;

    for i in 0..y.len() {
        let yi = y[i];
        let mui = mu[i].max(1e-10);

        let dev_i = match family {
            Family::Gaussian => (yi - mui).powi(2),
            Family::Binomial | Family::QuasiBinomial => {
                if yi > 0.0 && yi < 1.0 {
                    2.0 * (yi * (yi / mui).ln() + (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln())
                } else {
                    0.0
                }
            }
            Family::Poisson | Family::QuasiPoisson => {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * mui
                }
            }
            Family::Gamma | Family::GammaLog => 2.0 * ((yi - mui) / mui - (yi / mui).ln()),
            // TDist deviance: -2 * log p(y|μ,σ²,df) up to an additive constant.
            // Use the squared residual as a proxy (consistent scale comparison).
            Family::TDist { .. } => (yi - mui).powi(2),
            // Tweedie deviance for 1 < p < 2 (log link):
            //   d_i = 2 * [ y^(2-p)/((1-p)(2-p)) - y*μ^(1-p)/(1-p) + μ^(2-p)/(2-p) ]
            // = 2 * [ y^(2-p) - (2-p)*y*μ^(1-p) + (1-p)*μ^(2-p) ] / ((1-p)(2-p))
            Family::Tweedie { p } => {
                let twop = 2.0 - p;
                let onep = 1.0 - p;
                if yi <= 0.0 {
                    // For y=0: deviance = 2 * μ^(2-p) / (2-p)
                    2.0 * mui.powf(twop) / twop
                } else {
                    2.0 * (yi.powf(twop) / (onep * twop)
                        - yi * mui.powf(onep) / onep
                        + mui.powf(twop) / twop)
                }
            }
            // Inverse Gaussian deviance: d_i = (y - μ)² / (μ² · y)
            Family::InverseGaussian => {
                let mu_c = mui.max(1e-15);
                let yi_c = yi.max(1e-15);
                let diff = yi_c - mu_c;
                diff * diff / (mu_c * mu_c * yi_c)
            }
            // NB deviance per obs (matches mgcv `negbin$dev.resids` at
            // gam.fit3.r:2599-2602):
            //   2 · [y · log(max(1,y)/μ) - (y+θ) · log((y+θ)/(μ+θ))]
            // For y=0: y·log(...) → 0 (mgcv uses pmax(1, y) so log(1/μ)=-log μ
            // is multiplied by y=0, giving 0). The (y+θ)·log((y+θ)/(μ+θ))
            // term becomes θ·log(θ/(μ+θ)), so deviance = -2θ·log(θ/(μ+θ))
            // = 2θ·log((μ+θ)/θ) — POSITIVE (since μ>0 ⇒ (μ+θ)/θ > 1).
            Family::NegBin { theta } => {
                let mu_c = mui.max(1e-15);
                if yi > 0.0 {
                    2.0 * (yi * (yi / mu_c).ln()
                        - (yi + theta) * ((yi + theta) / (mu_c + theta)).ln())
                } else {
                    2.0 * theta * ((mu_c + theta) / theta).ln()
                }
            }
        };

        deviance += dev_i;
    }

    deviance
}

// ────────────────────────────────────────────────────────────────────────────
// Scaled t-distribution (scat) family — df profiling and IRLS
// ────────────────────────────────────────────────────────────────────────────

/// Profile log-likelihood of the scaled t-distribution over df ∈ [2, 100],
/// holding residuals and σ² fixed.
///
/// log p(y | μ, σ², ν) ∝ Σ_i [ lgamma((ν+1)/2) - lgamma(ν/2)
///     - 0.5 ln(ν π σ²)
///     - (ν+1)/2 · ln(1 + r_i²/(ν σ²)) ]
///
/// This is maximised over ν using Brent's method on [2, 100].
/// `residuals` = y - μ for each observation.
pub fn profile_df(residuals: &[f64], sigma2: f64) -> f64 {
    // Profile log-likelihood (negated for minimization)
    let neg_pll = |nu: f64| -> f64 {
        let n = residuals.len() as f64;
        let half_nu_p1 = (nu + 1.0) / 2.0;
        let half_nu = nu / 2.0;
        let log_sum: f64 = residuals
            .iter()
            .map(|&r| {
                let t2 = r * r / (nu * sigma2.max(1e-300));
                (1.0 + t2).ln()
            })
            .sum();
        // Negative profile log-likelihood (minimise ↔ maximise log-likelihood)
        -(n * (log_gamma(half_nu_p1) - log_gamma(half_nu) - 0.5 * (nu * std::f64::consts::PI * sigma2.max(1e-300)).ln())
            - half_nu_p1 * log_sum)
    };

    brent_minimize(neg_pll, 2.0, 100.0, 1e-4, 50)
}

/// Brent's method for univariate function minimization on [a, b].
/// Adapted from the classic algorithm (no external dependencies).
fn brent_minimize<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let golden = 0.381_966_011_250_105;
    let eps = 1e-10;

    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;

    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        let tol1 = tol * x.abs() + eps;
        let tol2 = 2.0 * tol1;

        if (x - mid).abs() <= tol2 - 0.5 * (b - a) {
            return x;
        }

        let mut take_golden = true;
        if e.abs() > tol1 {
            // Parabolic fit
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let mut q2 = 2.0 * (q - r);
            let pp = if q2 > 0.0 { -p } else { p };
            let q2a = q2.abs();
            if pp.abs() < (0.5 * q2a * e).abs() && pp > q2a * (a - x) && pp < q2a * (b - x) {
                d = pp / q2a;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < mid { tol1 } else { -tol1 };
                }
                take_golden = false;
            }
        }

        if take_golden {
            e = if x < mid { b - x } else { a - x };
            d = golden * e;
        }

        let u = x + if d.abs() >= tol1 { d } else if d > 0.0 { tol1 } else { -tol1 };
        let fu = f(u);

        if fu <= fx {
            if u < x { b = x; } else { a = x; }
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if u < x { a = u; } else { b = u; }
            if fu <= fw || w == x {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u; fv = fu;
            }
        }
    }
    x
}

/// Fit a penalized GAM with scaled t-distribution errors.
///
/// This is the outer loop for `Family::TDist` fitting. It alternates between:
///
/// 1. **Inner PIRLS** — fit β at fixed (df, σ²) using t-distribution IRLS
///    weights `w_i = (df+1) / (df + r_i²/σ²)` with identity link.
/// 2. **σ² update** — method-of-moments: `σ² = Σ w_i r_i² / (Σ w_i - p)`.
/// 3. **df update** (when `fixed_df` is None) — 1D Brent optimisation of the
///    profile log-likelihood over df ∈ [2, 100].
///
/// Returns a `PiRLSResult` with the converged β. The `weights` field holds the
/// final per-observation t-weights.
pub fn fit_pirls_tdist(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    fixed_df: Option<f64>,
    max_iter: usize,
    tolerance: f64,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }
    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string(),
        ));
    }

    // Validate df if fixed
    if let Some(df) = fixed_df {
        if df < 2.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be >= 2.0, got {}", df
            )));
        }
        if df > 100.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be <= 100.0, got {}", df
            )));
        }
    }

    // Build penalty total once
    let mut penalty_total = Array2::<f64>::zeros((p, p));
    for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
        penalty_j.scaled_add_to(&mut penalty_total, *lambda_j);
    }

    let num_penalties = lambda.len();
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + (num_penalties as f64).sqrt())
    };

    // Initialise β = 0, η = y (identity link), σ² from sample variance
    let mut beta = Array1::zeros(p);
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let mut sigma2 = y_var.max(1e-6);

    // df: start at user value or 5.0 (a reasonable default)
    let mut df = fixed_df.unwrap_or(5.0).clamp(2.0, 100.0);

    let mut converged = false;
    let mut iter = 0;

    for outer_iter in 0..max_iter {
        iter = outer_iter + 1;

        // ── Inner WLS: solve (X'WX + S) β = X'Wz with t-weights ─────────
        // For identity link: μ = η = Xβ, z = y (working response = y for identity).
        // t-weight: w_i = (df+1) / (df + r_i² / σ²) where r_i = y_i - η_i
        let eta: Array1<f64> = x.dot(&beta);
        let w: Array1<f64> = y
            .iter()
            .zip(eta.iter())
            .map(|(&yi, &etai)| {
                let r2 = (yi - etai).powi(2);
                let t2 = r2 / sigma2.max(1e-300);
                ((df + 1.0) / (df + t2)).max(1e-10)
            })
            .collect();

        // X'WX via dense triple product
        let xtwx = crate::reml::compute_xtwx(x, &w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // z = y for identity link
        let wz: Array1<f64> = w.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).collect();
        let xtwz = x.t().dot(&wz);

        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // ── Update σ² via method-of-moments ──────────────────────────────
        let eta_new: Array1<f64> = x.dot(&beta);
        let residuals: Vec<f64> = y
            .iter()
            .zip(eta_new.iter())
            .map(|(&yi, &etai)| yi - etai)
            .collect();

        // σ² = Σ w_i r_i² / Σ w_i   (simplified; ignores EDF correction for stability)
        let w_new: Vec<f64> = residuals
            .iter()
            .map(|&r| {
                let t2 = r * r / sigma2.max(1e-300);
                ((df + 1.0) / (df + t2)).max(1e-10)
            })
            .collect();
        let sum_wr2: f64 = w_new.iter().zip(residuals.iter()).map(|(&wi, &ri)| wi * ri * ri).sum();
        let sum_w: f64 = w_new.iter().sum::<f64>();
        let denom = (sum_w - p as f64).max(1.0);
        sigma2 = (sum_wr2 / denom).max(1e-6);

        // ── Update df via 1D Brent (skip if user fixed df) ───────────────
        if fixed_df.is_none() && outer_iter % 2 == 0 {
            // Profile df on every other outer iteration for efficiency
            df = profile_df(&residuals, sigma2);
        }

        // ── Convergence check ─────────────────────────────────────────────
        let max_change = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(b, b_old)| (b - b_old).abs())
            .fold(0.0f64, f64::max);

        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    // Final quantities
    let eta: Array1<f64> = x.dot(&beta);
    let fitted_values = eta.clone();

    let residuals_final: Vec<f64> = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    let weights: Array1<f64> = residuals_final
        .iter()
        .map(|&r| {
            let t2 = r * r / sigma2.max(1e-300);
            ((df + 1.0) / (df + t2)).max(1e-10)
        })
        .collect();

    // Deviance as weighted RSS (used for REML score; not the t-log-likelihood)
    let deviance: f64 = residuals_final.iter().map(|&r| r * r).sum();

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        deviance,
        iterations: iter,
        converged,
    })
}

/// Fit PiRLS using a discretized design for O(n*k + m*k^2) X'WX computation.
///
/// This replaces the naive O(n*p^2) X'WX with scatter-gather via compressed
/// basis storage. For large n (>= 500), this is 2-5x faster for the X'WX step.
///
/// The full design matrix `x` is still needed for computing the linear predictor
/// eta = X*beta when the discretized path has binning error. For exact (no-binning)
/// cases, eta is computed via the compressed storage.
///
/// # Arguments
/// * `disc` - Discretized design with compressed per-term basis matrices
/// * `cached_xtx` - Optional cached X'X (for Gaussian family)
pub fn fit_pirls_discretized(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    max_iter: usize,
    tolerance: f64,
    disc: &DiscretizedDesign,
    cached_xtx: Option<&Array2<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = x.ncols();

    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string(),
        ));
    }

    // Fast path for Gaussian family
    if matches!(family, Family::Gaussian) {
        return fit_pirls_gaussian_discretized(y, x, lambda, penalties, p, disc, cached_xtx);
    }

    // General IRLS path for non-Gaussian families with discretized X'WX
    let mut beta = Array1::zeros(p);
    let mut eta = disc.compute_eta(&beta);

    // Initialize eta based on family
    for i in 0..n {
        let safe_y = match family {
            Family::Binomial | Family::QuasiBinomial => y[i].max(0.01).min(0.99),
            Family::Poisson | Family::QuasiPoisson | Family::Gamma | Family::GammaLog
            | Family::Tweedie { .. } | Family::InverseGaussian | Family::NegBin { .. } => y[i].max(0.1),
            // TDist and Gaussian use identity link; initialize to y directly
            Family::Gaussian | Family::TDist { .. } => y[i],
        };
        eta[i] = family.link(safe_y);
    }

    let mut converged = false;
    let mut iter = 0;

    // Pre-compute penalty total (doesn't change between iterations)
    let mut penalty_total = Array2::<f64>::zeros((p, p));
    for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
        penalty_j.scaled_add_to(&mut penalty_total, *lambda_j);
    }
    let num_penalties = lambda.len();
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + (num_penalties as f64).sqrt())
    };

    for iteration in 0..max_iter {
        iter = iteration + 1;

        let mu: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();

        let mut z = Array1::zeros(n);
        let mut w = Array1::zeros(n);
        for i in 0..n {
            let dmu_deta = family.d_inverse_link(eta[i]);
            let variance = family.variance(mu[i]);
            if dmu_deta.abs() < 1e-10 {
                z[i] = eta[i];
            } else {
                z[i] = eta[i] + (y[i] - mu[i]) / dmu_deta;
            }
            w[i] = ((dmu_deta * dmu_deta) / variance.max(1e-10)).max(1e-10);
        }

        // X'WX via scatter-gather: O(n*k + m*k^2) instead of O(n*k^2)
        let xtwx = disc.compute_xtwx(&w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge: f64 = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // X'Wz via scatter-gather
        let wz: Array1<f64> = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi).collect();
        let xtwz = disc.compute_xtwz(&wz);

        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // eta via compressed gather
        eta = disc.compute_eta(&beta);

        let max_change = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(b, b_old)| (b - b_old).abs())
            .fold(0.0f64, f64::max);

        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    let fitted_values: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();
    let deviance = compute_deviance(y, &fitted_values, family);

    let weights: Array1<f64> = eta
        .iter()
        .map(|&e| {
            let mu = family.inverse_link(e);
            let dmu_deta = family.d_inverse_link(e);
            let variance = family.variance(mu);
            ((dmu_deta * dmu_deta) / variance.max(1e-10)).max(1e-10)
        })
        .collect();

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        deviance,
        iterations: iter,
        converged,
    })
}

/// Gaussian fast path using discretized design.
///
/// Combines the Gaussian 1-step solve with scatter-gather X'X computation.
fn fit_pirls_gaussian_discretized(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    p: usize,
    disc: &DiscretizedDesign,
    cached_xtx: Option<&Array2<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();

    // X'X: use cached version or compute via scatter-gather
    let xtx = if let Some(cached) = cached_xtx {
        cached.clone()
    } else {
        disc.compute_xtx()
    };

    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(xtx[[i, i]].abs());
    }

    let mut a = xtx;
    for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
        penalty_j.scaled_add_to(&mut a, *lambda_j);
    }

    let num_penalties = lambda.len();
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + (num_penalties as f64).sqrt())
    };
    let ridge: f64 = ridge_scale * max_diag;
    for i in 0..p {
        a[[i, i]] += ridge;
    }

    // X'y via scatter-gather
    let ones = Array1::ones(n);
    let xty = disc.compute_xtwy(&ones, y);

    let beta = solve(a, xty)?;

    // eta via compressed gather
    let eta = disc.compute_eta(&beta);
    let fitted_values = eta.clone();

    let deviance: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(yi, fi)| (yi - fi).powi(2))
        .sum();

    let weights = Array1::ones(n);

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        deviance,
        iterations: 1,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_family_functions() {
        let family = Family::Gaussian;
        assert!((family.variance(1.0) - 1.0).abs() < 1e-10);
        assert!((family.link(5.0) - 5.0).abs() < 1e-10);
        assert!((family.inverse_link(3.0) - 3.0).abs() < 1e-10);

        let family = Family::Poisson;
        assert!((family.variance(2.0) - 2.0).abs() < 1e-10);
        assert!((family.inverse_link(0.0) - 1.0).abs() < 1e-10);
    }

    /// Central-difference helper.
    fn cd<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() < tol * scale,
            "{}: analytic={} numeric={} diff={} (relative tol {})",
            msg,
            a,
            b,
            (a - b).abs(),
            tol
        );
    }

    #[test]
    fn test_dvar_against_finite_difference() {
        let h = 1e-6;
        let mus = [0.2, 0.5, 0.8, 1.5, 3.0];
        for &mu in &mus {
            for &fam in &[
                Family::Gaussian,
                Family::Binomial,
                Family::Poisson,
                Family::Gamma,
                Family::GammaLog,
            ] {
                if matches!(fam, Family::Binomial) && (mu <= 0.0 || mu >= 1.0) {
                    continue;
                }
                let analytic = fam.dvar(mu);
                let numeric = cd(|m| fam.variance(m), mu, h);
                assert_close(analytic, numeric, 1e-6, &format!("dvar {:?} mu={}", fam, mu));
            }
        }
    }

    #[test]
    fn test_d2var_against_finite_difference() {
        let h = 1e-4;
        let mus = [0.2, 0.5, 0.8, 1.5, 3.0];
        for &mu in &mus {
            for &fam in &[
                Family::Gaussian,
                Family::Binomial,
                Family::Poisson,
                Family::Gamma,
                Family::GammaLog,
            ] {
                if matches!(fam, Family::Binomial) && (mu <= 0.0 || mu >= 1.0) {
                    continue;
                }
                let analytic = fam.d2var(mu);
                let numeric = cd(|m| fam.dvar(m), mu, h);
                assert_close(analytic, numeric, 1e-6, &format!("d2var {:?} mu={}", fam, mu));
            }
        }
    }

    #[test]
    fn test_d2link_against_finite_difference() {
        // Second-difference of the link function. h chosen as ~ε^(1/4) ≈ 1e-4
        // for a 4-th-order accurate estimate; tolerance loose enough to absorb
        // the truncation/roundoff trade-off (Gamma's 2/μ³ at μ=0.2 is large).
        let h = 1e-4;
        let mus = [0.3, 0.5, 0.7, 1.5, 3.0];
        for &mu in &mus {
            for &fam in &[
                Family::Gaussian,
                Family::Binomial,
                Family::Poisson,
                Family::Gamma,
                Family::GammaLog,
            ] {
                if matches!(fam, Family::Binomial) && (mu <= 0.0 || mu >= 1.0) {
                    continue;
                }
                let analytic = fam.d2link(mu);
                let numeric =
                    (fam.link(mu + h) - 2.0 * fam.link(mu) + fam.link(mu - h)) / (h * h);
                assert_close(analytic, numeric, 1e-3, &format!("d2link {:?} mu={}", fam, mu));
            }
        }
    }

    #[test]
    fn test_d3link_against_finite_difference() {
        let h = 1e-3;
        let mus = [0.3, 0.5, 0.7, 1.5, 3.0];
        for &mu in &mus {
            for &fam in &[
                Family::Gaussian,
                Family::Binomial,
                Family::Poisson,
                Family::Gamma,
                Family::GammaLog,
            ] {
                if matches!(fam, Family::Binomial) && (mu <= 0.0 || mu >= 1.0) {
                    continue;
                }
                let analytic = fam.d3link(mu);
                let numeric = cd(|m| fam.d2link(m), mu, h);
                assert_close(analytic, numeric, 1e-3, &format!("d3link {:?} mu={}", fam, mu));
            }
        }
    }

    #[test]
    fn test_canonical_link_flags() {
        assert!(Family::Gaussian.is_canonical_link());
        assert!(Family::Binomial.is_canonical_link());
        assert!(Family::Poisson.is_canonical_link());
        assert!(Family::Gamma.is_canonical_link());
        assert!(!Family::GammaLog.is_canonical_link());
    }

    #[test]
    fn test_pirls_gaussian() {
        use crate::block_penalty::BlockPenalty;

        let n = 20;
        let p = 5;

        let x = Array2::from_shape_fn((n, p), |(i, j)| ((i as f64) * 0.1).powi(j as i32));

        let y: Array1<f64> = (0..n)
            .map(|i| {
                let xi = i as f64 * 0.1;
                xi + xi.powi(2) + 0.1 * (i as f64).sin()
            })
            .collect();

        let penalty = Array2::eye(p);
        let lambda = vec![0.01];
        let penalties = vec![BlockPenalty::new(penalty, 0, p)];

        let result = fit_pirls(&y, &x, &lambda, &penalties, Family::Gaussian, 100, 1e-6);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.converged);
        assert_eq!(result.coefficients.len(), p);
    }

    /// trigamma(x) for small x values from Abramowitz & Stegun tables.
    ///   trigamma(1) = π²/6 ≈ 1.6449340668
    ///   trigamma(2) = π²/6 - 1 ≈ 0.6449340668
    #[test]
    fn test_trigamma_known_values() {
        let pi2_over6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let t1 = trigamma(1.0);
        assert!(
            (t1 - pi2_over6).abs() < 1e-9,
            "trigamma(1) = {}, expected {}",
            t1,
            pi2_over6
        );
        let t2 = trigamma(2.0);
        let expected_2 = pi2_over6 - 1.0;
        assert!(
            (t2 - expected_2).abs() < 1e-9,
            "trigamma(2) = {}, expected {}",
            t2,
            expected_2
        );
    }

    /// estimate_phi_mgcv: Gaussian closed form must equal dp/(n-mp).
    #[test]
    fn test_estimate_phi_mgcv_gaussian() {
        let n = 50usize;
        let y = Array1::<f64>::ones(n);
        let dp = 120.0_f64;
        let mp = 4usize;
        let expected = dp / (n as f64 - mp as f64);
        let phi = Family::Gaussian.estimate_phi_mgcv(&y, dp, mp, 1.0, 1.0);
        assert!(
            (phi - expected).abs() < 1e-12,
            "Gaussian phi = {}, expected {}",
            phi,
            expected
        );
    }

    /// estimate_phi_mgcv: Poisson and Binomial always return 1.0.
    #[test]
    fn test_estimate_phi_mgcv_fixed_dispersion() {
        let y = Array1::<f64>::ones(10);
        assert_eq!(Family::Poisson.estimate_phi_mgcv(&y, 50.0, 3, 1.0, 2.0), 1.0);
        assert_eq!(Family::Binomial.estimate_phi_mgcv(&y, 50.0, 3, 1.0, 2.0), 1.0);
    }

    /// estimate_phi_mgcv Gamma: verify F(phi) ≈ 0 at the solved phi.
    /// Uses dp=100, n=200, mp=5, gamma=1.0.
    #[test]
    fn test_estimate_phi_mgcv_gamma_residual() {
        let n = 200usize;
        let y = Array1::<f64>::ones(n);
        let dp = 100.0_f64;
        let mp = 5usize;
        let phi_init = dp / (n as f64 - mp as f64);
        let phi = Family::Gamma.estimate_phi_mgcv(&y, dp, mp, 1.0, phi_init);

        // Verify F(phi) = dp + 2n[psi(1/phi) + ln phi] + mp*phi ≈ 0
        let n_f = n as f64;
        let mp_f = mp as f64;
        let f_at_phi = dp + 2.0 * n_f * (digamma(1.0 / phi) + phi.ln()) + mp_f * phi;
        let tol = 1e-8 * (dp.abs() + mp_f + 1.0);
        println!("Gamma phi = {:.10}, F(phi) = {:.3e}, tol = {:.3e}", phi, f_at_phi, tol);
        assert!(
            f_at_phi.abs() < tol,
            "F(phi) = {:.3e} is not near zero (tol {:.3e}); phi = {}",
            f_at_phi,
            tol,
            phi
        );
    }

    /// GammaLog should give same result as Gamma (same family equation).
    #[test]
    fn test_estimate_phi_mgcv_gammalog_same_as_gamma() {
        let n = 200usize;
        let y = Array1::<f64>::ones(n);
        let dp = 100.0_f64;
        let mp = 5usize;
        let phi_init = dp / (n as f64 - mp as f64);
        let phi_gamma = Family::Gamma.estimate_phi_mgcv(&y, dp, mp, 1.0, phi_init);
        let phi_gammalog = Family::GammaLog.estimate_phi_mgcv(&y, dp, mp, 1.0, phi_init);
        assert!(
            (phi_gamma - phi_gammalog).abs() < 1e-12,
            "Gamma phi = {}, GammaLog phi = {}",
            phi_gamma,
            phi_gammalog
        );
    }
}
