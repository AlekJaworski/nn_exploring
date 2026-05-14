//! PiRLS (Penalized Iteratively Reweighted Least Squares) algorithm for GAM fitting

use crate::block_penalty::BlockPenalty;
use crate::discrete::{
    compute_eta_discrete, compute_xtwx_discrete, compute_xtwy_discrete, DiscreteDesign,
};
use crate::linalg::solve;
use crate::reml::compute_xtwx_dispatch;
use crate::{GAMError, Result};
use ndarray::{Array1, Array2, ArrayView1};

/// REML / LAML score formula. Different families use structurally
/// different criteria (see `Family::score_formula`).
///
/// `assemble` combines the per-fit ingredients (`dp = D + β'Sβ`, saturated
/// log-likelihood `ls`, log-determinants `log|H|` and `log|S|+`, dispersion
/// `σ²`, model rank `Mp`) into a scalar score. The optimizer minimises
/// whichever variant `Family::score_formula` returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreFormula {
    /// gam.fit3 form (gam.fit3.r:616-617):
    ///     REML = Dp/(2σ²) - ls - Mp/2·log(2πσ²) + log|H|/2 - log|S|+/2
    ///
    /// For Gaussian/Gamma/Tweedie/Poisson/Binomial/etc. — the deviance is
    /// in raw response units (Σr² for Gaussian) and the dispersion φ=σ²
    /// scales the Dp/(2φ) term explicitly. The `-Mp/2·log(2πφ)` term is
    /// the σ²-profile correction.
    GamFit3,
    /// gam.fit5 form (gam.fit5.r):
    ///     LAML = Dp/2 - ls + log|H|/2 - log|S|+/2
    ///
    /// For extended families (TDist/scat, Quantile/ELF) where σ² is a
    /// *family-internal* parameter that already appears inside the
    /// deviance and saturated log-likelihood. Dp is in log-likelihood
    /// units (so Dp/2 already has the right scaling — no extra /σ² or
    /// `-Mp/2·log(2πσ²)` penalty term).
    GamFit5,
}

impl ScoreFormula {
    /// Combine the score ingredients into a scalar. The optimizer minimises
    /// the result.
    pub fn assemble(
        self,
        dp: f64,
        ls: f64,
        log_det_h: f64,
        log_det_s: f64,
        sigma2: f64,
        mp: usize,
    ) -> f64 {
        match self {
            ScoreFormula::GamFit3 => {
                let two_pi_phi = 2.0 * std::f64::consts::PI * sigma2;
                dp / (2.0 * sigma2) - ls - 0.5 * (mp as f64) * two_pi_phi.ln() + 0.5 * log_det_h
                    - 0.5 * log_det_s
            }
            ScoreFormula::GamFit5 => {
                0.5 * dp - ls + 0.5 * log_det_h
                    - 0.5 * log_det_s
                    - 0.5 * (mp as f64) * (2.0 * std::f64::consts::PI).ln()
            }
        }
    }
}

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
    TDist {
        df: f64,
        sigma2: f64,
    },
    /// Tweedie distribution with log link (1 < p < 2).
    ///
    /// Log link: η = log(μ), μ = exp(η). Variance function V(μ) = μ^p.
    /// The dispersion φ is profiled (not fixed at 1), similar to Gamma.
    ///
    /// The saturated log-likelihood is computed via the Dunn-Smyth (2005)
    /// series summation for 1 < p < 2 — a port of mgcv's `tweedious` C
    /// function (misc.c:170). For y=0 the density simplifies; for y>0 the
    /// series `W = Σ_j W_j` is summed using log-sum-exp to avoid overflow.
    Tweedie {
        p: f64,
    },
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
    NegBin {
        theta: f64,
    },
    /// Quantile (qgam-style) family using a smooth pinball loss.
    ///
    /// Implements the ELF (Extended Log-F) loss from Fasiolo et al. 2021 —
    /// the same calibrated smoothing of the pinball loss that R's qgam
    /// package uses on top of mgcv's basis/penalty machinery.
    ///
    /// For residual r = y - η (identity link), the negative log-likelihood is
    ///   L(r; τ, σ) = (η-y)(1-τ)/σ + log(1 + exp((y-η)/σ))
    /// which approaches the pinball loss `ρ_τ(r) = max(τ·r, (τ-1)·r)` as σ→0.
    /// The minimizer of E[L] is the τ-quantile of y|x rather than the mean.
    ///
    /// PIRLS weights/working-response are derived analytically (see
    /// `fit_pirls_quantile`):
    ///   s_i = 1/(1 + exp(-(y_i - η_i)/σ))
    ///   w_i = s_i(1-s_i)/σ²    (always positive, well-behaved Hessian)
    ///   z_i = η_i - σ(1 - τ - s_i)/(s_i(1-s_i))
    ///
    /// `tau` ∈ (0, 1) selects the target quantile; `sigma` controls the
    /// pinball-loss smoothing (smaller σ → sharper quantile, larger σ →
    /// smoother loss). v1 takes σ as user-provided or a heuristic default;
    /// full qgam-style σ calibration is a deferred followup.
    Quantile {
        tau: f64,
        sigma: f64,
    },
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
            // Quantile/ELF: weights are residual-dependent, not μ-dependent.
            // Return σ²·4 (max V = σ²/(s(1-s)) ≥ 4σ² since s(1-s) ≤ 1/4) so
            // that any code path defaulting to V(μ) gets a finite sentinel;
            // the actual per-obs weights are computed in `fit_pirls_quantile`.
            Family::Quantile { sigma, .. } => 4.0 * sigma * sigma,
        }
    }

    /// Link function g(μ).
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => mu,
            Family::Binomial | Family::QuasiBinomial => (mu / (1.0 - mu)).ln(),
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => mu.ln(),
            Family::Gamma => 1.0 / mu,
        }
    }

    /// Inverse link function g^(-1)(η).
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => eta,
            Family::Binomial | Family::QuasiBinomial => {
                let eta_safe = eta.max(-20.0).min(20.0);
                1.0 / (1.0 + (-eta_safe).exp())
            }
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => {
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
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 1.0,
            Family::Binomial | Family::QuasiBinomial => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            }
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => eta.exp(),
            Family::Gamma => -1.0 / (eta * eta),
        }
    }

    /// First derivative of variance function: dV/dμ.
    /// Used by mgcv's full-Newton PIRLS (`gam.fit3.r:507`) for non-canonical
    /// links to compute the α correction `α = 1 + (y−μ)·(V'/V + g''·dμ/dη)`.
    pub fn dvar(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
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
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
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
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => {
                let one_minus = 1.0 - mu;
                -1.0 / (mu * mu) + 1.0 / (one_minus * one_minus)
            }
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => -1.0 / (mu * mu),
            Family::Gamma => 2.0 / (mu * mu * mu),
        }
    }

    /// Third derivative of link function: d³g/dμ³.
    pub fn d3link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => {
                let one_minus = 1.0 - mu;
                2.0 / (mu * mu * mu) + 2.0 / (one_minus * one_minus * one_minus)
            }
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => 2.0 / (mu * mu * mu),
            Family::Gamma => -6.0 / (mu * mu * mu * mu),
        }
    }

    /// Third derivative of variance function: d³V/dμ³.
    /// Used by mgcv's α₂ derivative (`gdi.c:2546`) needed for the analytical
    /// Tk·KK' Hessian contribution (full Newton path).
    pub fn d3var(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => 0.0,
            Family::Poisson | Family::QuasiPoisson => 0.0,
            Family::Gamma | Family::GammaLog => 0.0,
            Family::Tweedie { p } => p * (p - 1.0) * (p - 2.0) * mu.powf(p - 3.0),
            // d³V/dμ³ = 6 for V(μ) = μ³
            Family::InverseGaussian => 6.0,
            Family::NegBin { .. } => 0.0,
        }
    }

    /// Fourth derivative of link function: d⁴g/dμ⁴.
    /// Used by mgcv's α₂ derivative (`gdi.c:2546`) — the analytical Tk·KK'
    /// Hessian contribution under the full Newton path.
    pub fn d4link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => 0.0,
            Family::Binomial | Family::QuasiBinomial => {
                // d3link = 2/μ³ + 2/(1-μ)³
                // d4link = -6/μ⁴ + d/dμ[2(1-μ)⁻³] = -6/μ⁴ + 6/(1-μ)⁴
                let one_minus = 1.0 - mu;
                let m2 = mu * mu;
                let om2 = one_minus * one_minus;
                -6.0 / (m2 * m2) + 6.0 / (om2 * om2)
            }
            Family::Poisson
            | Family::QuasiPoisson
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => {
                // d3link = 2/μ³ → d4link = -6/μ⁴
                let m2 = mu * mu;
                -6.0 / (m2 * m2)
            }
            Family::Gamma => {
                // d3link = -6/μ⁴ → d4link = 24/μ⁵
                let m2 = mu * mu;
                24.0 / (m2 * m2 * mu)
            }
        }
    }

    /// Which REML/LAML score formula this family uses.
    ///
    /// Two structurally different formulas live in mgcv: the gam.fit3 REML
    /// (Gaussian-style scaled by an external dispersion φ) and the gam.fit5
    /// LAML (extended families where σ² is a *family-internal* parameter
    /// that already appears inside the deviance/saturated-ls).
    ///
    /// Mapping:
    ///   - **GamFit3** — exponential families with profiled / fixed scale:
    ///     Gaussian, Gamma, GammaLog, Tweedie, Poisson, Binomial, NegBin,
    ///     QuasiPoisson, QuasiBinomial, InverseGaussian.
    ///   - **GamFit5** — extended families with internal scale: TDist
    ///     (mgcv `scat`), Quantile (qgam ELF). Their σ² is profiled inside
    ///     the family methods, not as the GLM dispersion φ.
    pub fn score_formula(&self) -> ScoreFormula {
        match self {
            Family::TDist { .. } | Family::Quantile { .. } => ScoreFormula::GamFit5,
            _ => ScoreFormula::GamFit3,
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
            | Family::TDist { .. }
            // Quantile family: identity link, weights derived analytically
            // from the Hessian — no extra Newton correction needed.
            | Family::Quantile { .. } => true,
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
            // Scaled-t saturated log-likelihood (mgcv scat$ls):
            //   ls = n · [lgamma((ν+1)/2) − lgamma(ν/2) − 0.5·log(π·ν·σ²)]
            Family::TDist { df, .. } => {
                let nu = *df;
                let half_nu_p1 = (nu + 1.0) / 2.0;
                let half_nu = nu / 2.0;
                n * (log_gamma(half_nu_p1)
                    - log_gamma(half_nu)
                    - 0.5 * (std::f64::consts::PI * nu * scale).ln())
            }
            // Quantile/ELF saturated log-likelihood — port of qgam elf.R:204-216
            // for the special case λ = σ (single-bandwidth parameterisation).
            //
            // qgam's general formula:
            //   ls_per_obs = (1-τ)·λ·log(1-τ)/σ + λ·τ·log(τ)/σ
            //                - log(λ) - log B(λ(1-τ)/σ, λτ/σ)
            // With λ = σ this collapses to:
            //   ls_per_obs = (1-τ)·log(1-τ) + τ·log(τ) - log σ - log B(1-τ, τ)
            //              = -H(τ) - log σ - log B(τ, 1-τ)
            // where H(τ) is the Bernoulli entropy. This differs from
            // -log 2 - log σ - log B by an entropy-vs-log-2 constant; for
            // τ=0.5 they coincide (H(0.5)=log 2) but for asymmetric τ the
            // entropy version is what mgcv's LAML wants.
            Family::Quantile { tau, sigma } => {
                let h_tau = -((1.0 - tau) * (1.0 - tau).ln() + tau * tau.ln());
                let log_beta = log_gamma(*tau) + log_gamma(1.0 - tau) - log_gamma(1.0);
                n * (-h_tau - sigma.ln() - log_beta)
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
                        let lgam_term =
                            log_gamma(yi_p_theta) - log_gamma(theta) - log_gamma(yi + 1.0);
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
                // Near p=2: j_max = y^(2-p)/(φ(2-p)) → ∞ making the Wright series
                // intractable. Use the Gamma-family limit (Tweedie → Gamma as p→2):
                //   ls_Gamma = n·[-lgamma(1/φ) - log(φ)/φ - 1/φ] - Σ log(y_i)
                if *p > 1.95 {
                    let n_f = y.len() as f64;
                    let inv_phi = 1.0 / phi;
                    let ls_per_obs = -log_gamma(inv_phi) - inv_phi * phi.ln() - inv_phi;
                    let sum_log_y: f64 = y.iter().filter(|&&yi| yi > 0.0).map(|&yi| yi.ln()).sum();
                    return n_f * ls_per_obs - sum_log_y;
                }
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
            // Quantile: ls is independent of φ (σ is the family parameter,
            // φ stays at 1 by convention). dls/dφ = 0 ⟹ no σ²-chain term.
            Family::Quantile { .. } => 0.0,
            // Tweedie dls/dφ: from ldTweedie0 R code (gam.fit3.r:2799):
            //   ld[,2] = -l_base/φ + dlogW/dφ
            // where l_base is the analytic density term at μ=y, and
            // dlogW/dφ = (dlogW/drho) * (1/φ)  [since rho = log φ].
            // We get dlogW/drho from tweedie_series as the second output.
            Family::Tweedie { p } => {
                let phi = scale;
                // Near p=2: use Gamma-limit derivative (exact at p=2).
                if *p > 1.95 {
                    let n = y.len() as f64;
                    let inv_phi = 1.0 / phi;
                    return n * (digamma(inv_phi) + phi.ln()) / (phi * phi);
                }
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
    result -= t / 12.0; // B2/2 = (1/6)/2
    t *= xinv2;
    result += t / 120.0; // B4/4 = (-1/30)/4  → +1/120
    t *= xinv2;
    result -= t / 252.0; // B6/6 = (1/42)/6   → -1/252
    t *= xinv2;
    result += t / 240.0; // B8/8 = (-1/30)/8  → +1/240
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
    /// For TDist, returns the enum-stored σ² (synced from `fit_pirls_tdist`'s
    /// converged value via `PirlsRefresh.sigma2` → smooth.rs outer Newton).
    /// This couples the REML score's dispersion to the same σ² that the
    /// inner PIRLS converged to, instead of recomputing a separate
    /// Pearson chi-squared `dp/(n-mp)`. Required for LAML self-consistency.
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
            Family::TDist { sigma2, .. } => (*sigma2).max(1e-8),
            // Quantile: σ is the family parameter; φ stays at 1 by convention.
            Family::Quantile { .. } => 1.0,
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
                // Near p=2: Wright series requires j_max = y^(2-p)/(φ(2-p)) → ∞.
                // Use the Gaussian closed-form φ̂ = Dp/(n-Mp) as a fast approximation
                // (exact at p=0 and p=2 where ls reduces to Gaussian/Gamma form).
                if p > 1.95 {
                    return dp / (n - mp_f).max(1.0);
                }
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
                    let fp = dp / (phi * phi * phi) - (dls_plus - dls_minus) / (2.0 * h)
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
pub(crate) fn log_gamma(x: f64) -> f64 {
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
    let log_eps_ln = log_eps.ln();

    // Precompute the j-only pieces of w_j and wp1_j. These depend on (p, phi)
    // but not on the observation, so the per-(i,j) special-fn calls
    // (`log_gamma`, `digamma`) collapse to one call per j shared across all
    // observations. For n=400 obs each touching ~20 j values this turns
    // ~8000 special-fn calls into ~30. Bound the cache to the maximum j
    // any observation will need (mode = y^(2-p)/(phi·(2-p))) plus a buffer
    // for the upsweep convergence tail.
    let mut j_max_obs: i64 = 1;
    for &yi in y.iter() {
        if yi > 0.0 {
            let xj = yi.powf(twop) / (phi * twop);
            let j_max = (xj.floor() as i64).max(1);
            if j_max > j_max_obs {
                j_max_obs = j_max;
            }
        }
    }
    // Buffer of 64 covers the upsweep tail comfortably; falls back to direct
    // computation if we ever exceed it (safety, not a hot path).
    let j_cache_size = (j_max_obs + 64).max(64) as usize;
    // wj0[j] = j*w_base - log_gamma(j+1) - log_gamma(-j*alpha)  (no obs dep)
    // wp1_const[j] = j*wp_base + (j/onep2) * digamma(-j*alpha)  (no obs dep)
    let mut wj0 = vec![0.0_f64; j_cache_size + 1];
    let mut wp1_const = vec![0.0_f64; j_cache_size + 1];
    for j in 1..=j_cache_size {
        let jf = j as f64;
        let neg_j_alpha = -jf * alpha;
        wj0[j] = jf * w_base - log_gamma(jf + 1.0) - log_gamma(neg_j_alpha);
        wp1_const[j] = jf * wp_base + (jf / onep2) * digamma(neg_j_alpha);
    }

    // Closures that fall back to direct evaluation past the cache (only
    // reached if the upsweep extends more than 64 past the global j_max
    // — unlikely for typical Tweedie data).
    let wj0_at = |j: usize| -> f64 {
        if j <= j_cache_size {
            wj0[j]
        } else {
            let jf = j as f64;
            jf * w_base - log_gamma(jf + 1.0) - log_gamma(-jf * alpha)
        }
    };
    let wp1_const_at = |j: usize| -> f64 {
        if j <= j_cache_size {
            wp1_const[j]
        } else {
            let jf = j as f64;
            jf * wp_base + (jf / onep2) * digamma(-jf * alpha)
        }
    };

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
        let wmax_j = j_max as usize;
        let wmax_val = wj0_at(wmax_j) - (j_max as f64) * alogy_i;

        let wmin = wmax_val + log_eps_ln;

        // Accumulate sums (scaled by exp(-wmax)):
        // wi = Σ exp(wj - wmax) = W / exp(wmax)
        // w1i = Σ exp(wj - wmax) * wb1_j  (for dlogW/drho numerator)
        // wdlogwdp = Σ exp(wj - wmax) * wp1_j  (for dlogW/dp numerator)
        let mut wi = 0.0f64;
        let mut w1i = 0.0f64;
        let mut wdlogwdp = 0.0f64;

        // Upsweep from j_max upward until wj < wmin
        let mut j = wmax_j;
        loop {
            let jf = j as f64;
            let wj = wj0_at(j) - jf * alogy_i;
            let wj_scaled = (wj - wmax_val).exp();
            wi += wj_scaled;
            w1i += wj_scaled * jf * wb1_base; // wb1_j = j * (-1/onep)
            let wp1 = wp1_const_at(j) - jf * logy1p2_i;
            wdlogwdp += wj_scaled * wp1;
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
        if wmax_j >= 1 {
            let mut j_ds: i64 = wmax_j as i64 - 1;
            while j_ds >= 1 {
                let jf = j_ds as f64;
                let ju = j_ds as usize;
                let wj = wj0_at(ju) - jf * alogy_i;
                let wj_scaled = (wj - wmax_val).exp();
                wi += wj_scaled;
                w1i += wj_scaled * jf * wb1_base;
                let wp1 = wp1_const_at(ju) - jf * logy1p2_i;
                wdlogwdp += wj_scaled * wp1;
                if wj < wmin {
                    break;
                }
                j_ds -= 1;
            }
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
    pub working_response: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Converged dispersion / family-scale parameter, when the inner
    /// fitter computed one. For `TDist` this is the MLE σ²; for
    /// other fitters this is `None` and the outer loop falls back to
    /// `estimate_phi_mgcv`.
    pub sigma2: Option<f64>,
    /// Converged df for `TDist` (output of inner Brent profile, or the
    /// caller-supplied fixed_df). Used by the outer loop to sync
    /// `Family::TDist::df` after PIRLS so subsequent score evaluations
    /// see the correct df. `None` for non-TDist fitters.
    pub df: Option<f64>,
}

/// Newton IRLS curvature factor `α = 1 + c_resid · (V₁ⁿ + g₂ⁿ)`.
///
/// `c_resid = y - μ`, `v1n = V'(μ)/V(μ)`, `g2n = g''(μ) · g'(μ)⁻¹ = d2link·dmu_deta`.
/// Callers handle the `α ≤ 0` case (pirls falls back to Fisher per-obs; reml's
/// `a1` computation clamps `α` to 1 to stay on the Newton form). Centralised so
/// the InvGauss + log alpha-clamp fix applies wherever the curvature term is
/// used.
#[inline]
pub(crate) fn newton_irls_alpha(c_resid: f64, v1n: f64, g2n: f64) -> f64 {
    1.0 + c_resid * (v1n + g2n)
}

/// Per-observation IRLS working weight `w` and working response `z`.
///
/// Encapsulates the standard GLM IRLS step for one observation: compute μ,
/// dμ/dη, V(μ), Fisher weight `wf = (dμ/dη)² / V(μ)`, optionally apply the
/// Newton curvature correction `α`, and fall back to Fisher when `α ≤ 0`
/// (e.g. InvGauss + log has α<0 on ~43% of obs for `α = 2y/μ − 1`).
///
/// `use_fisher` should be `true` whenever the caller wants pure Fisher
/// scoring (canonical-link families always do; non-canonical links use
/// Newton in the inner loop but Fisher for the final reporting pass).
#[inline]
pub(crate) fn compute_irls_wz(
    eta_i: f64,
    y_i: f64,
    family: Family,
    use_fisher: bool,
) -> (f64, f64) {
    let mu = family.inverse_link(eta_i);
    let dmu_deta = family.d_inverse_link(eta_i);
    let variance = family.variance(mu);
    let var_safe = variance.max(1e-10);
    if dmu_deta.abs() < 1e-10 {
        return (1e-10, eta_i);
    }

    let wf = (dmu_deta * dmu_deta) / var_safe;
    if use_fisher {
        let z = eta_i + (y_i - mu) / dmu_deta;
        let w = wf.max(1e-10);
        return (w, z);
    }

    let c_resid = y_i - mu;
    let v1n = family.dvar(mu) / var_safe;
    let g2n = family.d2link(mu) * dmu_deta;
    let alpha = newton_irls_alpha(c_resid, v1n, g2n);
    // Cholesky needs PSD weights; when Newton curvature goes negative,
    // fall back to Fisher scoring for that observation.
    if alpha <= 0.0 {
        let z = eta_i + c_resid / dmu_deta;
        let w = wf.max(1e-10);
        return (w, z);
    }
    let z = eta_i + c_resid / (dmu_deta * alpha);
    let w = wf * alpha;
    (w, z)
}

/// Per-observation Newton weight for REML score evaluation. Unlike
/// `compute_irls_wz` (which falls back to Fisher when `α ≤ 0` to keep the
/// inner-loop Cholesky PSD), this returns the raw Newton weight `wf · α`
/// regardless of sign. mgcv stores the analogous quantity as
/// `g$working.weights` (negative entries occur for log-link InvGauss in
/// ~43% of obs) and uses it for `log|H| = log|X'WX + S|` in the REML
/// formula. Pairs with `log_abs_det_symmetric` in `linalg.rs` for the
/// possibly-indefinite log-det.
#[inline]
pub(crate) fn compute_newton_score_weight(eta_i: f64, y_i: f64, family: Family) -> f64 {
    let mu = family.inverse_link(eta_i);
    let dmu_deta = family.d_inverse_link(eta_i);
    let variance = family.variance(mu);
    let var_safe = variance.max(1e-10);
    if dmu_deta.abs() < 1e-10 {
        return 0.0;
    }
    let wf = (dmu_deta * dmu_deta) / var_safe;
    if family.is_canonical_link() {
        return wf;
    }
    let c_resid = y_i - mu;
    let v1n = family.dvar(mu) / var_safe;
    let g2n = family.d2link(mu) * dmu_deta;
    let alpha = newton_irls_alpha(c_resid, v1n, g2n);
    wf * alpha
}

/// Vectorised `compute_newton_score_weight`. Returns the `g$working.weights`
/// analog at converged η: full Newton weights with no Fisher fallback for
/// negative α. Used by REML score evaluation to match mgcv's gam.fit3
/// `log|X'WX+S|` exactly under non-canonical links.
pub fn compute_newton_score_weights(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    family: Family,
) -> Array1<f64> {
    Array1::from_iter(
        eta.iter()
            .zip(y.iter())
            .map(|(&e, &yi)| compute_newton_score_weight(e, yi, family)),
    )
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
    fit_pirls_cached(y, x, lambda, penalties, family, max_iter, tolerance, None, None)
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
    prior_weights: Option<&Array1<f64>>,
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

    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "prior_weights length ({}) must match y length ({})",
                pw.len(),
                n
            )));
        }
    }

    // Fast path for Gaussian family: weights are constant (w=1), z=y on first iteration,
    // and PiRLS converges in exactly 1 step. Skip all the IRLS machinery.
    if matches!(family, Family::Gaussian) {
        return fit_pirls_gaussian_fast(y, x, lambda, penalties, p, cached_xtx, prior_weights);
    }

    // General IRLS path for non-Gaussian families
    // Initialize coefficients and linear predictor
    let mut beta = Array1::zeros(p);
    let mut eta = x.dot(&beta);

    // Initialize eta based on family
    for i in 0..n {
        let safe_y = match family {
            Family::Binomial | Family::QuasiBinomial => y[i].max(0.01).min(0.99),
            Family::Poisson
            | Family::QuasiPoisson
            | Family::Gamma
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => y[i].max(0.1),
            // Identity-link families: initialize η = y directly.
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => y[i],
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

        // GLM IRLS working pair with prior weights folded in. R4-refactor:
        // see `exp_family_irls_step`. `use_fisher=false` ⇒ Newton curvature
        // per row with per-row Fisher fallback when `α ≤ 0` (canonical
        // links auto-promote to pure Fisher inside `compute_irls_wz`).
        let step = exp_family_irls_step(
            y.view(),
            eta.view(),
            prior_weights.map(|pw| pw.view()),
            family,
            /* use_fisher = */ false,
        );
        let z = step.z;
        let w = step.w;

        // X'WX using BLAS (instead of manual triple-nested loop)
        let xtwx = compute_xtwx_dispatch(None, x, &w);

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

    // Compute deviance — weighted by prior_weights when supplied (mgcv
    // convention: D_w = Σ w_i · d_i where d_i is the per-obs deviance).
    let deviance = match prior_weights {
        Some(pw) => compute_weighted_deviance(y, &fitted_values, family, pw),
        None => compute_deviance(y, &fitted_values, family),
    };

    // Preserve the historical final scoring contract: outer REML callbacks
    // consume Fisher working quantities at the converged η, even if the inner
    // dense PIRLS iterations used Newton alpha correction. R4-refactor: see
    // `exp_family_irls_step`. `use_fisher=true` ⇒ pure Fisher working pair
    // for the REML scoring contract; prior weights are folded into w.
    let final_step = exp_family_irls_step(
        y.view(),
        eta.view(),
        prior_weights.map(|pw| pw.view()),
        family,
        /* use_fisher = */ true,
    );

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights: final_step.w,
        working_response: final_step.z,
        deviance,
        iterations: iter,
        converged,
        sigma2: None,
        df: None,
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
    prior_weights: Option<&Array1<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();

    // X'WX: with prior weights `w`, the weighted normal equations are
    //   (XᵀWX + λS) β = XᵀWy ,    W = diag(w).
    // Without prior weights, W = I and we can reuse the cached XᵀX.
    let xtwx = match prior_weights {
        None => {
            if let Some(cached) = cached_xtx {
                cached.clone()
            } else {
                x.t().dot(x)
            }
        }
        Some(w) => compute_xtwx_dispatch(None, x, w),
    };

    // Compute max diagonal for ridge scaling
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(xtwx[[i, i]].abs());
    }

    // Build (X'WX + Σλ_jS_j + ridge*I)
    let mut a = xtwx;
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

    // X'Wy using BLAS (or X'y when unweighted)
    let xtwy = match prior_weights {
        None => x.t().dot(y),
        Some(w) => {
            let wy: Array1<f64> = w.iter().zip(y.iter()).map(|(&wi, &yi)| wi * yi).collect();
            x.t().dot(&wy)
        }
    };

    // Solve for coefficients: β = A^{-1} X'Wy
    let beta = solve(a, xtwy)?;

    // eta = X*β
    let eta = x.dot(&beta);

    // For Gaussian, fitted_values = eta (identity link)
    let fitted_values = eta.clone();

    // Deviance = Σ w_i (y_i - μ_i)²  (unweighted ⇒ w_i = 1)
    let deviance: f64 = match prior_weights {
        None => y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, fi)| (yi - fi).powi(2))
            .sum(),
        Some(w) => y
            .iter()
            .zip(fitted_values.iter())
            .zip(w.iter())
            .map(|((yi, fi), wi)| wi * (yi - fi).powi(2))
            .sum(),
    };

    // PIRLS weights output:
    //   - unweighted: all 1.0 (Gaussian IRLS weight).
    //   - weighted:   prior weights, which are also the W in the converged
    //     XᵀWX. The outer REML / vcov consumers multiply this with X' to
    //     form X'WX, which is exactly what we want for the weighted fit.
    let weights = match prior_weights {
        None => Array1::ones(n),
        Some(w) => w.clone(),
    };

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        working_response: y.clone(),
        deviance,
        iterations: 1,
        converged: true,
        sigma2: None,
        df: None,
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
            // Scaled-t per-obs deviance (mgcv scat$dev.resids): (ν+1)·log1p(r²/(ν·σ²)).
            Family::TDist { df, sigma2 } => {
                let r = yi - mui;
                (df + 1.0) * (1.0 + r * r / (df * sigma2)).ln()
            }
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
                    2.0 * (yi.powf(twop) / (onep * twop) - yi * mui.powf(onep) / onep
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
            // Quantile/ELF deviance — port of qgam elf.R:122-138 with λ = σ.
            //
            // qgam's per-obs term:
            //   T = (1-τ)·λ·log(1-τ) + λ·τ·log(τ)
            //       - (1-τ)·(y-μ) + λ·log1pexp((y-μ)/λ)
            //   dev_per_obs = 2·T/σ
            //
            // With λ = σ this simplifies to:
            //   dev_per_obs = 2·[ -H(τ) - (1-τ)(y-μ)/σ + log1pexp((y-μ)/σ) ]
            // where the constant -H(τ) is qgam's "saturation offset" — at the
            // ELF likelihood mode (μ_max = y - σ·logit(1-τ), not μ=y) this
            // gives dev = 0. My earlier formulation used μ=y as the
            // saturation point which is only correct at τ=0.5; switching
            // to qgam's convention is what makes the REML score behave
            // sensibly (REML stops collapsing as σ→0).
            Family::Quantile { tau, sigma } => {
                let r = yi - mui;
                let r_over_sigma = r / sigma;
                let softplus = if r_over_sigma > 0.0 {
                    r_over_sigma + (-r_over_sigma).exp().ln_1p()
                } else {
                    r_over_sigma.exp().ln_1p()
                };
                let h_tau = -((1.0 - tau) * (1.0 - tau).ln() + tau * tau.ln());
                2.0 * (-h_tau - (1.0 - tau) * r / sigma + softplus)
            }
        };

        deviance += dev_i;
    }

    deviance
}

/// Prior-weighted deviance: `D_w = Σ w_i · d_i(y_i, μ_i)`.
///
/// Mgcv treats `weights=` as a per-row multiplier on the contribution
/// to the log-likelihood (equivalently to the deviance). This helper
/// is the per-row product of `prior_weights` and the per-family per-
/// observation deviance computed via `compute_deviance`'s per-row body.
/// `prior_weights.len()` MUST equal `y.len()`; caller is expected to
/// validate.
pub fn compute_weighted_deviance(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: Family,
    prior_weights: &Array1<f64>,
) -> f64 {
    let mut dev = 0.0;
    // We re-use compute_deviance's per-row math by calling it on
    // 1-element slices via a temporary — clearer than duplicating the
    // family switch. Hot enough to inline if it ever shows up in a
    // profile.
    for i in 0..y.len() {
        let yi = Array1::from(vec![y[i]]);
        let mi = Array1::from(vec![mu[i]]);
        dev += prior_weights[i] * compute_deviance(&yi, &mi, family);
    }
    dev
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
        -(n * (log_gamma(half_nu_p1)
            - log_gamma(half_nu)
            - 0.5 * (nu * std::f64::consts::PI * sigma2.max(1e-300)).ln())
            - half_nu_p1 * log_sum)
    };

    brent_minimize(neg_pll, 2.0, 100.0, 1e-4, 50)
}

/// Prior-weighted version of `profile_df`. Each per-row log-density
/// contribution is multiplied by `prior_weights[i]`, mirroring mgcv's
/// `weights=` convention (each observation behaves like w_i replications
/// of itself). Falls back to unweighted for `prior_weights = None`.
pub fn profile_df_weighted(
    residuals: &[f64],
    sigma2: f64,
    prior_weights: Option<&Array1<f64>>,
) -> f64 {
    let pw = match prior_weights {
        None => return profile_df(residuals, sigma2),
        Some(pw) => pw,
    };
    let neg_pll = |nu: f64| -> f64 {
        let sum_pw: f64 = pw.iter().sum::<f64>().max(1e-300);
        let half_nu_p1 = (nu + 1.0) / 2.0;
        let half_nu = nu / 2.0;
        let log_sum: f64 = residuals
            .iter()
            .zip(pw.iter())
            .map(|(&r, &w)| {
                let t2 = r * r / (nu * sigma2.max(1e-300));
                w * (1.0 + t2).ln()
            })
            .sum();
        -(sum_pw
            * (log_gamma(half_nu_p1)
                - log_gamma(half_nu)
                - 0.5 * (nu * std::f64::consts::PI * sigma2.max(1e-300)).ln())
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

        let u = x + if d.abs() >= tol1 {
            d
        } else if d > 0.0 {
            tol1
        } else {
            -tol1
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

/// Textbook EM t-distribution weight `w = (ν+1) / (ν + r²/σ²)`.
///
/// Used inside the inner-loop σ² MLE update (mgcv `gam.fit5` form) where the
/// score equation `Σ wᵖ·wᵗ·r² = σ² · Σ wᵖ` requires the EM weight specifically,
/// regardless of which working pair the IRLS step used. Floored at 1e-10 to
/// avoid pathological zero-weight rows.
#[inline]
pub(crate) fn t_weight_em(r: f64, sigma2: f64, df: f64) -> f64 {
    let t2 = r * r / sigma2.max(1e-300);
    ((df + 1.0) / (df + t2)).max(1e-10)
}

/// Newton (observed-info) working pair `(w, z)` from mgcv `scat$Dd`.
///
/// Returns `(½·D''μ, η − D'μ/D''μ)`. Falls back to the expected-info form
/// `(ν+1)/(σ²(ν+3))` when the observed curvature is non-PSD. Floored at 1e-10
/// to keep the Hessian invertible. Used by [`tdist_irls_step`] when the
/// caller is a gam.fit5-style outer Newton driver that supplies a fixed σ².
#[inline]
pub(crate) fn t_newton_working_pair(r: f64, eta: f64, sigma2: f64, df: f64) -> (f64, f64) {
    let sigma2 = sigma2.max(1e-300);
    let denom = df * sigma2 + r * r;
    let dmu = -2.0 * (df + 1.0) * r / denom;
    let observed_dmu2 = 2.0 * (df + 1.0) * (df * sigma2 - r * r) / (denom * denom);
    let expected_dmu2 = 2.0 * (df + 1.0) / (sigma2 * (df + 3.0));
    let dmu2 = if observed_dmu2.is_finite() && observed_dmu2 > 1e-12 {
        observed_dmu2
    } else {
        expected_dmu2.max(1e-12)
    };
    let z = eta - dmu / dmu2;
    ((0.5 * dmu2).max(1e-10), z)
}

/// mgcv `bgam.fitd`-style Fisher (expected-info) IRLS pair for the t-density.
///
/// Returns `(½·E[D''_η], η − D'_η/E[D''_η])` per mgcv's `dDeta` at identity
/// link with `EDeta2 = EDmu2 = 2·(ν+1)/(σ²·(ν+3))` (efam.r:3636, `oo$EDmu2`).
/// mgcv's `bgam.fitd` switches to Fisher weights when `rho != 0` (bam.r:638-640);
/// for `rho == 0` it uses observed-info `Dmu2/2`. We use Fisher
/// unconditionally for the t-density in the fREML driver because
/// observed-info can go negative (when `r² > ν·σ²`), and our
/// `compute_sl_fitchol_step` uses a plain LU solve (no pivoted Cholesky
/// fallback) that can't recover from the resulting indefinite X'WX —
/// non-finite β crashes within 2-3 Newton iters.
///
/// On the weighted-scat parity fixture this lands `(df, σ²)` within 0.2%
/// rel of mgcv (mgcv: 4.257, 0.0341; rust: 4.253, 0.0341) and λ within
/// ~25% rel (rust: 168, mgcv: 137). The remaining gap is the
/// observed-vs-Fisher difference in the per-row IRLS weights driving
/// Sl.fitChol; to close it byte-for-byte we'd need pivoted Cholesky with
/// ridge fallback in `compute_sl_fitchol_step` (mgcv's `Sl.fitChol` route).
/// Tracked as R-task in the worktree report.
#[inline]
fn t_bgam_fisher_pair(r: f64, eta: f64, sigma2: f64, df: f64) -> (f64, f64) {
    let sigma2 = sigma2.max(1e-300);
    // Dmu = -2·(ν+1)·r / (νσ² + r²)  (efam.r:3632).
    let denom = df * sigma2 + r * r;
    let dmu = -2.0 * (df + 1.0) * r / denom;
    // EDmu2 (efam.r:3636) = 2·(ν+1)/(σ²·(ν+3)). Strictly positive.
    let edmu2 = 2.0 * (df + 1.0) / (sigma2 * (df + 3.0));
    let w = 0.5 * edmu2;
    let z = eta - dmu / edmu2;
    (w, z)
}

/// mgcv `bgam.fitd`-style observed-info IRLS pair for the t-density.
///
/// Returns `(½·D''_η, η − D'_η/D''_η)` matching mgcv's `Dmu2 · 0.5` working
/// weight at `rho == 0` (bam.r:638-640) — the same per-row arithmetic
/// already shipped in `t_newton_working_pair`, lifted here under a name
/// that mirrors `t_bgam_fisher_pair` and inlined into the fREML driver.
///
/// **Indefinite-W posture (R8)**: on moderate-outlier rows
/// `r² > ν·σ²` makes `D''_η` go through zero and then negative. The
/// previous Fisher pair side-stepped this; here we let the row weight take
/// its true (possibly negative) value and rely on
/// `compute_sl_fitchol_step`'s pivoted-Chol route to handle the resulting
/// indefinite `X'WX`. If observed-info is non-finite for a row we still
/// fall back to Fisher on that row (otherwise the entire IRLS step is
/// NaN-poisoned).
///
/// The `min_w` floor below stays at `f64::MIN_POSITIVE` for finite-positive
/// rows — we deliberately do NOT clamp negative weights up. That allows
/// observed-info to express the curvature mgcv sees in `bgam.fitd` when
/// `rho == 0`, which is the main half-octave of λ that the Fisher path
/// was sub-optimising.
#[inline]
#[allow(dead_code)] // wired in via fastreml_irls_step in a later step
fn t_bgam_observed_info_pair(r: f64, eta: f64, sigma2: f64, df: f64) -> (f64, f64) {
    let sigma2 = sigma2.max(1e-300);
    // Dmu = -2·(ν+1)·r / (νσ² + r²)  (efam.r:3632, same as Fisher).
    let denom = df * sigma2 + r * r;
    let dmu = -2.0 * (df + 1.0) * r / denom;
    // Observed-info second derivative wrt η at identity link
    // (mgcv efam.r:3645, `Dmu2` for scat with identity-link η = μ):
    //   D''_η = 2·(ν+1)·(ν·σ² − r²) / (ν·σ² + r²)²
    // Strictly positive when `r² < ν·σ²`, zero at the inflection,
    // negative for the heavy-tail extremes.
    let observed_dmu2 = 2.0 * (df + 1.0) * (df * sigma2 - r * r) / (denom * denom);
    // mgcv's row-level Fisher fallback (efam.r:3641, "EDmu2" branch when
    // Dmu2 ≤ 0): used here only when the observed value is non-finite.
    let expected_dmu2 = 2.0 * (df + 1.0) / (sigma2 * (df + 3.0));
    let dmu2 = if observed_dmu2.is_finite() {
        observed_dmu2
    } else {
        expected_dmu2
    };
    // Avoid division by exact zero (NaN poisoning): nudge magnitude up by
    // f64::MIN_POSITIVE while preserving sign. The pivoted-Chol step
    // tolerates the resulting indefinite X'WX.
    let dmu2_safe = if dmu2.abs() < f64::MIN_POSITIVE {
        if dmu2 >= 0.0 {
            f64::MIN_POSITIVE
        } else {
            -f64::MIN_POSITIVE
        }
    } else {
        dmu2
    };
    let w = 0.5 * dmu2_safe;
    let z = eta - dmu / dmu2_safe;
    (w, z)
}

/// Per-observation working pair `(w, z)` for one t-distribution IRLS step.
///
/// Returned by [`tdist_irls_step`]. The values are the per-row working weight
/// and working response *after* multiplying any caller-supplied prior weights
/// into `w` — i.e. the vectors that the outer loop hands directly to the
/// X'WX / X'Wz assembly.
///
/// Refactor R2 (Path B prep): both `fit_pirls_tdist` (dense) and
/// `fit_pirls_tdist_discrete` route through this helper so the genuine
/// per-row math lives in one place.
pub(crate) struct TdistIrlsStep {
    /// Combined working weight `w_i = w_iᵗ · w_iᵖ` (prior weights folded in).
    pub w: Array1<f64>,
    /// Working response `z_i` (textbook EM: `z_i = y_i`; Newton path:
    /// `z_i = η_i − dμ/dμ²`).
    pub z: Array1<f64>,
}

/// Compute one t-distribution IRLS working pair `(w, z)` at fixed `(σ², df)`.
///
/// Implements the per-row arithmetic that was previously inlined into
/// [`fit_pirls_tdist`] and [`fit_pirls_tdist_discrete`]. Pure function: no
/// allocation other than the two returned vectors, no I/O, no global state.
///
/// `use_newton_working` selects between:
///   - `false` (textbook EM, used when the inner loop drives σ²/df itself):
///     `w_i = (df+1) / (df + r_i² / σ²)`, `z_i = y_i`.
///   - `true` (Newton observed-info, used when `fixed_sigma2` is supplied by
///     the gam.fit5 outer Newton): `w_i = ½ · D''μ(r_i)`, `z_i = η_i − D'μ/D''μ`,
///     with the expected-info fallback when the observed curvature is non-PSD.
///
/// The output weight has `prior_weights` (when supplied) already multiplied
/// in row-wise, matching mgcv's `weights=` semantics where the prior weight is
/// a per-row multiplier on the family's working weight.
///
/// **Lifted from** `fit_pirls_tdist` inner loop body — byte-identical
/// arithmetic.
pub(crate) fn tdist_irls_step(
    y: ArrayView1<f64>,
    eta: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    sigma2: f64,
    df: f64,
    use_newton_working: bool,
) -> TdistIrlsStep {
    let n = y.len();
    debug_assert_eq!(eta.len(), n, "eta length must match y");
    if let Some(pw) = prior_weights.as_ref() {
        debug_assert_eq!(pw.len(), n, "prior_weights length must match y");
    }

    // EM-IRLS t-weight: w_i = (ν+1) / (ν + r²/σ²).
    //
    // Derivation: from d log L / d β = 0 with the t-density we get the
    // weighted normal equations X'WX β = X'Wy where W = diag(w_i) above.
    // This is the standard EM majorisation for t-regression and the form
    // mgcv's gam.fit5 uses *implicitly* in the σ² score (the σ² MLE
    // condition Σw·r² = nσ² inverts to σ² = Σw·r²/n with this same w).
    //
    // The 2026-05-08 attempt to switch IRLS to observed-info Dmu2/2 broke
    // test_tdist_mgcv_parity (max relerr 0.10 → 0.44): observed-info has
    // units of 1/σ², whereas the rest of the pipeline (penalty matrix S,
    // ridge, λ) is calibrated for the unit-magnitude EM weight. The σ²
    // MLE update also requires the EM weight to recover the t-likelihood
    // MLE. Reverted; stayed with textbook EM-IRLS.
    let mut z_work = Array1::<f64>::zeros(n);
    // Per-row t-IRLS weight w_iᵗ. Computed independently of any prior
    // weights so the σ²/df updates see the textbook (unweighted) t-weight;
    // mgcv `weights=` semantics multiply on top of the family's working
    // weight, not into the family parameter MLE.
    let w_tdist: Array1<f64> = y
        .iter()
        .zip(eta.iter())
        .enumerate()
        .map(|(i, (&yi, &etai))| {
            let r = yi - etai;
            if use_newton_working {
                let (wi, zi) = t_newton_working_pair(r, etai, sigma2, df);
                z_work[i] = zi;
                wi
            } else {
                z_work[i] = yi;
                t_weight_em(r, sigma2, df)
            }
        })
        .collect();
    // Combine with prior_weights: mgcv treats `weights=` as a per-row
    // multiplier on the log-likelihood, so it multiplies the IRLS
    // working weight everywhere (X'WX, X'Wz). Without prior weights
    // we pass through w_tdist unchanged.
    let w: Array1<f64> = match prior_weights.as_ref() {
        Some(pw) => w_tdist
            .iter()
            .zip(pw.iter())
            .map(|(&wt, &wp)| wt * wp)
            .collect(),
        None => w_tdist,
    };

    TdistIrlsStep { w, z: z_work }
}

/// Per-observation IRLS working pair `(w, z)` for one exponential-family
/// PIRLS step, with caller-supplied prior weights already folded into `w`.
///
/// Returned by [`exp_family_irls_step`]. The fields are exactly the vectors a
/// caller passes to `X'WX` / `X'Wz` assembly — no further per-row arithmetic
/// is required after the helper returns.
pub(crate) struct ExpFamilyIrlsStep {
    /// Working weight `w_i = w_iᶠ · w_iᵖ` (prior weights folded in). `w_iᶠ`
    /// is the GLM Fisher (or Newton-with-Fisher-fallback) weight from
    /// [`compute_irls_wz`].
    pub w: Array1<f64>,
    /// Working response `z_i = η_i + (y_i − μ_i) / (dμ/dη)_i` (Fisher branch)
    /// or the Newton-corrected variant when `use_fisher=false` and the
    /// per-row curvature stays PSD.
    pub z: Array1<f64>,
}

/// Compute one exponential-family IRLS working pair `(w, z)` at the supplied
/// `(y, η)` with the given `Family`, folding `prior_weights` into `w` row-wise.
///
/// Implements the per-row arithmetic that was previously inlined into
/// [`fit_pirls_cached`] and [`fit_pirls_discretized`] (both inner loop and
/// final-scoring pass). Pure function: allocates only the two returned
/// vectors. The per-row math itself lives in [`compute_irls_wz`] and is
/// unchanged — this helper simply wraps the loop and the prior-weights
/// folding so the dispatch sites read as a single call.
///
/// `use_fisher` follows the same semantics as [`compute_irls_wz`]:
///   - `true` ⇒ Fisher scoring everywhere (used for canonical links and the
///     final-scoring pass that REML callbacks consume).
///   - `false` ⇒ Newton curvature correction per row, with per-row Fisher
///     fallback when `α ≤ 0`. Used only by `fit_pirls_cached`'s inner loop.
///
/// Prior weights enter mgcv's `weights=` semantics as a per-row multiplier
/// on the log-likelihood (exposure / replication count), so they multiply
/// the working weight everywhere (`X'WX`, `X'Wz`). When `prior_weights` is
/// `None`, the raw Fisher/Newton weight passes through unchanged.
///
/// **Lifted from** `fit_pirls_cached` / `fit_pirls_discretized` inner loop
/// bodies — byte-identical arithmetic, same as the in-place sequence of
/// `standard_glm_working_quantities` followed by the prior-weights fold.
pub(crate) fn exp_family_irls_step(
    y: ArrayView1<f64>,
    eta: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family: Family,
    use_fisher: bool,
) -> ExpFamilyIrlsStep {
    let n = y.len();
    debug_assert_eq!(eta.len(), n, "eta length must match y");
    if let Some(pw) = prior_weights.as_ref() {
        debug_assert_eq!(pw.len(), n, "prior_weights length must match y");
    }

    // Canonical-link families always use Fisher; non-canonical links pass
    // `use_fisher` through (mirroring `standard_glm_working_quantities`).
    let use_fisher_effective = use_fisher || family.is_canonical_link();

    let mut z = Array1::<f64>::zeros(n);
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (wi, zi) = compute_irls_wz(eta[i], y[i], family, use_fisher_effective);
        w[i] = wi;
        z[i] = zi;
    }

    // Fold prior weights into `w` row-wise. mgcv semantics: `weights=` is a
    // multiplier on the log-likelihood, so it multiplies the family's
    // working weight throughout the linear system.
    if let Some(pw) = prior_weights.as_ref() {
        for i in 0..n {
            w[i] *= pw[i];
        }
    }

    ExpFamilyIrlsStep { w, z }
}

/// Fit a penalized GAM with scaled t-distribution errors.
///
/// This is the outer loop for `Family::TDist` fitting. It alternates between:
///
/// 1. **Inner PIRLS** — fit β at fixed (df, σ²) using t-distribution IRLS
///    weights `w_i = (df+1) / (df + r_i²/σ²)` with identity link.
/// 2. **σ² MLE update** (when `fixed_sigma2` is None) — `σ² = Σ w_i r_i² / n`.
///    Skipped when σ² is supplied (gam.fit5-style outer Newton drives it).
/// 3. **df update** (when `fixed_df` is None) — 1D Brent on the profile
///    log-likelihood over df ∈ [2, 100]. Skipped when df is supplied
///    (outer Newton on log df at smooth.rs handles it).
///
/// Returns a `PiRLSResult` with the converged β. The `weights` field holds the
/// final per-observation t-weights.
pub fn fit_pirls_tdist(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    fixed_df: Option<f64>,
    fixed_sigma2: Option<f64>,
    max_iter: usize,
    tolerance: f64,
    prior_weights: Option<&Array1<f64>>,
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
    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "prior_weights length ({}) must match y length ({})",
                pw.len(),
                n
            )));
        }
    }

    // Validate df if fixed
    if let Some(df) = fixed_df {
        if df < 2.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be >= 2.0, got {}",
                df
            )));
        }
        if df > 100.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be <= 100.0, got {}",
                df
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

    // Initialise β = 0, η = y (identity link), σ² from caller / sample variance.
    let mut beta = Array1::zeros(p);
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 =
        y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let mut sigma2 = fixed_sigma2.unwrap_or_else(|| y_var.max(1e-6)).max(1e-6);

    // df: start at user value or 5.0 (a reasonable default)
    let mut df = fixed_df.unwrap_or(5.0).clamp(2.0, 100.0);

    // Default/fixed-df legacy path uses textbook EM weights. The
    // gam.fit5-style outer-LAML path (fixed σ² supplied by the outer
    // shape Newton) uses mgcv scat$Dd observed Hessian weights with the
    // expected Hessian fallback when the observed curvature is negative.
    let use_newton_working = fixed_sigma2.is_some();

    let mut converged = false;
    let mut iter = 0;

    for outer_iter in 0..max_iter {
        iter = outer_iter + 1;

        // ── Inner WLS: solve (X'WX + S) β = X'Wz with t weights ─────────
        let eta: Array1<f64> = x.dot(&beta);
        let step = tdist_irls_step(
            y.view(),
            eta.view(),
            prior_weights.map(|pw| pw.view()),
            sigma2,
            df,
            use_newton_working,
        );
        let w = step.w;
        let z_work = step.z;

        // X'WX via dense triple product
        let xtwx = crate::reml::compute_xtwx_dispatch(None, x, &w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        let wz: Array1<f64> = w
            .iter()
            .zip(z_work.iter())
            .map(|(&wi, &zi)| wi * zi)
            .collect();
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

        // ── MLE σ² update (mgcv's gam.fit5 inner-loop estimator) ──
        //
        // From d log L/d σ² = 0 (per-row contributions weighted by w_iᵖ):
        //   σ²_new = Σ w_iᵖ · w_iᵗ · r_i² / Σ w_iᵖ,   w_iᵗ = (ν+1)/(ν + r²/σ²)
        // iterated to a fixed point inside the outer PIRLS loop. This is
        // the MLE σ². For unweighted fits (w_iᵖ = 1) this reduces to the
        // textbook Σw·r²/n form. Skipped when `fixed_sigma2` is supplied —
        // the gam.fit5 outer Newton on log σ² (`smooth.rs`) drives σ²
        // instead, letting the LAML score's Jeffreys-like correction
        // shift σ² away from the MLE toward a finite-df minimum.
        if fixed_sigma2.is_none() {
            let w_new: Vec<f64> = residuals
                .iter()
                .map(|&r| t_weight_em(r, sigma2, df))
                .collect();
            let (sum_wr2, sum_pw): (f64, f64) = match prior_weights {
                Some(pw) => {
                    let mut a = 0.0;
                    let mut b = 0.0;
                    for i in 0..n {
                        a += pw[i] * w_new[i] * residuals[i] * residuals[i];
                        b += pw[i];
                    }
                    (a, b.max(1e-300))
                }
                None => {
                    let a: f64 = w_new
                        .iter()
                        .zip(residuals.iter())
                        .map(|(&wi, &ri)| wi * ri * ri)
                        .sum();
                    (a, n as f64)
                }
            };
            sigma2 = (sum_wr2 / sum_pw).max(1e-6);
        }
        let _ = p;

        // ── Update df via 1D Brent (skip if user fixed df) ───────────────
        if fixed_df.is_none() && outer_iter % 2 == 0 {
            // Profile df on every other outer iteration for efficiency.
            // With prior weights the profile log-likelihood is Σ w_iᵖ ·
            // log p_t(y_i | μ_i, σ², ν) — passed through `profile_df`.
            df = profile_df_weighted(&residuals, sigma2, prior_weights);
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

    // Final quantities. Reuse the IRLS-step helper at the converged β so the
    // final (weights, working_response) pair is bit-identical to the one the
    // outer REML loop will see when it re-evaluates the family weights.
    let eta: Array1<f64> = x.dot(&beta);
    let fitted_values = eta.clone();

    let residuals_final: Vec<f64> = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    let final_step = tdist_irls_step(
        y.view(),
        eta.view(),
        prior_weights.map(|pw| pw.view()),
        sigma2,
        df,
        use_newton_working,
    );
    let weights = final_step.w;
    // EM branch returns z_i = y_i, so the textbook working response
    // η_i + r_i = y_i matches z. Newton branch produces η_i − dμ/dμ² which
    // is exactly the helper's `z_work[i]`.
    let working_response = final_step.z;

    // Deviance as prior-weighted RSS (used for REML score; not the
    // t-log-likelihood). Mgcv treats `weights=` as a multiplier on the
    // per-row deviance contribution; reduces to plain Σr² when w_p = 1.
    let deviance: f64 = match prior_weights {
        Some(pw) => residuals_final
            .iter()
            .zip(pw.iter())
            .map(|(&r, &wp)| wp * r * r)
            .sum(),
        None => residuals_final.iter().map(|&r| r * r).sum(),
    };

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        working_response,
        deviance,
        iterations: iter,
        converged,
        // Surface the converged MLE σ² so the outer Newton REML loop can
        // sync `Family::TDist::sigma2` before it evaluates ls/deviance.
        sigma2: Some(sigma2),
        df: Some(df),
    })
}

/// Discrete-binning fast-path twin of [`fit_pirls_tdist`].
///
/// Mathematically byte-identical to `fit_pirls_tdist` — every per-row
/// quantity (t-weight, σ² MLE update, df Brent profile, residual
/// formation, working response, deviance) is computed on the same
/// length-n vectors. The only difference is the X'WX, X'Wy, and η = Xβ
/// assemblies, which run through the scatter-gather kernels in
/// `discrete.rs` instead of the dense BLAS triple-product.
///
/// On pure-dedup fixtures (`nunique ≤ max_bins_1d`) the result is
/// 1e-12-equal to `fit_pirls_tdist`. With quantile-grid binning the
/// design-doc accuracy floor is ~5e-3 on β.
///
/// This function is invoked from
/// [`crate::gam_optimized::FitCache::run_pirls_with_options`] when the
/// fit cache holds a `DiscreteDesign` and the family is `TDist`. Outside
/// that dispatch, callers should keep using `fit_pirls_tdist`.
pub fn fit_pirls_tdist_discrete(
    y: &Array1<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    fixed_df: Option<f64>,
    fixed_sigma2: Option<f64>,
    max_iter: usize,
    tolerance: f64,
    prior_weights: Option<&Array1<f64>>,
    disc: &DiscreteDesign,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = disc.total_basis;

    if disc.n != n {
        return Err(GAMError::DimensionMismatch(format!(
            "DiscreteDesign has n={} but y has {} elements",
            disc.n, n
        )));
    }
    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string(),
        ));
    }
    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "prior_weights length ({}) must match y length ({})",
                pw.len(),
                n
            )));
        }
    }

    // Validate df if fixed (identical to fit_pirls_tdist).
    if let Some(df) = fixed_df {
        if df < 2.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be >= 2.0, got {}",
                df
            )));
        }
        if df > 100.0 {
            return Err(GAMError::InvalidParameter(format!(
                "t-dist df must be <= 100.0, got {}",
                df
            )));
        }
    }

    // Build penalty total once (identical to fit_pirls_tdist).
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

    let mut beta = Array1::zeros(p);
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let y_var: f64 =
        y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let mut sigma2 = fixed_sigma2.unwrap_or_else(|| y_var.max(1e-6)).max(1e-6);

    let mut df = fixed_df.unwrap_or(5.0).clamp(2.0, 100.0);

    // Per-row IRLS weights/z share the helper used by the dense path
    // (`tdist_irls_step`). Only η differs: discrete uses the compressed
    // gather, dense uses `x.dot(&beta)`.
    let use_newton_working = fixed_sigma2.is_some();

    let mut converged = false;
    let mut iter = 0;

    for outer_iter in 0..max_iter {
        iter = outer_iter + 1;

        // ── Inner WLS: solve (X'WX + S) β = X'Wz with t weights ─────────
        // η uses the compressed-gather kernel instead of x.dot(&beta).
        let eta: Array1<f64> = compute_eta_discrete(disc, &beta);
        let step = tdist_irls_step(
            y.view(),
            eta.view(),
            prior_weights.map(|pw| pw.view()),
            sigma2,
            df,
            use_newton_working,
        );
        let w = step.w;
        let z_work = step.z;

        // X'WX via discrete scatter-gather (mgcv XWXd) — the basis-only
        // change vs `fit_pirls_tdist`. O(n + Σ m_a·p_a² + Σ m_a·m_b·(p_a+p_b)).
        let xtwx = compute_xtwx_discrete(disc, &w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // X'Wz via discrete scatter-gather (mgcv XWyd). compute_xtwy_discrete
        // forms Σ_i w_i·z_i per bin then gathers via X̃' — same arithmetic as
        // the un-binned X'·diag(w)·z.
        let xtwz = compute_xtwy_discrete(disc, &w, &z_work);

        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // ── Update σ² via MLE (same per-row math, recompute η on the
        // updated β via the discrete gather) ─────────────────────────────
        let eta_new: Array1<f64> = compute_eta_discrete(disc, &beta);
        let residuals: Vec<f64> = y
            .iter()
            .zip(eta_new.iter())
            .map(|(&yi, &etai)| yi - etai)
            .collect();

        if fixed_sigma2.is_none() {
            let w_new: Vec<f64> = residuals
                .iter()
                .map(|&r| t_weight_em(r, sigma2, df))
                .collect();
            let (sum_wr2, sum_pw): (f64, f64) = match prior_weights {
                Some(pw) => {
                    let mut a = 0.0;
                    let mut b = 0.0;
                    for i in 0..n {
                        a += pw[i] * w_new[i] * residuals[i] * residuals[i];
                        b += pw[i];
                    }
                    (a, b.max(1e-300))
                }
                None => {
                    let a: f64 = w_new
                        .iter()
                        .zip(residuals.iter())
                        .map(|(&wi, &ri)| wi * ri * ri)
                        .sum();
                    (a, n as f64)
                }
            };
            sigma2 = (sum_wr2 / sum_pw).max(1e-6);
        }
        let _ = p;

        // ── Update df via 1D Brent (same as fit_pirls_tdist) ─────────────
        if fixed_df.is_none() && outer_iter % 2 == 0 {
            df = profile_df_weighted(&residuals, sigma2, prior_weights);
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

    // Final quantities — η built via the compressed gather. Reuse the
    // IRLS-step helper at the converged β so (weights, working_response)
    // are bit-identical to one more inner step.
    let eta: Array1<f64> = compute_eta_discrete(disc, &beta);
    let fitted_values = eta.clone();

    let residuals_final: Vec<f64> = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    let final_step = tdist_irls_step(
        y.view(),
        eta.view(),
        prior_weights.map(|pw| pw.view()),
        sigma2,
        df,
        use_newton_working,
    );
    let weights = final_step.w;
    let working_response = final_step.z;

    let deviance: f64 = match prior_weights {
        Some(pw) => residuals_final
            .iter()
            .zip(pw.iter())
            .map(|(&r, &wp)| wp * r * r)
            .sum(),
        None => residuals_final.iter().map(|&r| r * r).sum(),
    };

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        working_response,
        deviance,
        iterations: iter,
        converged,
        sigma2: Some(sigma2),
        df: Some(df),
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Quantile (qgam-style) family — IRLS on ELF (Extended Log-F) loss
// ────────────────────────────────────────────────────────────────────────────

/// Custom IRLS loop for the qgam-style Quantile family.
///
/// At residual r = y - η (identity link) the per-obs negative log-likelihood is
///   L(r; τ, σ) = (η - y)(1-τ)/σ + log(1 + exp((y - η)/σ))
/// which is a smooth approximation to the pinball loss
///   ρ_τ(r) = max(τ·r, (τ-1)·r)
/// (recovered as σ→0). The minimizer of E[L(y - η(x))] is the τ-quantile of
/// y|x rather than the mean.
///
/// Per Fasiolo et al. 2021 the working IRLS quantities derived analytically are
///   s_i = sigmoid((y_i - η_i)/σ)
///   w_i = s_i(1-s_i) / σ²        (well-defined PSD Hessian)
///   z_i = η_i - σ(1-τ-s_i)/(s_i(1-s_i))
/// which fit cleanly into the standard PIRLS template.
///
/// `sigma` is taken from the family. If 0.0 (sentinel), it is auto-calibrated
/// at fit time as a robust scale of the residuals from the unpenalised
/// τ-quantile of y. Full qgam-style σ-calibration (cross-validated bandwidth)
/// is a deferred followup.
pub fn fit_pirls_quantile(
    y: &Array1<f64>,
    x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    tau: f64,
    sigma_user: f64,
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
    if tau <= 0.0 || tau >= 1.0 {
        return Err(GAMError::InvalidParameter(format!(
            "quantile tau must be in (0, 1), got {}",
            tau
        )));
    }

    // ── qgam-style warm-start (qgam.R::.init_gauss_fit + qgam.R:156) ──
    //
    // qgam first runs an unpenalised-loss Gaussian GAM, gets the residual
    // variance σ̂², then sets per-observation initial η[i] = qnorm(τ; μ̂_gauss[i], σ̂)
    // — the τ-quantile of N(μ̂_gauss(x), σ̂²). For us, the cleanest equivalent
    // is to use the empirical τ-quantile of the Gaussian-fit residuals as a
    // location shift (avoids needing a normal-quantile lookup; matches the
    // shape of qgam's mustart).
    //
    // The initial σ for the ELF loss is qgam's `co = err·√(2π·σ̂²)/(2·log 2)`
    // with err=0.05 default. This is much sharper than my earlier MAD-based
    // heuristic but comes paired with the warm-start, which keeps weights
    // out of saturation.

    // Build penalty total once (used for both the Gaussian init and the IRLS).
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

    // Step 1: Gaussian GAM fit at the supplied λ. Solves (X'X + Σ λᵢSᵢ + ridge·I) β = X'y.
    let xtx = crate::reml::compute_xtwx_dispatch(None, x, &Array1::ones(n));
    let mut a_gauss = &xtx + &penalty_total;
    let mut max_diag_g = 1.0_f64;
    for i in 0..p {
        max_diag_g = max_diag_g.max(a_gauss[[i, i]].abs());
    }
    let ridge_g = ridge_scale * max_diag_g;
    for i in 0..p {
        a_gauss[[i, i]] += ridge_g;
    }
    let xty: Array1<f64> = x.t().dot(y);
    let beta_gauss = solve(a_gauss.clone(), xty)?;

    // Step 2: Gaussian-fit residuals + variance.
    let mu_gauss: Array1<f64> = x.dot(&beta_gauss);
    let r_vec: Vec<f64> = y
        .iter()
        .zip(mu_gauss.iter())
        .map(|(&yi, &mi)| yi - mi)
        .collect();
    let sigma2_hat: f64 = r_vec.iter().map(|&ri| ri * ri).sum::<f64>() / (n as f64).max(1.0);

    // Step 3: empirical τ-quantile of residuals — the per-obs shift to apply
    // to the Gaussian fit so that the warm-start η ≈ μ̂_gauss(x) + q_τ(r).
    let mut r_sorted = r_vec.clone();
    r_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qi_r = ((n as f64 - 1.0) * tau).round() as usize;
    let q_r = r_sorted[qi_r.min(n - 1)];

    // Step 4: solve for β_init s.t. X·β_init ≈ μ̂_gauss + q_r·1
    //   = X β_gauss + q_r·1  ⟹  β_init = β_gauss + (X'X + S + ridge)⁻¹ X' (q_r·1)
    let q_const = Array1::from_elem(n, q_r);
    let xtq = x.t().dot(&q_const);
    let delta_beta = solve(a_gauss, xtq)?;
    let mut beta = &beta_gauss + &delta_beta;

    // Step 5: σ for the ELF loss — qgam's co formula with err=0.05 default,
    // bumped for extreme τ. qgam itself runs `tuneLearnFast` (cross-validated
    // bandwidth) — out of scope for v0.1 — but for extreme τ the default σ
    // is too sharp: most observations land well past the logit's saturation
    // and have ~zero weight, which makes the IRLS Hessian rank-deficient.
    // The 1/(4τ(1-τ)) factor (= 1 at τ=0.5, ≈ 5 at τ=0.05/0.95) widens σ
    // enough to keep observations contributing through the IRLS solve.
    // Sharper, properly-calibrated σ remains a deferred followup.
    let sigma = if sigma_user > 0.0 {
        sigma_user
    } else {
        let err = 0.05_f64;
        let sigma2_floor = sigma2_hat.max(1e-6);
        let co_default =
            err * (2.0 * std::f64::consts::PI * sigma2_floor).sqrt() / (2.0 * 2.0_f64.ln());
        let tail_scale = (1.0 / (4.0 * tau * (1.0 - tau))).max(1.0);
        co_default * tail_scale
    };
    let inv_sigma = 1.0 / sigma;

    // Helper: stable sigmoid
    fn sigmoid_stable(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    let mut converged = false;
    let mut iter = 0;

    for outer_iter in 0..max_iter {
        iter = outer_iter + 1;

        let eta: Array1<f64> = x.dot(&beta);

        // Per-obs IRLS quantities, computed in a saturation-safe way (qgam
        // R-source pattern, elf.R:159-162):
        //   w_i  = s_i(1-s_i)/σ²        (Dmu²/2; goes to 0 as |r/σ| → ∞ —
        //                                that's fine, those obs just drop
        //                                out of the IRLS solve)
        //   The Newton step solves (X'WX + S) β = X' (W·η + g) where g_i =
        //   (s_i - (1-τ))/σ is the "working gradient" — bounded in
        //   [-(1-τ)/σ, τ/σ] regardless of saturation. This avoids the w·z
        //   product blowing up when w → 0.
        let mut w = Array1::<f64>::zeros(n);
        let mut g = Array1::<f64>::zeros(n);
        for i in 0..n {
            let r = y[i] - eta[i];
            let s = sigmoid_stable(r * inv_sigma);
            w[i] = s * (1.0 - s) * inv_sigma * inv_sigma;
            g[i] = (s - (1.0 - tau)) * inv_sigma;
        }

        let xtwx = crate::reml::compute_xtwx_dispatch(None, x, &w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // RHS = X' (W·η + g). Equivalent to X'·W·z when w_i > 0 with
        // z_i = η_i + g_i / w_i, but well-defined at saturation.
        let weta_plus_g: Array1<f64> = w
            .iter()
            .zip(eta.iter())
            .zip(g.iter())
            .map(|((&wi, &etai), &gi)| wi * etai + gi)
            .collect();
        let xt_rhs = x.t().dot(&weta_plus_g);

        let beta_old = beta.clone();
        beta = solve(a, xt_rhs)?;

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

    let mut weights = Array1::<f64>::zeros(n);
    let mut working_response = Array1::<f64>::zeros(n);
    let mut deviance = 0.0;
    let log2 = 2.0_f64.ln();
    for i in 0..n {
        let r = y[i] - eta[i];
        let s = sigmoid_stable(r * inv_sigma);
        weights[i] = s * (1.0 - s) * inv_sigma * inv_sigma;
        let g = (s - (1.0 - tau)) * inv_sigma;
        working_response[i] = if weights[i] > 1e-10 {
            eta[i] + g / weights[i]
        } else {
            eta[i]
        };
        // Per-obs ELF deviance: 2·(L(r) - log 2)
        let r_over_sigma = r * inv_sigma;
        let softplus = if r_over_sigma > 0.0 {
            r_over_sigma + (-r_over_sigma).exp().ln_1p()
        } else {
            r_over_sigma.exp().ln_1p()
        };
        let l = -r * (1.0 - tau) * inv_sigma + softplus;
        deviance += 2.0 * (l - log2);
    }

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        working_response,
        deviance,
        iterations: iter,
        converged,
        sigma2: None,
        df: None,
    })
}

/// Result of joint location-scale ELF (`elflss`) fit.
pub struct PiRLSResultLSS {
    pub coefficients_loc: Array1<f64>,
    pub coefficients_scale: Array1<f64>,
    pub eta_loc: Array1<f64>,
    pub eta_scale: Array1<f64>,
    /// σ(x) = exp(η_scale) at training rows.
    pub sigma: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Per-location-smooth λ at the end of the fit. When the caller passed
    /// `s_loc_total` directly (fixed-λ path) this stays empty; when they
    /// went through `fit_pirls_quantile_lss_fs_tune` it holds the final
    /// FS-tuned λs.
    pub lambda_loc: Vec<f64>,
    /// Number of Fellner-Schall outer iterations actually performed
    /// (0 when no λ tuning was requested).
    pub fs_iterations: usize,
}

/// ELF (Extended Log-F) quantile IRLS with PER-OBSERVATION σ. Mirrors
/// `fit_pirls_quantile` but with σ as a vector and an Armijo backtracking
/// line search on the penalised ELF deviance, which keeps the IRLS stable
/// when per-obs σ varies widely (low-σ regions tend to dominate the
/// gradient and trigger overshoot without damping).
///
/// Penalised objective: D = Σ 2[-r·(1-τ)/σ + log(1+exp(r/σ)) - log 2]
///                          + β'·S_total·β.
///
/// Returns (β, fitted η, deviance, iterations, converged).
fn fit_pirls_quantile_perobs_sigma(
    y: &Array1<f64>,
    x: &Array2<f64>,
    s_total: &Array2<f64>,
    sigma_per: &Array1<f64>,
    tau: f64,
    beta_init: &Array1<f64>,
    max_iter: usize,
    tolerance: f64,
) -> Result<(Array1<f64>, Array1<f64>, f64, usize, bool)> {
    let n = y.len();
    let p = x.ncols();

    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5
    };

    fn sigmoid_stable(x: f64) -> f64 {
        if x >= 0.0 {
            let e = (-x).exp();
            1.0 / (1.0 + e)
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }

    let log2 = 2.0_f64.ln();
    let elf_deviance = |b: &Array1<f64>| -> f64 {
        let eta_t = x.dot(b);
        let mut total = 0.0_f64;
        for i in 0..n {
            let inv_si = 1.0 / sigma_per[i];
            let r = y[i] - eta_t[i];
            let u = r * inv_si;
            // Stable softplus.
            let sp = if u > 0.0 {
                u + (-u).exp().ln_1p()
            } else {
                u.exp().ln_1p()
            };
            let l = -r * (1.0 - tau) * inv_si + sp;
            if !l.is_finite() {
                return f64::INFINITY;
            }
            total += 2.0 * (l - log2);
        }
        let sb = s_total.dot(b);
        let pen: f64 = b.iter().zip(sb.iter()).map(|(&a, &v)| a * v).sum();
        total + pen
    };

    let mut beta = beta_init.clone();
    let mut obj_cur = elf_deviance(&beta);
    let mut converged = false;
    let mut iter = 0;

    for outer in 0..max_iter {
        iter = outer + 1;
        let eta: Array1<f64> = x.dot(&beta);

        let mut w = Array1::<f64>::zeros(n);
        let mut g = Array1::<f64>::zeros(n);
        for i in 0..n {
            let inv_si = 1.0 / sigma_per[i];
            let r = y[i] - eta[i];
            let s = sigmoid_stable(r * inv_si);
            w[i] = s * (1.0 - s) * inv_si * inv_si;
            g[i] = (s - (1.0 - tau)) * inv_si;
        }

        let xtwx = compute_xtwx_dispatch(None, x, &w);
        let mut a = &xtwx + s_total;
        let mut md: f64 = 1.0;
        for i in 0..p {
            md = md.max(a[[i, i]].abs());
        }
        for i in 0..p {
            a[[i, i]] += ridge_scale * md;
        }
        // X' (W·η + g) — well-defined at saturation.
        let rhs_pt: Array1<f64> = w
            .iter()
            .zip(eta.iter())
            .zip(g.iter())
            .map(|((&wi, &ei), &gi)| wi * ei + gi)
            .collect();
        let rhs = x.t().dot(&rhs_pt);
        let beta_proposed = solve(a, rhs)?;

        // Armijo backtracking on the penalised ELF deviance. The full
        // Newton step (α=1) is nearly always accepted, but at sharp σ_i
        // (especially when σ varies widely per-obs), a smaller step
        // prevents the diverge-to-extreme failure mode where weights
        // collapse near the new η and β explodes on the next iter.
        let direction: Array1<f64> = beta_proposed
            .iter()
            .zip(beta.iter())
            .map(|(&bn, &bo)| bn - bo)
            .collect();
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        let mut beta_new = beta.clone();
        let mut obj_new = obj_cur;
        for _ in 0..20 {
            for j in 0..p {
                beta_new[j] = beta[j] + alpha * direction[j];
            }
            obj_new = elf_deviance(&beta_new);
            if obj_new.is_finite() && obj_new <= obj_cur + 1e-10 {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // No descent direction found — declare convergence by stagnation.
            converged = true;
            break;
        }

        let max_change = beta_new
            .iter()
            .zip(beta.iter())
            .map(|(b, b_old)| (b - b_old).abs())
            .fold(0.0_f64, f64::max);
        beta = beta_new;
        obj_cur = obj_new;

        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    let eta: Array1<f64> = x.dot(&beta);
    let mut deviance = 0.0;
    for i in 0..n {
        let inv_si = 1.0 / sigma_per[i];
        let r = y[i] - eta[i];
        let u = r * inv_si;
        let softplus = if u > 0.0 {
            u + (-u).exp().ln_1p()
        } else {
            u.exp().ln_1p()
        };
        let l = -r * (1.0 - tau) * inv_si + softplus;
        deviance += 2.0 * (l - log2);
    }

    Ok((beta, eta, deviance, iter, converged))
}

/// Per-observation-σ ELF quantile fit — the location-only stage of the
/// gaulss-then-ELF qgam (≥1.3) pipeline for heteroskedastic τ-quantile
/// regression.
///
/// Takes σ_G(x) — the Gaussian conditional SD at each training row,
/// from a Python-side gaulss preprocessing (two REML-tuned Gaussian
/// GAMs: one on y for μ_G, one on log|y - μ_G| + 0.6351 for log σ_G).
/// Internally computes the per-obs σ used in the ELF IRLS via qgam's
/// rescaling: `σ_i = σ_global · σ_G(x_i) / mean(σ_G(x))` (qgam
/// elf.R:151), preserving σ_G's heteroskedastic shape while normalising
/// the global bandwidth.
///
/// `sigma_global` controls the σ scale:
/// - `Some(v)`: use that scalar directly (e.g. from K-fold CV).
/// - `None`: qgam's `err · sqrt(2π · varHat) / (2·log 2)` heuristic
///   with `varHat = mean(σ_G)²` and tail-widening
///   `max(1, 1/(4τ(1-τ)))` for extreme τ.
///
/// Why qgam ≥1.3 went this way: the Beta-normalised elflss likelihood
/// theoretically identifies σ via joint MLE, but in finite samples
/// joint (μ, σ) MLE biases σ̂ small at extreme τ, breaking calibration.
/// Externalising σ̂(x) to gaulss (well-behaved Gaussian MLE) and using
/// ELF only for the location avoids the degeneracy.
///
/// `s_loc_total` is the pre-summed location penalty Σ λ_i S_i at the
/// full design size; Python builds it from the per-smooth blocks and
/// REML-fitted lambdas of the location GAM.
///
/// Returns σ_global used (auto-computed if None passed) so Python can
/// surface it in the info dict.
pub fn fit_pirls_quantile_lss(
    y: &Array1<f64>,
    x_loc: &Array2<f64>,
    s_loc_total: &Array2<f64>,
    sigma_g_per_obs: &Array1<f64>,
    sigma_global: Option<f64>,
    tau: f64,
    max_iter: usize,
    tolerance: f64,
) -> Result<(PiRLSResultLSS, f64)> {
    let n = y.len();
    let p_loc = x_loc.ncols();

    if x_loc.nrows() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "X_loc has {} rows but y has {} elements",
            x_loc.nrows(),
            n
        )));
    }
    if sigma_g_per_obs.len() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "sigma_g_per_obs has {} elements but y has {} elements",
            sigma_g_per_obs.len(),
            n
        )));
    }
    if tau <= 0.0 || tau >= 1.0 {
        return Err(GAMError::InvalidParameter(format!(
            "quantile tau must be in (0, 1), got {}",
            tau
        )));
    }
    if s_loc_total.nrows() != p_loc || s_loc_total.ncols() != p_loc {
        return Err(GAMError::DimensionMismatch(format!(
            "s_loc_total must be ({0}, {0}), got ({1}, {2})",
            p_loc,
            s_loc_total.nrows(),
            s_loc_total.ncols()
        )));
    }

    // ── Compute σ_global if not user-supplied. ──
    // qgam heuristic: σ_global = err · √(2π · varHat) / (2·log 2),
    // with varHat ≡ mean(σ_G)² (qgam.R:156, varHat from gaulss σ̂_G²).
    // Tail widening at extreme τ — matches fit_pirls_quantile's scalar
    // default to keep IRLS weights well-conditioned.
    let sigma_g_mean: f64 = sigma_g_per_obs.iter().copied().sum::<f64>() / (n as f64).max(1.0);
    let sigma_g_mean = sigma_g_mean.max(1e-8);
    let sigma_global = sigma_global.unwrap_or_else(|| {
        let err = 0.05_f64;
        let var_hat = sigma_g_mean * sigma_g_mean;
        let base = err * (2.0 * std::f64::consts::PI * var_hat).sqrt() / (2.0 * 2.0_f64.ln());
        let tail_scale = (1.0 / (4.0 * tau * (1.0 - tau))).max(1.0);
        base * tail_scale
    });

    // Per-obs σ used in ELF: qgam's rescaling preserves heteroskedastic
    // shape, normalises mean to σ_global. σ_G floor of 1e-8 prevents
    // division by zero; no upper floor needed since the IRLS line
    // search handles the upper end via deviance check.
    let sigma_per_obs: Array1<f64> = sigma_g_per_obs
        .iter()
        .map(|&sg| sigma_global * sg.max(1e-8) / sigma_g_mean)
        .collect();

    // Match fit_pirls_quantile's ridge_scale formula.
    let num_penalties_proxy = (s_loc_total
        .diag()
        .iter()
        .filter(|&&v| v.abs() > 0.0)
        .count() as f64)
        .max(1.0);
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + num_penalties_proxy.sqrt())
    };

    // ── β_loc warm-start: β_init = β_gauss + δ where δ is the
    // per-obs τ-quantile shift in coefficient space (the σ-aware analogue
    // of fit_pirls_quantile's q_r·1 shift). ──
    let xtx = compute_xtwx_dispatch(None, x_loc, &Array1::ones(n));
    let mut a_init = &xtx + s_loc_total;
    let mut md: f64 = 1.0;
    for i in 0..p_loc {
        md = md.max(a_init[[i, i]].abs());
    }
    for i in 0..p_loc {
        a_init[[i, i]] += ridge_scale * md;
    }
    let xty = x_loc.t().dot(y);
    let beta_gauss = solve(a_init.clone(), xty)?;

    let mu: Array1<f64> = x_loc.dot(&beta_gauss);
    let mut r_over_sigma: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .zip(sigma_per_obs.iter())
        .map(|((&yi, &mi), &si)| (yi - mi) / si)
        .collect();
    r_over_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qi = ((n as f64 - 1.0) * tau).round() as usize;
    let q_z = r_over_sigma[qi.min(n - 1)];

    let shift: Array1<f64> = sigma_per_obs.iter().map(|&si| si * q_z).collect();
    let xt_shift = x_loc.t().dot(&shift);
    let delta_beta = solve(a_init, xt_shift)?;
    let beta_loc_init = &beta_gauss + &delta_beta;

    // ── Per-obs-σ ELF IRLS for β_loc. ──
    let (beta_loc, eta_loc, deviance, iter, converged) = fit_pirls_quantile_perobs_sigma(
        y,
        x_loc,
        s_loc_total,
        &sigma_per_obs,
        tau,
        &beta_loc_init,
        max_iter,
        tolerance,
    )?;

    let eta_scale: Array1<f64> = sigma_per_obs.iter().map(|&s| s.ln()).collect();

    // β_scale not estimated here — the σ comes from the external gaulss
    // preprocessing in Python. Return an empty β_scale; Python keeps the
    // gaulss GAM around for prediction.
    let result = PiRLSResultLSS {
        coefficients_loc: beta_loc,
        coefficients_scale: Array1::zeros(0),
        eta_loc,
        eta_scale,
        sigma: sigma_per_obs,
        deviance,
        iterations: iter,
        converged,
        lambda_loc: Vec::new(),
        fs_iterations: 0,
    };
    Ok((result, sigma_global))
}

/// Fit `fit_pirls_quantile_lss` with a Fellner-Schall outer loop that
/// re-tunes per-smooth λ_loc under the per-obs-σ ELF likelihood, instead
/// of inheriting the lambdas from the Gaussian-init GAM.
///
/// The outer loop alternates:
///   - **Inner IRLS** at fixed λ via [`fit_pirls_quantile_perobs_sigma`].
///   - **FS update** (Wood & Fasiolo 2017) at the converged β:
///     `λ_new = λ · φ · max(rank_i / λ_i − tr(A⁻¹ S_i), ε) / (β' S_i β)`
///     with `A = X' W X + Σ λ_j S_j` and `W` the per-obs ELF IRLS weights
///     `s(1−s)/σ²` from the inner converged state. φ = 1 (ELF dispersion
///     is fixed, σ is the family parameter).
///
/// The per-obs σ rescaling is identical to the fixed-λ path — the σ_global
/// auto-heuristic still uses `varHat = mean(σ_G)²` and tail widening at
/// extreme τ.
///
/// When the FS gradient stagnates (max log-λ change < `fs_tolerance`) the
/// outer loop exits early. A final IRLS pass at the tuned λs guarantees
/// `(β, η, σ̂)` are consistent with the returned `lambda_loc`.
///
/// Returns `(result, sigma_global_used)` like the fixed-λ entry point;
/// `result.lambda_loc` carries the tuned per-smooth λs and
/// `result.fs_iterations` the number of FS sweeps actually run.
pub fn fit_pirls_quantile_lss_fs_tune(
    y: &Array1<f64>,
    x_loc: &Array2<f64>,
    penalties_loc: &[BlockPenalty],
    lambda_init: &[f64],
    sigma_g_per_obs: &Array1<f64>,
    sigma_global: Option<f64>,
    tau: f64,
    max_outer: usize,
    max_inner: usize,
    tolerance: f64,
) -> Result<(PiRLSResultLSS, f64)> {
    use ndarray_linalg::{Cholesky, InverseInto, UPLO};

    let n = y.len();
    let p_loc = x_loc.ncols();

    if x_loc.nrows() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "X_loc has {} rows but y has {} elements",
            x_loc.nrows(),
            n
        )));
    }
    if sigma_g_per_obs.len() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "sigma_g_per_obs has {} elements but y has {} elements",
            sigma_g_per_obs.len(),
            n
        )));
    }
    if tau <= 0.0 || tau >= 1.0 {
        return Err(GAMError::InvalidParameter(format!(
            "quantile tau must be in (0, 1), got {}",
            tau
        )));
    }
    if penalties_loc.len() != lambda_init.len() {
        return Err(GAMError::DimensionMismatch(format!(
            "penalties_loc has {} entries but lambda_init has {}",
            penalties_loc.len(),
            lambda_init.len()
        )));
    }
    for pen in penalties_loc {
        if pen.total_size != p_loc {
            return Err(GAMError::DimensionMismatch(format!(
                "BlockPenalty.total_size={} != p_loc={}",
                pen.total_size, p_loc
            )));
        }
    }

    // ── σ_global + per-obs σ — identical to the fixed-λ path. ──
    let sigma_g_mean: f64 = sigma_g_per_obs.iter().copied().sum::<f64>() / (n as f64).max(1.0);
    let sigma_g_mean = sigma_g_mean.max(1e-8);
    let sigma_global = sigma_global.unwrap_or_else(|| {
        let err = 0.05_f64;
        let var_hat = sigma_g_mean * sigma_g_mean;
        let base = err * (2.0 * std::f64::consts::PI * var_hat).sqrt() / (2.0 * 2.0_f64.ln());
        let tail_scale = (1.0 / (4.0 * tau * (1.0 - tau))).max(1.0);
        base * tail_scale
    });
    let sigma_per_obs: Array1<f64> = sigma_g_per_obs
        .iter()
        .map(|&sg| sigma_global * sg.max(1e-8) / sigma_g_mean)
        .collect();

    // Precompute penalty ranks (used in FS rank/λ term).
    let mut penalty_ranks = Vec::with_capacity(penalties_loc.len());
    for pen in penalties_loc {
        let r = crate::reml::estimate_rank_eigen(pen) as f64;
        penalty_ranks.push(r.max(1.0));
    }

    let num_penalties_proxy = (penalties_loc.len() as f64).max(1.0);
    let ridge_scale = if std::env::var("MGCV_EXACT_FIT").is_ok() {
        1e-12
    } else {
        1e-5 * (1.0 + num_penalties_proxy.sqrt())
    };

    // ── Initial β via Gaussian-then-quantile-shift warm start at λ_init. ──
    let mut lambdas = lambda_init.to_vec();
    let mut s_total = Array2::<f64>::zeros((p_loc, p_loc));
    for (lam, pen) in lambdas.iter().zip(penalties_loc.iter()) {
        pen.scaled_add_to(&mut s_total, *lam);
    }

    let xtx = compute_xtwx_dispatch(None, x_loc, &Array1::ones(n));
    let mut a_init = &xtx + &s_total;
    let mut md: f64 = 1.0;
    for i in 0..p_loc {
        md = md.max(a_init[[i, i]].abs());
    }
    for i in 0..p_loc {
        a_init[[i, i]] += ridge_scale * md;
    }
    let beta_gauss = solve(a_init.clone(), x_loc.t().dot(y))?;
    let mu: Array1<f64> = x_loc.dot(&beta_gauss);
    let mut r_over_sigma: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .zip(sigma_per_obs.iter())
        .map(|((&yi, &mi), &si)| (yi - mi) / si)
        .collect();
    r_over_sigma.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let qi = ((n as f64 - 1.0) * tau).round() as usize;
    let q_z = r_over_sigma[qi.min(n - 1)];
    let shift: Array1<f64> = sigma_per_obs.iter().map(|&si| si * q_z).collect();
    let delta_beta = solve(a_init, x_loc.t().dot(&shift))?;
    let mut beta_loc = &beta_gauss + &delta_beta;

    // ── FS outer loop. ──
    //
    // Each sweep: (1) IRLS at the current λ to convergence; (2) FS update
    // at the converged β via the shared `smooth::fellner_schall_step`
    // helper (same formula as the GAM outer loop's FS path).
    let fs_tolerance = 1e-3_f64; // log-λ tolerance for outer convergence
    let mut fs_iter = 0usize;

    for outer in 0..max_outer.max(1) {
        fs_iter = outer + 1;

        // Build penalty total at current λ.
        let mut s_cur = Array2::<f64>::zeros((p_loc, p_loc));
        for (lam, pen) in lambdas.iter().zip(penalties_loc.iter()) {
            pen.scaled_add_to(&mut s_cur, *lam);
        }

        let (beta_new, eta_new, _dev_new, _iter_n, _conv_n) = fit_pirls_quantile_perobs_sigma(
            y,
            x_loc,
            &s_cur,
            &sigma_per_obs,
            tau,
            &beta_loc,
            max_inner,
            tolerance,
        )?;
        beta_loc = beta_new;

        // IRLS weights at converged β: w_i = s(1-s)/σ², σ = sigma_per_obs[i].
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let inv_si = 1.0 / sigma_per_obs[i];
            let r = y[i] - eta_new[i];
            let s = if r * inv_si >= 0.0 {
                1.0 / (1.0 + (-r * inv_si).exp())
            } else {
                let e = (r * inv_si).exp();
                e / (1.0 + e)
            };
            w[i] = s * (1.0 - s) * inv_si * inv_si;
        }
        let xtwx = compute_xtwx_dispatch(None, x_loc, &w);

        // A = X'WX + Σ λ_j S_j + ridge ⇒ Cholesky ⇒ A⁻¹.
        let mut a = &xtwx + &s_cur;
        let mut max_diag: f64 = 1.0;
        for i in 0..p_loc {
            max_diag = max_diag.max(a[[i, i]].abs());
        }
        let ridge = ridge_scale * max_diag;
        for i in 0..p_loc {
            a[[i, i]] += ridge;
        }
        let chol = match a.cholesky(UPLO::Lower) {
            Ok(l) => l,
            Err(_) => {
                let extra = 1e-3 * max_diag;
                for i in 0..p_loc {
                    a[[i, i]] += extra;
                }
                a.cholesky(UPLO::Lower)
                    .map_err(|_| GAMError::SingularMatrix)?
            }
        };
        let a_inv = chol.inv_into().map_err(|_| GAMError::SingularMatrix)?;

        // φ = 1 for ELF (σ is the family parameter, dispersion fixed at 1).
        let new_lambdas = crate::smooth::fellner_schall_step(
            penalties_loc,
            &penalty_ranks,
            &lambdas,
            &a_inv,
            &beta_loc,
            /*phi=*/ 1.0,
            /*log_step_clamp=*/ 3.0,
            /*lambda_bounds=*/ (1e-9, 1e7),
        );

        // Track max |Δlog λ| for the convergence check.
        let max_log_step = lambdas
            .iter()
            .zip(new_lambdas.iter())
            .map(|(&old, &new_lam)| (new_lam.ln() - old.ln()).abs())
            .fold(0.0_f64, f64::max);
        lambdas = new_lambdas;

        if max_log_step < fs_tolerance {
            break;
        }
    }

    // ── Final IRLS at tuned λ, so β / η / dev are consistent with lambda_loc. ──
    let mut s_final = Array2::<f64>::zeros((p_loc, p_loc));
    for (lam, pen) in lambdas.iter().zip(penalties_loc.iter()) {
        pen.scaled_add_to(&mut s_final, *lam);
    }
    let (beta_loc, eta_loc, deviance, iter_final, converged_final) =
        fit_pirls_quantile_perobs_sigma(
            y,
            x_loc,
            &s_final,
            &sigma_per_obs,
            tau,
            &beta_loc,
            max_inner,
            tolerance,
        )?;

    let eta_scale: Array1<f64> = sigma_per_obs.iter().map(|&s| s.ln()).collect();
    let result = PiRLSResultLSS {
        coefficients_loc: beta_loc,
        coefficients_scale: Array1::zeros(0),
        eta_loc,
        eta_scale,
        sigma: sigma_per_obs,
        deviance,
        iterations: iter_final,
        converged: converged_final,
        lambda_loc: lambdas,
        fs_iterations: fs_iter,
    };
    Ok((result, sigma_global))
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
    disc: &DiscreteDesign,
    cached_xtx: Option<&Array2<f64>>,
    prior_weights: Option<&Array1<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = x.ncols();

    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string(),
        ));
    }

    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "prior_weights length ({}) must match y length ({})",
                pw.len(),
                n
            )));
        }
    }

    // Fast path for Gaussian family
    if matches!(family, Family::Gaussian) {
        return fit_pirls_gaussian_discretized(y, x, lambda, penalties, p, disc, cached_xtx, prior_weights);
    }

    // General IRLS path for non-Gaussian families with discretized X'WX
    let mut beta = Array1::zeros(p);
    let mut eta = compute_eta_discrete(disc, &beta);

    // Initialize eta based on family
    for i in 0..n {
        let safe_y = match family {
            Family::Binomial | Family::QuasiBinomial => y[i].max(0.01).min(0.99),
            Family::Poisson
            | Family::QuasiPoisson
            | Family::Gamma
            | Family::GammaLog
            | Family::Tweedie { .. }
            | Family::InverseGaussian
            | Family::NegBin { .. } => y[i].max(0.1),
            // Identity-link families: initialize η = y directly.
            Family::Gaussian | Family::TDist { .. } | Family::Quantile { .. } => y[i],
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

        // GLM IRLS working pair with prior weights folded in. R4-refactor:
        // see `exp_family_irls_step`. The discretized path historically
        // uses `fisher_only=true` (canonical-link families dominate the
        // discrete hot path, and Fisher gives byte-identical results to
        // the Newton fallback there).
        let step = exp_family_irls_step(
            y.view(),
            eta.view(),
            prior_weights.map(|pw| pw.view()),
            family,
            /* use_fisher = */ true,
        );
        let z = step.z;
        let w = step.w;

        // X'WX via scatter-gather: O(n*k + m*k^2) instead of O(n*k^2)
        let xtwx = compute_xtwx_discrete(disc, &w);

        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        let mut a = xtwx + &penalty_total;
        let ridge: f64 = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // X'Wz via scatter-gather. compute_xtwy_discrete computes
        // X̃'·(Σ_i w_i·z_i·1[k=μ]) which is exactly X'Wz when its `w`
        // arg is the working weights and `y` is z — i.e. the scatter
        // forms `Σ_i w_i · z_i` per bin, then gathers via X̃'. This is
        // equivalent to the un-binned X'·diag(w)·z.
        let xtwz = compute_xtwy_discrete(disc, &w, &z);

        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // eta via compressed gather
        eta = compute_eta_discrete(disc, &beta);

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
    let deviance = match prior_weights {
        Some(pw) => compute_weighted_deviance(y, &fitted_values, family, pw),
        None => compute_deviance(y, &fitted_values, family),
    };

    // Final-scoring pass: pure Fisher working pair for the REML contract.
    // R4-refactor: see `exp_family_irls_step`. Mirrors the dense path's
    // final-scoring assembly.
    let final_step = exp_family_irls_step(
        y.view(),
        eta.view(),
        prior_weights.map(|pw| pw.view()),
        family,
        /* use_fisher = */ true,
    );

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights: final_step.w,
        working_response: final_step.z,
        deviance,
        iterations: iter,
        converged,
        sigma2: None,
        df: None,
    })
}

/// Gaussian fast path using discretized design.
///
/// Combines the Gaussian 1-step solve with scatter-gather X'X computation.
fn fit_pirls_gaussian_discretized(
    y: &Array1<f64>,
    _x: &Array2<f64>,
    lambda: &[f64],
    penalties: &[BlockPenalty],
    p: usize,
    disc: &DiscreteDesign,
    cached_xtx: Option<&Array2<f64>>,
    prior_weights: Option<&Array1<f64>>,
) -> Result<PiRLSResult> {
    let n = y.len();

    // X'WX: with W = diag(w) when prior weights are supplied, the
    // cached X'X (built with W=I) can't be reused.
    let xtwx = match prior_weights {
        None => {
            if let Some(cached) = cached_xtx {
                cached.clone()
            } else {
                let ones = Array1::ones(n);
                compute_xtwx_discrete(disc, &ones)
            }
        }
        Some(w) => compute_xtwx_discrete(disc, w),
    };

    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(xtwx[[i, i]].abs());
    }

    let mut a = xtwx;
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

    // X'Wy via scatter-gather (W=I when unweighted).
    let xtwy = match prior_weights {
        None => {
            let ones = Array1::ones(n);
            compute_xtwy_discrete(disc, &ones, y)
        }
        Some(w) => compute_xtwy_discrete(disc, w, y),
    };

    let beta = solve(a, xtwy)?;

    // eta via compressed gather
    let eta = compute_eta_discrete(disc, &beta);
    let fitted_values = eta.clone();

    let deviance: f64 = match prior_weights {
        None => y
            .iter()
            .zip(fitted_values.iter())
            .map(|(yi, fi)| (yi - fi).powi(2))
            .sum(),
        Some(w) => y
            .iter()
            .zip(fitted_values.iter())
            .zip(w.iter())
            .map(|((yi, fi), wi)| wi * (yi - fi).powi(2))
            .sum(),
    };

    let weights = match prior_weights {
        None => Array1::ones(n),
        Some(w) => w.clone(),
    };

    Ok(PiRLSResult {
        coefficients: beta,
        fitted_values,
        linear_predictor: eta,
        weights,
        working_response: y.clone(),
        deviance,
        iterations: 1,
        converged: true,
        sigma2: None,
        df: None,
    })
}

// ---------------------------------------------------------------------------
// B4 — fREML outer driver (port of mgcv's `bgam.fitd`).
// ---------------------------------------------------------------------------

/// Callback hook used by [`fit_pirls_fastreml`] to refresh extended-family
/// shape parameters (e.g. scat's `(σ², df)`) between outer iterations.
///
/// Given the current iterate (`y`, `μ̂`, optional prior weights, the latest
/// family, and the latest `log_phi`), the callback returns an updated
/// [`Family`] with refreshed shape params. For Gaussian / canonical-link
/// exponential families the callback is `None` and the family is held fixed
/// across iterations.
///
/// Mirrors mgcv's `estimate.theta` callback inside `bgam.fitd`
/// (R/bam.r:614-630). See `docs/B4_DESIGN.md` §2.
#[cfg(feature = "blas")]
pub type FastRemlThetaCallback<'a> = &'a mut dyn FnMut(
    &Array1<f64>,
    &Array1<f64>,
    Option<&Array1<f64>>,
    Family,
    f64,
) -> Result<Family>;

/// Configuration for the fREML outer driver. See [`fit_pirls_fastreml`].
#[cfg(feature = "blas")]
pub struct FastRemlConfig<'a> {
    /// Outer-iteration cap (mgcv `control$maxit`, default 200).
    pub max_outer_iter: usize,
    /// Outer-loop convergence tolerance (mgcv `control$epsilon`, default 1e-7).
    pub tol: f64,
    /// γ correction factor (default 1.0).
    pub gamma: f64,
    /// True for known-scale families (Binomial, Poisson, NegBin); false for
    /// Gaussian, Gamma, scat, Tweedie, InvGauss.
    pub phi_fixed: bool,
    /// Initial `log φ`. `None` ⇒ seed from `log(var(y) * 0.05)`.
    pub log_phi_init: Option<f64>,
    /// Optional shape-parameter callback (e.g. scat θ Newton). `None` ⇒ no-op.
    pub theta_callback: Option<FastRemlThetaCallback<'a>>,
}

#[cfg(feature = "blas")]
impl<'a> FastRemlConfig<'a> {
    /// Defaults matching mgcv's `bgam.fitd` control args for `method='fREML'`.
    pub fn default_for(phi_fixed: bool) -> Self {
        FastRemlConfig {
            max_outer_iter: 200,
            tol: 1e-7,
            gamma: 1.0,
            phi_fixed,
            log_phi_init: None,
            theta_callback: None,
        }
    }
}

/// Returned by [`fit_pirls_fastreml`].
#[cfg(feature = "blas")]
#[derive(Debug, Clone)]
pub struct FastRemlResult {
    pub beta: Array1<f64>,
    pub lambda: Vec<f64>,
    pub log_phi: f64,
    pub sigma2: f64,
    pub pp: Array2<f64>,
    pub edf: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub linear_predictor: Array1<f64>,
    pub final_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub deviance: f64,
    pub gcv_ubre: f64,
    pub iterations: usize,
    pub converged: bool,
    pub family_out: Family,
    pub db_drho: Array2<f64>,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
}

/// Compute η = X·β, dispatching to the discrete compressed-basis path when
/// a `DiscreteDesign` is supplied.
#[cfg(feature = "blas")]
#[inline]
fn fastreml_compute_eta(
    x: &Array2<f64>,
    discrete: Option<&crate::discrete::DiscreteDesign>,
    beta: &Array1<f64>,
) -> Array1<f64> {
    match discrete {
        Some(d) => crate::discrete::compute_eta_discrete(d, beta),
        None => x.dot(beta),
    }
}

/// Compute one IRLS working pair `(w, z)` for the fREML loop.
///
/// Dispatches on `family`:
///   - `Gaussian`: `w = 1` (folded with prior weights when supplied), `z = y`.
///   - `TDist`: scat IRLS via [`tdist_irls_step`].
///   - all other exponential families: [`exp_family_irls_step`] with Fisher
///     scoring (matches mgcv's `bgam.fitd` working-pair contract).
#[cfg(feature = "blas")]
fn fastreml_irls_step(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    family: Family,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    match family {
        Family::Gaussian => {
            let w = match prior_weights {
                Some(pw) => pw.clone(),
                None => Array1::<f64>::ones(n),
            };
            (w, y.clone())
        }
        Family::TDist { df, sigma2 } => {
            // mgcv `bgam.fitd` (bam.r:635-640) builds the IRLS pair from
            // `dDeta(y, μ, G$w, θ, family, 0)`. For `rho == 0` it uses
            // **observed-info** `Dmu2·0.5`; for AR1 (which we don't support)
            // it uses Fisher `EDmu2·0.5`. R7 used Fisher unconditionally
            // because plain LU + ridge in `compute_sl_fitchol_step` could
            // not handle the indefinite X'WX that observed-info produces.
            //
            // **R8 swap**: with pivoted Chol + rank-revealing dpstrf in
            // `compute_sl_fitchol_step`, indefinite/rank-deficient X'WX is
            // tolerated by truncating the null block of the (preconditioned)
            // factor. We can now match mgcv's `rho == 0` observed-info path,
            // which closes the residual ~23% rel λ gap from R7.
            //
            // **Per-row Fisher fallback** (mirrors mgcv efam.r:3641): when
            // a row's `D''_η` is non-finite (NaN/inf from extreme residuals),
            // substitute the row's expected-info value. Negative-but-finite
            // weights are kept verbatim — the pivoted-Chol solve handles them.
            //
            // **Whole-step fallback**: if MORE than half the rows yield
            // non-finite observed-info simultaneously (a pathological
            // β iterate), we switch the entire step to Fisher. This keeps
            // R7's good behavior on adversarial inputs without sacrificing
            // mgcv parity on well-conditioned scat fixtures.
            let mut w = Array1::<f64>::zeros(n);
            let mut z = Array1::<f64>::zeros(n);
            let mut bad_rows = 0usize;
            for i in 0..n {
                let r = y[i] - eta[i];
                let (wi, zi) = t_bgam_observed_info_pair(r, eta[i], sigma2, df);
                if wi.is_finite() && zi.is_finite() {
                    w[i] = wi;
                    z[i] = zi;
                } else {
                    bad_rows += 1;
                    // Leave w[i] = z[i] = 0 — bam.r:642-644 also does this.
                }
            }
            if bad_rows * 2 > n {
                // Pathological iterate: fall back to all-Fisher for this step.
                w.fill(0.0);
                z.fill(0.0);
                for i in 0..n {
                    let r = y[i] - eta[i];
                    let (wi, zi) = t_bgam_fisher_pair(r, eta[i], sigma2, df);
                    if wi.is_finite() && zi.is_finite() {
                        w[i] = wi;
                        z[i] = zi;
                    }
                }
            }
            if let Some(pw) = prior_weights {
                for i in 0..n {
                    w[i] *= pw[i];
                }
            }
            (w, z)
        }
        _ => {
            let step = exp_family_irls_step(
                y.view(),
                eta.view(),
                prior_weights.map(|pw| pw.view()),
                family,
                /*use_fisher*/ true,
            );
            (step.w, step.z)
        }
    }
}

/// Outer-driver port of mgcv's `bgam.fitd` (`method='fREML'`).
///
/// **Algorithmic mapping** (per `docs/B4_DESIGN.md` §3):
///
///   1. Initialise β = 0, η from family, `log φ` from sample variance.
///   2. Initialise smoothing params via [`crate::gam_optimized::initialize_lambda_smart`]
///      (mgcv's `initial.sp` analog at bam.r:687).
///   3. **Outer loop** (`bgam.fitd` line 563):
///      a. Take one IRLS step → `(w, z)` (`fastreml_irls_step`).
///      b. Form `X'WX`, `X'Wz`, `y'Wy` (or `z'Wz` — see note below) via the
///         discrete/dense dispatch helpers.
///      c. Call [`crate::reml::compute_sl_fitchol_step`] to get the
///         closed-form `(β̂, grad, Hess, step)` on `(ρ, log φ)`.
///      d. Apply mgcv's step-blending (bam.r:749-756): if the proposed
///         Newton step is uphill on REML (`g·step > dev·1e-7`), halve and
///         re-evaluate `Sl.fitChol`.
///      e. Refresh family θ via the callback (scat: 2-D Newton on
///         `(log σ², log(df-2))`).
///      f. Test convergence on `(dev change, log φ step)`.
///   4. Return `FastRemlResult` with β, λ, PP, EDF, fitted values, etc.
///
/// **Skipped vs mgcv** (per design §1 gaps, documented at callsite):
///   - `Sl.initial.repara` rotation — predictions/EDF are basis-invariant.
///   - AR1 `rho!=0` — out of scope (driver doesn't take a `rho` arg).
///   - Joint θ–φ mini-loop in iters 1–4 (bam.r:617-628) — `phi_fixed`
///     stays constant per fit.
///   - `coef`/`in.out`/`nei` warm-start args.
///
/// **Note on `z'Wz` vs `y'Wy`**: the closed-form Gaussian-equivalent
/// `D_p = z'Wz − β'X'Wz` substitutes `z = working response` in place of `y`.
/// For Gaussian (`z = y`) the two coincide. For non-Gaussian, mgcv's bam
/// path uses the working response throughout (bam.r:653).
#[cfg(feature = "blas")]
pub fn fit_pirls_fastreml(
    y: &Array1<f64>,
    x: &Array2<f64>,
    prior_weights: Option<&Array1<f64>>,
    sl: &[BlockPenalty],
    initial_family: Family,
    discrete: Option<&crate::discrete::DiscreteDesign>,
    config: &mut FastRemlConfig,
) -> Result<FastRemlResult> {
    use crate::reml::{compute_sl_fitchol_step, compute_xtwx_dispatch, compute_xtwy_dispatch};

    let n = y.len();
    let p = x.ncols();
    let m = sl.len();

    if x.nrows() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "X has {} rows but y has {}",
            x.nrows(),
            n
        )));
    }
    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "prior_weights length {} must match y length {}",
                pw.len(),
                n
            )));
        }
    }
    if m == 0 {
        return Err(GAMError::InvalidParameter(
            "fit_pirls_fastreml requires at least one penalty block".to_string(),
        ));
    }

    // ───── C1: Mp = p − Σ rank_k (mgcv `Sl.setup` Mp output, bam.r:544) ─────
    // Use estimate_rank_eigen — see block_penalty.rs::log_det_singleton_with_derivs
    // for the rationale (row-norm rank overcounts on banded penalties).
    let mut mp_signed: i64 = p as i64;
    for s in sl {
        mp_signed -= crate::reml::estimate_rank_eigen(s) as i64;
    }
    let mp = mp_signed.max(0) as usize;

    // ───── C2/C3: β = 0, η from family, log φ seeded from sample variance ────
    //
    // **Known-φ families (Poisson, Binomial, NegBin with fixed θ)**: mgcv's
    // `bgam.fitd` (bam.r:696) sets `log.phi = log(scale)` when `scale > 0`. The
    // user-facing `scale=1` default for these families means `log.phi = 0`,
    // i.e. `φ ≡ 1`. The score formula `crit = (dev/(φ·γ) − ldetS + ldetXXS)/2`
    // and the gradient term `(rss1+bSb1)/(φ·γ)` are both proportional to `1/φ`,
    // so seeding `log_phi` from the sample-variance heuristic (which is the
    // right initial guess for *Gaussian/Gamma* where φ is estimated) shifts the
    // λ optimum by a `1/φ_seed` factor. Pre-fix on this branch: rust Poisson(log)
    // λ landed at 95 vs mgcv's 160 (41% rel gap); Binomial(logit) at ~100× rel.
    let mut family = initial_family;
    let mut beta = Array1::<f64>::zeros(p);
    let y_mean: f64 = y.iter().sum::<f64>() / (n as f64).max(1.0);
    let y_var: f64 =
        y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let mut log_phi = config.log_phi_init.unwrap_or_else(|| {
        if config.phi_fixed {
            // Known-φ families: φ ≡ 1 per bam.r:696 (`log.phi = log(scale)` with
            // `scale = 1` for Poisson/Binomial/NegBin). Hard-pin to avoid the
            // sample-variance heuristic mis-scaling the score.
            0.0
        } else {
            (y_var * 0.05).max(1e-8).ln()
        }
    });

    // ───── C4: initial smoothing params via mgcv's `initial.sp` analog ───────
    // (bam.r:687). Per-smooth heuristic on the current y / X.
    let mut rho: Vec<f64> = sl
        .iter()
        .map(|s| crate::gam_optimized::initialize_lambda_smart(y, x, s).ln())
        .collect();

    // ───── State carried across the outer loop ───────────────────────────────
    let mut prev_dev = f64::INFINITY;
    let mut prev_crit = f64::INFINITY;
    let mut iter_used = 0usize;
    let mut converged = false;
    let mut last_prop: Option<crate::reml::SlFitCholResult> = None;
    let mut last_log_phi_step: f64 = 0.0;
    // Track the most recent accepted Newton step on (ρ, log φ) for the
    // log-φ convergence test (bam.r:678 second clause).

    // ───── C6: main outer loop (bam.r:563) ────────────────────────────────────
    for outer in 0..config.max_outer_iter {
        iter_used = outer + 1;

        // ─── C7: one IRLS step at current β. For Gaussian this is just
        // (w=prior, z=y). For scat we use tdist_irls_step. For exponential
        // families we use exp_family_irls_step.
        let eta = fastreml_compute_eta(x, discrete, &beta);
        let (w, z) = fastreml_irls_step(y, &eta, prior_weights, family);

        // ─── C9: form X'WX, X'Wz, y_w_y (here z'Wz) (bam.r:657-659). ────────
        let xx = compute_xtwx_dispatch(discrete, x, &w);
        let f = compute_xtwy_dispatch(discrete, x, &w, &z);
        let yy: f64 = w
            .iter()
            .zip(z.iter())
            .map(|(&wi, &zi)| wi * zi * zi)
            .sum();

        // ─── C12: B3 closed-form step on (ρ, log φ). ───────────────────────
        let rho_arr = Array1::from_vec(rho.clone());
        let proposal = compute_sl_fitchol_step(
            sl,
            xx.view(),
            f.view(),
            rho_arr.view(),
            yy,
            log_phi,
            config.phi_fixed,
            n as f64,
            mp,
            config.gamma,
        )?;

        // ─── C13: mgcv-style step-blending. On the first iter we accept the
        // proposal as-is (`Nstep == 0` at start). Subsequently, the bam.r
        // uphill check at line 749-756 is: if grad·step > dev·1e-7, halve
        // the step and re-evaluate Sl.fitChol at the halved point. We keep
        // it simple and use the just-computed `prev_dev` as the scale.
        //
        // Note: the `proposal.step` is already capped at |∞|≤4 by B3, so
        // we don't need a magnitude clamp here.
        let mut accepted_step = proposal.step.clone();
        let mut accepted_prop = proposal;
        if outer > 0 {
            // Uphill test on the proposed Newton step.
            let dev_scale = prev_dev.abs().max(1.0);
            let dot = accepted_prop.grad.dot(&accepted_step);
            if dot > dev_scale * 1e-7 {
                // Halve and re-evaluate. One halving suffices per
                // bam.r:749-756 (mgcv does not iterate; takes one trial).
                accepted_step.mapv_inplace(|x| 0.5 * x);
                let mut trial_rho: Vec<f64> = rho.clone();
                for (k, sk) in accepted_step.iter().take(m).enumerate() {
                    trial_rho[k] += sk;
                }
                let trial_log_phi = if config.phi_fixed {
                    log_phi
                } else {
                    log_phi + accepted_step[m]
                };
                let trial_rho_arr = Array1::from_vec(trial_rho.clone());
                let prop2 = compute_sl_fitchol_step(
                    sl,
                    xx.view(),
                    f.view(),
                    trial_rho_arr.view(),
                    yy,
                    trial_log_phi,
                    config.phi_fixed,
                    n as f64,
                    mp,
                    config.gamma,
                )?;
                accepted_prop = prop2;
            }
        }

        // Apply the (possibly halved) step to (ρ, log φ).
        for (k, sk) in accepted_step.iter().take(m).enumerate() {
            rho[k] += sk;
        }
        if !config.phi_fixed {
            last_log_phi_step = accepted_step[m];
            log_phi += accepted_step[m];
        } else {
            last_log_phi_step = 0.0;
        }

        // ─── C14: update β from the (possibly refreshed) proposal. ──────────
        beta = accepted_prop.beta.clone();
        let _ = (w, z); // IRLS pair consumed via xx/f; final pass recomputes.

        // ─── C15: refresh family-shape params via callback (scat θ Newton). ─
        if let Some(cb) = config.theta_callback.as_mut() {
            let mu_hat = fastreml_compute_eta(x, discrete, &beta);
            family = cb(y, &mu_hat, prior_weights, family, log_phi)?;
        }

        // ─── C17: non-finite β guard (bam.r:761-765). ───────────────────────
        if beta.iter().any(|b| !b.is_finite()) {
            return Err(GAMError::OptimizationFailed(
                "fit_pirls_fastreml: non-finite β in outer Newton step".to_string(),
            ));
        }

        // ─── C16: compute the fREML score at the new (ρ, log φ, β).
        // Per bam.r:767:
        //   crit = (dev/(φ·γ) − ldet_s + ldet_xxs) / 2
        // where dev for Gaussian-like is `yy − β'f` (residual + penalty)
        // and for general families is the working penalised RSS.
        //
        // We use the closed-form Dp = yy − β'f (matches mgcv's `dev`
        // assembly when `z` substitutes for `y`).
        let dev_now = (yy - beta.dot(&f)).max(0.0);
        let phi_now = log_phi.exp().max(1e-300);
        let crit_now =
            (dev_now / (phi_now * config.gamma) - accepted_prop.ldet_s + accepted_prop.ldet_xxs)
                * 0.5;

        // ─── C11: convergence test (bam.r:678). ─────────────────────────────
        if outer >= 2 {
            let rel = (prev_crit - crit_now).abs() / (0.1 + crit_now.abs());
            let log_phi_step_ok =
                config.phi_fixed || last_log_phi_step.abs() < config.tol * (log_phi.abs() + 1.0);
            if rel < config.tol && log_phi_step_ok {
                converged = true;
                last_prop = Some(accepted_prop);
                break;
            }
        }

        prev_dev = dev_now;
        prev_crit = crit_now;
        last_prop = Some(accepted_prop);
    }

    let prop = last_prop.expect("at least one outer iter ran");

    // ───── Post-processing (bam.r:782-895). ─────────────────────────────────
    // Final IRLS step to lock in (w, z) at the converged β so that the
    // returned X'WX (used for EDF) reflects the final state.
    let final_eta = fastreml_compute_eta(x, discrete, &beta);
    let (final_w, final_z) = fastreml_irls_step(y, &final_eta, prior_weights, family);
    let final_xtwx = compute_xtwx_dispatch(discrete, x, &final_w);

    // EDF = diag(PP · X'WX). For singleton blocks this matches mgcv's
    // `object$edf` (bam.r:870-880).
    let f_mat = prop.pp.dot(&final_xtwx);
    let mut edf = Array1::<f64>::zeros(p);
    for i in 0..p {
        edf[i] = f_mat[[i, i]];
    }

    // Fitted values: μ = inv_link(η) (identity for Gaussian / scat).
    let fitted_values: Array1<f64> = final_eta.iter().map(|&e| family.inverse_link(e)).collect();
    let deviance = compute_deviance(y, &fitted_values, family);

    Ok(FastRemlResult {
        beta,
        lambda: rho.iter().map(|r| r.exp()).collect(),
        log_phi,
        sigma2: log_phi.exp(),
        pp: prop.pp,
        edf,
        fitted_values,
        linear_predictor: final_eta,
        final_weights: final_w,
        working_response: final_z,
        deviance,
        gcv_ubre: prev_crit,
        iterations: iter_used,
        converged,
        family_out: family,
        db_drho: prop.db,
        grad: prop.grad,
        hess: prop.hess,
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
                assert_close(
                    analytic,
                    numeric,
                    1e-6,
                    &format!("dvar {:?} mu={}", fam, mu),
                );
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
                assert_close(
                    analytic,
                    numeric,
                    1e-6,
                    &format!("d2var {:?} mu={}", fam, mu),
                );
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
                let numeric = (fam.link(mu + h) - 2.0 * fam.link(mu) + fam.link(mu - h)) / (h * h);
                assert_close(
                    analytic,
                    numeric,
                    1e-3,
                    &format!("d2link {:?} mu={}", fam, mu),
                );
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
                assert_close(
                    analytic,
                    numeric,
                    1e-3,
                    &format!("d3link {:?} mu={}", fam, mu),
                );
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
        assert_eq!(
            Family::Poisson.estimate_phi_mgcv(&y, 50.0, 3, 1.0, 2.0),
            1.0
        );
        assert_eq!(
            Family::Binomial.estimate_phi_mgcv(&y, 50.0, 3, 1.0, 2.0),
            1.0
        );
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
        println!(
            "Gamma phi = {:.10}, F(phi) = {:.3e}, tol = {:.3e}",
            phi, f_at_phi, tol
        );
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

    /// `tdist_irls_step` per-row math vs hand-computed reference (R2 refactor
    /// guard). Fixture-free synthetic 50-point check covering:
    ///   - EM branch (use_newton_working = false): w_i = (df+1)/(df + r²/σ²),
    ///     z_i = y_i; prior weights multiplied into w.
    ///   - Newton branch (use_newton_working = true): observed-info w/z from
    ///     `scat$Dd` with expected-info fallback when curvature is non-PSD;
    ///     prior weights multiplied into w.
    /// Tolerance is 1e-12 absolute (pure scalar arithmetic, no BLAS).
    #[test]
    fn test_tdist_irls_step_hand_computed() {
        let n = 50usize;
        let sigma2 = 1.7_f64;
        let df = 5.5_f64;

        // Deterministic synthetic y/eta with a mix of small and large residuals
        // so both Newton branches (observed-PSD when r²<df·σ², expected-info
        // fallback when r²>df·σ², ≈3.06) are exercised. Indices alternate
        // sign + magnitude.
        let y: Array1<f64> = (0..n)
            .map(|i| 0.1 * (i as f64) - 0.07 * ((i * i) as f64))
            .collect();
        let eta: Array1<f64> = (0..n)
            .map(|i| {
                let base = 0.1 * (i as f64) - 0.07 * ((i * i) as f64);
                // Every 5th row gets a 4.5-magnitude perturbation so |r| > √(df·σ²);
                // others stay small to exercise the observed-PSD branch.
                let pert = if i % 5 == 0 { 4.5 } else { 0.25 } * ((i as f64).sin());
                base + pert
            })
            .collect();
        // Non-trivial prior weights (exclude 1.0 sentinel to verify the
        // multiplication path is hit).
        let pw: Array1<f64> = (0..n).map(|i| 0.5 + 0.03 * (i as f64)).collect();

        // ── EM branch ─────────────────────────────────────────────────────
        let step_em = tdist_irls_step(
            y.view(),
            eta.view(),
            Some(pw.view()),
            sigma2,
            df,
            /* use_newton_working = */ false,
        );
        for i in 0..n {
            let r = y[i] - eta[i];
            let w_t = ((df + 1.0) / (df + r * r / sigma2)).max(1e-10);
            let w_expected = w_t * pw[i];
            let z_expected = y[i];
            assert!(
                (step_em.w[i] - w_expected).abs() < 1e-12,
                "EM w[{}]: got {}, want {}",
                i,
                step_em.w[i],
                w_expected
            );
            assert!(
                (step_em.z[i] - z_expected).abs() < 1e-12,
                "EM z[{}]: got {}, want {}",
                i,
                step_em.z[i],
                z_expected
            );
        }

        // ── Newton branch ─────────────────────────────────────────────────
        let step_n = tdist_irls_step(
            y.view(),
            eta.view(),
            Some(pw.view()),
            sigma2,
            df,
            /* use_newton_working = */ true,
        );
        for i in 0..n {
            let r = y[i] - eta[i];
            let denom = df * sigma2 + r * r;
            let dmu = -2.0 * (df + 1.0) * r / denom;
            let observed_dmu2 = 2.0 * (df + 1.0) * (df * sigma2 - r * r) / (denom * denom);
            let expected_dmu2 = 2.0 * (df + 1.0) / (sigma2 * (df + 3.0));
            let dmu2 = if observed_dmu2.is_finite() && observed_dmu2 > 1e-12 {
                observed_dmu2
            } else {
                expected_dmu2.max(1e-12)
            };
            let w_n = (0.5 * dmu2).max(1e-10);
            let z_expected = eta[i] - dmu / dmu2;
            let w_expected = w_n * pw[i];
            assert!(
                (step_n.w[i] - w_expected).abs() < 1e-12,
                "Newton w[{}]: got {}, want {} (r={})",
                i,
                step_n.w[i],
                w_expected,
                r
            );
            assert!(
                (step_n.z[i] - z_expected).abs() < 1e-12,
                "Newton z[{}]: got {}, want {} (r={})",
                i,
                step_n.z[i],
                z_expected,
                r
            );
        }

        // Sanity: at least one row should have hit the expected-info fallback
        // (negative observed curvature when r² > df·σ²).
        let mut hit_fallback = false;
        for i in 0..n {
            let r = y[i] - eta[i];
            if r * r > df * sigma2 {
                hit_fallback = true;
                break;
            }
        }
        assert!(
            hit_fallback,
            "fixture should exercise the expected-info fallback at least once"
        );
    }

    /// `exp_family_irls_step` per-row math vs hand-computed reference (R4
    /// refactor guard). Fixture-free synthetic 100-point Poisson check.
    ///
    /// Poisson(log) is canonical, so `compute_irls_wz` runs the pure Fisher
    /// branch in both `use_fisher=true` and `use_fisher=false` calls (the
    /// helper internally OR's `use_fisher` with `is_canonical_link()`).
    /// The reference math is:
    ///   - μ_i = exp(η_i)
    ///   - dμ/dη_i = exp(η_i) = μ_i
    ///   - V(μ_i) = μ_i
    ///   - w_iᶠ = (dμ/dη)² / V = μ_i
    ///   - z_i  = η_i + (y_i − μ_i) / (dμ/dη) = η_i + (y_i − μ_i)/μ_i
    /// Prior weights multiply w but never z.
    /// Tolerance is 1e-12 absolute (pure scalar arithmetic, no BLAS).
    #[test]
    fn test_exp_family_irls_step_hand_computed_poisson() {
        let n = 100usize;
        let family = Family::Poisson;

        // Deterministic y/η. y must be ≥ 0 integers in spirit (Poisson counts);
        // we still pass floats because the IRLS step only sees y as a float.
        // η spans a moderate range so μ = exp(η) covers ~0.1 .. ~20.
        let y: Array1<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let eta: Array1<f64> = (0..n)
            .map(|i| -2.0 + 0.05 * (i as f64) + 0.3 * ((i as f64) * 0.1).sin())
            .collect();
        let pw: Array1<f64> = (0..n).map(|i| 0.4 + 0.02 * (i as f64)).collect();

        // ── With prior weights ────────────────────────────────────────────
        for &use_fisher in &[true, false] {
            let step = exp_family_irls_step(
                y.view(),
                eta.view(),
                Some(pw.view()),
                family,
                use_fisher,
            );
            for i in 0..n {
                let mu = eta[i].exp();
                // Fisher (canonical link): w = μ, z = η + (y − μ)/μ
                let w_expected = mu * pw[i];
                let z_expected = eta[i] + (y[i] - mu) / mu;
                assert!(
                    (step.w[i] - w_expected).abs() < 1e-12,
                    "Poisson w[{}] (use_fisher={}): got {}, want {} (mu={})",
                    i, use_fisher, step.w[i], w_expected, mu
                );
                assert!(
                    (step.z[i] - z_expected).abs() < 1e-12,
                    "Poisson z[{}] (use_fisher={}): got {}, want {} (mu={})",
                    i, use_fisher, step.z[i], z_expected, mu
                );
            }
        }

        // ── Without prior weights ─────────────────────────────────────────
        let step_nopw =
            exp_family_irls_step(y.view(), eta.view(), None, family, /* use_fisher */ true);
        for i in 0..n {
            let mu = eta[i].exp();
            let w_expected = mu;
            let z_expected = eta[i] + (y[i] - mu) / mu;
            assert!(
                (step_nopw.w[i] - w_expected).abs() < 1e-12,
                "Poisson(no pw) w[{}]: got {}, want {}",
                i, step_nopw.w[i], w_expected
            );
            assert!(
                (step_nopw.z[i] - z_expected).abs() < 1e-12,
                "Poisson(no pw) z[{}]: got {}, want {}",
                i, step_nopw.z[i], z_expected
            );
        }
    }
}
