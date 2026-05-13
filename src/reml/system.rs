//! Shared REML system primitives: X′WX / X′Wy assembly, the joint
//! linear system A·β = X′Wy (with A = X′WX + Σλ_jS_j), GLM deviance,
//! and the family-aware score-formula scalars used by the criterion,
//! gradient, and Hessian paths.
//!
//! These items were factored out of `reml/mod.rs` in May 2026 to keep
//! the inner-loop primitives in one place that R2 (the reparametrization
//! port) can adapt to operate on rotated `X / β / rS`.

use crate::block_penalty::BlockPenalty;
use crate::linalg::{inverse, solve};
use crate::GAMError;
use crate::Result;
use ndarray::{Array1, Array2};

/// Method for computing the scale parameter φ in REML
///
/// The scale parameter φ = RSS / (n - df) affects the Hessian scaling
/// and convergence behavior. Two methods are available:
///
/// - `Rank`: Uses penalty matrix ranks (constant, O(1) per iteration)
///   φ = RSS / (n - Σ rank(Sᵢ))
///   Fast but approximate; can cause issues when k >> n
///
/// - `EDF`: Uses Effective Degrees of Freedom (O(p³/3) per iteration)
///   φ = RSS / (n - EDF) where EDF = tr(A⁻¹·X'WX)
///   Exact method matching mgcv; better for ill-conditioned problems
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ScaleParameterMethod {
    /// Use penalty matrix ranks (fast, approximate)
    /// φ = RSS / (n - Σ rank(Sᵢ))
    #[default]
    Rank,
    /// Use Effective Degrees of Freedom (slower, exact)
    /// φ = RSS / (n - EDF) where EDF = tr(A⁻¹·X'WX)
    EDF,
}

/// Compute Effective Degrees of Freedom using the trace-Frobenius trick
///
/// EDF = tr(A⁻¹·X'WX)
///
/// Using Cholesky A = R'R:
/// EDF = tr(R⁻¹·R'⁻¹·X'WX) = ||R'⁻¹·L||²_F
/// where X'WX = L·L' (Cholesky factorization of X'WX)
///
/// # Arguments
/// * `r_t` - R' (transpose of Cholesky factor of A, lower triangular)
/// * `xtwx_chol` - Cholesky factor L of X'WX (lower triangular)
///
/// # Returns
/// EDF value (sum of squared elements of R'⁻¹·L)
#[cfg(feature = "blas")]
pub fn compute_edf_from_cholesky(r_t: &Array2<f64>, xtwx_chol: &Array2<f64>) -> Result<f64> {
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};

    // Solve R'·Y = L where L is the Cholesky factor of X'WX
    // R' is lower triangular, L is lower triangular
    let sol = r_t
        .solve_triangular(UPLO::Lower, Diag::NonUnit, xtwx_chol)
        .map_err(|e| GAMError::InvalidParameter(format!("EDF triangular solve failed: {:?}", e)))?;

    // EDF = ||Y||²_F = sum of all squared elements
    let edf: f64 = sol.iter().map(|x| x * x).sum();

    Ok(edf)
}

/// Compute Cholesky factor of X'WX for EDF computation
///
/// This should be pre-computed once at the start of optimization
/// since X'WX doesn't change during lambda optimization.
#[cfg(feature = "blas")]
pub fn compute_xtwx_cholesky(xtwx: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::{Cholesky, UPLO};

    // Add small ridge for numerical stability (X'WX might be ill-conditioned)
    let p = xtwx.nrows();
    let mut xtwx_reg = xtwx.clone();
    let max_diag = (0..p).map(|i| xtwx[[i, i]].abs()).fold(0.0f64, f64::max);
    let ridge = max_diag * 1e-10;
    for i in 0..p {
        xtwx_reg[[i, i]] += ridge;
    }

    // Compute Cholesky: X'WX = L·L' (L is lower triangular)
    let l = xtwx_reg
        .cholesky(UPLO::Lower)
        .map_err(|e| GAMError::InvalidParameter(format!("X'WX Cholesky failed: {:?}", e)))?;

    Ok(l)
}

/// Helper: Create weighted design matrix X_w[i,j] = sqrt(w[i]) * X[i,j]
/// Optimized with row-wise operations for better memory access patterns
#[inline]
fn create_weighted_x(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let mut x_weighted = x.to_owned();

    // Row-wise weighting: process each row at once for better cache locality
    for i in 0..n {
        let sqrt_wi = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= sqrt_wi;
        }
    }

    x_weighted
}

/// Compute X'WX efficiently without forming weighted matrix
/// This is a key optimization for large n: avoids redundant allocations
pub fn compute_xtwx(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let x_weighted = create_weighted_x(x, w);

    // Use BLAS matrix multiplication: X'WX = X_w' * X_w
    // This will automatically use optimized BLAS SYRK or GEMM
    x_weighted.t().dot(&x_weighted)
}

/// True GLM deviance D(y, μ) summed across observations, handling the
/// y∈{0,1} edge cases that `pirls::compute_deviance` short-circuits to
/// zero. Used by the mgcv-exact REML score (item 1 of #47) when an
/// `y_original` is supplied — switches the score from working-RSS
/// `Σw(y_working - Xβ)²` to true GLM deviance D(y_orig, μ̂(λ)).
///
/// Reference: mgcv/R/family.R `binomial$dev.resids`, `poisson$dev.resids`
/// etc. Here we sum the squared residuals (= deviance).
#[cfg(feature = "blas")]
pub fn glm_deviance(
    y_original: &Array1<f64>,
    mu: &Array1<f64>,
    family: crate::pirls::Family,
) -> f64 {
    use crate::pirls::Family;
    let mut dev = 0.0;
    for i in 0..y_original.len() {
        let yi = y_original[i];
        let mui = mu[i];
        let dev_i = match family {
            Family::Gaussian => (yi - mui).powi(2),
            Family::Binomial | Family::QuasiBinomial => {
                // Allow y in [0,1]; clamp μ for log safety.
                let mu_c = mui.clamp(1e-15, 1.0 - 1e-15);
                if yi <= 0.0 {
                    -2.0 * (1.0 - mu_c).ln()
                } else if yi >= 1.0 {
                    -2.0 * mu_c.ln()
                } else {
                    2.0 * (yi * (yi / mu_c).ln() + (1.0 - yi) * ((1.0 - yi) / (1.0 - mu_c)).ln())
                }
            }
            Family::Poisson | Family::QuasiPoisson => {
                let mu_c = mui.max(1e-15);
                if yi > 0.0 {
                    2.0 * (yi * (yi / mu_c).ln() - (yi - mu_c))
                } else {
                    2.0 * mu_c
                }
            }
            Family::Gamma | Family::GammaLog => {
                let mu_c = mui.max(1e-15);
                let yi_c = yi.max(1e-15);
                2.0 * ((yi_c - mu_c) / mu_c - (yi_c / mu_c).ln())
            }
            // Scaled-t per-observation deviance (mgcv scat$dev.resids):
            // (ν+1)·log1p(r²/(ν·σ²)). Keep this in sync with
            // pirls.rs::compute_deviance; mgcv-exact REML calls this path
            // for non-Gaussian true-response scoring.
            Family::TDist { df, sigma2 } => {
                let r = yi - mui;
                (df + 1.0) * (1.0 + r * r / (df * sigma2)).ln()
            }
            // Quantile/ELF deviance per qgam elf.R:122-138 with λ = σ
            // (matches pirls.rs::compute_deviance).
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
            // Tweedie deviance for 1 < p < 2
            Family::Tweedie { p } => {
                let twop = 2.0 - p;
                let onep = 1.0 - p;
                let mu_c = mui.max(1e-15);
                if yi <= 0.0 {
                    2.0 * mu_c.powf(twop) / twop
                } else {
                    2.0 * (yi.powf(twop) / (onep * twop) - yi * mu_c.powf(onep) / onep
                        + mu_c.powf(twop) / twop)
                }
            }
            // Inverse Gaussian deviance: d_i = (y - μ)² / (μ² · y)
            Family::InverseGaussian => {
                let mu_c = mui.max(1e-15);
                let yi_c = yi.max(1e-15);
                let diff = yi_c - mu_c;
                diff * diff / (mu_c * mu_c * yi_c)
            }
            // NB deviance (matches mgcv negbin$dev.resids at gam.fit3.r:2599-2602).
            // y=0 form is 2θ·log((μ+θ)/θ), positive since μ>0.
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
        dev += dev_i;
    }
    dev
}

/// Compute X'Wy efficiently using BLAS
pub fn compute_xtwy(x: &Array2<f64>, w: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let x_weighted = create_weighted_x(x, w);

    // Create weighted y vector: y_w[i] = sqrt(w[i]) * y[i]
    let n = x_weighted.nrows();
    let mut y_weighted = Array1::zeros(n);
    for i in 0..n {
        y_weighted[i] = y[i] * w[i].sqrt();
    }

    // Use BLAS matrix-vector product: X'Wy = X_w' * y_w
    x_weighted.t().dot(&y_weighted)
}

pub(crate) struct RemlSystem {
    pub(crate) a: Array2<f64>,
    pub(crate) beta: Array1<f64>,
    pub(crate) a_inv: Array2<f64>,
    pub(crate) tr_a: f64,
    /// X'Wy used to solve A·β = X'Wy. Cached so working_rss can use the
    /// p²-form (Gaussian) and `from_system` non-Gaussian path can avoid
    /// re-materialising fitted = X·β when caller already supplied xtwy.
    pub(crate) xtwy: Array1<f64>,
    /// Lazily computed `X·β`. Initialised on first call to `fitted(x)`.
    /// Stays uninitialised for the Gaussian working_rss p²-form path
    /// (the O(n·p) `X·β` is replaced by `y'Wy - 2β'X'Wy + β'X'WXβ`).
    fitted_cache: std::cell::OnceCell<Array1<f64>>,
}

impl RemlSystem {
    /// Lazily compute and cache `X·β`. For Gaussian + working_rss p²-form
    /// this is never called; non-Gaussian deviance paths trigger it once.
    pub(crate) fn fitted(&self, x: &Array2<f64>) -> &Array1<f64> {
        self.fitted_cache.get_or_init(|| x.dot(&self.beta))
    }

    /// Working-RSS deviance numerator: Σ w_i (y_i - x_iβ)². For Gaussian
    /// + canonical link this is the true deviance; for non-Gaussian families
    /// this is the IRLS working-response approximation used by the closed-form
    /// gradient/Hessian and as the Gaussian/None branch of the score function.
    ///
    /// Gaussian fast path: uses the p²-form `y'Wy - 2β'X'Wy + β'X'WXβ`,
    /// avoiding the O(n·p) `X·β` materialisation. Identity-preserving (algebraic
    /// expansion of the O(n) form).
    fn working_rss(
        &self,
        y: &Array1<f64>,
        w: &Array1<f64>,
        x: &Array2<f64>,
        xtwx: &Array2<f64>,
        family: crate::pirls::Family,
    ) -> f64 {
        if matches!(family, crate::pirls::Family::Gaussian) {
            let ywy: f64 = y
                .iter()
                .zip(w.iter())
                .map(|(yi, wi)| yi * yi * wi)
                .sum();
            let beta_xtwy = self.beta.dot(&self.xtwy);
            let xtwx_beta = xtwx.dot(&self.beta);
            let beta_xtwx_beta = self.beta.dot(&xtwx_beta);
            ywy - 2.0 * beta_xtwy + beta_xtwx_beta
        } else {
            let fitted = self.fitted(x);
            y.iter()
                .zip(fitted.iter())
                .zip(w.iter())
                .map(|((yi, fi), wi)| (yi - fi).powi(2) * wi)
                .sum()
        }
    }

    /// β'(ΣλS)β = Σ λ_j β'S_jβ — the penalty quadratic form, used in
    /// `Dp = D + β'(ΣλS)β` and to form the dispersion estimate.
    #[cfg(feature = "blas")]
    fn penalty_quadratic(&self, lambdas: &[f64], penalties_blocks: &[BlockPenalty]) -> f64 {
        lambdas
            .iter()
            .zip(penalties_blocks.iter())
            .map(|(l, pen)| l * pen.quadratic_form(&self.beta))
            .sum()
    }
}

pub(crate) fn assemble_reml_system(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    xtwx: &Array2<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwy: Option<&Array1<f64>>,
) -> Result<RemlSystem> {
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let xtwy: Array1<f64> = match cached_xtwy {
        Some(c) => c.clone(),
        None => compute_xtwy(x, w, y),
    };
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let solve_ridge = 1e-12 * max_diag;
    a_solve
        .diag_mut()
        .iter_mut()
        .for_each(|d| *d += solve_ridge);
    let beta = solve(a_solve, xtwy.clone())?;
    let a_inv = inverse(&a)?;
    let tr_a = (xtwx.dot(&a_inv)).diag().sum();

    Ok(RemlSystem {
        a,
        beta,
        a_inv,
        tr_a,
        xtwy,
        fitted_cache: std::cell::OnceCell::new(),
    })
}

/// Score-formula scalar quantities derived from a `RemlSystem`.
///
/// Centralises the mgcv-exact convention shared by the criterion and the
/// closed-form gradient/Hessian:
///   - `dev_num`: deviance numerator. Working-RSS for Gaussian or when no
///     original response is supplied; true GLM deviance D(y_orig, μ̂) when
///     `y_original` is passed for a non-Gaussian family.
///   - `bsb`: β'(ΣλS)β (penalty quadratic).
///   - `dp`: dev_num + bsb (the penalised deviance entering Dp/(2σ²)).
///   - `sigma2`: scale estimate. Binomial/Poisson/NegBin → 1. Gaussian →
///     RSS/(n - trA). Other dispersion-bearing families → mgcv's profiled
///     φ̂ via `estimate_phi_mgcv`.
#[cfg(feature = "blas")]
pub(crate) struct RemlScoreParts {
    pub(crate) dev_num: f64,
    pub(crate) bsb: f64,
    pub(crate) dp: f64,
    pub(crate) sigma2: f64,
}

#[cfg(feature = "blas")]
impl RemlScoreParts {
    /// Compute the family-aware deviance/scale parts at the converged β.
    /// Matches the `reml_criterion_multi_cached_mgcv_exact` and IFT path's
    /// σ² convention (item 1 of #47): for non-Gaussian + true response the
    /// deviance is `glm_deviance(y_orig, μ̂)`; otherwise working-RSS is used.
    pub(crate) fn from_system(
        system: &RemlSystem,
        y: &Array1<f64>,
        w: &Array1<f64>,
        x: &Array2<f64>,
        xtwx: &Array2<f64>,
        lambdas: &[f64],
        penalties_blocks: &[BlockPenalty],
        family: crate::pirls::Family,
        y_original: Option<&Array1<f64>>,
        mp: usize,
        n: usize,
    ) -> Self {
        let dev_num: f64 = match (family, y_original) {
            (crate::pirls::Family::Gaussian, _) | (_, None) => {
                system.working_rss(y, w, x, xtwx, family)
            }
            (fam, Some(y_orig)) => {
                let mu: Array1<f64> = system
                    .fitted(x)
                    .iter()
                    .map(|&eta| fam.inverse_link(eta))
                    .collect();
                glm_deviance(y_orig, &mu, fam)
            }
        };
        let bsb = system.penalty_quadratic(lambdas, penalties_blocks);
        let dp = dev_num + bsb;
        let y_for_phi = y_original.unwrap_or(y);
        let sigma2 = match family {
            crate::pirls::Family::Binomial
            | crate::pirls::Family::Poisson
            | crate::pirls::Family::NegBin { .. } => 1.0,
            _ => {
                let phi_init = dev_num / ((n as f64) - system.tr_a).max(1e-10);
                family.estimate_phi_mgcv(y_for_phi, dp, mp, 1.0, phi_init)
            }
        };
        Self {
            dev_num,
            bsb,
            dp,
            sigma2,
        }
    }

    /// Gaussian-style scale used by the closed-form gradient/Hessian.
    /// Mirrors the historical inline path: σ² = RSS / max(n - trA, 1e-10).
    /// `fixed_sigma2` overrides the computation (used when differentiating
    /// the score at a fixed σ² base point so FD-of-gradient and CF Hessian
    /// use the same σ² convention exactly).
    pub(crate) fn gaussian_only(
        system: &RemlSystem,
        y: &Array1<f64>,
        w: &Array1<f64>,
        x: &Array2<f64>,
        xtwx: &Array2<f64>,
        n: usize,
        fixed_sigma2: Option<f64>,
    ) -> Self {
        let dev_num = system.working_rss(y, w, x, xtwx, crate::pirls::Family::Gaussian);
        let sigma2 = fixed_sigma2
            .unwrap_or_else(|| dev_num / ((n as f64) - system.tr_a).max(1e-10));
        Self {
            dev_num,
            bsb: 0.0,
            dp: 0.0,
            sigma2,
        }
    }
}
