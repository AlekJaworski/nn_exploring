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
}

impl Family {
    /// Variance function V(μ). Same regardless of link choice (link
    /// affects how η maps to μ, not V(μ) itself).
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma | Family::GammaLog => mu * mu,
        }
    }

    /// Link function g(μ).
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => mu,
            Family::Binomial => (mu / (1.0 - mu)).ln(),
            Family::Poisson | Family::GammaLog => mu.ln(),
            Family::Gamma => 1.0 / mu,
        }
    }

    /// Inverse link function g^(-1)(η).
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian => eta,
            Family::Binomial => {
                let eta_safe = eta.max(-20.0).min(20.0);
                1.0 / (1.0 + (-eta_safe).exp())
            }
            Family::Poisson | Family::GammaLog => {
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
            Family::Gaussian => 1.0,
            Family::Binomial => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            }
            Family::Poisson | Family::GammaLog => eta.exp(),
            Family::Gamma => -1.0 / (eta * eta),
        }
    }

    /// First derivative of variance function: dV/dμ.
    /// Used by mgcv's full-Newton PIRLS (`gam.fit3.r:507`) for non-canonical
    /// links to compute the α correction `α = 1 + (y−μ)·(V'/V + g''·dμ/dη)`.
    pub fn dvar(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 0.0,
            Family::Binomial => 1.0 - 2.0 * mu,
            Family::Poisson => 1.0,
            Family::Gamma | Family::GammaLog => 2.0 * mu,
        }
    }

    /// Second derivative of variance function: d²V/dμ².
    /// Used by mgcv's α₁ derivative (`gdi.c:2548`) needed for the Tk
    /// weight-derivative term in the REML gradient.
    pub fn d2var(&self, _mu: f64) -> f64 {
        match self {
            Family::Gaussian => 0.0,
            Family::Binomial => -2.0,
            Family::Poisson => 0.0,
            Family::Gamma | Family::GammaLog => 2.0,
        }
    }

    /// Second derivative of link function: d²g/dμ².
    pub fn d2link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 0.0,
            Family::Binomial => {
                let one_minus = 1.0 - mu;
                -1.0 / (mu * mu) + 1.0 / (one_minus * one_minus)
            }
            Family::Poisson | Family::GammaLog => -1.0 / (mu * mu),
            Family::Gamma => 2.0 / (mu * mu * mu),
        }
    }

    /// Third derivative of link function: d³g/dμ³.
    pub fn d3link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 0.0,
            Family::Binomial => {
                let one_minus = 1.0 - mu;
                2.0 / (mu * mu * mu) + 2.0 / (one_minus * one_minus * one_minus)
            }
            Family::Poisson | Family::GammaLog => 2.0 / (mu * mu * mu),
            Family::Gamma => -6.0 / (mu * mu * mu * mu),
        }
    }

    /// True iff the link is canonical for this family. Determines whether
    /// PIRLS uses Fisher scoring (canonical) or full Newton (non-canonical),
    /// per `gam.fit3.r:118`.
    pub fn is_canonical_link(&self) -> bool {
        match self {
            Family::Gaussian | Family::Binomial | Family::Poisson | Family::Gamma => true,
            Family::GammaLog => false,
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
            Family::Binomial => y[i].max(0.01).min(0.99),
            Family::Poisson | Family::Gamma | Family::GammaLog => y[i].max(0.1),
            Family::Gaussian => unreachable!(),
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

        // Compute working response z and IRLS weights w in a single pass
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
            Family::Binomial => {
                if yi > 0.0 && yi < 1.0 {
                    2.0 * (yi * (yi / mui).ln() + (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln())
                } else {
                    0.0
                }
            }
            Family::Poisson => {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * mui
                }
            }
            Family::Gamma | Family::GammaLog => 2.0 * ((yi - mui) / mui - (yi / mui).ln()),
        };

        deviance += dev_i;
    }

    deviance
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
            Family::Binomial => y[i].max(0.01).min(0.99),
            Family::Poisson | Family::Gamma | Family::GammaLog => y[i].max(0.1),
            Family::Gaussian => unreachable!(),
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
}
