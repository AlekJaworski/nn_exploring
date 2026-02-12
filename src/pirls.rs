//! PiRLS (Penalized Iteratively Reweighted Least Squares) algorithm for GAM fitting

use crate::block_penalty::BlockPenalty;
use crate::linalg::solve;
use crate::reml::compute_xtwx;
use crate::{GAMError, Result};
use ndarray::{Array1, Array2};

/// Family and link function for GLM
#[derive(Debug, Clone, Copy)]
pub enum Family {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
}

impl Family {
    /// Variance function V(μ)
    pub fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
        }
    }

    /// Canonical link function g(μ)
    pub fn link(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => mu,
            Family::Binomial => (mu / (1.0 - mu)).ln(),
            Family::Poisson => mu.ln(),
            Family::Gamma => 1.0 / mu,
        }
    }

    /// Inverse link function g^(-1)(η)
    pub fn inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian => eta,
            Family::Binomial => {
                // Clamp eta to avoid overflow in exp
                let eta_safe = eta.max(-20.0).min(20.0);
                1.0 / (1.0 + (-eta_safe).exp())
            }
            Family::Poisson => {
                // Clamp to avoid overflow
                let eta_safe = eta.min(20.0);
                eta_safe.exp()
            }
            Family::Gamma => {
                // Ensure eta is not too close to zero
                let eta_safe = if eta.abs() < 1e-10 { 1e-10 } else { eta };
                1.0 / eta_safe
            }
        }
    }

    /// Derivative of inverse link function
    pub fn d_inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            }
            Family::Poisson => eta.exp(),
            Family::Gamma => -1.0 / (eta * eta),
        }
    }
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
            Family::Poisson | Family::Gamma => y[i].max(0.1),
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
    let ridge_scale = 1e-5 * (1.0 + (num_penalties as f64).sqrt());

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
    let ridge_scale = 1e-5 * (1.0 + (num_penalties as f64).sqrt());
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
fn compute_deviance(y: &Array1<f64>, mu: &Array1<f64>, family: Family) -> f64 {
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
            Family::Gamma => 2.0 * ((yi - mui) / mui - (yi / mui).ln()),
        };

        deviance += dev_i;
    }

    deviance
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
