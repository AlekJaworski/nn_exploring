//! PiRLS (Penalized Iteratively Reweighted Least Squares) algorithm for GAM fitting

use ndarray::{Array1, Array2};
use crate::{Result, GAMError};
use crate::linalg::solve;

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
            },
            Family::Poisson => {
                // Clamp to avoid overflow
                let eta_safe = eta.min(20.0);
                eta_safe.exp()
            },
            Family::Gamma => {
                // Ensure eta is not too close to zero
                let eta_safe = if eta.abs() < 1e-10 { 1e-10 } else { eta };
                1.0 / eta_safe
            },
        }
    }

    /// Derivative of inverse link function
    pub fn d_inverse_link(&self, eta: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            },
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
    penalties: &[Array2<f64>],
    family: Family,
    max_iter: usize,
    tolerance: f64,
) -> Result<PiRLSResult> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(GAMError::DimensionMismatch(
            format!("X has {} rows but y has {} elements", x.nrows(), n)
        ));
    }

    if lambda.len() != penalties.len() {
        return Err(GAMError::DimensionMismatch(
            "Number of lambdas must match number of penalty matrices".to_string()
        ));
    }

    // Initialize coefficients and linear predictor
    let mut beta = Array1::zeros(p);
    let mut eta = x.dot(&beta);

    // Initialize eta based on family
    for i in 0..n {
        let safe_y = match family {
            Family::Binomial => y[i].max(0.01).min(0.99),  // Avoid 0 and 1
            Family::Poisson | Family::Gamma => y[i].max(0.1),  // Avoid 0
            Family::Gaussian => y[i],
        };
        eta[i] = family.link(safe_y);
    }

    let mut converged = false;
    let mut iter = 0;

    for iteration in 0..max_iter {
        iter = iteration + 1;

        // Compute fitted values μ = g^(-1)(η)
        let mu: Array1<f64> = eta.iter()
            .map(|&e| family.inverse_link(e))
            .collect();

        // Compute working response z = η + (y - μ) / g'(μ)
        let mut z = Array1::zeros(n);
        for i in 0..n {
            let dmu_deta = family.d_inverse_link(eta[i]);
            if dmu_deta.abs() < 1e-10 {
                z[i] = eta[i];
            } else {
                z[i] = eta[i] + (y[i] - mu[i]) / dmu_deta;
            }
        }

        // Compute IRLS weights w = (g'(μ))^2 / V(μ)
        let mut w = Array1::zeros(n);
        for i in 0..n {
            let dmu_deta = family.d_inverse_link(eta[i]);
            let variance = family.variance(mu[i]);
            w[i] = (dmu_deta * dmu_deta) / variance.max(1e-10);
            w[i] = w[i].max(1e-10); // Ensure positive weights
        }

        // Construct weighted normal equations: (X'WX + Σ λ_j S_j)β = X'Wz
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            for j in 0..p {
                for k in 0..p {
                    xtwx[[j, k]] += x[[i, j]] * w[i] * x[[i, k]];
                }
            }
        }

        // Compute max diagonal for ridge scaling before moving xtwx
        let mut max_diag: f64 = 1.0;
        for i in 0..p {
            max_diag = max_diag.max(xtwx[[i, i]].abs());
        }

        // Add penalty terms
        let mut penalty_total = Array2::<f64>::zeros((p, p));
        for (lambda_j, penalty_j) in lambda.iter().zip(penalties.iter()) {
            penalty_total = penalty_total + &(penalty_j * *lambda_j);
        }

        let mut a = xtwx + penalty_total;

        // Add adaptive ridge for numerical stability (like mgcv does)
        // This prevents singularity issues with rank-deficient penalty matrices
        // Scale ridge by number of penalties for multidimensional cases
        let num_penalties = lambda.len();
        let ridge_scale = 1e-5 * (1.0 + (num_penalties as f64).sqrt());
        let ridge: f64 = ridge_scale * max_diag;
        for i in 0..p {
            a[[i, i]] += ridge;
        }

        // Compute X'Wz
        let mut xtwz = Array1::zeros(p);
        for j in 0..p {
            for i in 0..n {
                xtwz[j] += x[[i, j]] * w[i] * z[i];
            }
        }

        // Solve for new coefficients
        let beta_old = beta.clone();
        beta = solve(a, xtwz)?;

        // Update linear predictor
        eta = x.dot(&beta);

        // Check convergence
        let max_change = beta.iter().zip(beta_old.iter())
            .map(|(b, b_old)| (b - b_old).abs())
            .fold(0.0f64, f64::max);

        if max_change < tolerance {
            converged = true;
            break;
        }
    }

    // Compute final fitted values
    let fitted_values: Array1<f64> = eta.iter()
        .map(|&e| family.inverse_link(e))
        .collect();

    // Compute deviance
    let deviance = compute_deviance(y, &fitted_values, family);

    // Compute final weights
    let weights: Array1<f64> = eta.iter()
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
            },
            Family::Poisson => {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * mui
                }
            },
            Family::Gamma => {
                2.0 * ((yi - mui) / mui - (yi / mui).ln())
            },
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
        let n = 20;
        let p = 5;

        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            ((i as f64) * 0.1).powi(j as i32)
        });

        let y: Array1<f64> = (0..n).map(|i| {
            let xi = i as f64 * 0.1;
            xi + xi.powi(2) + 0.1 * (i as f64).sin()
        }).collect();

        let penalty = Array2::eye(p);
        let lambda = vec![0.01];
        let penalties = vec![penalty];

        let result = fit_pirls(
            &y,
            &x,
            &lambda,
            &penalties,
            Family::Gaussian,
            100,
            1e-6
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.converged);
        assert_eq!(result.coefficients.len(), p);
    }
}
