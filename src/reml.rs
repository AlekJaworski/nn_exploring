//! REML (Restricted Maximum Likelihood) criterion for smoothing parameter selection

use ndarray::{Array1, Array2};
use crate::Result;
use crate::linalg::{solve, determinant, inverse};

/// Estimate the rank of a matrix by counting diagonal elements above threshold
/// This is a rough approximation - proper implementation would use SVD
fn estimate_rank(matrix: &Array2<f64>) -> usize {
    let n = matrix.nrows().min(matrix.ncols());
    let mut rank = 0;

    // Count non-zero diagonal elements (rough estimate)
    for i in 0..n {
        if matrix[[i, i]].abs() > 1e-8 {
            rank += 1;
        }
    }

    // If all diagonal elements are zero, count non-zero off-diagonal elements
    if rank == 0 {
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if matrix[[i, j]].abs() > 1e-8 {
                    rank += 1;
                    break;
                }
            }
        }
    }

    rank
}

/// Compute the REML criterion for smoothing parameter selection
///
/// The REML criterion is:
/// REML = n*log(RSS) + log|X'WX + λS| - log|S|
///
/// Where:
/// - RSS: residual sum of squares
/// - X: design matrix
/// - W: weight matrix (from IRLS)
/// - λ: smoothing parameter
/// - S: penalty matrix
pub fn reml_criterion(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambda: f64,
    penalty: &Array2<f64>,
    beta: Option<&Array1<f64>>,
) -> Result<f64> {
    let n = y.len();
    let p = x.ncols();

    // Compute weighted design matrix: sqrt(W) * X
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Compute coefficients if not provided
    let beta_computed;
    let beta = if let Some(b) = beta {
        b
    } else {
        // Solve: (X'WX + λS)β = X'Wy
        let xtw = x_weighted.t().to_owned();
        let xtwx = xtw.dot(&x_weighted);

        let mut a = xtwx + &(penalty * lambda);

        let y_weighted: Array1<f64> = y.iter().zip(w.iter())
            .map(|(yi, wi)| yi * wi)
            .collect();

        let b = xtw.dot(&y_weighted);

        beta_computed = solve(a, b)?;
        &beta_computed
    };

    // Compute fitted values
    let fitted = x.dot(beta);

    // Compute residuals and RSS
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute X'WX + λS
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);
    let a = xtwx + &(penalty * lambda);

    // Compute log determinants
    let log_det_a = determinant(&a)?.ln();

    // For penalty matrix with rank r, the correct term is log|λS| = r*log(λ) + log|S_+|
    // where S_+ is the pseudo-determinant (product of non-zero eigenvalues)
    //
    // For simplicity, we estimate the rank by counting eigenvalues > threshold
    // A proper implementation would use SVD or eigendecomposition
    let rank_s = estimate_rank(penalty);

    // The REML criterion is:
    // REML = n*log(RSS/n) + log|X'WX + λS| - log|λS| - ...
    //      = n*log(RSS/n) + log|X'WX + λS| - rank(S)*log(λ) - log|S_+|
    //
    // For now, ignore the constant log|S_+| term and use:
    let log_lambda_s = if lambda > 1e-10 && rank_s > 0 {
        (rank_s as f64) * lambda.ln()
    } else {
        0.0
    };

    // REML = n*log(RSS/n) + log|X'WX + λS| - rank(S)*log(λ)
    let reml = (n as f64) * (rss / n as f64).ln() + log_det_a - log_lambda_s;

    Ok(reml)
}

/// Compute GCV (Generalized Cross-Validation) criterion as alternative to REML
///
/// GCV = n * RSS / (n - tr(A))^2
/// where A is the influence matrix
pub fn gcv_criterion(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambda: f64,
    penalty: &Array2<f64>,
) -> Result<f64> {
    let n = y.len();
    let p = x.ncols();

    // Compute weighted design matrix
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Solve for coefficients
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);
    let mut a = xtwx + &(penalty * lambda);

    let y_weighted: Array1<f64> = y.iter().zip(w.iter())
        .map(|(yi, wi)| yi * wi)
        .collect();

    let b = xtw.dot(&y_weighted);

    let a_for_solve = a.clone();
    let beta = solve(a_for_solve, b)?;

    // Compute fitted values and residuals
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute effective degrees of freedom (trace of influence matrix)
    // EDF = tr(H) where H = X(X'WX + λS)^(-1)X'W
    let a_inv = inverse(&a)?;

    // Compute X'W (not sqrt(W))
    let mut xtw_full = Array2::zeros((p, n));
    for i in 0..n {
        for j in 0..p {
            xtw_full[[j, i]] = x[[i, j]] * w[i];
        }
    }

    // H = X * (X'WX + λS)^(-1) * X'W
    let h_temp = x.dot(&a_inv);
    let influence = h_temp.dot(&xtw_full);

    // Trace of H
    let mut edf = 0.0;
    for i in 0..n {
        edf += influence[[i, i]];
    }

    // GCV = n * RSS / (n - edf)^2
    let gcv = (n as f64) * rss / ((n as f64) - edf).powi(2);

    Ok(gcv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reml_criterion() {
        let n = 10;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let w = Array1::ones(n);
        let penalty = Array2::eye(p);
        let lambda = 0.1;

        let result = reml_criterion(&y, &x, &w, lambda, &penalty, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gcv_criterion() {
        let n = 10;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let w = Array1::ones(n);
        let penalty = Array2::eye(p);
        let lambda = 0.1;

        let result = gcv_criterion(&y, &x, &w, lambda, &penalty);
        assert!(result.is_ok());
    }
}
