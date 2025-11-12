//! REML (Restricted Maximum Likelihood) criterion for smoothing parameter selection

use ndarray::{Array1, Array2};
use crate::Result;
use crate::linalg::{solve, determinant, inverse};

/// Estimate the rank of a matrix using row norms as approximation to singular values
/// For symmetric matrices like penalty matrices, this gives a reasonable estimate
fn estimate_rank(matrix: &Array2<f64>) -> usize {
    let n = matrix.nrows().min(matrix.ncols());

    // For symmetric matrices, compute row-wise squared norms as eigenvalue estimates
    // This is much faster than SVD but gives reasonable rank estimates
    let mut row_norms = Vec::with_capacity(n);
    for i in 0..n {
        let mut norm_sq = 0.0;
        for j in 0..matrix.ncols() {
            norm_sq += matrix[[i, j]].powi(2);
        }
        row_norms.push(norm_sq);
    }

    // Sort in descending order
    row_norms.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Count rows with significant norm (threshold scaled by largest norm)
    let threshold = 1e-12 * row_norms[0].max(1.0);
    let mut rank = 0;
    for &norm_sq in &row_norms {
        if norm_sq > threshold {
            rank += 1;
        } else {
            break;
        }
    }

    // CR splines have rank k-2 for second derivative penalty
    // If we got k, reduce to k-2 for CR penalty matrices specifically
    // (This is a heuristic based on known structure)
    if rank == n && n >= 2 {
        // Check if last two row norms are significantly smaller
        if row_norms[n-1] < 1e-10 * row_norms[0] && row_norms[n-2] < 1e-10 * row_norms[0] {
            rank = n - 2;
        }
    }

    rank.max(1) // At least rank 1
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

/// Compute the REML criterion for multiple smoothing parameters
///
/// The REML criterion with multiple penalties is:
/// REML = n*log(RSS/n) + log|X'WX + Σλᵢ·Sᵢ| - Σrank(Sᵢ)·log(λᵢ)
///
/// Where:
/// - RSS: residual sum of squares
/// - X: design matrix
/// - W: weight matrix (from IRLS)
/// - λᵢ: smoothing parameters
/// - Sᵢ: penalty matrices
pub fn reml_criterion_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
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

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        a = a + &(penalty * *lambda);
    }

    // Compute coefficients if not provided
    let beta_computed;
    let beta = if let Some(b) = beta {
        b
    } else {
        let y_weighted: Array1<f64> = y.iter().zip(w.iter())
            .map(|(yi, wi)| yi * wi)
            .collect();

        let b = xtw.dot(&y_weighted);
        beta_computed = solve(a.clone(), b)?;
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

    // Compute log|X'WX + Σλᵢ·Sᵢ|
    // Add small ridge term to ensure numerical stability
    let ridge = 1e-6;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let log_det_a = determinant(&a_reg)?.ln();

    // Compute -Σrank(Sᵢ)·log(λᵢ)
    let mut log_lambda_sum = 0.0;
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        if *lambda > 1e-10 {
            let rank_s = estimate_rank(penalty);
            if rank_s > 0 {
                log_lambda_sum += (rank_s as f64) * lambda.ln();
            }
        }
    }

    // REML = n*log(RSS/n) + log|X'WX + Σλᵢ·Sᵢ| - Σrank(Sᵢ)·log(λᵢ)
    let reml = (n as f64) * (rss / n as f64).ln() + log_det_a - log_lambda_sum;

    Ok(reml)
}

/// Compute the gradient of REML with respect to log(λᵢ)
///
/// Returns: ∂REML/∂log(λᵢ) for i = 1..m
///
/// The derivative is:
/// ∂REML/∂log(λᵢ) = -n/(RSS) · ∂RSS/∂log(λᵢ) + tr((X'WX + Σλⱼ·Sⱼ)⁻¹·λᵢ·Sᵢ) - rank(Sᵢ)
pub fn reml_gradient_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array1<f64>> {
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute weighted design matrix
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        a = a + &(penalty * *lambda);
    }

    // Solve for coefficients
    let y_weighted: Array1<f64> = y.iter().zip(w.iter())
        .map(|(yi, wi)| yi * wi)
        .collect();

    let b = xtw.dot(&y_weighted);
    let beta = solve(a.clone(), b)?;

    // Compute fitted values and RSS
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute A^(-1)
    // Add small ridge term to ensure numerical stability (especially for perfectly regular data)
    let ridge = 1e-6;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute gradient for each λᵢ
    let mut gradient = Array1::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];
        let rank_i = estimate_rank(penalty_i);

        // tr((X'WX + Σλⱼ·Sⱼ)⁻¹·λᵢ·Sᵢ)
        let lambda_s_i = penalty_i * lambda_i;
        let temp = a_inv.dot(&lambda_s_i);
        let mut trace = 0.0;
        for j in 0..p {
            trace += temp[[j, j]];
        }

        // ∂RSS/∂log(λᵢ) computation requires implicit differentiation
        // For now, use approximation: ∂REML/∂log(λᵢ) ≈ tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ)
        //
        // Full derivative would be:
        // ∂REML/∂log(λᵢ) = -n·∂RSS/∂log(λᵢ)/RSS + tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ)
        //
        // where ∂RSS/∂log(λᵢ) = -2λᵢ·β'·Sᵢ·β

        // Compute β'·Sᵢ·β
        let s_beta = penalty_i.dot(&beta);
        let beta_s_beta: f64 = beta.iter().zip(s_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();

        // ∂RSS/∂log(λᵢ) = -2λᵢ·β'·Sᵢ·β
        let drss_dlog_lambda = -2.0 * lambda_i * beta_s_beta;

        // Full gradient
        gradient[i] = -(n as f64) * drss_dlog_lambda / rss + trace - (rank_i as f64);
    }

    Ok(gradient)
}

/// Compute the Hessian of REML with respect to log(λᵢ), log(λⱼ)
///
/// Returns: ∂²REML/∂log(λᵢ)∂log(λⱼ) for i,j = 1..m
///
/// This is a symmetric m x m matrix
pub fn reml_hessian_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array2<f64>> {
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute weighted design matrix
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        a = a + &(penalty * *lambda);
    }

    // Compute A^(-1)
    // Add small ridge term to ensure numerical stability
    let ridge = 1e-6;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute Hessian
    let mut hessian = Array2::zeros((m, m));

    for i in 0..m {
        for j in i..m {  // Only compute upper triangle (symmetric)
            let lambda_i = lambdas[i];
            let lambda_j = lambdas[j];
            let penalty_i = &penalties[i];
            let penalty_j = &penalties[j];

            // Compute ∂²REML/∂log(λᵢ)∂log(λⱼ)
            // ≈ -tr((A⁻¹·λᵢ·Sᵢ)·(A⁻¹·λⱼ·Sⱼ))
            //
            // This is a simplification. Full Hessian requires second derivatives
            // of RSS and more complex implicit differentiation.

            let lambda_s_i = penalty_i * lambda_i;
            let lambda_s_j = penalty_j * lambda_j;

            let a_inv_si = a_inv.dot(&lambda_s_i);
            let a_inv_sj = a_inv.dot(&lambda_s_j);

            let product = a_inv_si.dot(&a_inv_sj);

            let mut trace = 0.0;
            for k in 0..p {
                trace += product[[k, k]];
            }

            hessian[[i, j]] = -trace;

            // Fill symmetric entry
            if i != j {
                hessian[[j, i]] = hessian[[i, j]];
            }
        }
    }

    Ok(hessian)
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
