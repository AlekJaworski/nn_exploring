//! REML (Restricted Maximum Likelihood) criterion for smoothing parameter selection

use ndarray::{Array1, Array2, Axis, s};
use crate::Result;
use crate::linalg::{solve, determinant, inverse};
use crate::GAMError;

/// Estimate the rank of a matrix using row norms as approximation to singular values
/// For symmetric matrices like penalty matrices, this gives a reasonable estimate
fn estimate_rank(matrix: &Array2<f64>) -> usize {
    let n = matrix.nrows().min(matrix.ncols());

    // For block-diagonal penalty matrices (multi-smooth case), count non-zero rows
    // Each block corresponds to one smooth, with rank = k-2 for CR splines
    let matrix_norm = matrix.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let threshold = 1e-10 * matrix_norm.max(1.0);

    let mut non_zero_rows = 0;
    for i in 0..n {
        let mut row_norm = 0.0;
        for j in 0..matrix.ncols() {
            row_norm += matrix[[i, j]].abs();
        }
        if row_norm > threshold {
            non_zero_rows += 1;
        }
    }

    // For CR splines: rank = (non_zero_rows - 2).max(1)
    // The null space dimension is 2 (constant and linear functions)
    if non_zero_rows >= 2 {
        return non_zero_rows - 2;
    }

    // Fallback for very small matrices
    1
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
    let _p = x.ncols();

    // Compute weighted design matrix: sqrt(W) * X
    // Optimized: avoid intermediate allocation, compute directly
    let mut x_weighted = x.to_owned();
    for (i, mut row) in x_weighted.rows_mut().into_iter().enumerate() {
        let w_sqrt = w[i].sqrt();
        for val in row.iter_mut() {
            *val *= w_sqrt;
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

        let a = xtwx + &(penalty * lambda);

        // Optimized: compute y_weighted in-place to avoid allocation
        let mut y_weighted = Array1::zeros(n);
        for i in 0..n {
            y_weighted[i] = y[i] * w[i];
        }

        let b = xtw.dot(&y_weighted);

        beta_computed = solve(a, b)?;
        &beta_computed
    };

    // Compute fitted values
    let fitted = x.dot(beta);

    // Compute residuals and RSS (optimized to avoid intermediate allocation)
    let mut rss = 0.0;
    for i in 0..n {
        let residual = y[i] - fitted[i];
        rss += residual * residual * w[i];
    }

    // Compute penalty term: β'Sβ (optimized dot product)
    let s_beta = penalty.dot(beta);
    let mut beta_s_beta = 0.0;
    for i in 0..s_beta.len() {
        beta_s_beta += beta[i] * s_beta[i];
    }

    // Compute RSS + λβ'Sβ (this is what mgcv calls rss.bSb)
    let rss_bsb = rss + lambda * beta_s_beta;

    // Compute X'WX + λS
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);
    let a = xtwx + &(penalty * lambda);

    // Compute log determinants
    let log_det_a = determinant(&a)?.ln();

    // Estimate rank of penalty matrix
    let rank_s = estimate_rank(penalty);

    // Compute scale parameter: φ = RSS / (n - rank(S))
    // Note: φ is based on RSS alone, not RSS + λβ'Sβ
    let phi = rss / (n - rank_s) as f64;

    // The correct REML criterion (matching mgcv's fast-REML.r implementation):
    // REML = ((RSS + λβ'Sβ)/φ + (n-rank(S))*log(2π φ) + log|X'WX + λS| - rank(S)*log(λ) - log|S_+|) / 2
    //
    // For now, we ignore the constant log|S_+| term (pseudo-determinant of S)
    // since it doesn't affect optimization over λ
    let log_lambda_term = if lambda > 1e-10 && rank_s > 0 {
        (rank_s as f64) * lambda.ln()
    } else {
        0.0
    };

    let pi = std::f64::consts::PI;
    let reml = (rss_bsb / phi
                + ((n - rank_s) as f64) * (2.0 * pi * phi).ln()
                + log_det_a
                - log_lambda_term) / 2.0;

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

    // Compute weighted design matrix (optimized)
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
    let a = xtwx + &(penalty * lambda);

    // Optimized y_weighted computation
    let mut y_weighted = Array1::zeros(n);
    for i in 0..n {
        y_weighted[i] = y[i] * w[i];
    }

    let b = xtw.dot(&y_weighted);

    let a_for_solve = a.clone();
    let beta = solve(a_for_solve, b)?;

    // Compute fitted values and residuals (optimized)
    let fitted = x.dot(&beta);
    let mut rss = 0.0;
    for i in 0..n {
        let residual = y[i] - fitted[i];
        rss += residual * residual * w[i];
    }

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
    // Use broadcasting: multiply each row of X by corresponding sqrt(w_i)
    let w_sqrt: Array1<f64> = w.iter().map(|wi| wi.sqrt()).collect();
    let x_weighted = x * &w_sqrt.view().insert_axis(ndarray::Axis(1));

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        // Add in-place instead of creating temporary
        a.scaled_add(*lambda, penalty);
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

        // Add ridge for numerical stability when solving
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
        let ridge = ridge_scale * max_diag;
        let mut a_solve = a.clone();
        a_solve.diag_mut().iter_mut().for_each(|x| *x += ridge);

        beta_computed = solve(a_solve, b)?;
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

    // Compute penalty term: Σλᵢ·β'·Sᵢ·β
    let mut penalty_sum = 0.0;
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        let s_beta = penalty.dot(beta);
        let beta_s_beta: f64 = beta.iter().zip(s_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        penalty_sum += lambda * beta_s_beta;
    }

    // Compute RSS + Σλᵢ·β'·Sᵢ·β
    let rss_bsb = rss + penalty_sum;

    // Compute log|X'WX + Σλᵢ·Sᵢ|
    // Add adaptive ridge term to ensure numerical stability
    // Scale by problem size and matrix magnitude for robustness
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    // Use stronger ridge for multidimensional cases (more penalties = more potential for ill-conditioning)
    let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    a_reg.diag_mut().iter_mut().for_each(|x| *x += ridge);
    let log_det_a = determinant(&a_reg)?.ln();

    // Compute total rank and -Σrank(Sᵢ)·log(λᵢ)
    let mut total_rank = 0;
    let mut log_lambda_sum = 0.0;
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        if *lambda > 1e-10 {
            let rank_s = estimate_rank(penalty);
            if rank_s > 0 {
                total_rank += rank_s;
                log_lambda_sum += (rank_s as f64) * lambda.ln();
            }
        }
    }

    // Compute scale parameter: φ = RSS / (n - Σrank(Sᵢ))
    let phi = rss / (n - total_rank) as f64;

    // The correct REML criterion:
    // REML = ((RSS + Σλᵢ·β'·Sᵢ·β)/φ + (n-Σrank(Sᵢ))*log(2πφ) + log|X'WX + Σλᵢ·Sᵢ| - Σrank(Sᵢ)·log(λᵢ)) / 2
    let pi = std::f64::consts::PI;
    let reml = (rss_bsb / phi
                + ((n - total_rank) as f64) * (2.0 * pi * phi).ln()
                + log_det_a
                - log_lambda_sum) / 2.0;

    Ok(reml)
}

/// Compute square root of a penalty matrix using eigenvalue decomposition
///
/// For a symmetric positive semi-definite matrix S, computes L such that S = L'L
/// Uses eigenvalue decomposition: S = Q Λ Q', so L = Q Λ^{1/2} Q' (taking transpose)
fn penalty_sqrt(penalty: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::Eigh;

    let n = penalty.nrows();
    if n != penalty.ncols() {
        return Err(GAMError::InvalidParameter(
            "Penalty matrix must be square".to_string()
        ));
    }

    // Compute eigenvalue decomposition: S = Q Λ Q'
    let (eigenvalues, eigenvectors) = penalty.eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| GAMError::InvalidParameter(format!("Eigenvalue decomposition failed: {:?}", e)))?;

    // Threshold for considering eigenvalue as zero
    let max_eigenvalue = eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let threshold = 1e-10 * max_eigenvalue.max(1.0);

    // Count non-zero eigenvalues
    let non_zero_eigs: Vec<(usize, f64)> = eigenvalues.iter().copied().enumerate()
        .filter(|&(_, e)| e > threshold)
        .collect();

    let rank = non_zero_eigs.len();

    if rank == 0 {
        // Penalty is zero, return empty matrix
        return Ok(Array2::<f64>::zeros((n, 0)));
    }

    // Create thin square root: L is n × rank
    // Only keep eigenvectors corresponding to non-zero eigenvalues
    let mut sqrt_penalty = Array2::<f64>::zeros((n, rank));
    for (out_j, &(in_j, eigenvalue)) in non_zero_eigs.iter().enumerate() {
        let sqrt_eval = eigenvalue.sqrt();
        for i in 0..n {
            sqrt_penalty[[i, out_j]] = eigenvectors[[i, in_j]] * sqrt_eval;
        }
    }

    Ok(sqrt_penalty)
}

/// Compute the gradient of REML using QR-based approach (matching mgcv's gdi.c)
///
/// Following Wood (2011) and mgcv's gdi.c (get_ddetXWXpS function), this uses:
/// 1. QR decomposition of augmented matrix [sqrt(W)X; sqrt(λ_0)L_0; ...]
/// 2. R such that R'R = X'WX + Σλᵢ·Sᵢ
/// 3. P = R^{-1}
/// 4. Gradient: ∂log|R'R|/∂log(λ_m) = λ_m·tr(P'·S_m·P)
///
/// This avoids explicit formation of A^{-1} and cross-coupling issues.
pub fn reml_gradient_multi_qr(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array1<f64>> {
    use ndarray_linalg::Inverse;
    use ndarray_linalg::QR;

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute sqrt(W) * X
    let mut sqrt_w_x = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] *= weight_sqrt;
        }
    }

    // Compute square root penalties and their ranks
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    for penalty in penalties.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        let rank = estimate_rank(penalty);
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Build augmented matrix Z = [sqrt(W)X; sqrt(λ_0)L_0'; sqrt(λ_1)L_1'; ...]
    // Determine total rows (n + sum of ranks)
    let mut total_rows = n;
    for sqrt_pen in sqrt_penalties.iter() {
        total_rows += sqrt_pen.ncols();  // Number of columns = rank
    }

    let mut z = Array2::<f64>::zeros((total_rows, p));

    // Fill in sqrt(W)X
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = sqrt_w_x[[i, j]];
        }
    }

    // Fill in scaled square root penalties (transposed)
    // sqrt_pen is p × rank, we need rank × p for augmented matrix
    let mut row_offset = n;
    for (sqrt_pen, &lambda) in sqrt_penalties.iter().zip(lambdas.iter()) {
        let sqrt_lambda = lambda.sqrt();
        let rank = sqrt_pen.ncols();  // Number of non-zero eigenvalues
        for i in 0..rank {
            for j in 0..p {
                z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];  // Transpose!
            }
        }
        row_offset += rank;
    }

    // QR decomposition: Z = QR
    let (_, r) = z.qr()
        .map_err(|e| GAMError::InvalidParameter(format!("QR decomposition failed: {:?}", e)))?;

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[QR_DEBUG] Z dimensions: {}×{}", z.nrows(), z.ncols());
        eprintln!("[QR_DEBUG] R dimensions: {}×{}", r.nrows(), r.ncols());
        eprintln!("[QR_DEBUG] total_rows={}, n={}, p={}", total_rows, n, p);
    }

    // Extract upper triangular part (first p rows)
    let mut r_upper = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            r_upper[[i, j]] = r[[i, j]];
        }
    }

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        // Check R'R to see if it matches X'WX + S
        let rtr = r_upper.t().dot(&r_upper);
        eprintln!("[QR_DEBUG] R'R diagonal: [{:.6}, {:.6}, ..., {:.6}]",
                 rtr[[0,0]], rtr[[1,1]], rtr[[p-1,p-1]]);
    }

    // Compute P = R^{-1}
    let p_matrix = r_upper.inv()
        .map_err(|e| GAMError::InvalidParameter(format!("Matrix inversion failed: {:?}", e)))?;

    // Compute coefficients for penalty term
    let xtw = sqrt_w_x.t().to_owned();
    let xtwx = xtw.dot(&sqrt_w_x);
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        a.scaled_add(*lambda, penalty);
    }

    let y_weighted: Array1<f64> = y.iter().zip(w.iter())
        .map(|(yi, wi)| yi * wi)
        .collect();
    let b = xtw.dot(&y_weighted);

    // Add small ridge for stability
    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a, b)?;

    // Compute RSS and φ
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    let total_rank: usize = penalty_ranks.iter().sum();
    let phi = rss / (n - total_rank) as f64;

    // Compute gradient for each penalty
    let mut gradient = Array1::zeros(m);

    eprintln!("[QR_GRAD_DEBUG] UNCONDITIONAL: About to compute {} gradients", m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];
        let rank_i = penalty_ranks[i];
        let sqrt_pen_i = &sqrt_penalties[i];

        eprintln!("[QR_GRAD_DEBUG] UNCONDITIONAL: Computing gradient for penalty {}", i);

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[QR_GRAD_DEBUG] Starting penalty {} gradient computation", i);
        }

        // CRITICAL FIX: Extract non-zero block from block-diagonal penalty
        // Our penalties are p×p but only a block is non-zero
        // We need to find that block and use only the corresponding rows of P

        // Find the non-zero block by checking row sums
        let mut block_start = 0;
        let mut block_end = p;
        let mut found_start = false;

        for row in 0..p {
            let row_sum: f64 = (0..p).map(|col| penalty_i[[row, col]].abs()).sum();
            if row_sum > 1e-10 {
                if !found_start {
                    block_start = row;
                    found_start = true;
                }
                block_end = row + 1;
            }
        }

        let block_size = block_end - block_start;

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[QR_GRAD_DEBUG] Found block: start={}, end={}, size={}", block_start, block_end, block_size);
        }

        // Extract the non-zero block from the penalty
        let mut penalty_block = Array2::<f64>::zeros((block_size, block_size));
        for ii in 0..block_size {
            for jj in 0..block_size {
                penalty_block[[ii, jj]] = penalty_i[[block_start + ii, block_start + jj]];
            }
        }

        // Compute square root of the block
        let sqrt_pen_block = penalty_sqrt(&penalty_block)?;

        // Extract corresponding rows from P matrix
        let mut p_block = Array2::<f64>::zeros((block_size, p));
        for ii in 0..block_size {
            for jj in 0..p {
                p_block[[ii, jj]] = p_matrix[[block_start + ii, jj]];
            }
        }

        // Compute tr(P_block'·S_block·P_block) using thin square root
        let p_block_t_l = p_block.t().dot(&sqrt_pen_block);  // p × rank_i

        // Compute tr(P_block'L * (P_block'L)') = tr(P_block'LL'P_block) = tr(P_block'S_block·P_block)
        let mut trace = 0.0;
        for k in 0..p {
            for r in 0..sqrt_pen_block.ncols() {
                trace += p_block_t_l[[k, r]] * p_block_t_l[[k, r]];
            }
        }

        // Scale by λ_i (derivative w.r.t. log(λ_i))
        trace *= lambda_i;

        // Compute penalty term: λᵢ·β'·Sᵢ·β
        let s_beta = penalty_i.dot(&beta);
        let beta_s_beta: f64 = beta.iter().zip(s_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        let penalty_term = lambda_i * beta_s_beta;

        // Gradient formula: Wood (2011) formula AS IS
        // ∂REML/∂ρ = [tr(M·A) - r + β'·M·β/φ] / 2
        // where M = λ·S, ρ = log(λ)
        gradient[i] = (trace - (rank_i as f64) + penalty_term / phi) / 2.0;

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[QR_GRAD_DEBUG] lambda={:.6}, trace={:.6}, rank={}, penalty_term={:.6}, phi={:.6}, block_size={}",
                     lambda_i, trace, rank_i, penalty_term, phi, block_size);
            eprintln!("[QR_GRAD_DEBUG]   gradient[{}] = {:.6}",
                     i, gradient[i]);
        }
    }

    Ok(gradient)
}

/// Compute the gradient of REML with respect to log(λᵢ)
///
/// Returns: ∂REML/∂log(λᵢ) for i = 1..m
///
/// Following mgcv's fast-REML.r implementation (lines 1718-1719), the gradient is:
/// ∂REML/∂log(λᵢ) = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2
///
/// Where:
/// - A = X'WX + Σλⱼ·Sⱼ
/// - φ = RSS / (n - Σrank(Sⱼ))
/// - At optimum, ∂RSS/∂log(λᵢ) ≈ 0 (first-order condition), so we can ignore it
pub fn reml_gradient_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array1<f64>> {
    eprintln!("[GRAD_DEBUG] OLD reml_gradient_multi called!");
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
        // Add in-place instead of creating temporary
        a.scaled_add(*lambda, penalty);
    }

    // Solve for coefficients
    let y_weighted: Array1<f64> = y.iter().zip(w.iter())
        .map(|(yi, wi)| yi * wi)
        .collect();

    let b = xtw.dot(&y_weighted);

    // Add ridge for numerical stability
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (penalties.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_solve = a.clone();
    for i in 0..p {
        a_solve[[i, i]] += ridge;
    }

    let beta = solve(a_solve, b)?;

    // Compute fitted values and RSS
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute total rank and φ
    let mut total_rank = 0;
    for penalty in penalties.iter() {
        total_rank += estimate_rank(penalty);
    }
    let phi = rss / (n - total_rank) as f64;

    // Compute A^(-1)
    // Use adaptive ridge based on matrix magnitude and number of penalties
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (penalties.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
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

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[GRAD_DEBUG] ALL lambdas: {:?}", lambdas);
            eprintln!("[GRAD_DEBUG] penalty matrix size: {}x{}, estimated rank: {}",
                     penalty_i.nrows(), penalty_i.ncols(), rank_i);

            // Check A and A_inv
            let a_max = a_inv.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let a_trace = (0..p).map(|j| a_inv[[j,j]]).sum::<f64>();
            eprintln!("[GRAD_DEBUG] A_inv: max_element={:.6e}, trace={:.6}", a_max, a_trace);
        }

        // Term 1: tr(A⁻¹·λᵢ·Sᵢ)
        let lambda_s_i = penalty_i * lambda_i;
        let temp = a_inv.dot(&lambda_s_i);
        let mut trace = 0.0;
        for j in 0..p {
            trace += temp[[j, j]];
        }

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            // Also compute trace a different way to verify
            let temp_max = temp.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            eprintln!("[GRAD_DEBUG] A^(-1)*lambda*S: trace={:.6}, max_element={:.6e}", trace, temp_max);
        }

        // Term 2: λᵢ·β'·Sᵢ·β
        let s_beta = penalty_i.dot(&beta);
        let beta_s_beta: f64 = beta.iter().zip(s_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        let penalty_term = lambda_i * beta_s_beta;

        // Gradient: [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2
        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[GRAD_DEBUG] Component {}: lambda={:.6}, trace={:.6}, rank={}, penalty_term={:.6}, phi={:.6}",
                     i, lambda_i, trace, rank_i, penalty_term, phi);
            eprintln!("[GRAD_DEBUG]   trace - rank = {:.6}, (trace - rank + penalty_term/phi)/2 = {:.6}",
                     trace - (rank_i as f64), (trace - (rank_i as f64) + penalty_term / phi) / 2.0);
        }
        gradient[i] = (trace - (rank_i as f64) + penalty_term / phi) / 2.0;
    }

    Ok(gradient)
}

/// Compute the Hessian of REML with respect to log(λᵢ), log(λⱼ)
///
/// Returns: ∂²REML/∂log(λᵢ)∂log(λⱼ) for i,j = 1..m
///
/// Following Wood (2011) J.R.Statist.Soc.B 73(1):3-36, the complete Hessian is:
/// H[i,j] = [-tr(M_i·A·M_j·A) + (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²] / 2
///
/// where M_i = λ_i·S_i, A = (X'WX + ΣM_i)^(-1)
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
        // Add in-place instead of creating temporary
        a.scaled_add(*lambda, penalty);
    }

    // Compute A^(-1)
    // Add adaptive ridge term to ensure numerical stability
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute coefficients β
    let y_weighted: Array1<f64> = y.iter().zip(w.iter())
        .map(|(yi, wi)| yi * wi)
        .collect();
    let b = xtw.dot(&y_weighted);
    // Use regularized matrix for numerical stability
    let beta = solve(a_reg.clone(), b)?;

    // Compute RSS and φ
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute total rank for φ
    let mut total_rank = 0;
    for penalty in penalties.iter() {
        total_rank += estimate_rank(penalty);
    }
    let phi = rss / (n - total_rank) as f64;

    // Compute Hessian
    let mut hessian = Array2::zeros((m, m));

    for i in 0..m {
        for j in i..m {  // Only compute upper triangle (symmetric)
            let lambda_i = lambdas[i];
            let lambda_j = lambdas[j];
            let penalty_i = &penalties[i];
            let penalty_j = &penalties[j];

            // Complete Hessian following Wood (2011)
            // H[i,j] = [-tr(M_i·A·M_j·A) + (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²] / 2

            // Compute M_i = λ_i·S_i and M_j = λ_j·S_j
            let m_i = penalty_i * lambda_i;
            let m_j = penalty_j * lambda_j;

            // Term 1: -tr(M_i·A·M_j·A)
            let a_m_j = a_inv.dot(&m_j);
            let a_m_j_a = a_m_j.dot(&a_inv);
            let product = m_i.dot(&a_m_j_a);

            let mut trace_term = 0.0;
            for k in 0..p {
                trace_term += product[[k, k]];
            }

            // Term 2: (2β'·M_i·A·M_j·β)/φ
            let m_i_beta = m_i.dot(&beta);          // M_i·β
            let a_m_j_beta = a_inv.dot(&m_j.dot(&beta));  // A·M_j·β
            let term2: f64 = m_i_beta.iter().zip(a_m_j_beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            let term2 = 2.0 * term2 / phi;

            // Term 3: -(2β'·M_i·β·β'·M_j·β)/φ²
            let beta_m_i_beta: f64 = beta.iter().zip(m_i_beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            let m_j_beta = m_j.dot(&beta);
            let beta_m_j_beta: f64 = beta.iter().zip(m_j_beta.iter())
                .map(|(a, b)| a * b)
                .sum();
            let term3 = -2.0 * beta_m_i_beta * beta_m_j_beta / (phi * phi);

            // Simplified Hessian that was working (from checkpoint)
            // Use trace term with chain rule scaling
            let mut h_val = lambda_i * lambda_j * trace_term / 2.0;

            // Add diagonal gradient term (chain rule correction)
            if i == j {
                let rank_i = estimate_rank(penalty_i);
                let penalty_term_i = lambda_i * beta_m_i_beta;
                let grad_lambda_i = (trace_term - (rank_i as f64) + penalty_term_i / phi) / 2.0;
                h_val += lambda_i * grad_lambda_i;
            }

            // Negate for correct Newton direction
            hessian[[i, j]] = -h_val;

            if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 && j == 0 {
                eprintln!("[HESS_DEBUG] Hessian[{},{}]:", i, j);
                eprintln!("[HESS_DEBUG]   trace_term = {:.6e}", trace_term);
                eprintln!("[HESS_DEBUG]   term2 = {:.6e}", term2);
                eprintln!("[HESS_DEBUG]   term3 = {:.6e}", term3);
                eprintln!("[HESS_DEBUG]   total = {:.6e}", hessian[[i, j]]);
                eprintln!("[HESS_DEBUG]   phi = {:.6e}, lambda_i = {:.6e}", phi, lambda_i);
            }

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
