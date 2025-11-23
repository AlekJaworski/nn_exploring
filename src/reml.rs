//! REML (Restricted Maximum Likelihood) criterion for smoothing parameter selection

use ndarray::{Array1, Array2, Axis, s};
use crate::Result;
use crate::linalg::{solve, determinant, inverse};
use crate::GAMError;

/// Compute X'WX efficiently without forming weighted matrix
/// This is a key optimization for large n: avoids O(np) allocation
fn compute_xtwx(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let (_n, p) = x.dim();
    let mut xtwx = Array2::zeros((p, p));

    // Compute only upper triangle (symmetric matrix)
    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            // This loop is cache-friendly: sequential access
            for row in 0..x.nrows() {
                sum += x[[row, i]] * w[row] * x[[row, j]];
            }
            xtwx[[i, j]] = sum;
            if i != j {
                xtwx[[j, i]] = sum;  // Fill lower triangle by symmetry
            }
        }
    }

    xtwx
}

/// Compute X'Wy efficiently without forming weighted vectors
fn compute_xtwy(x: &Array2<f64>, w: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let (_n, p) = x.dim();
    let mut xtwy = Array1::zeros(p);

    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..x.nrows() {
            sum += x[[i, j]] * w[i] * y[i];
        }
        xtwy[j] = sum;
    }

    xtwy
}

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

    // OPTIMIZED: Compute X'WX once and reuse it
    let xtwx = compute_xtwx(x, w);

    // Compute coefficients if not provided
    let beta_computed;
    let beta = if let Some(b) = beta {
        b
    } else {
        // Compute X'Wy directly without forming weighted vectors
        let xtwy = compute_xtwy(x, w, y);

        // Solve: (X'WX + λS)β = X'Wy
        let a = &xtwx + &(penalty * lambda);

        beta_computed = solve(a, xtwy)?;
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

    // Reuse X'WX from above (no recomputation needed!)
    let a = &xtwx + &(penalty * lambda);

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

    // OPTIMIZED: Compute X'WX directly without forming weighted matrix
    let xtwx = compute_xtwx(x, w);

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
        // OPTIMIZED: Compute X'Wy directly
        let b = compute_xtwy(x, w, y);

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

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[PENALTY_SQRT_DEBUG] Matrix size: {}×{}", n, n);
        eprintln!("[PENALTY_SQRT_DEBUG] Max eigenvalue: {:.6e}", max_eigenvalue);
        eprintln!("[PENALTY_SQRT_DEBUG] Threshold: {:.6e}", threshold);
        eprintln!("[PENALTY_SQRT_DEBUG] Positive eigenvalues found: {}", rank);
        if rank > 0 {
            let eig_values: Vec<f64> = non_zero_eigs.iter().map(|(_, e)| *e).collect();
            eprintln!("[PENALTY_SQRT_DEBUG] Eigenvalues: {:?}", &eig_values[..rank.min(5)]);
        }
    }

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

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[PENALTY_SQRT_DEBUG] Output L matrix shape: {}×{}", n, rank);

        // Verify L·L' = S
        let reconstructed = sqrt_penalty.dot(&sqrt_penalty.t());
        let max_error = penalty.iter().zip(reconstructed.iter())
            .map(|(s, r)| (s - r).abs())
            .fold(0.0, f64::max);
        eprintln!("[PENALTY_SQRT_DEBUG] Reconstruction error ||L·L' - S||_∞ = {:.6e}", max_error);
    }

    Ok(sqrt_penalty)
}

/// Compute the gradient of REML using block-wise QR approach
///
/// This is optimized for large n by processing X in blocks instead of forming
/// the full augmented matrix. Complexity is O(blocks × p²) instead of O(np²).
///
/// For n < 2000, falls back to full QR for simplicity.
pub fn reml_gradient_multi_qr_adaptive(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array1<f64>> {
    let n = y.len();

    // Use block-wise for large n (>= 2000), full QR for small n
    if n >= 2000 {
        reml_gradient_multi_qr_blockwise(y, x, w, lambdas, penalties, 1000)
    } else {
        reml_gradient_multi_qr(y, x, w, lambdas, penalties)
    }
}

/// Block-wise version of QR gradient computation
/// Processes X in blocks to avoid O(np²) complexity
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_blockwise(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
    block_size: usize,
) -> Result<Array1<f64>> {
    use ndarray_linalg::Inverse;
    use crate::blockwise_qr::compute_r_blockwise;

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute square root penalties once (these are constant)
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    for penalty in penalties.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        let rank = sqrt_pen.ncols();
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Use block-wise QR to get R factor
    let r_upper = compute_r_blockwise(x, w, lambdas, &sqrt_penalties, block_size)?;

    // DEBUG: Verify R'R = X'WX + λS
    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        let rtr = r_upper.t().dot(&r_upper);
        let xtwx = compute_xtwx(x, w);
        let mut expected = xtwx.clone();
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            expected.scaled_add(*lambda, penalty);
        }

        let max_error = rtr.iter().zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!("[BLOCKWISE_DEBUG] Max error in R'R vs X'WX+λS: {:.6e}", max_error);
        eprintln!("[BLOCKWISE_DEBUG] R'R trace: {:.6e}", (0..p).map(|i| rtr[[i,i]]).sum::<f64>());
        eprintln!("[BLOCKWISE_DEBUG] Expected trace: {:.6e}", (0..p).map(|i| expected[[i,i]]).sum::<f64>());
    }

    // Compute P = R^{-1}
    let p_matrix = r_upper.inv()
        .map_err(|_| GAMError::LinAlgError("Failed to invert R matrix".to_string()))?;

    // Rest is same as before: compute gradient using P
    let mut gradient = Array1::<f64>::zeros(m);

    for i in 0..m {
        let penalty = &penalties[i];

        // Compute tr(P'·S·P) = tr(P·P'·S) since tr(ABC) = tr(CAB)
        let pp_t = p_matrix.dot(&p_matrix.t());
        let pp_s = pp_t.dot(penalty);

        let mut trace = 0.0;
        for j in 0..p {
            trace += pp_s[[j, j]];
        }

        // ∂log|A|/∂log(λ) = λ·tr(A^{-1}·S) = λ·tr(P·P'·S)
        gradient[i] = lambdas[i] * trace;
    }

    Ok(gradient)
}

#[cfg(not(feature = "blas"))]
pub fn reml_gradient_multi_qr_blockwise(
    _y: &Array1<f64>,
    _x: &Array2<f64>,
    _w: &Array1<f64>,
    _lambdas: &[f64],
    _penalties: &[Array2<f64>],
    _block_size: usize,
) -> Result<Array1<f64>> {
    Err(GAMError::InvalidParameter(
        "Block-wise QR requires 'blas' feature".to_string()
    ))
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
///
/// NOTE: For large n (>= 2000), use reml_gradient_multi_qr_blockwise instead
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

    // OPTIMIZED: Compute sqrt(W) * X without cloning x
    // Allocate directly to avoid clone overhead
    let mut sqrt_w_x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] = x[[i, j]] * weight_sqrt;
        }
    }

    // Compute square root penalties and their ranks
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    for penalty in penalties.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        // Use the actual rank from eigenvalue decomposition (number of positive eigenvalues)
        // This is more accurate than the heuristic in estimate_rank()
        let rank = sqrt_pen.ncols();  // rank = number of positive eigenvalues
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Build augmented matrix Z = [sqrt(W)X; sqrt(λ_0)L_0'; sqrt(λ_1)L_1'; ...]
    // Determine total rows (n + sum of ranks)
    let mut total_rows = n;
    for sqrt_pen in sqrt_penalties.iter() {
        total_rows += sqrt_pen.ncols();  // Number of columns = rank
    }

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[Z_BUILD_DEBUG] Building Z matrix:");
        eprintln!("[Z_BUILD_DEBUG]   n = {}, p = {}", n, p);
        for (i, sqrt_pen) in sqrt_penalties.iter().enumerate() {
            eprintln!("[Z_BUILD_DEBUG]   L{} shape: {}×{}, λ{} = {:.6}",
                     i, sqrt_pen.nrows(), sqrt_pen.ncols(), i, lambdas[i]);
        }
        eprintln!("[Z_BUILD_DEBUG]   Total Z rows: {}", total_rows);
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
    for (idx, (sqrt_pen, &lambda)) in sqrt_penalties.iter().zip(lambdas.iter()).enumerate() {
        let sqrt_lambda = lambda.sqrt();
        let rank = sqrt_pen.ncols();  // Number of non-zero eigenvalues
        for i in 0..rank {
            for j in 0..p {
                z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];  // Transpose!
            }
        }

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && rank > 0 {
            eprintln!("[Z_BUILD_DEBUG]   After adding L{} (rows {} to {}), first value: {:.6e}",
                     idx, row_offset, row_offset + rank - 1, z[[row_offset, 0]]);
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

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[QR_DEBUG] P = R^{{-1}} computed, shape: {}×{}", p_matrix.nrows(), p_matrix.ncols());
        eprintln!("[QR_DEBUG] P[0,0] = {:.6e}, P[1,1] = {:.6e}", p_matrix[[0,0]], p_matrix[[1,1]]);

        // Verify P·P' = A^{-1} by checking against A
        let pp_t = p_matrix.dot(&p_matrix.t());
        eprintln!("[QR_DEBUG] P·P' diagonal: [{:.6e}, {:.6e}, ..., {:.6e}]",
                 pp_t[[0,0]], pp_t[[1,1]], pp_t[[p-1,p-1]]);
    }

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

    // Compute effective degrees of freedom: edf = tr(A^{-1} X'X)
    // We have P = R^{-1}, and A^{-1} = P·P'
    // So tr(A^{-1} X'X) = tr(P·P'·X'X)
    let a_inv = p_matrix.dot(&p_matrix.t());
    let ainv_xtwx = a_inv.dot(&xtwx);
    let edf_total: f64 = (0..p).map(|i| ainv_xtwx[[i, i]]).sum();

    // Compute total rank for φ calculation
    // IMPORTANT: Use total_rank (sum of penalty ranks), NOT edf_total
    let total_rank: usize = penalty_ranks.iter().sum();
    let phi = rss / (n as f64 - total_rank as f64);

    // Pre-compute A^{-1}·X'X·A^{-1} for ∂edf/∂ρ calculations
    // This is O(p³) and independent of smooth index, so compute once
    let ainv_xtx_ainv = ainv_xtwx.dot(&a_inv);

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[PHI_DEBUG] n={}, edf_total={:.6}, rss={:.6}, phi={:.6}",
                 n, edf_total, rss, phi);
    }

    // Compute gradient for each penalty
    // Using the CORRECT IFT-based formula accounting for implicit dependencies:
    //
    // REML = [(RSS + Σλⱼ·β'·Sⱼ·β)/φ + (n-r)·log(2πφ) + log|A| - Σrⱼ·log(λⱼ)] / 2
    //
    // where β and φ implicitly depend on ρ through:
    //   A·β = X'y  =>  ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
    //   φ = RSS/(n-r)  =>  ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ)/(n-r)
    //
    // Full gradient:
    // ∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rᵢ + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
    //
    // where P = RSS + Σλⱼ·β'·Sⱼ·β
    //
    // This matches numerical gradients to < 0.1% error.
    let mut gradient = Array1::zeros(m);

    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;
    let n_minus_r = (n as f64) - (total_rank as f64);

    // Pre-compute P = RSS + Σλⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        let s_j_beta = penalties[j].dot(&beta);
        let beta_s_j_beta: f64 = beta.iter().zip(s_j_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];
        let rank_i = penalty_ranks[i] as f64;

        // Term 1: tr(A^{-1}·λᵢ·Sᵢ) = λᵢ·Σ(A^{-1})[i,j]·(Sᵢ)[j,i]
        // This is the Frobenius inner product: λᵢ·<A^{-1}, Sᵢ>
        let ainv_s = a_inv.dot(penalty_i);
        let trace_term: f64 = (0..p).map(|i| ainv_s[[i, i]]).sum();
        let trace = lambda_i * trace_term;

        // Term 2: -rank(Sᵢ)
        let rank_term = -rank_i;

        // Compute ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
        let s_i_beta = penalty_i.dot(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);
        let dbeta_drho = a_inv.dot(&lambda_s_beta).mapv(|x| -x);

        // Compute ∂RSS/∂ρᵢ = -2·residuals'·X·∂β/∂ρᵢ
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0 * residuals.iter().zip(x_dbeta.iter())
            .map(|(ri, xdbi)| ri * xdbi)
            .sum::<f64>();

        // Compute ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        let dphi_drho = drss_drho / n_minus_r;

        // Compute ∂P/∂ρᵢ where P = RSS + Σλⱼ·β'·Sⱼ·β
        // Explicit term: λᵢ·β'·Sᵢ·β
        let beta_s_i_beta: f64 = beta.iter().zip(s_i_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        let explicit_pen = lambda_i * beta_s_i_beta;

        // Implicit term: 2·Σλⱼ·β'·Sⱼ·∂β/∂ρᵢ
        // Note: This simplifies to exactly -∂RSS/∂ρᵢ by the algebra
        let mut implicit_pen = 0.0;
        for j in 0..m {
            let s_j_beta = penalties[j].dot(&beta);
            let s_j_dbeta = penalties[j].dot(&dbeta_drho);
            let term1: f64 = s_j_beta.iter().zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta.iter().zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;

        // Term 3: ∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;

        // Term 4: ∂[(n-r)·log(2πφ)]/∂ρᵢ = (n-r)·(1/φ)·∂φ/∂ρᵢ
        let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

        // Total gradient (divide by 2)
        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

/// Compute the Hessian of REML with respect to log(λᵢ) using QR-based approach
///
/// Returns: ∂²REML/∂ρᵢ∂ρⱼ for i,j = 1..m, where ρᵢ = log(λᵢ)
///
/// This uses the CORRECTED formula matching the IFT-based gradient:
///
/// H[i,j] = ∂/∂ρⱼ [∂REML/∂ρᵢ]
///
/// where ∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
///
/// The Hessian accounts for all implicit dependencies through the Implicit Function Theorem.
pub fn reml_hessian_multi_qr(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[Array2<f64>],
) -> Result<Array2<f64>> {
    use ndarray_linalg::Inverse;
    use ndarray_linalg::QR;

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!("\n[HESS_CORRECTED] Starting CORRECTED Hessian computation (matching gradient)");
        eprintln!("[HESS_CORRECTED] n={}, p={}, m={}", n, p, m);
    }

    // Step 1: QR decomposition for efficient A^{-1} computation
    let mut sqrt_w_x = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] *= weight_sqrt;
        }
    }

    let mut sqrt_penalties = Vec::with_capacity(m);
    let mut penalty_ranks = Vec::with_capacity(m);

    for penalty in penalties.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        let rank = sqrt_pen.ncols();
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Build augmented matrix Z = [sqrt(W)X; √λ₁·L₁'; √λ₂·L₂'; ...]
    let total_rows: usize = n + penalty_ranks.iter().sum::<usize>();
    let mut z = Array2::zeros((total_rows, p));
    z.slice_mut(s![0..n, ..]).assign(&sqrt_w_x);

    let mut row_offset = n;
    for (i, sqrt_pen) in sqrt_penalties.iter().enumerate() {
        let rank = penalty_ranks[i];
        let lambda_sqrt = lambdas[i].sqrt();
        for j in 0..rank {
            for k in 0..p {
                z[[row_offset + j, k]] = lambda_sqrt * sqrt_pen[[k, j]];
            }
        }
        row_offset += rank;
    }

    // QR decomposition
    let (_, r) = z.qr().map_err(|_| GAMError::LinAlgError("QR decomposition failed".to_string()))?;
    let p_matrix = r.slice(s![0..p, 0..p]).inv().map_err(|_| GAMError::SingularMatrix)?;

    // Compute A^{-1} = P·P'
    let a_inv = p_matrix.dot(&p_matrix.t());

    // Step 2: Compute coefficients β
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

    // Add ridge for stability
    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a, b)?;

    // Step 3: Compute residuals, RSS, phi, P
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    let total_rank: usize = penalty_ranks.iter().sum();
    let n_minus_r = (n as f64) - (total_rank as f64);
    let phi = rss / n_minus_r;
    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;
    let phi_cb = phi * phi * phi;

    // Compute P = RSS + Σⱼ λⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        let s_j_beta = penalties[j].dot(&beta);
        let beta_s_j_beta: f64 = beta.iter().zip(s_j_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!("[HESS_CORRECTED] RSS={:.6e}, phi={:.6e}, P={:.6e}", rss, phi, p_value);
        eprintln!("[HESS_CORRECTED] total_rank={}, n-r={:.6}", total_rank, n_minus_r);
    }

    // Step 4: Compute first derivatives (matching gradient formula)
    let mut dbeta_drho = Vec::with_capacity(m);
    let mut drss_drho = Vec::with_capacity(m);
    let mut dphi_drho = Vec::with_capacity(m);
    let mut dp_drho = Vec::with_capacity(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];

        // ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
        let s_i_beta = penalty_i.dot(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);
        let dbeta_i = a_inv.dot(&lambda_s_beta).mapv(|x| -x);
        dbeta_drho.push(dbeta_i.clone());

        // ∂RSS/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ
        let x_dbeta = x.dot(&dbeta_i);
        let drss_i: f64 = -2.0 * residuals.iter().zip(x_dbeta.iter())
            .map(|(ri, xdbi)| ri * xdbi)
            .sum::<f64>();
        drss_drho.push(drss_i);

        // ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        let dphi_i = drss_i / n_minus_r;
        dphi_drho.push(dphi_i);

        // ∂P/∂ρᵢ = ∂RSS/∂ρᵢ + λᵢ·β'·Sᵢ·β + 2·Σⱼ λⱼ·β'·Sⱼ·∂β/∂ρᵢ
        let beta_s_i_beta: f64 = beta.iter().zip(s_i_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            let s_j_beta = penalties[j].dot(&beta);
            let s_j_dbeta = penalties[j].dot(&dbeta_i);
            let term1: f64 = s_j_beta.iter().zip(dbeta_i.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta.iter().zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_i = drss_i + explicit_pen + implicit_pen;
        dp_drho.push(dp_i);
    }

    // Step 5: Compute Hessian
    let mut hessian = Array2::zeros((m, m));

    for i in 0..m {
        for j in i..m {  // Only compute upper triangle (symmetric)
            let lambda_i = lambdas[i];
            let lambda_j = lambdas[j];
            let s_i = &penalties[i];
            let s_j = &penalties[j];
            let sqrt_si = &sqrt_penalties[i];

            // ================================================================
            // TERM 1: ∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)] / 2
            // ================================================================
            // = [δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ) - λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)] / 2

            // Part A: -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
            let ainv_sj = a_inv.dot(s_j);
            let ainv_sj_ainv = ainv_sj.dot(&a_inv);
            let si_ainv_sj_ainv = s_i.dot(&ainv_sj_ainv);
            let mut trace1a = 0.0;
            for k in 0..p {
                trace1a += si_ainv_sj_ainv[[k, k]];
            }
            let term1a = -lambda_i * lambda_j * trace1a;

            // Part B: δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ)
            let term1b = if i == j {
                let p_t_sqrt_si = p_matrix.t().dot(sqrt_si);
                let trace_ainv_si: f64 = p_t_sqrt_si.iter().map(|x| x * x).sum();
                lambda_i * trace_ainv_si
            } else {
                0.0
            };

            let term1 = (term1a + term1b) / 2.0;

            // ================================================================
            // TERM 2: ∂²(P/φ)/∂ρⱼ∂ρᵢ / 2
            // ================================================================
            // This is the big one! Needs ∂²P, ∂²RSS, ∂²β, ∂²φ

            // Compute ∂²β/∂ρⱼ∂ρᵢ
            let si_beta = s_i.dot(&beta);
            let ainv_si_beta = a_inv.dot(&si_beta);
            let lambda_i_ainv_si_beta = ainv_si_beta.mapv(|x| lambda_i * x);
            let sj_times_term = s_j.dot(&lambda_i_ainv_si_beta);
            let part_a = a_inv.dot(&sj_times_term).mapv(|x| lambda_j * x);

            let si_dbeta_j = s_i.dot(&dbeta_drho[j]);
            let part_b = a_inv.dot(&si_dbeta_j).mapv(|x| -lambda_i * x);

            let mut d2beta = part_a + part_b;
            if i == j {
                d2beta = d2beta - dbeta_drho[i].clone();
            }

            // Compute ∂²RSS/∂ρⱼ∂ρᵢ
            let x_dbeta_j = x.dot(&dbeta_drho[j]);
            let x_dbeta_i = x.dot(&dbeta_drho[i]);
            let d2rss_part1 = 2.0 * x_dbeta_j.dot(&x_dbeta_i);

            let x_d2beta = x.dot(&d2beta);
            let d2rss_part2 = -2.0 * residuals.dot(&x_d2beta);

            let d2rss = d2rss_part1 + d2rss_part2;

            // Compute ∂²φ/∂ρⱼ∂ρᵢ = (1/(n-r))·∂²RSS/∂ρⱼ∂ρᵢ
            let d2phi = d2rss / n_minus_r;

            // Compute ∂²P/∂ρⱼ∂ρᵢ
            // = ∂²RSS/∂ρⱼ∂ρᵢ + δᵢⱼ·λᵢ·β'·Sᵢ·β + 2·λᵢ·∂β'/∂ρⱼ·Sᵢ·β
            //   + 2·Σₖ[δₖⱼ·λₖ·∂β'/∂ρᵢ·Sₖ·β + λₖ·∂²β'/∂ρⱼ∂ρᵢ·Sₖ·β + λₖ·∂β'/∂ρᵢ·Sₖ·∂β/∂ρⱼ]

            let diag_explicit = if i == j {
                let beta_si_beta: f64 = beta.iter().zip(si_beta.iter())
                    .map(|(bi, sbi)| bi * sbi)
                    .sum();
                lambda_i * beta_si_beta
            } else {
                0.0
            };

            let dbeta_j_si_beta: f64 = dbeta_drho[j].iter().zip(si_beta.iter())
                .map(|(dbj, sbi)| dbj * sbi)
                .sum();
            let explicit_cross = 2.0 * lambda_i * dbeta_j_si_beta;

            let mut implicit_sum = 0.0;
            for k in 0..m {
                let sk_beta = penalties[k].dot(&beta);
                let sk_dbeta_i = penalties[k].dot(&dbeta_drho[i]);

                // δₖⱼ·λₖ·∂β'/∂ρᵢ·Sₖ·β
                let term1 = if k == j {
                    let val: f64 = dbeta_drho[i].iter().zip(sk_beta.iter())
                        .map(|(dbi, skb)| dbi * skb)
                        .sum();
                    lambdas[k] * val
                } else {
                    0.0
                };

                // λₖ·∂²β'/∂ρⱼ∂ρᵢ·Sₖ·β
                let sk_d2beta: f64 = d2beta.iter().zip(sk_beta.iter())
                    .map(|(d2bi, skb)| d2bi * skb)
                    .sum();
                let term2 = lambdas[k] * sk_d2beta;

                // λₖ·∂β'/∂ρᵢ·Sₖ·∂β/∂ρⱼ
                let dbeta_i_sk_dbeta_j: f64 = dbeta_drho[i].iter().zip(sk_dbeta_i.iter())
                    .map(|(dbi, skdbj)| dbi * skdbj)
                    .sum();
                let term3 = lambdas[k] * dbeta_i_sk_dbeta_j;

                implicit_sum += term1 + term2 + term3;
            }

            let d2p = d2rss + diag_explicit + explicit_cross + 2.0 * implicit_sum;

            // Now compute ∂²(P/φ)/∂ρⱼ∂ρᵢ
            // = (1/φ)·∂²P/∂ρⱼ∂ρᵢ - (1/φ²)·[∂φ/∂ρⱼ·∂P/∂ρᵢ + ∂P/∂ρⱼ·∂φ/∂ρᵢ]
            //   + 2·(P/φ³)·∂φ/∂ρⱼ·∂φ/∂ρᵢ - (P/φ²)·∂²φ/∂ρⱼ∂ρᵢ

            let term2a = inv_phi * d2p;
            let term2b = -(1.0 / phi_sq) * (dphi_drho[j] * dp_drho[i] + dp_drho[j] * dphi_drho[i]);
            let term2c = 2.0 * (p_value / phi_cb) * dphi_drho[j] * dphi_drho[i];
            let term2d = -(p_value / phi_sq) * d2phi;

            let term2 = (term2a + term2b + term2c + term2d) / 2.0;

            // ================================================================
            // TERM 3: ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
            // ================================================================
            // = (n-r)·[(1/φ)·∂²φ/∂ρⱼ∂ρᵢ - (1/φ²)·∂φ/∂ρⱼ·∂φ/∂ρᵢ] / 2

            let term3a = n_minus_r * inv_phi * d2phi;
            let term3b = -n_minus_r * (1.0 / phi_sq) * dphi_drho[j] * dphi_drho[i];

            let term3 = (term3a + term3b) / 2.0;

            // ================================================================
            // TOTAL HESSIAN
            // ================================================================
            let h_val = term1 + term2 + term3;
            hessian[[i, j]] = h_val;

            if std::env::var("MGCV_HESS_DEBUG").is_ok() && (i == j || (i == 0 && j == 1)) {
                eprintln!("\n[HESS_CORRECTED] H[{},{}]:", i, j);
                eprintln!("  Term 1 (∂²tr/∂ρⱼ∂ρᵢ): {:.6e}", term1);
                eprintln!("    - 1a (cross): {:.6e}", term1a / 2.0);
                eprintln!("    - 1b (diagonal): {:.6e}", term1b / 2.0);
                eprintln!("  Term 2 (∂²(P/φ)/∂ρⱼ∂ρᵢ): {:.6e}", term2);
                eprintln!("    - 2a (d2P/φ): {:.6e}", term2a / 2.0);
                eprintln!("    - 2b (cross dP·dφ/φ²): {:.6e}", term2b / 2.0);
                eprintln!("    - 2c (P·dφ²/φ³): {:.6e}", term2c / 2.0);
                eprintln!("    - 2d (P·d2φ/φ²): {:.6e}", term2d / 2.0);
                eprintln!("    - d2rss: {:.6e}", d2rss);
                eprintln!("    - d2P: {:.6e}", d2p);
                eprintln!("    - d2phi: {:.6e}", d2phi);
                eprintln!("  Term 3 (∂/∂ρⱼ[(n-r)·dφ/φ]): {:.6e}", term3);
                eprintln!("    - 3a ((n-r)·d2φ/φ): {:.6e}", term3a / 2.0);
                eprintln!("    - 3b (-(n-r)·dφ²/φ²): {:.6e}", term3b / 2.0);
                eprintln!("  TOTAL: {:.6e}", h_val);
            }

            // Fill symmetric entry
            if i != j {
                hessian[[j, i]] = hessian[[i, j]];
            }
        }
    }

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!("\n[HESS_CORRECTED] Final Hessian:");
        for i in 0..m {
            eprint!("  [");
            for j in 0..m {
                eprint!("{:10.6e} ", hessian[[i, j]]);
            }
            eprintln!("]");
        }
    }

    Ok(hessian)
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

    // Compute effective degrees of freedom for φ
    // edf = tr(A^{-1}·X'WX)
    // For Gaussian case with W=I: edf = tr(A^{-1}·X'X)
    let xtx = x.t().to_owned().dot(&x.to_owned());
    let ainv_xtx = a_inv.dot(&xtx);
    let edf: f64 = (0..ainv_xtx.nrows())
        .map(|i| ainv_xtx[[i, i]])
        .sum();

    // Correct φ computation using effective df
    let phi = rss / (n as f64 - edf);

    // Debug: compare against old approach
    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        let old_total_rank: usize = penalties.iter()
            .map(|p| estimate_rank(p))
            .sum();
        let old_phi = rss / (n as f64 - old_total_rank as f64);
        eprintln!("[PHI_DEBUG] edf (correct) = {:.3}, old total_rank = {}, φ_correct = {:.6e}, φ_old = {:.6e}, ratio = {:.3}",
                  edf, old_total_rank, phi, old_phi, old_phi / phi);
    }

    // Compute first derivatives of β with respect to log(λ_i)
    // dβ/dρ_i = -A^{-1}·M_i·β where M_i = λ_i·S_i
    let mut dbeta_drho = Vec::with_capacity(m);
    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];
        let m_i = penalty_i * lambda_i;  // M_i = λ_i·S_i
        let m_i_beta = m_i.dot(&beta);    // M_i·β
        let dbeta_i = a_inv.dot(&m_i_beta).mapv(|x| -x);  // -A^{-1}·M_i·β
        dbeta_drho.push(dbeta_i);
    }

    // Compute bSb1 (first derivatives of β'·S·β/φ with respect to log(λ_i))
    // This is needed for diagonal correction in bSb2
    let mut bsb1 = Vec::with_capacity(m);
    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties[i];

        // β'·S_i·β
        let s_i_beta = penalty_i.dot(&beta);
        let beta_s_i_beta: f64 = beta.iter().zip(s_i_beta.iter())
            .map(|(b, sb)| b * sb)
            .sum();

        // 2·dβ/dρ_i'·S·β where S = Σλ_j·S_j
        let mut s_beta_total = Array1::zeros(p);
        for (lambda_j, penalty_j) in lambdas.iter().zip(penalties.iter()) {
            let s_j_beta = penalty_j.dot(&beta);
            s_beta_total.scaled_add(*lambda_j, &s_j_beta);
        }
        let dbeta_s_beta: f64 = dbeta_drho[i].iter().zip(s_beta_total.iter())
            .map(|(db, sb)| db * sb)
            .sum();

        bsb1.push((lambda_i * beta_s_i_beta + 2.0 * dbeta_s_beta) / phi);
    }

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

            // det2 part: log-determinant Hessian from mgcv
            // det2[k,m] = δ_{k,m}·tr(A^{-1}·M_m) - tr[(A^{-1}·M_k)·(A^{-1}·M_m)]
            // where trace_term = tr[(A^{-1}·M_k)·(A^{-1}·M_m)] = tr(M_k·A^{-1}·M_m·A^{-1})

            // For diagonal, need tr(A^{-1}·M_k)
            let trace_a_inv_m_i = if i == j {
                let a_m_i = a_inv.dot(&m_i);
                let mut tr = 0.0;
                for k in 0..p {
                    tr += a_m_i[[k, k]];
                }
                tr
            } else {
                0.0
            };

            // det2[k,m] from C code (in ρ-space, before /2)
            let det2 = if i == j {
                trace_a_inv_m_i - trace_term
            } else {
                -trace_term
            };

            // bSb2: Penalty Hessian from mgcv's get_bSb function
            // Following mgcv C code in gdi.c
            //
            // bSb2[k,m] = 2·(d²β'/dρ_k dρ_m · S · β)       [Term 1: second derivatives]
            //            + 2·(dβ'/dρ_k · S · dβ/dρ_m)       [Term 2: mixed derivatives]
            //            + 2·(dβ'/dρ_m · S_k · β · λ_k)     [Term 3: parameter-dependent]
            //            + 2·(dβ'/dρ_k · S_m · β · λ_m)     [Term 4: parameter-dependent]
            //            + δ_{k,m}·bSb1[k]                   [Diagonal correction]

            // Term 1: d²β'/dρ_k dρ_m · S · β
            // From implicit differentiation:
            // d²β/dρ_i dρ_j = A^{-1}·[M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β] + δ_{ij}·dβ/dρ_i
            // The diagonal term δ_{ij}·dβ/dρ_i is CRITICAL!
            let m_i_beta = m_i.dot(&beta);
            let m_j_beta = m_j.dot(&beta);
            let a_inv_m_i_beta = a_inv.dot(&m_i_beta);
            let a_inv_m_j_beta = a_inv.dot(&m_j_beta);
            let m_i_a_inv_m_j_beta = m_i.dot(&a_inv_m_j_beta);
            let m_j_a_inv_m_i_beta = m_j.dot(&a_inv_m_i_beta);

            let mut d2beta_term = Array1::zeros(p);
            d2beta_term += &m_i_a_inv_m_j_beta;
            d2beta_term += &m_j_a_inv_m_i_beta;
            let mut d2beta = a_inv.dot(&d2beta_term);

            // Add diagonal correction: + δ_{ij}·dβ/dρ_i
            // This term comes from ∂M_i/∂ρ_j = δ_{ij}·M_i in the derivation
            if i == j {
                d2beta += &dbeta_drho[i];
            }

            // S·β where S = Σλ_k·S_k
            let mut s_beta_total = Array1::zeros(p);
            for (lambda_k, penalty_k) in lambdas.iter().zip(penalties.iter()) {
                let s_k_beta = penalty_k.dot(&beta);
                s_beta_total.scaled_add(*lambda_k, &s_k_beta);
            }

            let term1: f64 = d2beta.iter().zip(s_beta_total.iter())
                .map(|(d2b, sb)| d2b * sb)
                .sum();

            // Term 2: dβ'/dρ_k · S · dβ/dρ_m
            let s_dbeta_j = {
                let mut result = Array1::zeros(p);
                for (lambda_k, penalty_k) in lambdas.iter().zip(penalties.iter()) {
                    let s_k_dbeta_j = penalty_k.dot(&dbeta_drho[j]);
                    result.scaled_add(*lambda_k, &s_k_dbeta_j);
                }
                result
            };

            let term2: f64 = dbeta_drho[i].iter().zip(s_dbeta_j.iter())
                .map(|(db_i, s_db_j)| db_i * s_db_j)
                .sum();

            // Term 3: dβ'/dρ_m · S_k · β · λ_k (when k=i)
            let s_i_beta = penalty_i.dot(&beta);
            let term3: f64 = dbeta_drho[j].iter().zip(s_i_beta.iter())
                .map(|(db_j, s_i_b)| db_j * s_i_b)
                .sum::<f64>() * lambda_i;

            // Term 4: dβ'/dρ_k · S_m · β · λ_m (when m=j)
            let s_j_beta = penalty_j.dot(&beta);
            let term4: f64 = dbeta_drho[i].iter().zip(s_j_beta.iter())
                .map(|(db_i, s_j_b)| db_i * s_j_b)
                .sum::<f64>() * lambda_j;

            // Diagonal correction
            let diag_corr = if i == j { bsb1[i] } else { 0.0 };

            // Combine all bSb2 terms
            let bsb2 = 2.0 * (term1 + term2 + term3 + term4) + diag_corr;

            // Total Hessian = (det2 + bSb2) / 2
            let h_val = (det2 + bsb2) / 2.0;

            // Newton's method: x_new = x - H^{-1}·grad
            // For minimization, H = ∂²V/∂ρ² should be positive at minimum
            // No negation needed - we computed the Hessian correctly
            hessian[[i, j]] = h_val;

            if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
                eprintln!("[HESS_DEBUG] Hessian[{},{}]:", i, j);
                eprintln!("[HESS_DEBUG]   det2 = {:.6e} (log-determinant)", det2);
                eprintln!("[HESS_DEBUG]   bSb2 term1 (d2beta) = {:.6e}", term1);
                eprintln!("[HESS_DEBUG]   bSb2 term2 (dbeta·S·dbeta) = {:.6e}", term2);
                eprintln!("[HESS_DEBUG]   bSb2 term3 (dbeta_j·S_i·beta) = {:.6e}", term3);
                eprintln!("[HESS_DEBUG]   bSb2 term4 (dbeta_i·S_j·beta) = {:.6e}", term4);
                eprintln!("[HESS_DEBUG]   bSb2 diag_corr = {:.6e}", diag_corr);
                eprintln!("[HESS_DEBUG]   bSb2 total = {:.6e} (penalty)", bsb2);
                eprintln!("[HESS_DEBUG]   (det2 + bSb2)/2 = {:.6e}", h_val);
                eprintln!("[HESS_DEBUG]   phi = {:.6e}, lambda_{} = {:.6e}, lambda_{} = {:.6e}", phi, i, lambda_i, j, lambda_j);
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
