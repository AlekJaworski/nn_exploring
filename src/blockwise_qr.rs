/// Block-wise QR decomposition for large datasets
///
/// Instead of forming the full augmented matrix Z = [sqrt(W)X; sqrt(λ)L; ...]
/// and computing QR all at once, we process X in blocks and update R incrementally.
///
/// This reduces complexity from O(np²) to O(blocks × p²) where blocks << n
///
/// Algorithm (following Wood et al. 2015):
/// 1. Start with R₀ from penalty: R₀'R₀ = λS
/// 2. For each block B of sqrt(W)X:
///    - Compute QR of [R_old; B] to get [Q; R_new]
///    - R_new'R_new = R_old'R_old + B'B
/// 3. Final R satisfies R'R = X'WX + λS
use ndarray::{Array1, Array2};
use crate::{Result, GAMError};

#[cfg(feature = "blas")]
use ndarray_linalg::QR;

/// Compute R factor incrementally from blocks
/// Returns R such that R'R = X'WX + λS
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `w` - Weights (n)
/// * `lambdas` - Smoothing parameters
/// * `sqrt_penalties` - Square root penalty matrices (pre-computed)
/// * `block_size` - Number of rows to process at once (default: 1000)
#[cfg(feature = "blas")]
pub fn compute_r_blockwise(
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    sqrt_penalties: &[Array2<f64>],
    block_size: usize,
) -> Result<Array2<f64>> {
    let (n, p) = x.dim();
    let m = lambdas.len();

    // Step 1: Initialize R from penalties
    // R₀'R₀ = Σ λᵢSᵢ = Σ λᵢ(Lᵢ'Lᵢ) where Lᵢ is sqrt(Sᵢ)
    // We build augmented matrix [sqrt(λ₀)L₀'; sqrt(λ₁)L₁'; ...] and QR it

    // Count total rows for penalty part
    let mut penalty_rows = 0;
    for sqrt_pen in sqrt_penalties.iter() {
        penalty_rows += sqrt_pen.ncols(); // rank of penalty
    }

    if penalty_rows > 0 {
        // Build penalty augmentation matrix
        let mut z_penalty = Array2::<f64>::zeros((penalty_rows, p));
        let mut row_offset = 0;

        for (sqrt_pen, &lambda) in sqrt_penalties.iter().zip(lambdas.iter()) {
            let sqrt_lambda = lambda.sqrt();
            let rank = sqrt_pen.ncols();

            // Fill in sqrt(λ)L' (transposed)
            for i in 0..rank {
                for j in 0..p {
                    z_penalty[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];
                }
            }
            row_offset += rank;
        }

        // Get R₀ from QR of penalty part
        let (_, r) = z_penalty.qr()
            .map_err(|_| GAMError::LinAlgError("QR decomposition failed".to_string()))?;

        // Extract upper triangular part (first p rows)
        let mut r_current = Array2::<f64>::zeros((p, p));
        let r_rows = r.nrows();
        let r_cols = r.ncols();

        for i in 0..p.min(r_rows) {
            for j in i..p.min(r_cols) {
                r_current[[i, j]] = r[[i, j]];
            }
        }

        // Step 2: Process X in blocks and update R
        let num_blocks = (n + block_size - 1) / block_size;

        for block_idx in 0..num_blocks {
            let start_row = block_idx * block_size;
            let end_row = ((block_idx + 1) * block_size).min(n);
            let block_rows = end_row - start_row;

            // Extract block and weight it
            let mut block = Array2::<f64>::zeros((block_rows, p));
            for i in 0..block_rows {
                let weight_sqrt = w[start_row + i].sqrt();
                for j in 0..p {
                    block[[i, j]] = x[[start_row + i, j]] * weight_sqrt;
                }
            }

            // Stack [R_current; block] and compute QR
            let total_rows = p + block_rows;
            let mut stacked = Array2::<f64>::zeros((total_rows, p));

            // Copy R_current into top
            for i in 0..p {
                for j in 0..p {
                    stacked[[i, j]] = r_current[[i, j]];
                }
            }

            // Copy block into bottom
            for i in 0..block_rows {
                for j in 0..p {
                    stacked[[p + i, j]] = block[[i, j]];
                }
            }

            // QR decomposition to get updated R
            let (_, r_new) = stacked.qr()
                .map_err(|_| GAMError::LinAlgError("Block QR update failed".to_string()))?;

            // Extract new R (upper triangular, first p rows)
            // Note: r_new dimensions are min(total_rows, p) × p
            let r_rows = r_new.nrows();
            let r_cols = r_new.ncols();

            for i in 0..p.min(r_rows) {
                for j in i..p.min(r_cols) {
                    r_current[[i, j]] = r_new[[i, j]];
                }
            }
        }

        Ok(r_current)
    } else {
        // No penalties, just process X in blocks
        // Start with empty R (will be built from first block)
        let num_blocks = (n + block_size - 1) / block_size;
        let mut r_current: Option<Array2<f64>> = None;

        for block_idx in 0..num_blocks {
            let start_row = block_idx * block_size;
            let end_row = ((block_idx + 1) * block_size).min(n);
            let block_rows = end_row - start_row;

            // Extract block and weight it
            let mut block = Array2::<f64>::zeros((block_rows, p));
            for i in 0..block_rows {
                let weight_sqrt = w[start_row + i].sqrt();
                for j in 0..p {
                    block[[i, j]] = x[[start_row + i, j]] * weight_sqrt;
                }
            }

            if let Some(ref r_old) = r_current {
                // Stack [R_old; block] and compute QR
                let total_rows = p + block_rows;
                let mut stacked = Array2::<f64>::zeros((total_rows, p));

                for i in 0..p {
                    for j in 0..p {
                        stacked[[i, j]] = r_old[[i, j]];
                    }
                }

                for i in 0..block_rows {
                    for j in 0..p {
                        stacked[[p + i, j]] = block[[i, j]];
                    }
                }

                let (_, r_new) = stacked.qr()
                    .map_err(|_| GAMError::LinAlgError("Block QR update failed".to_string()))?;

                let mut r_extracted = Array2::<f64>::zeros((p, p));
                let r_rows = r_new.nrows();
                let r_cols = r_new.ncols();

                for i in 0..p.min(r_rows) {
                    for j in i..p.min(r_cols) {
                        r_extracted[[i, j]] = r_new[[i, j]];
                    }
                }
                r_current = Some(r_extracted);
            } else {
                // First block: just QR it
                let (_, r_first) = block.qr()
                    .map_err(|_| GAMError::LinAlgError("First block QR failed".to_string()))?;

                let mut r_extracted = Array2::<f64>::zeros((p, p));
                let r_rows = r_first.nrows();
                let r_cols = r_first.ncols();

                for i in 0..p.min(block_rows).min(r_rows) {
                    for j in i..p.min(r_cols) {
                        r_extracted[[i, j]] = r_first[[i, j]];
                    }
                }
                r_current = Some(r_extracted);
            }
        }

        r_current.ok_or(GAMError::InvalidParameter("No blocks processed".to_string()))
    }
}

#[cfg(not(feature = "blas"))]
pub fn compute_r_blockwise(
    _x: &Array2<f64>,
    _w: &Array1<f64>,
    _lambdas: &[f64],
    _sqrt_penalties: &[Array2<f64>],
    _block_size: usize,
) -> Result<Array2<f64>> {
    Err(GAMError::InvalidParameter(
        "Block-wise QR requires 'blas' feature".to_string()
    ))
}
