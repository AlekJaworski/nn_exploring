//! Block-diagonal penalty representation for efficient high-dimensional GAM fitting.
//!
//! In additive models with d independent smooths, each penalty matrix S_i is
//! total_basis × total_basis but only has a k_i × k_i non-zero block on the
//! diagonal. Storing and operating on the full matrix wastes O((d·k)² - k²)
//! work per penalty. This module provides a sparse block representation that
//! reduces all penalty operations from O(p²) to O(k²) where p = Σk_i.

use ndarray::{s, Array1, Array2};

/// A penalty matrix stored as a single non-zero block on the diagonal.
///
/// Represents a p×p matrix that is zero everywhere except for a k×k block
/// starting at position (offset, offset).
#[derive(Debug, Clone)]
pub struct BlockPenalty {
    /// The non-zero k×k block (already scaled by penalty_scale if applicable)
    pub block: Array2<f64>,
    /// Column/row offset in the full p×p matrix
    pub offset: usize,
    /// Total size of the full matrix (p = total_basis)
    pub total_size: usize,
}

impl BlockPenalty {
    /// Create a new block penalty.
    ///
    /// # Arguments
    /// * `block` - The k×k non-zero penalty block
    /// * `offset` - Starting row/column index in the full matrix
    /// * `total_size` - Dimension p of the full p×p matrix
    pub fn new(block: Array2<f64>, offset: usize, total_size: usize) -> Self {
        debug_assert_eq!(block.nrows(), block.ncols(), "Block must be square");
        debug_assert!(
            offset + block.nrows() <= total_size,
            "Block exceeds total matrix size"
        );
        BlockPenalty {
            block,
            offset,
            total_size,
        }
    }

    /// Size of the non-zero block (k)
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block.nrows()
    }

    /// Total rows of the conceptual full matrix
    #[inline]
    pub fn nrows(&self) -> usize {
        self.total_size
    }

    /// Total columns of the conceptual full matrix
    #[inline]
    pub fn ncols(&self) -> usize {
        self.total_size
    }

    // =========================================================================
    // Core operations used by REML, PiRLS, and smooth parameter optimization
    // =========================================================================

    /// In-place: `target += scale * self`
    ///
    /// Only touches the k×k block region of target. O(k²) instead of O(p²).
    /// This is the most-called operation (~18 sites in reml.rs + pirls.rs).
    pub fn scaled_add_to(&self, target: &mut Array2<f64>, scale: f64) {
        let k = self.block_size();
        let end = self.offset + k;
        let mut slice = target.slice_mut(s![self.offset..end, self.offset..end]);
        slice.scaled_add(scale, &self.block);
    }

    /// Compute `self · v` where v is a vector of length p.
    ///
    /// Returns a vector of length p that is zero outside [offset..offset+k].
    /// O(k²) instead of O(p²). Used ~25 times in reml.rs for β'Sβ terms.
    pub fn dot_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let k = self.block_size();
        let end = self.offset + k;
        let v_slice = v.slice(s![self.offset..end]);
        let result_block = self.block.dot(&v_slice);

        let mut result = Array1::zeros(self.total_size);
        result.slice_mut(s![self.offset..end]).assign(&result_block);
        result
    }

    /// Compute the quadratic form `v' · self · v`.
    ///
    /// Equivalent to `v.dot(&self.dot_vec(v))` but avoids allocating the intermediate.
    /// O(k²) instead of O(p²).
    pub fn quadratic_form(&self, v: &Array1<f64>) -> f64 {
        let k = self.block_size();
        let end = self.offset + k;
        let v_slice = v.slice(s![self.offset..end]);
        let sv = self.block.dot(&v_slice);
        v_slice.dot(&sv)
    }

    /// Compute `self * scale` returning a new BlockPenalty.
    ///
    /// O(k²) instead of O(p²). Used ~6 times for λ·S construction.
    pub fn scale(&self, scale: f64) -> BlockPenalty {
        BlockPenalty {
            block: &self.block * scale,
            offset: self.offset,
            total_size: self.total_size,
        }
    }

    /// Compute the trace of `dense_matrix · self` (or equivalently `self · dense_matrix`
    /// for symmetric self).
    ///
    /// `tr(M · S)` only depends on columns `offset..offset+k` of M and
    /// rows `offset..offset+k` of S. O(k²) instead of O(p²).
    ///
    /// Used for `tr(A⁻¹·Sᵢ)` in Fellner-Schall and gradient computations.
    pub fn trace_product(&self, dense: &Array2<f64>) -> f64 {
        let k = self.block_size();
        let end = self.offset + k;
        // tr(M * S) = Σ_ij M[i,j] * S[j,i]
        // S is only non-zero for i,j in [offset..end]
        // So tr = Σ_{i in block} Σ_{j in block} M[i,j] * S[j,i]
        // But we want tr over ALL rows, so:
        // tr(M * S) = Σ_{i=0..p} (M * S)[i,i] = Σ_{i=0..p} Σ_{j} M[i,j]*S[j,i]
        // S[j,i] nonzero only for j,i in [offset..end]
        // So tr = Σ_{i in block} Σ_{j in block} M[i,j] * S[j,i]
        let m_block = dense.slice(s![self.offset..end, self.offset..end]);
        // tr(M_block * S_block) where both are k×k
        let mut trace = 0.0;
        for i in 0..k {
            for j in 0..k {
                trace += m_block[[i, j]] * self.block[[j, i]];
            }
        }
        trace
    }

    /// Compute `dense · self` returning a full p×p matrix.
    ///
    /// The result is dense (M·S has non-zero columns only in [offset..offset+k]
    /// but all rows can be non-zero). However, only k columns of `dense` are needed.
    /// O(p·k) instead of O(p²). Used for `A⁻¹·Sⱼ` in Hessian.
    pub fn left_mul_dense(&self, dense: &Array2<f64>) -> Array2<f64> {
        let p = self.total_size;
        let k = self.block_size();
        let end = self.offset + k;

        // Result = dense · self
        // result[i, j] = Σ_l dense[i,l] * self[l,j]
        // self[l,j] nonzero only for l,j in [offset..end]
        // So result[:, j] = dense[:, offset..end] · self.block[:, j-offset]  for j in [offset..end]
        // result[:, j] = 0 for j outside [offset..end]

        let dense_cols = dense.slice(s![.., self.offset..end]);
        let product = dense_cols.dot(&self.block); // p × k

        let mut result = Array2::zeros((p, p));
        result.slice_mut(s![.., self.offset..end]).assign(&product);
        result
    }

    /// Estimate the rank of this penalty matrix.
    ///
    /// Uses the same row-norm heuristic as the dense `estimate_rank` but only
    /// on the k×k block. O(k²) instead of O(p²).
    pub fn estimate_rank(&self) -> usize {
        let k = self.block_size();
        let mut max_val: f64 = 0.0;
        for i in 0..k {
            for j in 0..k {
                max_val = max_val.max(self.block[[i, j]].abs());
            }
        }

        if max_val < 1e-10 {
            return 0;
        }

        let threshold = max_val * 1e-6;
        let mut rank = 0;
        for i in 0..k {
            let row_norm: f64 = (0..k).map(|j| self.block[[i, j]].abs()).sum();
            if row_norm > threshold {
                rank += 1;
            }
        }
        rank
    }

    /// Convert to full dense p×p matrix.
    ///
    /// Fallback for operations not yet optimized for block structure.
    pub fn to_dense(&self) -> Array2<f64> {
        let k = self.block_size();
        let end = self.offset + k;
        let mut full = Array2::zeros((self.total_size, self.total_size));
        full.slice_mut(s![self.offset..end, self.offset..end])
            .assign(&self.block);
        full
    }

    /// Access the block as a read-only view.
    #[inline]
    pub fn block_view(&self) -> ndarray::ArrayView2<f64> {
        self.block.view()
    }

    /// Compute element at position [i, j] in the full matrix.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let end = self.offset + self.block_size();
        if i >= self.offset && i < end && j >= self.offset && j < end {
            self.block[[i - self.offset, j - self.offset]]
        } else {
            0.0
        }
    }

    /// Infinity norm (max absolute row sum) of the block.
    /// Same as the infinity norm of the full matrix since all other rows are zero.
    pub fn inf_norm(&self) -> f64 {
        let k = self.block_size();
        let mut max_row_sum: f64 = 0.0;
        for i in 0..k {
            let row_sum: f64 = (0..k).map(|j| self.block[[i, j]].abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }
        max_row_sum
    }

    /// Iterator over diagonal elements of the block (in the full matrix,
    /// these are at positions offset..offset+k).
    pub fn block_diagonal_iter(&self) -> impl Iterator<Item = f64> + '_ {
        let k = self.block_size();
        (0..k).map(move |i| self.block[[i, i]])
    }

    /// Trace of the penalty matrix (= trace of the block).
    pub fn trace(&self) -> f64 {
        self.block_diagonal_iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_penalty() -> BlockPenalty {
        // 3x3 block starting at offset 2, in a 8x8 matrix
        let block = Array2::from_shape_vec(
            (3, 3),
            vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
        )
        .unwrap();
        BlockPenalty::new(block, 2, 8)
    }

    #[test]
    fn test_to_dense() {
        let bp = make_test_penalty();
        let dense = bp.to_dense();
        assert_eq!(dense.shape(), &[8, 8]);

        // Check block values
        assert_eq!(dense[[2, 2]], 4.0);
        assert_eq!(dense[[2, 3]], -1.0);
        assert_eq!(dense[[3, 4]], -1.0);
        assert_eq!(dense[[4, 4]], 4.0);

        // Check zeros outside block
        assert_eq!(dense[[0, 0]], 0.0);
        assert_eq!(dense[[1, 2]], 0.0);
        assert_eq!(dense[[5, 5]], 0.0);
    }

    #[test]
    fn test_scaled_add_to() {
        let bp = make_test_penalty();
        let mut target = Array2::eye(8);
        bp.scaled_add_to(&mut target, 2.0);

        // Diagonal inside block: 1.0 + 2.0 * 4.0 = 9.0
        assert_eq!(target[[2, 2]], 9.0);
        assert_eq!(target[[3, 3]], 9.0);
        assert_eq!(target[[4, 4]], 9.0);
        // Off-diagonal in block: 0.0 + 2.0 * (-1.0) = -2.0
        assert_eq!(target[[2, 3]], -2.0);
        // Outside block unchanged
        assert_eq!(target[[0, 0]], 1.0);
        assert_eq!(target[[5, 5]], 1.0);
    }

    #[test]
    fn test_dot_vec() {
        let bp = make_test_penalty();
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = bp.dot_vec(&v);

        // Expected: block * v[2..5] = [[4,-1,0],[-1,4,-1],[0,-1,4]] * [3,4,5]
        // [4*3 - 1*4 + 0*5, -1*3 + 4*4 - 1*5, 0*3 - 1*4 + 4*5]
        // = [8, 8, 16]   wait let me recalc:
        // [12-4+0, -3+16-5, 0-4+20] = [8, 8, 16]
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 8.0);
        assert_eq!(result[3], 8.0);
        assert_eq!(result[4], 16.0);
        assert_eq!(result[5], 0.0);
    }

    #[test]
    fn test_quadratic_form() {
        let bp = make_test_penalty();
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let qf = bp.quadratic_form(&v);
        // v' S v = v[2..5]' * block * v[2..5] = [3,4,5] * [8,8,16] = 24 + 32 + 80 = 136
        let expected = v.dot(&bp.dot_vec(&v));
        assert!((qf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_trace_product() {
        let bp = make_test_penalty();
        let dense = Array2::eye(8) * 2.0;
        let trace = bp.trace_product(&dense);
        // tr(2I * S) = 2 * tr(S) = 2 * (4+4+4) = 24
        assert!((trace - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_consistency_with_dense() {
        let bp = make_test_penalty();
        let dense_s = bp.to_dense();
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // dot_vec consistency
        let block_result = bp.dot_vec(&v);
        let dense_result = dense_s.dot(&v);
        for i in 0..8 {
            assert!(
                (block_result[i] - dense_result[i]).abs() < 1e-10,
                "dot_vec mismatch at {}: {} vs {}",
                i,
                block_result[i],
                dense_result[i]
            );
        }

        // scaled_add_to consistency
        let mut target_block = Array2::eye(8);
        let mut target_dense = Array2::eye(8);
        bp.scaled_add_to(&mut target_block, 3.0);
        target_dense.scaled_add(3.0, &dense_s);
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (target_block[[i, j]] - target_dense[[i, j]]).abs() < 1e-10,
                    "scaled_add_to mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }
}
