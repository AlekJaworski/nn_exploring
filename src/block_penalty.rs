//! Block-diagonal penalty representation for efficient high-dimensional GAM fitting.
//!
//! In additive models with d independent smooths, each penalty matrix S_i is
//! total_basis × total_basis but only has a k_i × k_i non-zero block on the
//! diagonal. Storing and operating on the full matrix wastes O((d·k)² - k²)
//! work per penalty. This module provides a sparse block representation that
//! reduces all penalty operations from O(p²) to O(k²) where p = Σk_i.

use ndarray::{s, Array1, Array2};

/// Per-block "initial reparameterization" transform from mgcv's `Sl.initial.repara`
/// (R/fast-REML.r:517). For a singleton block, the eigendecomposition
/// `S_k = U · diag(D) · U'` gives a basis rotation `D_k` such that the rotated
/// penalty has 1's on the rank-r entries and 0's on the null. The transform is
/// `sp`-independent and computed once at block-setup time.
///
/// In the rotated basis, the linear system `(X'WX + S(ρ)) β = f` becomes better
/// conditioned because S is reduced to a partial identity. We rotate `XX`, `f`
/// at the entry of `compute_sl_fitchol_step` and rotate `β`, `db`, `PP` back at
/// exit, so the external contract is unchanged.
///
/// We only store the dense (eigen) case used by mgcv's eigen-stabilised path
/// (line 288-302). The Cholesky-pivoted `singleStrans` branch (line 280-287)
/// is not yet ported — that branch adds a `−log|D|` term to the score via
/// `Sl[[b]]$ldet`, which we deliberately set to zero (line 296).
#[derive(Debug, Clone)]
pub struct ReparaTransform {
    /// Forward transform `D` (k×k). Maps from the original basis to the
    /// reparameterised basis. `D = U · diag(1/sqrt(D_+))` with `D_+` clamped
    /// to 1 on null components (mgcv line 300-301). On a diagonal S this
    /// degenerates to `diag(1/sqrt(D_+))`.
    pub d_mat: Array2<f64>,
    /// Inverse transform `Di = D^{-1}`. mgcv stores it explicitly (line 302)
    /// to avoid the matrix inverse at every call.
    pub di_mat: Array2<f64>,
    /// Estimated rank of S_k from the eigen-spectrum. The leading `rank`
    /// components are the penalised range; the trailing `k - rank` are the
    /// null space.
    pub rank: usize,
}

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
    /// Optional per-block reparameterisation transform (mgcv `Sl.initial.repara`).
    /// Populated on demand via [`BlockPenalty::setup_initial_repara`]. When
    /// `None`, rotation operations are no-ops, so existing call sites that
    /// haven't opted in see identical behaviour.
    pub repara: Option<ReparaTransform>,
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
            repara: None,
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
            // Scaling λ·S doesn't change the rotation basis (eigenvectors of
            // λS = eigenvectors of S, eigenvalues scale by λ). Re-deriving on
            // demand keeps the API simple; production callers compute repara
            // on the unit-scaled S anyway (mgcv's `Sl.setup` operates on
            // unit-S, then folds λ in via `lambda`).
            repara: None,
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

    /// Compute `(log|λS|+, d/dρ log|λS|+, d²/dρ² log|λS|+)` for a singleton
    /// block, where `ρ = log(λ)` is the log-smoothing-parameter.
    ///
    /// For a single-`S` block, the penalty under λ-scaling is `λ·S`, so
    /// `log|λS|+ = rank·log(λ) + log|S|+ = rank·ρ + log_pseudo_det_unit`,
    /// where `log_pseudo_det_unit` is the (constant-in-ρ) pseudo-determinant
    /// of the unit-scaled `S`. Hence the derivatives are `(rank, 0)`.
    ///
    /// This corresponds to the singleton branch of mgcv's `ldetS`
    /// (`R/fast-REML.r:762`). Multi-`S` blocks (rare; tensor smooths with
    /// multiple penalty marginals) require an eigendecomposition of
    /// `Σ_k λ_k S_k` and are not yet supported — see `todo!()` below.
    ///
    /// Reuses `crate::reml::pseudo_determinant` for the unit-S log-det.
    #[cfg(feature = "blas")]
    pub fn log_det_singleton_with_derivs(&self, rho: f64) -> (f64, f64, f64) {
        // True (eigen) rank — matches mgcv `ldetS` (fast-REML.r:834). The
        // legacy row-norm `estimate_rank()` overcounts on banded penalties
        // (2nd-diff k=10 → 10 vs true 8), creating a (rank_row − rank_eigen)·ρ
        // drift vs the rest of the REML code which uses estimate_rank_eigen.
        let rank = crate::reml::estimate_rank_eigen(self) as f64;
        // Unit-scaled S log-pseudo-determinant — independent of ρ.
        // `pseudo_determinant` returns log Π_{i: λ_i>0} λ_i for the block's S.
        let log_pseudo_det_unit = crate::reml::pseudo_determinant(self)
            .expect("pseudo_determinant should succeed on a finite penalty block");
        (rank * rho + log_pseudo_det_unit, rank, 0.0)
    }

    /// Non-BLAS fallback: pseudo-determinant isn't available without
    /// `ndarray-linalg`, so the singleton log-det reduces to `rank·ρ` only.
    /// This branch exists for compile-time completeness; production fits
    /// always run with the `blas` feature.
    #[cfg(not(feature = "blas"))]
    pub fn log_det_singleton_with_derivs(&self, rho: f64) -> (f64, f64, f64) {
        let rank = self.estimate_rank() as f64;
        (rank * rho, rank, 0.0)
    }

    // =========================================================================
    // Initial reparameterisation (mgcv R/fast-REML.r `Sl.initial.repara`)
    // =========================================================================

    /// Compute and cache the per-block initial reparameterisation transform.
    ///
    /// Mirrors the eigen-stabilised branch of mgcv's `Sl.setup` (R/fast-REML.r
    /// lines 268-302). The diagonal-S shortcut (line 268-278) is detected and
    /// uses the cheap `1/sqrt(diag)` path; otherwise we run an eigendecomposition
    /// of the k×k block and form `D = U · diag(1/sqrt(D_+))` with the null
    /// components clamped to 1.
    ///
    /// Idempotent — calling twice on the same block is a no-op after the first
    /// call (return early when `self.repara` is already populated).
    ///
    /// **Not implemented yet**: multi-S blocks (tensor smooths with multiple
    /// penalty marginals). B3's `log_det_singleton_with_derivs` already panics
    /// with `todo!()` on multi-S; repara follows the same precedent.
    #[cfg(feature = "blas")]
    pub fn setup_initial_repara(&mut self) {
        use ndarray_linalg::Eigh;

        if self.repara.is_some() {
            return;
        }
        let k = self.block_size();
        if k == 0 {
            self.repara = Some(ReparaTransform {
                d_mat: Array2::<f64>::zeros((0, 0)),
                di_mat: Array2::<f64>::zeros((0, 0)),
                rank: 0,
            });
            return;
        }

        // Detect diagonal S — mgcv line 268. Use a tight tolerance relative to
        // the diagonal magnitude.
        let max_abs = (0..k)
            .map(|i| self.block[[i, i]].abs())
            .fold(0.0f64, f64::max);
        let tol = (1e-14 * max_abs).max(0.0);
        let mut is_diag = true;
        'outer: for i in 0..k {
            for j in 0..k {
                if i != j && self.block[[i, j]].abs() > tol {
                    is_diag = false;
                    break 'outer;
                }
            }
        }

        if is_diag {
            // mgcv lines 273-278: D[ind] = 1/sqrt(D[ind]); D[!ind] = 1.
            let mut d_diag = Array1::<f64>::zeros(k);
            let mut rank = 0usize;
            let threshold = 1e-10 * max_abs.max(1.0);
            for i in 0..k {
                let v = self.block[[i, i]];
                if v > threshold {
                    d_diag[i] = 1.0 / v.sqrt();
                    rank += 1;
                } else {
                    d_diag[i] = 1.0;
                }
            }
            // Embed diagonal as a dense k×k. The matrix branch carries it
            // uniformly with the eigen path so the rotation helpers don't
            // have to branch on representation.
            let mut d_mat = Array2::<f64>::zeros((k, k));
            let mut di_mat = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                d_mat[[i, i]] = d_diag[i];
                di_mat[[i, i]] = 1.0 / d_diag[i];
            }
            self.repara = Some(ReparaTransform {
                d_mat,
                di_mat,
                rank,
            });
            return;
        }

        // Dense S: eigen branch (mgcv lines 289-302).
        // ndarray-linalg returns eigenvalues ascending. mgcv's `eigen()` returns
        // DESCENDING by default — `ind[1:rank] <- TRUE` (line 298) puts the
        // range-space at the TOP of the basis. We REVERSE both arrays here so
        // the "leading rank columns" of D correspond to the range space and
        // the null components sit at the bottom — matching mgcv's index
        // convention. This is essential for the rotated-S → partial-identity
        // shortcut at the top of `compute_sl_fitchol_step` to be valid
        // (`diag(1..1, 0..0)` with 1's at indices `0..rank`).
        let (eigvals_asc, eigvecs_asc) = match self.block.eigh(ndarray_linalg::UPLO::Upper) {
            Ok(p) => p,
            Err(_) => {
                // Pathological block: fall back to identity (no-op rotation).
                let d_mat = Array2::<f64>::eye(k);
                let di_mat = Array2::<f64>::eye(k);
                self.repara = Some(ReparaTransform {
                    d_mat,
                    di_mat,
                    rank: 0,
                });
                return;
            }
        };
        // Reverse eigvals and reorder eigvecs columns to match descending order.
        let mut eigvals = Array1::<f64>::zeros(k);
        let mut eigvecs = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            eigvals[j] = eigvals_asc[k - 1 - j];
            for i in 0..k {
                eigvecs[[i, j]] = eigvecs_asc[[i, k - 1 - j]];
            }
        }
        let max_eig = eigvals.iter().copied().fold(0.0f64, f64::max);
        let rank_threshold = max_eig * f64::EPSILON.powf(0.8);
        let mut scale = Array1::<f64>::zeros(k);
        let mut rank = 0usize;
        for (i, &ev) in eigvals.iter().enumerate() {
            if ev > rank_threshold && ev > 0.0 {
                scale[i] = 1.0 / ev.sqrt();
                rank += 1;
            } else {
                scale[i] = 1.0;
            }
        }
        // D[:, j] = U[:, j] · scale[j]   (eigvecs is U with descending eigvals).
        let mut d_mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                d_mat[[i, j]] = eigvecs[[i, j]] * scale[j];
            }
        }
        // Di = U' / scale (mgcv line 302). Di[j, i] = U[i, j] / scale[j].
        let mut di_mat = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let inv_scale = 1.0 / scale[j];
            for i in 0..k {
                di_mat[[j, i]] = eigvecs[[i, j]] * inv_scale;
            }
        }
        self.repara = Some(ReparaTransform {
            d_mat,
            di_mat,
            rank,
        });
    }

    /// Non-BLAS fallback: no eigendecomposition available, so the transform
    /// is the identity (rotation is a no-op).
    #[cfg(not(feature = "blas"))]
    pub fn setup_initial_repara(&mut self) {
        if self.repara.is_some() {
            return;
        }
        let k = self.block_size();
        let d_mat = Array2::<f64>::eye(k);
        let di_mat = Array2::<f64>::eye(k);
        self.repara = Some(ReparaTransform {
            d_mat,
            di_mat,
            rank: 0,
        });
    }

    /// Forward-rotate a symmetric "outer-product" matrix in place.
    ///
    /// Maps `X[ind, ind] ← D' · X[ind, ind] · D` where `ind = offset..offset+k`.
    /// Equivalent to mgcv's `Sl.initial.repara(X, inverse=FALSE, both.sides=TRUE,
    /// cov=FALSE)` for the block region (R/fast-REML.r:568-571).
    ///
    /// Use this for `XX = X'WX` at the entry of the score evaluator.
    pub fn rotate_xx_in_place(&self, x: &mut Array2<f64>) {
        let repara = match &self.repara {
            Some(r) => r,
            None => return,
        };
        let k = self.block_size();
        if k == 0 {
            return;
        }
        let p = self.total_size;
        let off = self.offset;
        let end = off + k;
        debug_assert_eq!(x.nrows(), p);
        debug_assert_eq!(x.ncols(), p);

        // Step 1: row-side  X[ind, :] ← D' · X[ind, :]  (operate over ALL p cols).
        // Slice out the k×p row block, replace with D' · block.
        let row_block = x.slice(s![off..end, ..]).to_owned();
        let new_rows = repara.d_mat.t().dot(&row_block);
        x.slice_mut(s![off..end, ..]).assign(&new_rows);

        // Step 2: col-side  X[:, ind] ← X[:, ind] · D  (operate over ALL p rows).
        let col_block = x.slice(s![.., off..end]).to_owned();
        let new_cols = col_block.dot(&repara.d_mat);
        x.slice_mut(s![.., off..end]).assign(&new_cols);
    }

    /// Forward-rotate a model-matrix-shaped vector / column in place.
    ///
    /// Maps `v[ind] ← D' · v[ind]`. Use this for `f = X'Wy` at the entry of the
    /// score evaluator. Mirrors mgcv's vector branch at line 577-579
    /// (`both.sides=TRUE` vector path: "vector to be treated like model matrix
    /// X").
    pub fn rotate_f_in_place(&self, v: &mut Array1<f64>) {
        let repara = match &self.repara {
            Some(r) => r,
            None => return,
        };
        let k = self.block_size();
        if k == 0 {
            return;
        }
        let off = self.offset;
        let end = off + k;
        let block = v.slice(s![off..end]).to_owned();
        let new_block = repara.d_mat.t().dot(&block);
        v.slice_mut(s![off..end]).assign(&new_block);
    }

    /// Inverse-rotate a parameter (coefficient) vector in place.
    ///
    /// Maps `v[ind] ← D · v[ind]`. Use this for `β̂` returned from the rotated
    /// solve. Mirrors mgcv's parameter-vector branch at line 558-562
    /// (`inverse=TRUE`, vector, matrix-D case: `Sl[[b]]$D %*% X[ind]`).
    ///
    /// Predictions invariance: at the rotated optimum `(XX_rot + S_rot) β_rot = f_rot`
    /// with `XX_rot = D' XX D`, `f_rot = D' f`, `S_rot = D' S D`. Multiplying both
    /// sides on the left by `D` and using `D D' XX D = D (D' XX D)`, the
    /// unrotated coefficient is `β = D · β_rot`, so `X · β = X · D · β_rot`. We
    /// rotate `β_rot` back here to restore the original-basis coefficients.
    pub fn inverse_rotate_beta_in_place(&self, v: &mut Array1<f64>) {
        let repara = match &self.repara {
            Some(r) => r,
            None => return,
        };
        let k = self.block_size();
        if k == 0 {
            return;
        }
        let off = self.offset;
        let end = off + k;
        let block = v.slice(s![off..end]).to_owned();
        let new_block = repara.d_mat.dot(&block);
        v.slice_mut(s![off..end]).assign(&new_block);
    }

    /// Inverse-rotate a covariance-shape matrix in place.
    ///
    /// Maps `X[ind, ind] ← D · X[ind, ind] · D'` where `ind = offset..offset+k`.
    /// Mirrors mgcv's covariance branch with `inverse=TRUE, both.sides=TRUE,
    /// cov=TRUE` (R/fast-REML.r:528-540).
    ///
    /// Use this for `PP = (X'X+S)⁻¹` and for each column of `db = dβ/dρ` (mgcv
    /// treats `db` columns as covariance-style vectors, applying the same
    /// `inverse=TRUE, cov=TRUE, both.sides=TRUE` rotation at bam.r:801).
    pub fn inverse_rotate_cov_in_place(&self, x: &mut Array2<f64>) {
        let repara = match &self.repara {
            Some(r) => r,
            None => return,
        };
        let k = self.block_size();
        if k == 0 {
            return;
        }
        let p = self.total_size;
        let off = self.offset;
        let end = off + k;
        debug_assert_eq!(x.nrows(), p);
        debug_assert_eq!(x.ncols(), p);

        // Step 1: row-side  X[ind, :] ← D · X[ind, :].
        let row_block = x.slice(s![off..end, ..]).to_owned();
        let new_rows = repara.d_mat.dot(&row_block);
        x.slice_mut(s![off..end, ..]).assign(&new_rows);

        // Step 2: col-side  X[:, ind] ← X[:, ind] · D'.
        let col_block = x.slice(s![.., off..end]).to_owned();
        let new_cols = col_block.dot(&repara.d_mat.t());
        x.slice_mut(s![.., off..end]).assign(&new_cols);
    }

    /// Inverse-rotate a single dβ/dρ column in place.
    ///
    /// `db[:, k]` is a length-`p` vector; mgcv applies the same cov-style
    /// rotation as for a matrix slice (bam.r:800-801 passes
    /// `as.numeric(prop$db[,i])` to `Sl.initial.repara` with `inverse=TRUE,
    /// both.sides=TRUE, cov=TRUE`). For a vector under those flags, the
    /// parameter-vector branch (line 557-562) applies `X[ind] ← D · X[ind]`,
    /// matching `inverse_rotate_beta_in_place`. The `both.sides=TRUE, cov=TRUE`
    /// flags don't change behaviour on a vector — they only affect the matrix
    /// branch. So this is a thin alias documenting the call site.
    #[inline]
    pub fn inverse_rotate_db_column_in_place(&self, v: &mut Array1<f64>) {
        self.inverse_rotate_beta_in_place(v);
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

    /// FD parity for `compute_ldet_s_with_derivs` on a 3-block fixture
    /// with mixed ranks. The singleton-block formula gives
    /// `d/dρ_k log|λS|+ = rank_k` exactly, so we check that against a
    /// central-difference of the scalar.
    #[cfg(feature = "blas")]
    #[test]
    fn log_det_singleton_with_derivs_fd_parity() {
        use crate::reml::compute_ldet_s_with_derivs;

        // Block 1: 3×3 tridiagonal (rank 3, det > 0)
        let b1 = Array2::from_shape_vec(
            (3, 3),
            vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
        )
        .unwrap();
        // Block 2: 4×4 with one zero row → rank 3 (a singular SPSD case
        // exercises the pseudo-determinant path).
        let b2 = Array2::from_shape_vec(
            (4, 4),
            vec![
                2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        // Block 3: 2×2 SPD (rank 2)
        let b3 = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap();

        let total = 3 + 4 + 2;
        let bp1 = BlockPenalty::new(b1, 0, total);
        let bp2 = BlockPenalty::new(b2, 3, total);
        let bp3 = BlockPenalty::new(b3, 7, total);
        let sl = vec![bp1, bp2, bp3];

        let rho = vec![0.7, -0.3, 1.4];
        let (val, d1, d2) = compute_ldet_s_with_derivs(&sl, &rho);

        // Analytical: d1[k] = rank_k, d2 = 0.
        let ranks = [3usize, 3, 2];
        for (k, r) in ranks.iter().enumerate() {
            assert!(
                (d1[k] - (*r as f64)).abs() < 1e-12,
                "analytical d1[{}]: got {}, expected {}",
                k,
                d1[k],
                r
            );
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(d2[[i, j]], 0.0, "d2[{},{}] should be 0 (singleton)", i, j);
            }
        }

        // FD check: central-difference on each ρ_k against scalar `val`.
        let h = 1e-6;
        for k in 0..3 {
            let mut rho_plus = rho.clone();
            let mut rho_minus = rho.clone();
            rho_plus[k] += h;
            rho_minus[k] -= h;
            let (val_plus, _, _) = compute_ldet_s_with_derivs(&sl, &rho_plus);
            let (val_minus, _, _) = compute_ldet_s_with_derivs(&sl, &rho_minus);
            let fd = (val_plus - val_minus) / (2.0 * h);
            assert!(
                (fd - d1[k]).abs() < 1e-6,
                "FD d1[{}]: analytical {}, FD {}, |diff| {}",
                k,
                d1[k],
                fd,
                (fd - d1[k]).abs()
            );
        }

        // Sanity: total value should equal Σ_k (rank_k·ρ_k + log_pseudo_det_k).
        // We don't hard-code log_pseudo_det values; just verify monotone
        // behaviour in ρ — increasing ρ_0 by Δ increases `val` by rank_0·Δ.
        let delta = 0.25;
        let mut rho_shift = rho.clone();
        rho_shift[0] += delta;
        let (val_shift, _, _) = compute_ldet_s_with_derivs(&sl, &rho_shift);
        let expected = val + (ranks[0] as f64) * delta;
        assert!(
            (val_shift - expected).abs() < 1e-12,
            "shift sanity: got {}, expected {}",
            val_shift,
            expected
        );
    }

    /// Round-trip: rotate `XX` forward (`D' · XX · D` on block) and verify the
    /// rotated matrix matches the analytical sandwich computed via dense
    /// `D'·XX·D` on a full p×p `D`-embedding. This is the load-bearing
    /// correctness check: it pins the block-level rotation against the
    /// reference formula without mixing in unrelated inverse-cov semantics.
    ///
    /// NOTE: there is no "round-trip XX → XX" via forward + inverse cov
    /// rotation. mgcv's `inverse_rotate_cov` is `D · · D'`, which is the
    /// inverse for PP (since `A_rot⁻¹ = Di A⁻¹ Di'` and inverting gives
    /// `A⁻¹ = D · PP_rot · D'`). Composing forward then inverse-cov gives
    /// `D (D' XX D) D' = (DD') XX (DD')`, which is XX only when D is
    /// orthogonal (it isn't, since D scales by `1/sqrt(D_+)`).
    #[cfg(feature = "blas")]
    #[test]
    fn repara_round_trip_xx() {
        // 3 blocks of different sizes at different offsets.
        let p = 12;
        let mut sl = vec![
            BlockPenalty::new(
                Array2::from_shape_vec(
                    (3, 3),
                    vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
                )
                .unwrap(),
                0,
                p,
            ),
            BlockPenalty::new(
                Array2::from_shape_vec((4, 4), {
                    // Singular block: rank 3, last row/col zero.
                    let mut v = vec![
                        2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ];
                    // Force exact symmetry (paranoia).
                    for i in 0..4 {
                        for j in (i + 1)..4 {
                            v[j * 4 + i] = v[i * 4 + j];
                        }
                    }
                    v
                })
                .unwrap(),
                3,
                p,
            ),
            BlockPenalty::new(
                Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 3.0]).unwrap(),
                7,
                p,
            ),
        ];
        for block in sl.iter_mut() {
            block.setup_initial_repara();
        }

        // Synthetic XX (symmetric).
        let mut xx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                xx[[i, j]] = 1.0 + ((i * 7 + j * 13) as f64).sin();
            }
            xx[[i, i]] += 10.0;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (xx[[i, j]] + xx[[j, i]]);
                xx[[i, j]] = avg;
                xx[[j, i]] = avg;
            }
        }
        let xx_orig = xx.clone();

        // Reference: build the full p×p block-diagonal D embedding, compute
        // D' · XX · D directly.
        let mut d_full = Array2::<f64>::eye(p);
        for b in sl.iter() {
            let repara = b.repara.as_ref().unwrap();
            let off = b.offset;
            let k = b.block_size();
            for i in 0..k {
                for j in 0..k {
                    d_full[[off + i, off + j]] = repara.d_mat[[i, j]];
                }
            }
        }
        let ref_xx = d_full.t().dot(&xx_orig).dot(&d_full);

        // Compose in-place block rotations.
        for b in sl.iter() {
            b.rotate_xx_in_place(&mut xx);
        }

        let mut max_diff = 0.0f64;
        for i in 0..p {
            for j in 0..p {
                max_diff = max_diff.max((xx[[i, j]] - ref_xx[[i, j]]).abs());
            }
        }
        eprintln!(
            "[repara_round_trip_xx] block vs full-D' XX D max |diff| = {:.2e}",
            max_diff
        );
        assert!(
            max_diff < 1e-12,
            "Block-rotation vs full-D mismatch {:.2e} exceeds 1e-12",
            max_diff
        );

        // Round-trip XX ↔ XX using the genuine inverse `Di' · · Di`.
        // This is NOT exposed as a public method (mgcv never round-trips XX
        // either — only PP, β, db are inverse-rotated back), but verifying
        // it here pins the algebra.
        let mut di_full = Array2::<f64>::eye(p);
        for b in sl.iter() {
            let repara = b.repara.as_ref().unwrap();
            let off = b.offset;
            let k = b.block_size();
            for i in 0..k {
                for j in 0..k {
                    di_full[[off + i, off + j]] = repara.di_mat[[i, j]];
                }
            }
        }
        let xx_back = di_full.t().dot(&xx).dot(&di_full);
        let mut rt = 0.0f64;
        for i in 0..p {
            for j in 0..p {
                rt = rt.max((xx_back[[i, j]] - xx_orig[[i, j]]).abs());
            }
        }
        eprintln!(
            "[repara_round_trip_xx] Di'·(D'XX·D)·Di = XX max |diff| = {:.2e}",
            rt
        );
        assert!(
            rt < 1e-10,
            "Genuine XX round-trip Di'·rotated·Di failed: {:.2e}",
            rt
        );
    }

    /// Round-trip on a vector (`f` forward + inverse round-trips via `β`-style
    /// inverse rotation since `Di · (D' v)` = ... wait, that's `Di · D' · v`,
    /// not the identity. The correct round-trip for `f` uses a separate inverse
    /// dual to `D'` that we don't currently expose (since the score pipeline
    /// never needs it — `f` doesn't come back from the solve, only β does).
    /// Instead we verify the dual round-trips:
    ///
    ///   - `β` round-trip:  forward = `Di · β`, inverse = `D · β_rot`, returns β.
    ///   - `f` forward sandwich consistency:  `f_rot = D' f` then `Di' f_rot = f`.
    #[cfg(feature = "blas")]
    #[test]
    fn repara_round_trip_vectors() {
        let p = 6;
        let mut sl = vec![BlockPenalty::new(
            Array2::from_shape_vec((4, 4), {
                let mut v = vec![
                    3.0, -1.0, 0.0, 0.0, -1.0, 3.0, -1.0, 0.0, 0.0, -1.0, 3.0, -1.0, 0.0, 0.0,
                    -1.0, 3.0,
                ];
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        v[j * 4 + i] = v[i * 4 + j];
                    }
                }
                v
            })
            .unwrap(),
            1,
            p,
        )];
        sl[0].setup_initial_repara();

        // β-style round-trip: rotate f-style forward, then inverse-as-β.
        // Mathematically: β = D · (Di · β)  — invert with D⁻¹ first to mimic
        // the solve, then rotate back with D.
        let beta_orig = Array1::<f64>::from_vec((0..p).map(|i| (i as f64) * 0.3 - 0.5).collect());
        // Forward analogue: β_rot = Di · β. We don't have an explicit "forward
        // β" method (the solve produces β_rot directly from rotated XX/f).
        // For this test, we emulate it by applying `Di · β` manually via the
        // public `repara` data.
        let repara = sl[0].repara.as_ref().unwrap();
        let mut beta_rot = beta_orig.clone();
        {
            let off = 1usize;
            let end = off + 4;
            let block = beta_rot.slice(s![off..end]).to_owned();
            let new = repara.di_mat.dot(&block);
            beta_rot.slice_mut(s![off..end]).assign(&new);
        }
        // Inverse-rotate via public API.
        sl[0].inverse_rotate_beta_in_place(&mut beta_rot);
        let mut max_diff = 0.0f64;
        for i in 0..p {
            max_diff = max_diff.max((beta_rot[i] - beta_orig[i]).abs());
        }
        eprintln!(
            "[repara_round_trip_vectors] β round-trip max |diff| = {:.2e}",
            max_diff
        );
        assert!(
            max_diff < 1e-12,
            "β round-trip max diff {:.2e} exceeds 1e-12",
            max_diff
        );

        // f forward + matching inverse via Di'  (manual, since f's inverse
        // counterpart isn't on the public API).
        let f_orig = Array1::<f64>::from_vec((0..p).map(|i| (i as f64) * 0.2 + 1.0).collect());
        let mut f_rot = f_orig.clone();
        sl[0].rotate_f_in_place(&mut f_rot);
        // Inverse: f = Di' · f_rot   (since f_rot = D' f).
        let mut f_back = f_rot.clone();
        {
            let off = 1usize;
            let end = off + 4;
            let block = f_back.slice(s![off..end]).to_owned();
            let new = repara.di_mat.t().dot(&block);
            f_back.slice_mut(s![off..end]).assign(&new);
        }
        let mut max_diff_f = 0.0f64;
        for i in 0..p {
            max_diff_f = max_diff_f.max((f_back[i] - f_orig[i]).abs());
        }
        eprintln!(
            "[repara_round_trip_vectors] f round-trip max |diff| = {:.2e}",
            max_diff_f
        );
        assert!(
            max_diff_f < 1e-12,
            "f round-trip max diff {:.2e} exceeds 1e-12",
            max_diff_f
        );
    }

    /// Verify that the rotated penalty matrix `D' · S · D` equals a partial
    /// identity (1's on the leading `rank` entries, 0 elsewhere), which is the
    /// mgcv invariant from R/fast-REML.r:309-311.
    #[cfg(feature = "blas")]
    #[test]
    fn repara_yields_partial_identity_S() {
        // Eigen branch: dense block.
        let block_dense = Array2::from_shape_vec(
            (3, 3),
            vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
        )
        .unwrap();
        let mut bp = BlockPenalty::new(block_dense.clone(), 0, 3);
        bp.setup_initial_repara();
        let repara = bp.repara.as_ref().unwrap();
        // S_rot = D' · S · D  (since D was computed s.t. this is partial-I).
        let s_rot = repara.d_mat.t().dot(&block_dense).dot(&repara.d_mat);
        let k = 3;
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j && i < repara.rank { 1.0 } else { 0.0 };
                let diff = (s_rot[[i, j]] - expected).abs();
                assert!(
                    diff < 1e-10,
                    "S_rot[{},{}] = {}, expected {}: |diff| = {:.2e}",
                    i,
                    j,
                    s_rot[[i, j]],
                    expected,
                    diff
                );
            }
        }

        // Diagonal branch: 1's and 0's mixed.
        let mut block_diag = Array2::<f64>::zeros((4, 4));
        block_diag[[0, 0]] = 9.0;
        block_diag[[1, 1]] = 0.0; // null
        block_diag[[2, 2]] = 4.0;
        block_diag[[3, 3]] = 0.0; // null
        let mut bp2 = BlockPenalty::new(block_diag.clone(), 0, 4);
        bp2.setup_initial_repara();
        let repara2 = bp2.repara.as_ref().unwrap();
        assert_eq!(repara2.rank, 2);
        let s_rot2 = repara2.d_mat.t().dot(&block_diag).dot(&repara2.d_mat);
        // After repara, S_rot has 1's at positions 0 and 2 (the rank-carrying
        // entries) and 0 elsewhere — the diagonal `D[ind] = 1/sqrt(diag)`
        // makes `(1/sqrt(d))² · d = 1`.
        assert!((s_rot2[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((s_rot2[[2, 2]] - 1.0).abs() < 1e-12);
        assert!(s_rot2[[1, 1]].abs() < 1e-12);
        assert!(s_rot2[[3, 3]].abs() < 1e-12);
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
