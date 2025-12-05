//! Chunked QR decomposition for efficient GAM fitting
//!
//! Implements incremental QR factorization following Wood et al. (2015)
//! "Generalized Additive Models for Large Data Sets"

use ndarray::{Array1, Array2, s};
use crate::{Result, GAMError};

#[cfg(feature = "blas")]
use ndarray_linalg::QR as QRDecomp;

/// QR factorization that can be updated incrementally
///
/// Stores Q and R factors where X = QR
/// Supports adding new rows (chunks) to the decomposition
#[derive(Debug, Clone)]
pub struct IncrementalQR {
    /// R factor (upper triangular, p×p)
    pub r: Array2<f64>,
    /// Q'y accumulator (p×1)
    pub qty: Array1<f64>,
    /// Sum of squared y values
    pub yty: f64,
    /// Number of rows processed so far
    pub n_rows: usize,
    /// Number of columns (basis functions)
    pub n_cols: usize,
}

impl IncrementalQR {
    /// Create a new incremental QR decomposition
    ///
    /// # Arguments
    /// * `n_cols` - Number of columns (basis functions)
    pub fn new(n_cols: usize) -> Self {
        Self {
            r: Array2::zeros((n_cols, n_cols)),
            qty: Array1::zeros(n_cols),
            yty: 0.0,
            n_rows: 0,
            n_cols,
        }
    }

    /// Update QR decomposition with a new chunk of data
    ///
    /// # Arguments
    /// * `x_chunk` - Design matrix chunk (n_chunk × p)
    /// * `y_chunk` - Response vector chunk (n_chunk)
    /// * `w_chunk` - Weights chunk (n_chunk), optional
    ///
    /// Updates R and Q'y incrementally without forming full Q
    pub fn update_chunk(
        &mut self,
        x_chunk: &Array2<f64>,
        y_chunk: &Array1<f64>,
        w_chunk: Option<&Array1<f64>>,
    ) -> Result<()> {
        let n_chunk = x_chunk.nrows();

        if x_chunk.ncols() != self.n_cols {
            return Err(GAMError::DimensionMismatch(
                format!("Expected {} columns, got {}", self.n_cols, x_chunk.ncols())
            ));
        }

        if y_chunk.len() != n_chunk {
            return Err(GAMError::DimensionMismatch(
                "y_chunk length must match x_chunk rows".to_string()
            ));
        }

        // Apply weights if provided
        let (x_weighted, y_weighted) = if let Some(w) = w_chunk {
            if w.len() != n_chunk {
                return Err(GAMError::DimensionMismatch(
                    "w_chunk length must match x_chunk rows".to_string()
                ));
            }

            // W^{1/2} * X and W^{1/2} * y
            let mut x_w = x_chunk.clone();
            let mut y_w = y_chunk.clone();

            for i in 0..n_chunk {
                let sqrt_w = w[i].sqrt();
                for j in 0..self.n_cols {
                    x_w[[i, j]] *= sqrt_w;
                }
                y_w[i] *= sqrt_w;
            }

            (x_w, y_w)
        } else {
            (x_chunk.clone(), y_chunk.clone())
        };

        // Stack [R; X_chunk] vertically
        let n_total = self.n_cols + n_chunk;
        let mut stacked = Array2::zeros((n_total, self.n_cols));

        // Top part: current R
        stacked.slice_mut(s![..self.n_cols, ..]).assign(&self.r);

        // Bottom part: new X chunk
        stacked.slice_mut(s![self.n_cols.., ..]).assign(&x_weighted);

        // Compute QR of stacked matrix
        #[cfg(feature = "blas")]
        {
            let (_, r_new) = stacked.qr()
                .map_err(|_| GAMError::LinAlgError("QR decomposition failed".to_string()))?;

            self.r = r_new;
        }

        #[cfg(not(feature = "blas"))]
        {
            // Fallback: use custom QR (we'll implement this if needed)
            return Err(GAMError::NotImplemented(
                "Incremental QR requires BLAS feature".to_string()
            ));
        }

        // Update Q'y by computing the same Q from the stacked QR
        // We need to apply the same transformations that created R_new to [Q'y_old; y_chunk]
        #[cfg(feature = "blas")]
        {
            // Stack [Q'y_old; y_chunk]
            let mut qty_stacked = Array1::zeros(n_total);
            qty_stacked.slice_mut(s![..self.n_cols]).assign(&self.qty);
            qty_stacked.slice_mut(s![self.n_cols..]).assign(&y_weighted);

            // Recompute QR with y included to get Q'y
            // More efficient would be to extract Q and apply it, but this is simpler for now
            let (q, _) = stacked.qr()
                .map_err(|_| GAMError::LinAlgError("QR decomposition failed".to_string()))?;

            // Q'y = Q^T * qty_stacked
            self.qty = q.t().dot(&qty_stacked);
        }

        // Update sum of squares
        self.yty += y_weighted.dot(&y_weighted);

        self.n_rows += n_chunk;

        Ok(())
    }

    /// Get the current R factor
    pub fn r(&self) -> &Array2<f64> {
        &self.r
    }

    /// Get Q'y
    pub fn qty(&self) -> &Array1<f64> {
        &self.qty
    }

    /// Compute coefficients: β = R^{-1} Q'y
    pub fn coefficients(&self) -> Result<Array1<f64>> {
        // Back-substitution to solve R β = Q'y
        back_substitute(&self.r, &self.qty)
    }

    /// Compute trace of A^{-1} · S via QR factorization
    ///
    /// Where A = R'R (from QR decomposition of augmented design matrix)
    /// This avoids forming the explicit inverse.
    ///
    /// Uses: tr(A^{-1}·S) = tr((R'R)^{-1}·S) = tr(R^{-1}·R^{-T}·S)
    ///
    /// Algorithm:
    /// 1. For each column j of S:
    ///    - Solve R' w_j = S[:,j] (forward substitution)
    ///    - Solve R v_j = w_j (back substitution)
    ///    - Sum v_j[j] (diagonal elements)
    ///
    /// # Arguments
    /// * `penalty` - Penalty matrix S (p×p)
    ///
    /// # Returns
    /// trace(A^{-1} · S)
    pub fn trace_ainv_s(&self, penalty: &Array2<f64>) -> Result<f64> {
        let p = self.n_cols;

        if penalty.nrows() != p || penalty.ncols() != p {
            return Err(GAMError::DimensionMismatch(
                format!("Penalty must be {}×{}", p, p)
            ));
        }

        let mut trace = 0.0;

        // For each column of S
        for j in 0..p {
            let s_col = penalty.column(j).to_owned();

            // Solve R' w = s_col (forward substitution with transposed R)
            let w = forward_substitute_transpose(&self.r, &s_col)?;

            // Solve R v = w (back substitution)
            let v = back_substitute(&self.r, &w)?;

            // Add diagonal element v[j] to trace
            trace += v[j];
        }

        Ok(trace)
    }
}

/// Solve upper triangular system R x = b via back-substitution
fn back_substitute(r: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = r.nrows();

    if r.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix dimensions must match".to_string()
        ));
    }

    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += r[[i, j]] * x[j];
        }

        if r[[i, i]].abs() < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        x[i] = (b[i] - sum) / r[[i, i]];
    }

    Ok(x)
}

/// Solve lower triangular system R' x = b via forward substitution
/// (where R is upper triangular, so R' is lower triangular)
fn forward_substitute_transpose(r: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = r.nrows();

    if r.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix dimensions must match".to_string()
        ));
    }

    let mut x = Array1::zeros(n);

    // Solve R' x = b where R' is lower triangular
    // R'[i,j] = R[j,i], and R is upper triangular, so R'[i,j] = 0 for j > i
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += r[[j, i]] * x[j];  // R'[i,j] = R[j,i]
        }

        if r[[i, i]].abs() < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        x[i] = (b[i] - sum) / r[[i, i]];
    }

    Ok(x)
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_incremental_qr_single_chunk() {
        // Simple test: single chunk should give same result as direct QR
        let x = Array2::from_shape_vec((5, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
            13.0, 14.0, 15.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut inc_qr = IncrementalQR::new(3);
        inc_qr.update_chunk(&x, &y, None).unwrap();

        // Check R is upper triangular
        let r = inc_qr.r();
        for i in 0..3 {
            for j in 0..i {
                assert!(r[[i, j]].abs() < 1e-10, "R should be upper triangular");
            }
        }

        // Check that R'R ≈ X'X (QR property)
        let xtx = x.t().dot(&x);
        let rtr = r.t().dot(r);

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(rtr[[i, j]], xtx[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_incremental_qr_two_chunks() {
        // Test that processing in chunks gives same R'R as full matrix
        let x = Array2::from_shape_vec((6, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Process in two chunks
        let mut inc_qr = IncrementalQR::new(2);

        let x1 = x.slice(s![0..3, ..]).to_owned();
        let y1 = y.slice(s![0..3]).to_owned();
        inc_qr.update_chunk(&x1, &y1, None).unwrap();

        let x2 = x.slice(s![3..6, ..]).to_owned();
        let y2 = y.slice(s![3..6]).to_owned();
        inc_qr.update_chunk(&x2, &y2, None).unwrap();

        // Compare with full X'X
        let xtx = x.t().dot(&x);
        let rtr = inc_qr.r().t().dot(inc_qr.r());

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(rtr[[i, j]], xtx[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_back_substitute() {
        // Test back-substitution with known solution
        let r = Array2::from_shape_vec((3, 3), vec![
            2.0, 1.0, 1.0,
            0.0, 3.0, 2.0,
            0.0, 0.0, 4.0,
        ]).unwrap();

        // R x = b where x = [1, 2, 3]
        // b = [2*1 + 1*2 + 1*3, 3*2 + 2*3, 4*3] = [7, 12, 12]
        let b = Array1::from_vec(vec![7.0, 12.0, 12.0]);

        let x = back_substitute(&r, &b).unwrap();

        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incremental_qr_weighted() {
        // Test with weights
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let w = Array1::from_vec(vec![1.0, 2.0, 1.0, 2.0]);

        let mut inc_qr = IncrementalQR::new(2);
        inc_qr.update_chunk(&x, &y, Some(&w)).unwrap();

        // Compute weighted X'WX manually
        let mut xtwx = Array2::zeros((2, 2));
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..2 {
                    xtwx[[j, k]] += w[i] * x[[i, j]] * x[[i, k]];
                }
            }
        }

        let rtr = inc_qr.r().t().dot(inc_qr.r());

        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(rtr[[i, j]], xtwx[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_incremental_qr_coefficients() {
        // Test that incremental QR gives same coefficients as batch solve
        // System: X β = y, solve for β using QR
        let x = Array2::from_shape_vec((6, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
            5.0, 6.0,
            6.0, 7.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        // Incremental QR (process in two chunks)
        let mut inc_qr = IncrementalQR::new(2);

        let x1 = x.slice(s![0..3, ..]).to_owned();
        let y1 = y.slice(s![0..3]).to_owned();
        inc_qr.update_chunk(&x1, &y1, None).unwrap();

        let x2 = x.slice(s![3..6, ..]).to_owned();
        let y2 = y.slice(s![3..6]).to_owned();
        inc_qr.update_chunk(&x2, &y2, None).unwrap();

        let beta_inc = inc_qr.coefficients().unwrap();

        // Batch QR for comparison
        use ndarray_linalg::LeastSquaresSvd;
        let beta_batch = x.least_squares(&y).unwrap().solution;

        // Compare coefficients
        for i in 0..2 {
            assert_abs_diff_eq!(beta_inc[i], beta_batch[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_forward_substitute_transpose() {
        // Test forward substitution with R' (lower triangular)
        let r = Array2::from_shape_vec((3, 3), vec![
            2.0, 1.0, 1.0,
            0.0, 3.0, 2.0,
            0.0, 0.0, 4.0,
        ]).unwrap();

        // R' x = b where x = [1, 2, 3]
        // R' = [2, 0, 0; 1, 3, 0; 1, 2, 4]
        // b = [2*1, 1*1 + 3*2, 1*1 + 2*2 + 4*3] = [2, 7, 17]
        let b = Array1::from_vec(vec![2.0, 7.0, 17.0]);

        let x = forward_substitute_transpose(&r, &b).unwrap();

        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace_ainv_s() {
        // Test trace computation via QR vs explicit inverse
        use crate::linalg::inverse;

        // Use well-conditioned data
        let x = Array2::from_shape_vec((6, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Build QR
        let mut inc_qr = IncrementalQR::new(3);
        inc_qr.update_chunk(&x, &y, None).unwrap();

        // Test penalty matrix
        let penalty = Array2::from_shape_vec((3, 3), vec![
            2.0, 0.5, 0.0,
            0.5, 3.0, 0.5,
            0.0, 0.5, 2.0,
        ]).unwrap();

        // Compute trace via QR
        let trace_qr = inc_qr.trace_ainv_s(&penalty).unwrap();

        // Compute trace via explicit inverse (for validation)
        let a = inc_qr.r().t().dot(inc_qr.r());
        let a_inv = inverse(&a).unwrap();
        let ainv_s = a_inv.dot(&penalty);
        let mut trace_inv = 0.0;
        for i in 0..3 {
            trace_inv += ainv_s[[i, i]];
        }

        // Compare
        assert_abs_diff_eq!(trace_qr, trace_inv, epsilon = 1e-8);
    }

    #[test]
    fn test_trace_ainv_s_identity() {
        // Special case: tr(A^{-1} · I) should equal tr(A^{-1})
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 1.0,
            2.0, 1.0,
            3.0, 1.0,
            4.0, 1.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mut inc_qr = IncrementalQR::new(2);
        inc_qr.update_chunk(&x, &y, None).unwrap();

        let identity = Array2::eye(2);
        let trace_qr = inc_qr.trace_ainv_s(&identity).unwrap();

        // Compute trace of A^{-1} explicitly
        use crate::linalg::inverse;
        let a = inc_qr.r().t().dot(inc_qr.r());
        let a_inv = inverse(&a).unwrap();
        let trace_inv = a_inv[[0, 0]] + a_inv[[1, 1]];

        assert_abs_diff_eq!(trace_qr, trace_inv, epsilon = 1e-8);
    }

    #[test]
    fn test_incremental_qr_multiple_chunks() {
        // Test with many small chunks to stress-test the incremental updates
        // Use well-conditioned data (not collinear)
        let x = Array2::from_shape_vec((10, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            2.0, 1.0, 0.0,
            1.0, 2.0, 0.0,
            1.0, 1.0, 2.0,
        ]).unwrap();

        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Process in 5 chunks of 2 rows each
        let mut inc_qr = IncrementalQR::new(3);
        for i in 0..5 {
            let start = i * 2;
            let end = start + 2;
            let x_chunk = x.slice(s![start..end, ..]).to_owned();
            let y_chunk = y.slice(s![start..end]).to_owned();
            inc_qr.update_chunk(&x_chunk, &y_chunk, None).unwrap();
        }

        // Compare X'X
        let xtx = x.t().dot(&x);
        let rtr = inc_qr.r().t().dot(inc_qr.r());

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(rtr[[i, j]], xtx[[i, j]], epsilon = 1e-6);
            }
        }

        // Check we can solve for coefficients
        let beta = inc_qr.coefficients();
        assert!(beta.is_ok(), "Should be able to compute coefficients");
    }
}
