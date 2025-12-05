//! Basic linear algebra operations for GAM fitting

use ndarray::{Array1, Array2, s};
use crate::{Result, GAMError};

#[cfg(feature = "blas")]
use ndarray_linalg::{Solve, Determinant, Inverse};

/// Solve linear system Ax = b
/// Uses BLAS/LAPACK when available for large matrices (n >= 1000)
/// Falls back to Gaussian elimination for small matrices (BLAS overhead dominates for n < 1000)
pub fn solve(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        // BLAS crossover point is around n=1000
        // Below this, Gaussian elimination is faster due to BLAS overhead
        if n >= 1000 {
            solve_blas(a, b)
        } else {
            solve_gaussian(a, b)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fall back to Gaussian elimination
        solve_gaussian(a, b)
    }
}

#[cfg(feature = "blas")]
fn solve_blas(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square and match RHS".to_string()
        ));
    }

    // Use general LU solver via Solve trait
    // In ndarray-linalg 0.17, solve() works directly with 1D arrays
    match Solve::solve(&a, &b) {
        Ok(x) => Ok(x),
        Err(_) => Err(GAMError::SingularMatrix)
    }
}

// Always available for hybrid BLAS approach (used for small matrices even when BLAS is enabled)
fn solve_gaussian(mut a: Array2<f64>, mut b: Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square and match RHS".to_string()
        ));
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[[k, k]].abs();

        for i in (k + 1)..n {
            let val = a[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Use more relaxed threshold for ill-conditioned penalized systems
        // mgcv uses even more relaxed thresholds with ridge regularization
        if max_val < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = a.row(k).to_owned();
            let max_row = a.row(max_idx).to_owned();

            for j in 0..n {
                a[[k, j]] = max_row[j];
                a[[max_idx, j]] = temp_row[j];
            }
            b.swap(k, max_idx);
        }

        // Eliminate (optimized but safe)
        let pivot = a[[k, k]];
        let pivot_row = a.row(k).to_owned();  // Cache pivot row

        for i in (k + 1)..n {
            let factor = a[[i, k]] / pivot;

            // Update row i using cached pivot row
            for j in (k + 1)..n {
                a[[i, j]] -= factor * pivot_row[j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += a[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Compute matrix determinant
/// Uses BLAS/LAPACK for large matrices (n >= 1000), LU decomposition for small matrices
pub fn determinant(a: &Array2<f64>) -> Result<f64> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        if n >= 1000 {
            determinant_blas(a)
        } else {
            determinant_lu(a)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        determinant_lu(a)
    }
}

#[cfg(feature = "blas")]
fn determinant_blas(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string()
        ));
    }

    Determinant::det(&a.to_owned())
        .map_err(|_| GAMError::SingularMatrix)
}

// Always available for hybrid BLAS approach
fn determinant_lu(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string()
        ));
    }

    let mut lu = a.clone();
    let mut sign = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = lu[[k, k]].abs();

        for i in (k + 1)..n {
            let val = lu[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return Ok(0.0); // Singular matrix has det = 0
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = lu.row(k).to_owned();
            let max_row = lu.row(max_idx).to_owned();

            for j in 0..n {
                lu[[k, j]] = max_row[j];
                lu[[max_idx, j]] = temp_row[j];
            }
            sign = -sign;
        }

        // Eliminate (safe version with pivot row caching)
        let pivot = lu[[k, k]];
        let pivot_row = lu.row(k).to_owned();

        for i in (k + 1)..n {
            lu[[i, k]] /= pivot;
            let multiplier = lu[[i, k]];

            // Update row i
            for j in (k + 1)..n {
                lu[[i, j]] -= multiplier * pivot_row[j];
            }
        }
    }

    // Determinant is product of diagonal elements
    let mut det = sign;
    for i in 0..n {
        det *= lu[[i, i]];
    }

    Ok(det)
}

/// Compute matrix inverse
/// Uses BLAS/LAPACK for large matrices (n >= 1000), Gauss-Jordan for small matrices
pub fn inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        if n >= 1000 {
            inverse_blas(a)
        } else {
            inverse_gauss_jordan(a)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        inverse_gauss_jordan(a)
    }
}

#[cfg(feature = "blas")]
fn inverse_blas(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string()
        ));
    }

    Inverse::inv(&a.to_owned())
        .map_err(|_| GAMError::SingularMatrix)
}

// Always available for hybrid BLAS approach
fn inverse_gauss_jordan(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string()
        ));
    }

    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I] (safe version using slicing)
    aug.slice_mut(s![.., 0..n]).assign(a);
    for i in 0..n {
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = aug[[k, k]].abs();

        for i in (k + 1)..n {
            let val = aug[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Use more relaxed threshold for numerical stability
        if max_val < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = aug.row(k).to_owned();
            let max_row = aug.row(max_idx).to_owned();

            for j in 0..(2 * n) {
                aug[[k, j]] = max_row[j];
                aug[[max_idx, j]] = temp_row[j];
            }
        }

        // Scale pivot row (safe)
        let pivot = aug[[k, k]];
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * n) {
            aug[[k, j]] *= inv_pivot;
        }

        // Eliminate column (safe with row caching)
        let pivot_row = aug.row(k).to_owned();

        for i in 0..n {
            if i != k {
                let factor = aug[[i, k]];
                for j in 0..(2 * n) {
                    aug[[i, j]] -= factor * pivot_row[j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Apply sum-to-zero identifiability constraint using simplified QR approach
///
/// This implements mgcv's approach to absorbing identifiability constraints
/// into the parameterization. For sum-to-zero constraint Σf(xi) = 0, we:
/// 1. Create constraint matrix C = [1, 1, ..., 1]^T / sqrt(k)  (normalized)
/// 2. Build orthogonal complement using Gram-Schmidt
/// 3. Return Q (k x k-1) matrix where columns are orthonormal basis for constraint complement
///
/// The constrained basis is then: X_constrained = X * Q
/// The constrained penalty is: S_constrained = Q^T * S * Q
///
/// # Arguments
/// * `k` - Number of basis functions (before constraint)
///
/// # Returns
/// * Q matrix (k x k-1) with orthonormal columns orthogonal to [1,1,...,1]
pub fn sum_to_zero_constraint_matrix(k: usize) -> Result<Array2<f64>> {
    // Build orthonormal basis for the orthogonal complement of [1,1,...,1]
    // Start with standard basis vectors and orthogonalize against [1,1,...,1]/sqrt(k)

    let mut q = Array2::zeros((k, k - 1));
    let sqrt_k = (k as f64).sqrt();

    // For each basis vector (except the first which is [1,1,...,1])
    for j in 0..(k - 1) {
        // Start with standard basis vector e_{j+1} (0,0,...,1,0,...,0)
        let mut v = Array1::zeros(k);
        v[j + 1] = 1.0;

        // Subtract projection onto [1,1,...,1]/sqrt(k)
        // projection = (v · [1/sqrt(k),...,1/sqrt(k)]) * [1/sqrt(k),...,1/sqrt(k)]
        let dot_with_ones = v.sum() / sqrt_k;  // v · [1/sqrt(k),...]
        for i in 0..k {
            v[i] -= dot_with_ones / sqrt_k;  // subtract projection
        }

        // Orthogonalize against previously computed columns
        for prev_j in 0..j {
            let prev_col = q.column(prev_j);
            let dot: f64 = v.iter().zip(prev_col.iter()).map(|(a, b)| a * b).sum();
            for i in 0..k {
                v[i] -= dot * prev_col[i];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(GAMError::SingularMatrix);
        }

        for i in 0..k {
            q[[i, j]] = v[i] / norm;
        }
    }

    Ok(q)
}

/// Apply identifiability constraint to penalty matrix
///
/// Given penalty matrix S (k x k) and constraint matrix Q (k x k-1),
/// compute: S_constrained = Q^T * S * Q
///
/// # Arguments
/// * `penalty` - Original penalty matrix (k x k)
/// * `q_matrix` - Constraint matrix from QR decomposition (k x k-1)
///
/// # Returns
/// * Constrained penalty matrix (k-1 x k-1)
pub fn apply_constraint_to_penalty(
    penalty: &Array2<f64>,
    q_matrix: &Array2<f64>
) -> Result<Array2<f64>> {
    // S_constrained = Q^T * S * Q
    let s_q = penalty.dot(q_matrix);
    let constrained_penalty = q_matrix.t().dot(&s_q);

    Ok(constrained_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_solve() {
        let a = Array2::from_shape_vec((3, 3), vec![
            2.0, 1.0, 1.0,
            1.0, 3.0, 2.0,
            1.0, 2.0, 2.0,
        ]).unwrap();

        let b = Array1::from_vec(vec![4.0, 6.0, 5.0]);

        let x = solve(a.clone(), b.clone()).unwrap();

        // Check Ax = b
        let ax = a.dot(&x);
        for i in 0..3 {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_determinant() {
        let a = Array2::from_shape_vec((2, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
        ]).unwrap();

        let det = determinant(&a).unwrap();
        assert_abs_diff_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![
            4.0, 7.0,
            2.0, 6.0,
        ]).unwrap();

        let inv = inverse(&a).unwrap();
        let product = a.dot(&inv);

        // Check that A * A^(-1) = I
        assert_abs_diff_eq!(product[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 0]], 0.0, epsilon = 1e-10);
    }
}
