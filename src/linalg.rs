//! Basic linear algebra operations for GAM fitting

use ndarray::{Array1, Array2, s};
use crate::{Result, GAMError};

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
pub fn solve(mut a: Array2<f64>, mut b: Array1<f64>) -> Result<Array1<f64>> {
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

        if max_val < 1e-12 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows
        if max_idx != k {
            for j in 0..n {
                let temp = a[[k, j]];
                a[[k, j]] = a[[max_idx, j]];
                a[[max_idx, j]] = temp;
            }
            let temp = b[k];
            b[k] = b[max_idx];
            b[max_idx] = temp;
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = a[[i, k]] / a[[k, k]];
            for j in (k + 1)..n {
                a[[i, j]] -= factor * a[[k, j]];
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

/// Compute matrix determinant using LU decomposition
pub fn determinant(a: &Array2<f64>) -> Result<f64> {
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

        // Swap rows
        if max_idx != k {
            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_idx, j]];
                lu[[max_idx, j]] = temp;
            }
            sign = -sign;
        }

        // Eliminate
        for i in (k + 1)..n {
            lu[[i, k]] /= lu[[k, k]];
            for j in (k + 1)..n {
                lu[[i, j]] -= lu[[i, k]] * lu[[k, j]];
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

/// Compute matrix inverse using Gauss-Jordan elimination
pub fn inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string()
        ));
    }

    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
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

        if max_val < 1e-12 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows
        if max_idx != k {
            for j in 0..(2 * n) {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_idx, j]];
                aug[[max_idx, j]] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[[k, k]];
        for j in 0..(2 * n) {
            aug[[k, j]] /= pivot;
        }

        // Eliminate column
        for i in 0..n {
            if i != k {
                let factor = aug[[i, k]];
                for j in 0..(2 * n) {
                    aug[[i, j]] -= factor * aug[[k, j]];
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
