//! Utility functions for GAM fitting

use ndarray::{Array1, Array2};

/// Check if a matrix is positive definite
pub fn is_positive_definite(matrix: &Array2<f64>, tolerance: f64) -> bool {
    // Simplified check - a proper implementation would compute eigenvalues
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return false;
    }

    // Check diagonal elements are positive
    for i in 0..n {
        if matrix[[i, i]] <= tolerance {
            return false;
        }
    }

    // Check symmetry
    for i in 0..n {
        for j in 0..i {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                return false;
            }
        }
    }

    true
}

/// Add a small ridge penalty to ensure positive definiteness
pub fn add_ridge(matrix: &mut Array2<f64>, ridge: f64) {
    let n = matrix.nrows();
    for i in 0..n {
        matrix[[i, i]] += ridge;
    }
}

/// Standardize a vector to have mean 0 and standard deviation 1
pub fn standardize(x: &Array1<f64>) -> (Array1<f64>, f64, f64) {
    let n = x.len() as f64;
    let mean = x.sum() / n;
    let variance = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt().max(1e-10);

    let standardized = x.iter()
        .map(|xi| (xi - mean) / std)
        .collect();

    (standardized, mean, std)
}

/// Standardize a matrix column-wise
pub fn standardize_matrix(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let (n, p) = x.dim();
    let mut means = Array1::zeros(p);
    let mut stds = Array1::zeros(p);
    let mut standardized = x.clone();

    for j in 0..p {
        let col = x.column(j);
        let mean = col.sum() / n as f64;
        let variance = col.iter()
            .map(|xi| (xi - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std = variance.sqrt().max(1e-10);

        means[j] = mean;
        stds[j] = std;

        for i in 0..n {
            standardized[[i, j]] = (x[[i, j]] - mean) / std;
        }
    }

    (standardized, means, stds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_standardize() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (standardized, mean, _std) = standardize(&x);

        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(standardized.sum(), 0.0, epsilon = 1e-10);

        let variance = standardized.iter()
            .map(|xi| xi.powi(2))
            .sum::<f64>() / standardized.len() as f64;
        assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_positive_definite() {
        let mut matrix = Array2::eye(5);
        assert!(is_positive_definite(&matrix, 1e-10));

        matrix[[0, 0]] = -1.0;
        assert!(!is_positive_definite(&matrix, 1e-10));
    }

    #[test]
    fn test_add_ridge() {
        let mut matrix = Array2::zeros((3, 3));
        add_ridge(&mut matrix, 0.1);

        for i in 0..3 {
            assert_abs_diff_eq!(matrix[[i, i]], 0.1, epsilon = 1e-10);
        }
    }
}
