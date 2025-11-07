//! Penalty matrix construction for smoothing splines

use ndarray::{Array1, Array2, s};
use crate::{Result, GAMError};

/// Construct penalty matrix for cubic splines
///
/// This creates a second derivative penalty matrix for smoothing splines
pub fn cubic_spline_penalty(num_basis: usize, knots: &Array1<f64>) -> Result<Array2<f64>> {
    let mut penalty = Array2::zeros((num_basis, num_basis));

    // For cubic splines, penalty is based on integrated second derivative
    // S = integral (f''(x))^2 dx
    //
    // This can be computed analytically from the B-spline basis

    let n_knots = knots.len();
    if n_knots < 2 {
        return Err(GAMError::InvalidParameter(
            "Need at least 2 knots for penalty matrix".to_string()
        ));
    }

    // Use finite difference approximation of second derivative
    // D² = [1, -2, 1] / h²
    //
    // For the penalty matrix (integrated squared second derivative):
    // S = ∫ (D²f)² dx ≈ (D²)ᵀ * W * D²
    // where W is integration weight matrix (diagonal with weights h)
    //
    // This gives net scaling of 1/h (not 1/h²)
    // See Wood (2017) "Generalized Additive Models" Section 5.3

    for i in 1..(num_basis - 1) {
        for j in 1..(num_basis - 1) {
            if i == j {
                penalty[[i, j]] = 2.0;
            } else if (i as i32 - j as i32).abs() == 1 {
                penalty[[i, j]] = -1.0;
            }
        }
    }

    // Scale by 1/h² for second derivative penalty on B-spline coefficients
    // Note: For B-spline BASIS coefficients (not function values), the
    // finite difference approximation to ∫[f''(x)]² dx uses 1/h² scaling
    if n_knots > 1 {
        let avg_spacing = (knots[n_knots - 1] - knots[0]) / (n_knots - 1) as f64;
        penalty = penalty / (avg_spacing * avg_spacing);  // 1/h² scaling
    }

    // Normalize penalty matrix by Frobenius norm (sqrt of sum of squared elements)
    // This ensures the penalty matrix has consistent scale across different k values
    // mgcv normalizes penalties for numerical stability
    let mut frob_norm_sq = 0.0;
    for i in 0..num_basis {
        for j in 0..num_basis {
            frob_norm_sq += penalty[[i, j]] * penalty[[i, j]];
        }
    }
    let frob_norm = frob_norm_sq.sqrt();
    if frob_norm > 1e-10 {
        penalty = penalty / frob_norm;
    }

    Ok(penalty)
}

/// Construct penalty matrix for thin plate splines
///
/// For thin plate splines, the penalty is based on the integrated squared
/// second derivatives
pub fn thin_plate_penalty(num_basis: usize, dim: usize) -> Result<Array2<f64>> {
    let mut penalty = Array2::zeros((num_basis, num_basis));

    if dim == 1 {
        // For 1D, similar to cubic spline
        for i in 2..num_basis {
            for j in 2..num_basis {
                // Radial basis functions (excluding polynomial terms)
                if i == j {
                    penalty[[i, j]] = 1.0;
                }
            }
        }
    } else {
        // For higher dimensions, the penalty is more complex
        // Simplified version: penalize non-polynomial part
        let poly_terms = if dim == 1 { 2 } else { (dim + 1) * (dim + 2) / 2 };

        for i in poly_terms..num_basis {
            for j in poly_terms..num_basis {
                if i == j {
                    penalty[[i, j]] = 1.0;
                }
            }
        }
    }

    Ok(penalty)
}

/// Construct penalty matrix for cubic regression splines (cr basis like mgcv)
///
/// For cubic regression splines with cardinal natural cubic spline basis,
/// the penalty matrix S_ij = integral (h_i''(x) * h_j''(x)) dx
/// where h_i is the i-th cardinal basis function.
pub fn cr_spline_penalty(num_basis: usize, knots: &Array1<f64>) -> Result<Array2<f64>> {
    if knots.len() != num_basis {
        return Err(GAMError::InvalidParameter(
            format!("Number of knots ({}) must equal number of basis functions ({}) for cr splines",
                    knots.len(), num_basis)
        ));
    }

    let mut penalty = Array2::zeros((num_basis, num_basis));
    let n = num_basis - 1;

    // For natural cubic splines, we can compute the penalty matrix analytically
    // The second derivative of a cubic spline in interval [x_i, x_{i+1}] is linear
    // For cardinal basis functions, we need to integrate the product of second derivatives

    // Compute knot spacings
    let mut h = vec![0.0; n];
    for i in 0..n {
        h[i] = knots[i + 1] - knots[i];
    }

    // Build penalty matrix using the second derivative formula for natural cubic splines
    // This is based on Wood (2017) Section 5.3.1
    for i in 0..num_basis {
        for j in i..num_basis {
            let mut integral = 0.0;

            // For each interval [knots[k], knots[k+1]]
            for k in 0..n {
                // The second derivatives of basis functions i and j in this interval
                // For natural cubic splines, the second derivative is piecewise linear

                // Contribution from interval k
                if k > 0 && k < n {
                    let weight = if i == k && j == k {
                        2.0 / (3.0 * h[k]) + 2.0 / (3.0 * h[k - 1])
                    } else if i == k && j == k + 1 {
                        1.0 / (3.0 * h[k])
                    } else if i == k + 1 && j == k {
                        1.0 / (3.0 * h[k])
                    } else if i == k - 1 && j == k {
                        1.0 / (3.0 * h[k - 1])
                    } else if i == k && j == k - 1 {
                        1.0 / (3.0 * h[k - 1])
                    } else if (i as i32 - j as i32).abs() == 1 && (i == k || j == k) {
                        1.0 / (6.0 * h[k.min(k - 1)])
                    } else {
                        0.0
                    };
                    integral += weight;
                } else if k == 0 {
                    // First interval
                    let weight = if i == 0 && j == 0 {
                        2.0 / (3.0 * h[0])
                    } else if (i == 0 && j == 1) || (i == 1 && j == 0) {
                        1.0 / (3.0 * h[0])
                    } else {
                        0.0
                    };
                    integral += weight;
                } else if k == n - 1 {
                    // Last interval
                    let weight = if i == n && j == n {
                        2.0 / (3.0 * h[n - 1])
                    } else if (i == n - 1 && j == n) || (i == n && j == n - 1) {
                        1.0 / (3.0 * h[n - 1])
                    } else {
                        0.0
                    };
                    integral += weight;
                }
            }

            penalty[[i, j]] = integral;
            penalty[[j, i]] = integral; // Symmetric
        }
    }

    // Normalize penalty matrix by Frobenius norm (sqrt of sum of squared elements)
    // This ensures the penalty matrix has consistent scale across different k values
    // mgcv normalizes penalties for numerical stability
    let mut frob_norm_sq = 0.0;
    for i in 0..num_basis {
        for j in 0..num_basis {
            frob_norm_sq += penalty[[i, j]] * penalty[[i, j]];
        }
    }
    let frob_norm = frob_norm_sq.sqrt();
    if frob_norm > 1e-10 {
        penalty = penalty / frob_norm;
    }

    Ok(penalty)
}

/// Compute the penalty matrix S for a given basis
pub fn compute_penalty(basis_type: &str, num_basis: usize, knots: Option<&Array1<f64>>, dim: usize) -> Result<Array2<f64>> {
    match basis_type {
        "cubic" => {
            let knots = knots.ok_or_else(|| GAMError::InvalidParameter(
                "Cubic spline penalty requires knots".to_string()
            ))?;
            cubic_spline_penalty(num_basis, knots)
        },
        "cr" | "cubic_regression" => {
            let knots = knots.ok_or_else(|| GAMError::InvalidParameter(
                "Cubic regression spline penalty requires knots".to_string()
            ))?;
            cr_spline_penalty(num_basis, knots)
        },
        "tps" | "thin_plate" => {
            thin_plate_penalty(num_basis, dim)
        },
        _ => Err(GAMError::InvalidParameter(
            format!("Unknown basis type: {}", basis_type)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline_penalty() {
        let knots = Array1::linspace(0.0, 1.0, 10);
        let penalty = cubic_spline_penalty(12, &knots).unwrap();

        assert_eq!(penalty.shape(), &[12, 12]);

        // Penalty matrix should be symmetric
        for i in 0..12 {
            for j in 0..12 {
                assert!((penalty[[i, j]] - penalty[[j, i]]).abs() < 1e-10);
            }
        }

        // Should be positive semi-definite (non-negative eigenvalues)
        // This is a basic structural check
        assert!(penalty[[5, 5]] >= 0.0);
    }

    #[test]
    fn test_thin_plate_penalty() {
        let penalty = thin_plate_penalty(10, 1).unwrap();

        assert_eq!(penalty.shape(), &[10, 10]);

        // Should be symmetric
        for i in 0..10 {
            for j in 0..10 {
                assert!((penalty[[i, j]] - penalty[[j, i]]).abs() < 1e-10);
            }
        }
    }
}
