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

    // Simplified: Use finite difference approximation of second derivative
    // For a more rigorous implementation, we would integrate the second derivatives
    // of the B-spline basis functions

    for i in 1..(num_basis - 1) {
        for j in 1..(num_basis - 1) {
            if i == j {
                penalty[[i, j]] = 2.0;
            } else if (i as i32 - j as i32).abs() == 1 {
                penalty[[i, j]] = -1.0;
            }
        }
    }

    // Scale by average knot spacing
    if n_knots > 1 {
        let avg_spacing = (knots[n_knots - 1] - knots[0]) / (n_knots - 1) as f64;
        penalty = penalty / (avg_spacing * avg_spacing);
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

/// Compute the penalty matrix S for a given basis
pub fn compute_penalty(basis_type: &str, num_basis: usize, knots: Option<&Array1<f64>>, dim: usize) -> Result<Array2<f64>> {
    match basis_type {
        "cubic" => {
            let knots = knots.ok_or_else(|| GAMError::InvalidParameter(
                "Cubic spline penalty requires knots".to_string()
            ))?;
            cubic_spline_penalty(num_basis, knots)
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
