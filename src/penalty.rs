//! Penalty matrix construction for smoothing splines

use ndarray::{Array1, Array2, s};
use crate::{Result, GAMError};

/// Compute B-spline basis function using Cox-de Boor recursion
fn b_spline_basis(x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
    if k == 0 {
        if i < t.len() - 1 {
            if i == t.len() - 2 {
                if x >= t[i] && x <= t[i + 1] { 1.0 } else { 0.0 }
            } else {
                if x >= t[i] && x < t[i + 1] { 1.0 } else { 0.0 }
            }
        } else {
            0.0
        }
    } else {
        let mut result = 0.0;
        if i + k < t.len() {
            let denom1 = t[i + k] - t[i];
            if denom1.abs() > 1e-10 {
                result += (x - t[i]) / denom1 * b_spline_basis(x, i, k - 1, t);
            }
        }
        if i + k + 1 < t.len() {
            let denom2 = t[i + k + 1] - t[i + 1];
            if denom2.abs() > 1e-10 {
                result += (t[i + k + 1] - x) / denom2 * b_spline_basis(x, i + 1, k - 1, t);
            }
        }
        result
    }
}

/// Compute B-spline first derivative
fn b_spline_derivative(x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
    if k == 0 {
        0.0
    } else {
        let mut result = 0.0;
        if i + k < t.len() {
            let denom1 = t[i + k] - t[i];
            if denom1.abs() > 1e-10 {
                result += (k as f64) / denom1 * b_spline_basis(x, i, k - 1, t);
            }
        }
        if i + k + 1 < t.len() {
            let denom2 = t[i + k + 1] - t[i + 1];
            if denom2.abs() > 1e-10 {
                result -= (k as f64) / denom2 * b_spline_basis(x, i + 1, k - 1, t);
            }
        }
        result
    }
}

/// Compute B-spline second derivative
fn b_spline_second_derivative(x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
    if k <= 1 {
        0.0
    } else {
        let mut result = 0.0;
        if i + k < t.len() {
            let denom1 = t[i + k] - t[i];
            if denom1.abs() > 1e-10 {
                result += (k as f64) / denom1 * b_spline_derivative(x, i, k - 1, t);
            }
        }
        if i + k + 1 < t.len() {
            let denom2 = t[i + k + 1] - t[i + 1];
            if denom2.abs() > 1e-10 {
                result -= (k as f64) / denom2 * b_spline_derivative(x, i + 1, k - 1, t);
            }
        }
        result
    }
}

/// Construct penalty matrix for cubic splines using analytical B-spline integrals
///
/// Computes S_ij = ∫ B''_i(x) B''_j(x) dx analytically using numerical integration
/// This is the correct penalty matrix that mgcv uses (not finite differences)
pub fn cubic_spline_penalty(num_basis: usize, knots: &Array1<f64>) -> Result<Array2<f64>> {
    let mut penalty = Array2::zeros((num_basis, num_basis));

    let n_knots = knots.len();
    if n_knots < 2 {
        return Err(GAMError::InvalidParameter(
            "Need at least 2 knots for penalty matrix".to_string()
        ));
    }

    // Create extended knot vector for cubic B-splines (degree 3)
    let degree = 3;
    let mut extended_knots = Array1::zeros(n_knots + 2 * degree);

    // Repeat boundary knots
    let ext_len = extended_knots.len();
    for i in 0..degree {
        extended_knots[i] = knots[0];
        extended_knots[ext_len - 1 - i] = knots[n_knots - 1];
    }
    for i in 0..n_knots {
        extended_knots[degree + i] = knots[i];
    }

    // Compute S_ij = ∫ B''_i(x) B''_j(x) dx using Gaussian quadrature
    // For each pair of basis functions, integrate over their overlapping support

    let x_min = knots[0];
    let x_max = knots[n_knots - 1];

    // Use 10-point Gaussian quadrature per interval for high accuracy
    let n_quad = 10;
    let quad_points = gauss_legendre_points(n_quad);

    for i in 0..num_basis {
        for j in i..num_basis {
            let mut integral = 0.0;

            // Integrate over each knot interval
            // B-splines have compact support, so we only integrate where both are non-zero
            for k in 0..(n_knots - 1) {
                let a = knots[k];
                let b = knots[k + 1];
                let h = b - a;

                // Transform Gaussian quadrature points from [-1, 1] to [a, b]
                for &(xi, wi) in &quad_points {
                    let x = a + 0.5 * h * (xi + 1.0);
                    let d2_bi = b_spline_second_derivative(x, i, degree, &extended_knots);
                    let d2_bj = b_spline_second_derivative(x, j, degree, &extended_knots);
                    integral += wi * d2_bi * d2_bj * 0.5 * h;
                }
            }

            penalty[[i, j]] = integral;
            penalty[[j, i]] = integral; // Symmetric
        }
    }

    // Normalize by largest eigenvalue (sum of absolute row values as approximation)
    // mgcv normalizes penalty matrices for numerical stability
    let mut max_row_sum = 0.0;
    for i in 0..num_basis {
        let mut row_sum = 0.0;
        for j in 0..num_basis {
            row_sum += penalty[[i, j]].abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }
    if max_row_sum > 1e-10 {
        penalty = penalty / max_row_sum;
    }

    Ok(penalty)
}

/// Gauss-Legendre quadrature points and weights on [-1, 1]
/// Returns (point, weight) pairs for n-point quadrature
fn gauss_legendre_points(n: usize) -> Vec<(f64, f64)> {
    match n {
        2 => vec![
            (-0.5773502691896257, 1.0),
            (0.5773502691896257, 1.0),
        ],
        3 => vec![
            (-0.7745966692414834, 0.5555555555555556),
            (0.0, 0.8888888888888888),
            (0.7745966692414834, 0.5555555555555556),
        ],
        5 => vec![
            (-0.9061798459386640, 0.2369268850561891),
            (-0.5384693101056831, 0.4786286704993665),
            (0.0, 0.5688888888888889),
            (0.5384693101056831, 0.4786286704993665),
            (0.9061798459386640, 0.2369268850561891),
        ],
        10 => vec![
            (-0.9739065285171717, 0.0666713443086881),
            (-0.8650633666889845, 0.1494513491505806),
            (-0.6794095682990244, 0.2190863625159820),
            (-0.4333953941292472, 0.2692667193099963),
            (-0.1488743389816312, 0.2955242247147529),
            (0.1488743389816312, 0.2955242247147529),
            (0.4333953941292472, 0.2692667193099963),
            (0.6794095682990244, 0.2190863625159820),
            (0.8650633666889845, 0.1494513491505806),
            (0.9739065285171717, 0.0666713443086881),
        ],
        _ => panic!("Unsupported number of quadrature points: {}", n),
    }
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

    // CR penalty already has correct scaling from analytical formula
    // No additional normalization needed
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
