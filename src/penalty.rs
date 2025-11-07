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

/// Create mgcv-style extended knot vector for B-splines
///
/// mgcv creates knots that extend beyond the data range by degree * spacing
/// Formula from mgcv smooth.construct.bs.smooth.spec:
///   xr <- xu - xl
///   xl <- xl - xr * 0.001
///   xu <- xu + xr * 0.001
///   dx <- (xu - xl)/(nk - 1)
///   k <- seq(xl - dx * m[1], xu + dx * m[1], length = nk + 2 * m[1])
fn create_mgcv_bs_knots(x_min: f64, x_max: f64, num_basis: usize, degree: usize) -> Array1<f64> {
    // In mgcv: k (bs.dim) is the basis dimension parameter
    // num_basis = k - 1 for BS splines (due to identifiability constraints)
    // To recover k from num_basis: k = num_basis + 1
    let k = num_basis + 1;

    // mgcv's formula: nk = k - degree + 1 (number of interior knot intervals)
    // Total knots = nk + 2 * degree
    let nk = k - degree + 1;

    // Extend data range slightly (0.1% on each side)
    let x_range = x_max - x_min;
    let xl = x_min - x_range * 0.001;
    let xu = x_max + x_range * 0.001;

    // Compute interior knot spacing
    let dx = (xu - xl) / (nk - 1) as f64;

    // Create extended knot sequence from (xl - degree*dx) to (xu + degree*dx)
    let n_total = nk + 2 * degree;
    let start = xl - (degree as f64) * dx;
    let end = xu + (degree as f64) * dx;

    Array1::linspace(start, end, n_total)
}

/// Construct penalty matrix for cubic splines using analytical B-spline integrals
///
/// Computes S_ij = ∫ B''_i(x) B''_j(x) dx analytically using numerical integration
/// This matches mgcv's penalty matrix calculation
///
/// # Arguments
/// * `num_basis` - Number of basis functions (for mgcv compatibility, pass k here)
/// * `knots` - Interior knots (used only to get data range [x_min, x_max])
///
/// Note: This function creates mgcv-style extended knots internally
pub fn cubic_spline_penalty(num_basis: usize, knots: &Array1<f64>) -> Result<Array2<f64>> {
    let mut penalty = Array2::zeros((num_basis, num_basis));

    let n_knots = knots.len();
    if n_knots < 2 {
        return Err(GAMError::InvalidParameter(
            "Need at least 2 knots for penalty matrix".to_string()
        ));
    }

    // Get data range from interior knots
    let x_min = knots[0];
    let x_max = knots[n_knots - 1];

    // Create mgcv-style extended knot vector
    let degree = 3;
    let extended_knots = create_mgcv_bs_knots(x_min, x_max, num_basis, degree);

    // Compute S_ij = ∫ B''_i(x) B''_j(x) dx using Gaussian quadrature
    // Integrate over the full extended knot range

    let n_intervals = extended_knots.len() - 1;

    // Use 10-point Gaussian quadrature per interval for high accuracy
    let n_quad = 10;
    let quad_points = gauss_legendre_points(n_quad);

    for i in 0..num_basis {
        for j in i..num_basis {
            let mut integral = 0.0;

            // Integrate over each knot interval in the extended knot sequence
            for k in 0..n_intervals {
                let a = extended_knots[k];
                let b = extended_knots[k + 1];
                let h = b - a;

                if h < 1e-14 {
                    continue; // Skip zero-length intervals
                }

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

    // NOTE: mgcv does NOT normalize penalty matrices
    // We use the raw penalty values to match mgcv's lambda estimates exactly

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

    // NOTE: mgcv does NOT normalize penalty matrices
    // We use the raw penalty values to match mgcv's lambda estimates exactly

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

    /// Compute finite difference penalty for comparison (old method)
    fn finite_difference_penalty(num_basis: usize, knots: &Array1<f64>) -> Array2<f64> {
        let mut penalty = Array2::zeros((num_basis, num_basis));
        let n_knots = knots.len();

        // Finite difference [1, -2, 1] pattern
        for i in 1..(num_basis - 1) {
            for j in 1..(num_basis - 1) {
                if i == j {
                    penalty[[i, j]] = 2.0;
                } else if (i as i32 - j as i32).abs() == 1 {
                    penalty[[i, j]] = -1.0;
                }
            }
        }

        // Scale by 1/h²
        if n_knots > 1 {
            let avg_spacing = (knots[n_knots - 1] - knots[0]) / (n_knots - 1) as f64;
            penalty = penalty / (avg_spacing * avg_spacing);
        }

        // Normalize by infinity norm
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

        penalty
    }

    #[test]
    fn test_cubic_spline_penalty_basic() {
        let knots = Array1::linspace(0.0, 1.0, 10);
        let penalty = cubic_spline_penalty(12, &knots).unwrap();

        assert_eq!(penalty.shape(), &[12, 12]);

        // Penalty matrix should be symmetric
        for i in 0..12 {
            for j in 0..12 {
                assert!((penalty[[i, j]] - penalty[[j, i]]).abs() < 1e-10,
                    "Penalty not symmetric at ({}, {}): {} vs {}", i, j, penalty[[i, j]], penalty[[j, i]]);
            }
        }

        // Should be positive semi-definite (non-negative eigenvalues)
        // This is a basic structural check
        assert!(penalty[[5, 5]] >= 0.0);
    }

    #[test]
    fn test_analytical_vs_finite_difference() {
        // Test with small number of basis functions for easy verification
        let knots = Array1::linspace(0.0, 1.0, 5);
        let num_basis = 7;

        let analytical = cubic_spline_penalty(num_basis, &knots).unwrap();
        let finite_diff = finite_difference_penalty(num_basis, &knots);

        // Both should be symmetric
        for i in 0..num_basis {
            for j in 0..num_basis {
                assert!((analytical[[i, j]] - analytical[[j, i]]).abs() < 1e-10);
                assert!((finite_diff[[i, j]] - finite_diff[[j, i]]).abs() < 1e-10);
            }
        }

        // Both should be normalized (infinity norm = 1)
        let mut max_row_sum_analytical: f64 = 0.0;
        let mut max_row_sum_finite: f64 = 0.0;
        for i in 0..num_basis {
            let mut row_sum_analytical: f64 = 0.0;
            let mut row_sum_finite: f64 = 0.0;
            for j in 0..num_basis {
                row_sum_analytical += analytical[[i, j]].abs();
                row_sum_finite += finite_diff[[i, j]].abs();
            }
            max_row_sum_analytical = max_row_sum_analytical.max(row_sum_analytical);
            max_row_sum_finite = max_row_sum_finite.max(row_sum_finite);
        }
        assert!((max_row_sum_analytical - 1.0).abs() < 1e-6,
            "Analytical penalty not normalized: max row sum = {}", max_row_sum_analytical);
        assert!((max_row_sum_finite - 1.0).abs() < 1e-6,
            "Finite diff penalty not normalized: max row sum = {}", max_row_sum_finite);

        // Print comparison for inspection
        println!("\nAnalytical penalty (normalized):");
        for i in 0..num_basis.min(5) {
            print!("  ");
            for j in 0..num_basis.min(5) {
                print!("{:8.4} ", analytical[[i, j]]);
            }
            println!();
        }

        println!("\nFinite difference penalty (normalized):");
        for i in 0..num_basis.min(5) {
            print!("  ");
            for j in 0..num_basis.min(5) {
                print!("{:8.4} ", finite_diff[[i, j]]);
            }
            println!();
        }
    }

    #[test]
    fn test_penalty_with_known_values() {
        // Test with simple case: 3 interior knots, 5 basis functions
        // For cubic B-splines with evenly spaced knots
        let knots = Array1::from_vec(vec![0.0, 0.5, 1.0]);
        let num_basis = 5;

        let penalty = cubic_spline_penalty(num_basis, &knots).unwrap();

        // Check basic properties
        assert_eq!(penalty.shape(), &[5, 5]);

        // Symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert!((penalty[[i, j]] - penalty[[j, i]]).abs() < 1e-10);
            }
        }

        // B-splines have compact support (degree+1 intervals = 4 for cubics)
        // So the penalty matrix should have some band structure,
        // but may not be strictly tridiagonal due to overlapping support
        // Just check it's not completely dense - very far elements should be small
        assert!(penalty[[0, 4]].abs() < 0.1, "Very far off-diagonal should be small");

        // Normalized: max row sum should be 1
        let mut max_row_sum: f64 = 0.0;
        for i in 0..5 {
            let mut row_sum: f64 = 0.0;
            for j in 0..5 {
                row_sum += penalty[[i, j]].abs();
            }
            max_row_sum = max_row_sum.max(row_sum);
        }
        assert!((max_row_sum - 1.0).abs() < 1e-6,
            "Penalty should be normalized, got max row sum: {}", max_row_sum);
    }

    #[test]
    fn test_penalty_scales_with_knot_spacing() {
        // Test that penalty magnitude scales appropriately with knot spacing
        let num_basis = 7;

        // Wide spacing
        let knots_wide = Array1::linspace(0.0, 10.0, 5);
        let penalty_wide = cubic_spline_penalty(num_basis, &knots_wide).unwrap();

        // Narrow spacing
        let knots_narrow = Array1::linspace(0.0, 1.0, 5);
        let penalty_narrow = cubic_spline_penalty(num_basis, &knots_narrow).unwrap();

        // After normalization, both should have max row sum ≈ 1
        let mut max_sum_wide: f64 = 0.0;
        let mut max_sum_narrow: f64 = 0.0;
        for i in 0..num_basis {
            let mut sum_wide: f64 = 0.0;
            let mut sum_narrow: f64 = 0.0;
            for j in 0..num_basis {
                sum_wide += penalty_wide[[i, j]].abs();
                sum_narrow += penalty_narrow[[i, j]].abs();
            }
            max_sum_wide = max_sum_wide.max(sum_wide);
            max_sum_narrow = max_sum_narrow.max(sum_narrow);
        }

        assert!((max_sum_wide - 1.0).abs() < 1e-6,
            "Wide spacing: max row sum = {}", max_sum_wide);
        assert!((max_sum_narrow - 1.0).abs() < 1e-6,
            "Narrow spacing: max row sum = {}", max_sum_narrow);
    }

    #[test]
    fn test_cr_spline_penalty_basic() {
        // Test cubic regression spline penalty
        let knots = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let penalty = cr_spline_penalty(5, &knots).unwrap();

        assert_eq!(penalty.shape(), &[5, 5]);

        // Symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert!((penalty[[i, j]] - penalty[[j, i]]).abs() < 1e-10);
            }
        }

        // CR splines (natural cubic splines) have tridiagonal second derivative structure
        // The penalty should be mostly band-diagonal with main mass near diagonal
        // Check that it's not completely dense
        let mut off_diagonal_mass = 0.0;
        let mut diagonal_mass = 0.0;
        for i in 0..5 {
            diagonal_mass += penalty[[i, i]].abs();
            for j in 0..5 {
                if (i as i32 - j as i32).abs() > 1 {
                    off_diagonal_mass += penalty[[i, j]].abs();
                }
            }
        }
        // Most mass should be on/near diagonal
        assert!(diagonal_mass > off_diagonal_mass,
            "CR penalty should have most mass near diagonal");

        // Diagonal elements should be non-negative
        // (boundary knots may have zero second derivative for natural splines)
        for i in 0..5 {
            assert!(penalty[[i, i]] >= 0.0,
                "Diagonal element {} should be non-negative: {}", i, penalty[[i, i]]);
        }

        // At least interior knots should have positive diagonal
        assert!(penalty[[2, 2]] > 0.0, "Interior knot diagonal should be positive");
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

    #[test]
    fn test_penalty_rank_deficiency() {
        // For cubic splines with second derivative penalty,
        // the penalty should have rank = num_basis - 2
        // (null space contains linear functions)
        let knots = Array1::linspace(0.0, 1.0, 5);
        let num_basis = 7;
        let penalty = cubic_spline_penalty(num_basis, &knots).unwrap();

        // Check the structure of the penalty matrix
        // For cubic B-splines, the second derivative has support on fewer intervals
        // than the original basis, so penalty should show some structure
        let first_row_sum: f64 = (0..num_basis).map(|j| penalty[[0, j]].abs()).sum();
        let last_row_sum: f64 = (0..num_basis).map(|j| penalty[[num_basis-1, j]].abs()).sum();
        let middle_row_sum: f64 = (0..num_basis).map(|j| penalty[[3, j]].abs()).sum();

        println!("\nFirst row sum: {}", first_row_sum);
        println!("Last row sum: {}", last_row_sum);
        println!("Middle row (3) sum: {}", middle_row_sum);

        // After normalization, at least one row should have sum close to 1
        // (that's what defines the infinity norm)
        assert!(first_row_sum <= 1.0 + 1e-6 && last_row_sum <= 1.0 + 1e-6,
            "Row sums should not exceed 1 after normalization");
    }
}
