//! Basis functions for smoothing splines

use ndarray::{Array1, Array2};
use crate::Result;

/// Trait for basis function implementations
/// Note: Send + Sync is required for PyO3 thread safety with pyclass (PyO3 0.27+)
pub trait BasisFunction: Send + Sync {
    /// Evaluate the basis functions at given points
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>>;

    /// Get the number of basis functions
    fn num_basis(&self) -> usize;

    /// Get the knot positions (if applicable)
    fn knots(&self) -> Option<&Array1<f64>>;
}

/// Cubic regression spline basis
pub struct CubicSpline {
    /// Knot locations
    knots: Array1<f64>,
    /// Number of basis functions
    num_basis: usize,
    /// Boundary conditions: "natural" or "periodic"
    boundary: BoundaryCondition,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundaryCondition {
    Natural,
    Periodic,
}

impl CubicSpline {
    /// Create a new cubic spline basis with specified knots
    pub fn new(knots: Array1<f64>, boundary: BoundaryCondition) -> Self {
        let num_basis = knots.len() + 2; // For cubic splines with natural boundaries
        Self {
            knots,
            num_basis,
            boundary,
        }
    }

    /// Create a cubic spline with evenly spaced knots
    pub fn with_num_knots(min: f64, max: f64, num_knots: usize, boundary: BoundaryCondition) -> Self {
        let knots = Array1::linspace(min, max, num_knots);
        Self::new(knots, boundary)
    }

    /// Create a cubic spline with quantile-based knots (like mgcv)
    /// Places knots at quantiles of the data distribution for better adaptation
    ///
    /// For B-splines with repeated boundary knots, this places interior knots
    /// strictly between the data boundaries to avoid numerical issues.
    pub fn with_quantile_knots(x_data: &Array1<f64>, num_knots: usize, boundary: BoundaryCondition) -> Self {
        // Sort data to compute quantiles
        let mut sorted_x = x_data.to_vec();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_x.len();
        let x_min = sorted_x[0];
        let x_max = sorted_x[n - 1];

        let mut knots = Vec::with_capacity(num_knots);

        // For B-splines with repeated boundary knots, place interior knots
        // strictly between boundaries (not at them) to avoid degeneracy
        // Use positions 1/(num_knots+1), 2/(num_knots+1), ..., num_knots/(num_knots+1)
        for i in 0..num_knots {
            // Compute quantile position (strictly interior, not at boundaries)
            let q = (i + 1) as f64 / (num_knots + 1) as f64;
            let pos = q * (n - 1) as f64;
            let idx = pos.floor() as usize;

            // Linear interpolation between data points
            let knot = if idx >= n - 1 {
                sorted_x[n - 1]
            } else {
                let frac = pos - idx as f64;
                sorted_x[idx] * (1.0 - frac) + sorted_x[idx + 1] * frac
            };

            knots.push(knot);
        }

        // Ensure knots are strictly interior by clamping
        for knot in &mut knots {
            if *knot <= x_min {
                *knot = x_min + (x_max - x_min) * 1e-6;
            }
            if *knot >= x_max {
                *knot = x_max - (x_max - x_min) * 1e-6;
            }
        }

        Self::new(Array1::from_vec(knots), boundary)
    }

    /// Cubic B-spline basis function
    fn b_spline_basis(&self, x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
        if k == 0 {
            if i < t.len() - 1 {
                // Handle boundary: last interval includes the endpoint
                if i == t.len() - 2 {
                    if x >= t[i] && x <= t[i + 1] {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    if x >= t[i] && x < t[i + 1] {
                        1.0
                    } else {
                        0.0
                    }
                }
            } else {
                0.0
            }
        } else {
            let mut result = 0.0;

            if i + k < t.len() {
                let denom1 = t[i + k] - t[i];
                if denom1.abs() > 1e-10 {
                    result += (x - t[i]) / denom1
                        * self.b_spline_basis(x, i, k - 1, t);
                }
            }

            if i + k + 1 < t.len() {
                let denom2 = t[i + k + 1] - t[i + 1];
                if denom2.abs() > 1e-10 {
                    result += (t[i + k + 1] - x) / denom2
                        * self.b_spline_basis(x, i + 1, k - 1, t);
                }
            }

            result
        }
    }

    /// Derivative of cubic B-spline basis function
    fn b_spline_derivative(&self, x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
        if k == 0 {
            0.0
        } else {
            let mut result = 0.0;

            if i + k < t.len() {
                let denom1 = t[i + k] - t[i];
                if denom1.abs() > 1e-10 {
                    result += (k as f64) / denom1
                        * self.b_spline_basis(x, i, k - 1, t);
                }
            }

            if i + k + 1 < t.len() {
                let denom2 = t[i + k + 1] - t[i + 1];
                if denom2.abs() > 1e-10 {
                    result -= (k as f64) / denom2
                        * self.b_spline_basis(x, i + 1, k - 1, t);
                }
            }

            result
        }
    }
}

impl BasisFunction for CubicSpline {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let mut design_matrix = Array2::zeros((n, self.num_basis));

        // Extend knots for cubic B-splines (degree 3)
        let degree = 3;
        let mut extended_knots = Array1::zeros(self.knots.len() + 2 * degree);

        // Repeat boundary knots
        let knots_len = self.knots.len();
        let ext_len = extended_knots.len();
        for i in 0..degree {
            extended_knots[i] = self.knots[0];
            extended_knots[ext_len - 1 - i] = self.knots[knots_len - 1];
        }
        for i in 0..self.knots.len() {
            extended_knots[degree + i] = self.knots[i];
        }

        // Get boundary values for extrapolation
        let x_min = self.knots[0];
        let x_max = self.knots[knots_len - 1];

        // Evaluate basis functions with linear extrapolation (like mgcv)
        let eps = 1e-10;  // Small tolerance for boundary detection

        // Evaluate slightly inside boundaries to avoid repeated knot issues
        // Both boundaries handled symmetrically
        let x_boundary_left = x_min + 2.0 * eps;
        let x_boundary_right = x_max - 2.0 * eps;

        for (i, &xi) in x.iter().enumerate() {
            if xi < x_min - eps {
                // Linear extrapolation below range
                // b_j(x) ≈ b_j(x_boundary_left) + b_j'(x_boundary_left) * (x - x_boundary_left)
                for j in 0..self.num_basis {
                    let basis_val = self.b_spline_basis(x_boundary_left, j, degree, &extended_knots);
                    let basis_deriv = self.b_spline_derivative(x_boundary_left, j, degree, &extended_knots);
                    design_matrix[[i, j]] = basis_val + basis_deriv * (xi - x_boundary_left);
                }
            } else if xi > x_max + eps {
                // Linear extrapolation above range
                // b_j(x) ≈ b_j(x_boundary_right) + b_j'(x_boundary_right) * (x - x_boundary_right)
                for j in 0..self.num_basis {
                    let basis_val = self.b_spline_basis(x_boundary_right, j, degree, &extended_knots);
                    let basis_deriv = self.b_spline_derivative(x_boundary_right, j, degree, &extended_knots);
                    design_matrix[[i, j]] = basis_val + basis_deriv * (xi - x_boundary_right);
                }
            } else {
                // Within range: normal evaluation
                // Clamp to avoid numerical issues at exact boundaries
                // Use tiny offsets from both boundaries to avoid degenerate repeated knots
                let x_eval = if xi < x_min + eps {
                    x_boundary_left
                } else if xi > x_max - eps {
                    x_boundary_right
                } else {
                    xi
                };

                for j in 0..self.num_basis {
                    design_matrix[[i, j]] = self.b_spline_basis(x_eval, j, degree, &extended_knots);
                }
            }
        }

        Ok(design_matrix)
    }

    fn num_basis(&self) -> usize {
        self.num_basis
    }

    fn knots(&self) -> Option<&Array1<f64>> {
        Some(&self.knots)
    }
}

/// Cubic regression spline basis (cardinal natural cubic splines, like mgcv's "cr")
///
/// Uses cardinal basis functions where each basis function is 1 at one knot and 0 at all others.
/// Each basis function is a natural cubic spline (zero second derivatives at boundaries).
pub struct CubicRegressionSpline {
    /// Knot locations
    knots: Array1<f64>,
    /// Number of basis functions (equal to number of knots)
    num_basis: usize,
}

impl CubicRegressionSpline {
    /// Create a new cubic regression spline basis with specified knots
    pub fn new(knots: Array1<f64>) -> Self {
        let num_basis = knots.len();
        Self {
            knots,
            num_basis,
        }
    }

    /// Create cubic regression spline with evenly spaced knots
    pub fn with_num_knots(min: f64, max: f64, num_knots: usize) -> Self {
        let knots = Array1::linspace(min, max, num_knots);
        Self::new(knots)
    }

    /// Create cubic regression spline with quantile-based knots (like mgcv)
    pub fn with_quantile_knots(x_data: &Array1<f64>, num_knots: usize) -> Self {
        // Sort data to compute quantiles
        let mut sorted_x = x_data.to_vec();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_x.len();
        let mut knots = Vec::with_capacity(num_knots);

        for i in 0..num_knots {
            // Compute quantile position
            let q = i as f64 / (num_knots - 1) as f64;
            let pos = q * (n - 1) as f64;
            let idx = pos.floor() as usize;

            // Linear interpolation between data points
            let knot = if idx >= n - 1 {
                sorted_x[n - 1]
            } else {
                let frac = pos - idx as f64;
                sorted_x[idx] * (1.0 - frac) + sorted_x[idx + 1] * frac
            };

            knots.push(knot);
        }

        Self::new(Array1::from_vec(knots))
    }

    /// Solve tridiagonal system for natural cubic spline coefficients
    /// This computes the second derivatives at knots for a natural cubic spline
    fn solve_tridiagonal(&self, h: &[f64], alpha: &[f64]) -> Vec<f64> {
        let n = self.knots.len() - 1;
        let mut c = vec![0.0; n + 1];
        let mut l = vec![0.0; n + 1];
        let mut mu = vec![0.0; n + 1];
        let mut z = vec![0.0; n + 1];

        // Forward elimination
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;

        for i in 1..n {
            l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n] = 1.0;
        z[n] = 0.0;
        c[n] = 0.0;

        // Back substitution
        for j in (0..n).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }

        c
    }

    /// Evaluate a single natural cubic spline with given values at knots
    fn evaluate_natural_spline(&self, x: f64, values: &[f64]) -> f64 {
        let n = self.knots.len() - 1;

        // Find the interval
        let mut interval = 0;
        for i in 0..n {
            if x >= self.knots[i] && x <= self.knots[i + 1] {
                interval = i;
                break;
            }
            if x > self.knots[i + 1] && i == n - 1 {
                interval = n - 1;
                break;
            }
        }

        // Handle extrapolation (linear continuation)
        if x < self.knots[0] {
            // Linear extrapolation at left boundary
            let h = self.knots[1] - self.knots[0];
            let slope = (values[1] - values[0]) / h;
            return values[0] + slope * (x - self.knots[0]);
        } else if x > self.knots[n] {
            // Linear extrapolation at right boundary
            let h = self.knots[n] - self.knots[n - 1];
            let slope = (values[n] - values[n - 1]) / h;
            return values[n] + slope * (x - self.knots[n]);
        }

        // Compute h values (knot spacings)
        let mut h = vec![0.0; n];
        for i in 0..n {
            h[i] = self.knots[i + 1] - self.knots[i];
        }

        // Compute alpha values for the spline system
        let mut alpha = vec![0.0; n + 1];
        for i in 1..n {
            alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i])
                - (3.0 / h[i - 1]) * (values[i] - values[i - 1]);
        }

        // Solve for second derivatives
        let c = self.solve_tridiagonal(&h, &alpha);

        // Compute b and d coefficients
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            b[i] = (values[i + 1] - values[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        // Evaluate the spline at x
        let i = interval;
        let dx = x - self.knots[i];
        values[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx
    }
}

impl BasisFunction for CubicRegressionSpline {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let mut design_matrix = Array2::zeros((n, self.num_basis));

        // For each basis function j (corresponding to knot j)
        for j in 0..self.num_basis {
            // Create values vector: 1 at knot j, 0 everywhere else
            let mut values = vec![0.0; self.num_basis];
            values[j] = 1.0;

            // Evaluate this cardinal natural cubic spline at each x point
            for (i, &xi) in x.iter().enumerate() {
                design_matrix[[i, j]] = self.evaluate_natural_spline(xi, &values);
            }
        }

        Ok(design_matrix)
    }

    fn num_basis(&self) -> usize {
        self.num_basis
    }

    fn knots(&self) -> Option<&Array1<f64>> {
        Some(&self.knots)
    }
}

/// Thin plate regression spline basis
pub struct ThinPlateSpline {
    /// Dimension of the covariate space
    dim: usize,
    /// Number of basis functions (rank of approximation)
    num_basis: usize,
    /// Knot locations (for low-rank approximation)
    knots: Option<Array2<f64>>,
}

impl ThinPlateSpline {
    /// Create a new thin plate spline basis
    pub fn new(dim: usize, num_basis: usize) -> Self {
        Self {
            dim,
            num_basis,
            knots: None,
        }
    }

    /// Set knot locations for low-rank approximation
    pub fn with_knots(mut self, knots: Array2<f64>) -> Self {
        self.knots = Some(knots);
        self
    }

    /// Thin plate spline radial basis function
    fn tps_basis(&self, r: f64) -> f64 {
        if r < 1e-10 {
            0.0
        } else {
            if self.dim == 1 {
                r.powi(3)
            } else if self.dim == 2 {
                r.powi(2) * r.ln()
            } else {
                // For higher dimensions, use r^(2m-d) log(r) where m = 2
                let power = 2 * 2 - self.dim as i32;
                if power > 0 {
                    r.powi(power) * r.ln()
                } else {
                    r.ln()
                }
            }
        }
    }
}

impl BasisFunction for ThinPlateSpline {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        // For 1D case
        if self.dim != 1 {
            return Err(crate::GAMError::InvalidParameter(
                "ThinPlateSpline::evaluate currently only supports 1D".to_string()
            ));
        }

        let n = x.len();
        let mut design_matrix = Array2::zeros((n, self.num_basis));

        // Use data points as knots if not specified
        let knots = if let Some(ref k) = self.knots {
            k.column(0).to_owned()
        } else {
            // Use evenly spaced knots
            let min = x.iter().copied().fold(f64::INFINITY, f64::min);
            let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Array1::linspace(min, max, self.num_basis)
        };

        // Polynomial part (constant + linear for 1D)
        for i in 0..n {
            design_matrix[[i, 0]] = 1.0;
            if self.num_basis > 1 {
                design_matrix[[i, 1]] = x[i];
            }
        }

        // Radial basis functions
        let poly_terms = 2.min(self.num_basis);
        for i in 0..n {
            for j in poly_terms..self.num_basis {
                let r = (x[i] - knots[j - poly_terms]).abs();
                design_matrix[[i, j]] = self.tps_basis(r);
            }
        }

        Ok(design_matrix)
    }

    fn num_basis(&self) -> usize {
        self.num_basis
    }

    fn knots(&self) -> Option<&Array1<f64>> {
        None // Returns 2D knots, not implemented in trait
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline_creation() {
        let knots = Array1::linspace(0.0, 1.0, 10);
        let spline = CubicSpline::new(knots, BoundaryCondition::Natural);
        assert!(spline.num_basis() > 0);
    }

    #[test]
    fn test_cubic_spline_evaluation() {
        let spline = CubicSpline::with_num_knots(0.0, 1.0, 5, BoundaryCondition::Natural);
        let x = Array1::linspace(0.0, 1.0, 20);
        let basis = spline.evaluate(&x).unwrap();

        assert_eq!(basis.nrows(), 20);
        assert_eq!(basis.ncols(), spline.num_basis());
    }

    #[test]
    fn test_thin_plate_spline() {
        let tps = ThinPlateSpline::new(1, 10);
        let x = Array1::linspace(0.0, 1.0, 20);
        let basis = tps.evaluate(&x).unwrap();

        assert_eq!(basis.nrows(), 20);
        assert_eq!(basis.ncols(), 10);
    }
}
