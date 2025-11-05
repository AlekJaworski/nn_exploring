//! Basis functions for smoothing splines

use ndarray::{Array1, Array2};
use crate::Result;

/// Trait for basis function implementations
pub trait BasisFunction: Send {
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

    /// Cubic B-spline basis function
    fn b_spline_basis(&self, x: f64, i: usize, k: usize, t: &Array1<f64>) -> f64 {
        if k == 0 {
            if i < t.len() - 1 && x >= t[i] && x < t[i + 1] {
                1.0
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

        // Evaluate basis functions
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..self.num_basis {
                design_matrix[[i, j]] = self.b_spline_basis(xi, j, degree, &extended_knots);
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
    use approx::assert_abs_diff_eq;

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
