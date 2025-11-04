use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{Eigh, UPLO};
use crate::errors::{MgcvError, Result};

/// Trait for smooth basis functions
pub trait Basis {
    /// Construct the basis matrix X from data
    fn basis_matrix(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>>;

    /// Get the penalty matrix S
    fn penalty_matrix(&self) -> Result<Array2<f64>>;

    /// Number of basis functions
    fn n_basis(&self) -> usize;
}

/// Cubic regression spline basis
pub struct CubicSpline {
    knots: Array1<f64>,
    n_basis: usize,
}

impl CubicSpline {
    /// Create a new cubic spline with given number of knots
    pub fn new(x: &ArrayView1<f64>, n_knots: usize) -> Result<Self> {
        if n_knots < 4 {
            return Err(MgcvError::InvalidParameter(
                "Need at least 4 knots for cubic spline".to_string(),
            ));
        }

        // Create evenly spaced knots
        let mut x_sorted = x.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let x_min = x_sorted[0];
        let x_max = x_sorted[x_sorted.len() - 1];

        let mut knots = Array1::zeros(n_knots);
        for i in 0..n_knots {
            knots[i] = x_min + (x_max - x_min) * (i as f64) / ((n_knots - 1) as f64);
        }

        Ok(Self {
            knots,
            n_basis: n_knots,
        })
    }

    fn spline_basis(&self, x: f64, knots: &Array1<f64>) -> Array1<f64> {
        let n = knots.len();
        let mut basis = Array1::zeros(n);

        // Natural cubic spline basis functions
        for i in 0..n {
            if i == 0 {
                basis[i] = 1.0;
            } else if i == 1 {
                basis[i] = x;
            } else {
                // Recursive B-spline construction
                let t = (x - knots[0]) / (knots[n - 1] - knots[0]);
                basis[i] = t.powi(i as i32);
            }
        }

        basis
    }
}

impl Basis for CubicSpline {
    fn basis_matrix(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let mut X = Array2::zeros((n, self.n_basis));

        for (i, &xi) in x.iter().enumerate() {
            let basis = self.spline_basis(xi, &self.knots);
            for j in 0..self.n_basis {
                X[[i, j]] = basis[j];
            }
        }

        Ok(X)
    }

    fn penalty_matrix(&self) -> Result<Array2<f64>> {
        let m = self.n_basis;
        let mut S = Array2::zeros((m, m));

        // Second derivative penalty (integrated squared second derivative)
        for i in 2..m {
            for j in 2..m {
                let ii = i as f64;
                let jj = j as f64;
                S[[i, j]] = ii * (ii - 1.0) * jj * (jj - 1.0) / ((ii + jj - 3.0) * (ii + jj - 2.0) * (ii + jj - 1.0));
            }
        }

        Ok(S)
    }

    fn n_basis(&self) -> usize {
        self.n_basis
    }
}

/// Thin plate regression spline
pub struct ThinPlateSpline {
    knots: Array2<f64>,
    n_basis: usize,
    dimension: usize,
}

impl ThinPlateSpline {
    /// Create a new thin plate spline
    pub fn new(x: &Array2<f64>, n_knots: usize) -> Result<Self> {
        let dimension = x.ncols();
        if dimension < 1 {
            return Err(MgcvError::InvalidParameter(
                "Data must have at least one dimension".to_string(),
            ));
        }

        // Use k-means or grid sampling to select knots
        // For simplicity, we'll use uniform sampling here
        let n = x.nrows();
        let step = n.max(n_knots) / n_knots;
        let mut knots = Array2::zeros((n_knots, dimension));

        for i in 0..n_knots {
            let idx = (i * step).min(n - 1);
            for j in 0..dimension {
                knots[[i, j]] = x[[idx, j]];
            }
        }

        Ok(Self {
            knots,
            n_basis: n_knots + dimension + 1,
            dimension,
        })
    }

    fn radial_basis(&self, r: f64) -> f64 {
        if self.dimension == 1 {
            r.powi(3)
        } else if self.dimension == 2 {
            if r > 0.0 {
                r.powi(2) * r.ln()
            } else {
                0.0
            }
        } else {
            r.powi(2 * self.dimension as i32 - 1)
        }
    }

    fn distance(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Basis for ThinPlateSpline {
    fn basis_matrix(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>> {
        // For 1D case
        if self.dimension != 1 {
            return Err(MgcvError::InvalidParameter(
                "This method only works for 1D data. Use basis_matrix_2d for multidimensional data".to_string(),
            ));
        }

        let n = x.len();
        let m = self.n_basis;
        let mut X = Array2::zeros((n, m));

        // Polynomial part
        for i in 0..n {
            X[[i, 0]] = 1.0;
            X[[i, 1]] = x[i];
        }

        // Radial basis functions
        let k = self.knots.nrows();
        for i in 0..n {
            for j in 0..k {
                let r = (x[i] - self.knots[[j, 0]]).abs();
                X[[i, j + 2]] = self.radial_basis(r);
            }
        }

        Ok(X)
    }

    fn penalty_matrix(&self) -> Result<Array2<f64>> {
        let k = self.knots.nrows();
        let m = self.n_basis;
        let mut S = Array2::zeros((m, m));

        // Penalty only on the radial basis part
        for i in 0..k {
            for j in 0..k {
                let knot_i = self.knots.row(i);
                let knot_j = self.knots.row(j);
                let r = self.distance(&knot_i, &knot_j);
                S[[i + self.dimension + 1, j + self.dimension + 1]] = self.radial_basis(r);
            }
        }

        Ok(S)
    }

    fn n_basis(&self) -> usize {
        self.n_basis
    }
}

/// P-spline (penalized B-spline)
pub struct PSpline {
    knots: Array1<f64>,
    degree: usize,
    n_basis: usize,
}

impl PSpline {
    /// Create a new P-spline
    pub fn new(x: &ArrayView1<f64>, n_basis: usize, degree: usize) -> Result<Self> {
        if degree < 1 {
            return Err(MgcvError::InvalidParameter(
                "Degree must be at least 1".to_string(),
            ));
        }

        let mut x_sorted = x.to_vec();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let x_min = x_sorted[0];
        let x_max = x_sorted[x_sorted.len() - 1];

        // Create knot sequence
        let n_interior_knots = n_basis - degree - 1;
        let n_knots = n_interior_knots + 2 * (degree + 1);
        let mut knots = Array1::zeros(n_knots);

        // Replicate boundary knots
        for i in 0..=degree {
            knots[i] = x_min;
            knots[n_knots - 1 - i] = x_max;
        }

        // Interior knots
        for i in 0..n_interior_knots {
            let t = (i + 1) as f64 / (n_interior_knots + 1) as f64;
            knots[degree + 1 + i] = x_min + t * (x_max - x_min);
        }

        Ok(Self {
            knots,
            degree,
            n_basis,
        })
    }

    fn b_spline(&self, x: f64, i: usize, k: usize) -> f64 {
        if k == 0 {
            if x >= self.knots[i] && x < self.knots[i + 1] {
                1.0
            } else {
                0.0
            }
        } else {
            let denom1 = self.knots[i + k] - self.knots[i];
            let denom2 = self.knots[i + k + 1] - self.knots[i + 1];

            let term1 = if denom1.abs() > 1e-10 {
                (x - self.knots[i]) / denom1 * self.b_spline(x, i, k - 1)
            } else {
                0.0
            };

            let term2 = if denom2.abs() > 1e-10 {
                (self.knots[i + k + 1] - x) / denom2 * self.b_spline(x, i + 1, k - 1)
            } else {
                0.0
            };

            term1 + term2
        }
    }
}

impl Basis for PSpline {
    fn basis_matrix(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let mut X = Array2::zeros((n, self.n_basis));

        for (row, &xi) in x.iter().enumerate() {
            for col in 0..self.n_basis {
                X[[row, col]] = self.b_spline(xi, col, self.degree);
            }
        }

        Ok(X)
    }

    fn penalty_matrix(&self) -> Result<Array2<f64>> {
        let m = self.n_basis;
        let mut D = Array2::zeros((m - 2, m));

        // Second order difference penalty
        for i in 0..m - 2 {
            D[[i, i]] = 1.0;
            D[[i, i + 1]] = -2.0;
            D[[i, i + 2]] = 1.0;
        }

        // S = D^T D
        let dt = D.t();
        let S = dt.dot(&D);

        Ok(S)
    }

    fn n_basis(&self) -> usize {
        self.n_basis
    }
}
