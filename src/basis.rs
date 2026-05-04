//! Basis functions for smoothing splines

use crate::Result;
use ndarray::{Array1, Array2};

/// Trait for basis function implementations
/// Note: Send + Sync is required for PyO3 thread safety with pyclass (PyO3 0.27+)
pub trait BasisFunction: Send + Sync {
    /// Evaluate the basis functions at given points (standard fit-time
    /// evaluation, no extrapolation handling).
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>>;

    /// Evaluate at given points for *prediction* — for most bases this
    /// is identical to `evaluate`, but B-splines need linear
    /// extrapolation outside the inner-knot range to match mgcv's
    /// `Predict.matrix.pspline.smooth` (smooth.r:1952-1983). Default
    /// impl just delegates to `evaluate`; override on B-spline.
    fn evaluate_for_predict(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.evaluate(x)
    }

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
    pub fn with_num_knots(
        min: f64,
        max: f64,
        num_knots: usize,
        boundary: BoundaryCondition,
    ) -> Self {
        let knots = Array1::linspace(min, max, num_knots);
        Self::new(knots, boundary)
    }

    /// Create a cubic spline with quantile-based knots (like mgcv)
    /// Places knots at quantiles of the data distribution for better adaptation
    ///
    /// For B-splines with repeated boundary knots, this places interior knots
    /// strictly between the data boundaries to avoid numerical issues.
    pub fn with_quantile_knots(
        x_data: &Array1<f64>,
        num_knots: usize,
        boundary: BoundaryCondition,
    ) -> Self {
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
                    result += (x - t[i]) / denom1 * self.b_spline_basis(x, i, k - 1, t);
                }
            }

            if i + k + 1 < t.len() {
                let denom2 = t[i + k + 1] - t[i + 1];
                if denom2.abs() > 1e-10 {
                    result += (t[i + k + 1] - x) / denom2 * self.b_spline_basis(x, i + 1, k - 1, t);
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
                    result += (k as f64) / denom1 * self.b_spline_basis(x, i, k - 1, t);
                }
            }

            if i + k + 1 < t.len() {
                let denom2 = t[i + k + 1] - t[i + 1];
                if denom2.abs() > 1e-10 {
                    result -= (k as f64) / denom2 * self.b_spline_basis(x, i + 1, k - 1, t);
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
        let eps = 1e-10; // Small tolerance for boundary detection

        // Evaluate slightly inside boundaries to avoid repeated knot issues
        // Both boundaries handled symmetrically
        let x_boundary_left = x_min + 2.0 * eps;
        let x_boundary_right = x_max - 2.0 * eps;

        for (i, &xi) in x.iter().enumerate() {
            if xi < x_min - eps {
                // Linear extrapolation below range
                // b_j(x) ≈ b_j(x_boundary_left) + b_j'(x_boundary_left) * (x - x_boundary_left)
                for j in 0..self.num_basis {
                    let basis_val =
                        self.b_spline_basis(x_boundary_left, j, degree, &extended_knots);
                    let basis_deriv =
                        self.b_spline_derivative(x_boundary_left, j, degree, &extended_knots);
                    design_matrix[[i, j]] = basis_val + basis_deriv * (xi - x_boundary_left);
                }
            } else if xi > x_max + eps {
                // Linear extrapolation above range
                // b_j(x) ≈ b_j(x_boundary_right) + b_j'(x_boundary_right) * (x - x_boundary_right)
                for j in 0..self.num_basis {
                    let basis_val =
                        self.b_spline_basis(x_boundary_right, j, degree, &extended_knots);
                    let basis_deriv =
                        self.b_spline_derivative(x_boundary_right, j, degree, &extended_knots);
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
        Self { knots, num_basis }
    }

    /// Create cubic regression spline with evenly spaced knots
    pub fn with_num_knots(min: f64, max: f64, num_knots: usize) -> Self {
        let knots = Array1::linspace(min, max, num_knots);
        Self::new(knots)
    }

    /// Create cubic regression spline with quantile-based knots (like mgcv).
    ///
    /// Matches mgcv's `smooth.construct.cr.smooth.spec` (smooth.r:36-43):
    /// quantiles are computed on the **unique** values of x_data, not
    /// the raw sample. This avoids duplicate knots when the feature
    /// has mass concentrated at one value (e.g. ~90% zeros in
    /// real-estate concessions data). Returns evenly-spaced fallback
    /// when fewer than `num_knots` unique values exist (mgcv would
    /// error in that case; we degrade gracefully).
    pub fn with_quantile_knots(x_data: &Array1<f64>, num_knots: usize) -> Self {
        // Deduplicate (sorted) using a small epsilon for float-equality.
        let mut sorted_x = x_data.to_vec();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut unique: Vec<f64> = Vec::with_capacity(sorted_x.len());
        for &v in &sorted_x {
            if unique.last().is_none_or(|&last| (v - last).abs() > 1e-12) {
                unique.push(v);
            }
        }

        // Fallback: not enough unique values → evenly-spaced over the
        // observed range. mgcv stops here; we keep the call valid by
        // degrading. Penalty matrix will still be well-conditioned.
        if unique.len() < num_knots {
            let lo = *unique.first().unwrap_or(&0.0);
            let hi = *unique.last().unwrap_or(&1.0);
            let knots = Array1::linspace(lo, hi, num_knots);
            return Self::new(knots);
        }

        // Quantiles of unique values.
        let n = unique.len();
        let mut knots = Vec::with_capacity(num_knots);
        for i in 0..num_knots {
            let q = i as f64 / (num_knots - 1) as f64;
            let pos = q * (n - 1) as f64;
            let idx = pos.floor() as usize;
            let knot = if idx >= n - 1 {
                unique[n - 1]
            } else {
                let frac = pos - idx as f64;
                unique[idx] * (1.0 - frac) + unique[idx + 1] * frac
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

        // Compute b and d coefficients for each interval
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];
        for i in 0..n {
            b[i] = (values[i + 1] - values[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        // Handle extrapolation using true spline derivative at boundaries
        if x < self.knots[0] {
            // Linear extrapolation at left boundary
            // True derivative at x_0: f'(x_0) = b[0] (since dx=0 at the knot)
            let slope = b[0];
            return values[0] + slope * (x - self.knots[0]);
        } else if x > self.knots[n] {
            // Linear extrapolation at right boundary
            // True derivative at x_n: f'(x_n) = b[n-1] + 2*c[n-1]*h + 3*d[n-1]*h^2
            let hn = h[n - 1];
            let slope = b[n - 1] + 2.0 * c[n - 1] * hn + 3.0 * d[n - 1] * hn * hn;
            return values[n] + slope * (x - self.knots[n]);
        }

        // Find the interval for interior points
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

        // Evaluate the spline at x
        let i = interval;
        let dx = x - self.knots[i];
        values[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx
    }
}

impl BasisFunction for CubicRegressionSpline {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let k = self.num_basis;
        let num_intervals = k - 1;
        let mut design_matrix = Array2::zeros((n, k));

        // Pre-compute knot spacings h ONCE
        let h: Vec<f64> = (0..num_intervals)
            .map(|i| self.knots[i + 1] - self.knots[i])
            .collect();

        let knot_first = self.knots[0];
        let knot_last = self.knots[num_intervals];

        // Pre-compute all spline coefficients for all k basis functions.
        // Store in flat arrays indexed by [interval * k + basis_fn] for cache locality
        // when evaluating all basis functions at the same interval.
        let mut vals_at = vec![0.0f64; k * k]; // vals_at[interval * k + j] = values[j][interval]
        let mut b_coeff = vec![0.0f64; num_intervals * k]; // b_coeff[interval * k + j]
        let mut c_coeff = vec![0.0f64; k * k]; // c_coeff[interval * k + j]
        let mut d_coeff = vec![0.0f64; num_intervals * k]; // d_coeff[interval * k + j]
        let mut left_slopes = vec![0.0f64; k];
        let mut right_slopes = vec![0.0f64; k];
        let mut vals_0 = vec![0.0f64; k]; // values[j][0] for left extrapolation
        let mut vals_last = vec![0.0f64; k]; // values[j][num_intervals] for right extrapolation

        for j in 0..k {
            // Cardinal basis: values[j] = 1, all others = 0
            let mut values = vec![0.0; k];
            values[j] = 1.0;

            // Compute alpha (RHS of tridiagonal system)
            let mut alpha = vec![0.0; k];
            for i in 1..num_intervals {
                alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i])
                    - (3.0 / h[i - 1]) * (values[i] - values[i - 1]);
            }

            // Solve tridiagonal system for second derivatives c
            let c = self.solve_tridiagonal(&h, &alpha);

            // Store values and compute b, d coefficients
            for i in 0..num_intervals {
                let b_i = (values[i + 1] - values[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
                let d_i = (c[i + 1] - c[i]) / (3.0 * h[i]);

                vals_at[i * k + j] = values[i];
                b_coeff[i * k + j] = b_i;
                c_coeff[i * k + j] = c[i];
                d_coeff[i * k + j] = d_i;
            }
            // Store last interval endpoint values
            vals_at[num_intervals * k + j] = values[num_intervals]; // though not used in interior

            vals_0[j] = values[0];
            vals_last[j] = values[num_intervals];

            // Pre-compute extrapolation slopes
            left_slopes[j] = b_coeff[0 * k + j]; // b[0] for basis j
            let hn = h[num_intervals - 1];
            let last_i = num_intervals - 1;
            right_slopes[j] = b_coeff[last_i * k + j]
                + 2.0 * c_coeff[last_i * k + j] * hn
                + 3.0 * d_coeff[last_i * k + j] * hn * hn;
        }

        // Evaluate at all n data points — iterate by ROW for cache-friendly design matrix access
        for (i, &xi) in x.iter().enumerate() {
            let row_start = i * k; // design_matrix is row-major, row i starts at offset i*k
            let row = &mut design_matrix.as_slice_mut().unwrap()[row_start..row_start + k];

            if xi < knot_first {
                let dx = xi - knot_first;
                for j in 0..k {
                    row[j] = vals_0[j] + left_slopes[j] * dx;
                }
            } else if xi > knot_last {
                let dx = xi - knot_last;
                for j in 0..k {
                    row[j] = vals_last[j] + right_slopes[j] * dx;
                }
            } else {
                // Binary search for the interval
                let mut lo = 0usize;
                let mut hi = num_intervals;
                while lo < hi - 1 {
                    let mid = (lo + hi) / 2;
                    if xi < self.knots[mid] {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let interval = lo;
                let dx = xi - self.knots[interval];
                let dx2 = dx * dx;
                let dx3 = dx2 * dx;

                // All coefficients for this interval are contiguous in memory
                let base = interval * k;
                for j in 0..k {
                    row[j] = vals_at[base + j]
                        + b_coeff[base + j] * dx
                        + c_coeff[base + j] * dx2
                        + d_coeff[base + j] * dx3;
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

/// B-spline basis (de Boor recursion). Mirrors mgcv's `bs="bs"` with
/// default `m=c(3,2)`: order-(m[1]+1)=4 (cubic) basis with `nk = k -
/// m[1] + 1` interior knots evenly spaced over the data range, plus
/// `m[1]` boundary knots on each side at the same spacing. Total knot
/// count is `k + m[1] + 1` (mgcv's `nk + 2*m[1]`).
///
/// Reference: mgcv smooth.r:1990-2030, splines::spline.des / de Boor.
pub struct BSplineBasis {
    /// Augmented knot vector (length k + order). `order = m[1]+1` (4 for cubic).
    knots: Array1<f64>,
    /// Number of basis functions = number of distinct B-splines of given order.
    num_basis: usize,
    /// B-spline order (= polynomial degree + 1). 4 for cubic.
    order: usize,
}

impl BSplineBasis {
    /// Construct mgcv-style `bs="bs"` knots. With `m[1]=3` (cubic order
    /// 3 in mgcv's convention; basis order = 4), data range `[xl, xu]`,
    /// `k` basis functions:
    ///
    /// * `nk = k - m[1] + 1` interior knots evenly spaced from
    ///   `xl - 0.001*range` to `xu + 0.001*range`.
    /// * `m[1]` boundary knots on each side at the same spacing.
    ///
    /// Matches `smooth.r:2020-2024` literally.
    pub fn with_data_range(x_data: &Array1<f64>, num_basis: usize) -> Self {
        let m1: usize = 3; // mgcv default (cubic)
        let order = m1 + 1; // basis order = 4
        let nk = num_basis - m1 + 1; // interior knots
        if nk < 1 {
            // mgcv would error here; fall back to a sane default.
            let xl = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
            let xu = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let knots = Array1::linspace(xl, xu, num_basis + order);
            return Self { knots, num_basis, order };
        }
        let mut xl = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut xu = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let xr = xu - xl;
        xl -= xr * 0.001;
        xu += xr * 0.001;
        let dx = (xu - xl) / ((nk - 1) as f64);
        // smooth.r:2024: k <- seq(xl - dx*m[1], xu + dx*m[1], length=nk + 2*m[1])
        let total = nk + 2 * m1;
        let lo = xl - dx * (m1 as f64);
        let hi = xu + dx * (m1 as f64);
        let knots = Array1::linspace(lo, hi, total);
        Self { knots, num_basis, order }
    }

    /// Evaluate the d-th derivative of every B-spline basis function at
    /// every point in `x`. d=0 gives the basis matrix itself; d=2 gives
    /// the 2nd-derivative basis used to assemble the standard penalty.
    fn evaluate_derivative(&self, x: &Array1<f64>, deriv: usize) -> Array2<f64> {
        let n = x.len();
        let p = self.num_basis;
        let order = self.order;
        let mut out = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let xi = x[i];
            // Order-1 (constant): B_{j,1}(x) = 1 if t_j <= x < t_{j+1} else 0.
            // For x slightly past the rightmost knot, allow `<= last_k` so
            // boundary evaluations don't all collapse to zero. For x outside
            // the augmented knot range entirely, recursion naturally
            // returns 0 — matches mgcv's `splines::spline.des` semantics.
            let mut b = vec![0.0_f64; p + order - 1];
            let nk_total = self.knots.len();
            let last_k = self.knots[nk_total - 1];
            for j in 0..(p + order - 1) {
                let lo = self.knots[j];
                let hi = self.knots[j + 1];
                let inside = (lo <= xi && xi < hi)
                    || (j + 1 == p + order - 1 && (xi - last_k).abs() < 1e-15);
                if inside {
                    b[j] = 1.0;
                }
            }
            // For points strictly outside the augmented knot range, all
            // base entries stay 0 → final basis is all 0 (correct).
            let xi_clamped = xi;
            // Pure-spline recursion up to (order - deriv).
            let recur_to = order - deriv;
            for ord in 2..=recur_to {
                let mut nb = vec![0.0_f64; p + order - ord];
                for j in 0..(p + order - ord) {
                    let denom1 = self.knots[j + ord - 1] - self.knots[j];
                    let denom2 = self.knots[j + ord] - self.knots[j + 1];
                    let term1 = if denom1.abs() > 1e-15 {
                        (xi_clamped - self.knots[j]) / denom1 * b[j]
                    } else {
                        0.0
                    };
                    let term2 = if denom2.abs() > 1e-15 {
                        (self.knots[j + ord] - xi_clamped) / denom2 * b[j + 1]
                    } else {
                        0.0
                    };
                    nb[j] = term1 + term2;
                }
                b = nb;
            }
            // Derivative recursion: ∂B_{j,m}/∂x = (m-1)·[ B_{j,m-1}/(t_{j+m-1}-t_j)
            //                                          - B_{j+1,m-1}/(t_{j+m}-t_{j+1}) ]
            // Apply `deriv` times.
            let mut current_order = recur_to;
            for _ in 0..deriv {
                let next_order = current_order + 1;
                let mut nb = vec![0.0_f64; p + order - next_order];
                let mult = (next_order - 1) as f64;
                for j in 0..(p + order - next_order) {
                    let denom1 = self.knots[j + next_order - 1] - self.knots[j];
                    let denom2 = self.knots[j + next_order] - self.knots[j + 1];
                    let t1 = if denom1.abs() > 1e-15 { b[j] / denom1 } else { 0.0 };
                    let t2 = if denom2.abs() > 1e-15 { b[j + 1] / denom2 } else { 0.0 };
                    nb[j] = mult * (t1 - t2);
                }
                b = nb;
                current_order = next_order;
            }
            for j in 0..p {
                out[[i, j]] = b[j];
            }
        }
        out
    }
}

impl BasisFunction for BSplineBasis {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        Ok(self.evaluate_derivative(x, 0))
    }

    /// Predict-time evaluation with mgcv-style linear extrapolation.
    /// `Predict.matrix.pspline.smooth` (smooth.r:1952-1983): for x
    /// outside the inner-knot range `[ll, ul]`, replace the standard
    /// de Boor recursion (which gives zero past the augmented knot
    /// range and an awkward boundary-region polynomial elsewhere) with
    /// a 1st-order Taylor extension built from boundary basis values
    /// and slopes:
    ///   X[i] = D[ll] + (x - ll) · D'[ll]   when x < ll
    ///   X[i] = D[ul] + (x - ul) · D'[ul]   when x > ul
    /// In-range x uses the standard de Boor recursion unchanged.
    ///
    /// `ll` / `ul` are the first / last *interior* knots (knots[ord]
    /// and knots[len-ord-1] in 0-indexed terms).
    fn evaluate_for_predict(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let nk = self.knots.len();
        let ord = self.order;
        // mgcv's `Predict.matrix.Bspline.smooth` does `object$m <-
        // object$m - 1` BEFORE delegating to pspline.predict
        // (smooth.r:2099). The effective `m` going into the ll/ul
        // computation is `(order - 1)`, so:
        //   ll = knots[m+1] in R 1-indexed = knots[ord-1] in 0-indexed
        //   ul = knots[len-m]                = knots[len-ord]
        // For our cubic basis (ord=4) with 24 knots this gives
        // ll=knots[3]≈0.0044, ul=knots[20]≈1.001 — covers the data
        // range, so training points never trip the extrap path.
        let ll = self.knots[ord - 1];
        let ul = self.knots[nk - ord];
        let n = x.len();
        let p = self.num_basis;

        let any_below = x.iter().any(|&xi| xi < ll);
        let any_above = x.iter().any(|&xi| xi > ul);
        if !any_below && !any_above {
            return Ok(self.evaluate_derivative(x, 0));
        }

        let bnds = Array1::from_vec(vec![ll, ul]);
        let basis_at_bnds = self.evaluate_derivative(&bnds, 0);
        let dbasis_at_bnds = self.evaluate_derivative(&bnds, 1);

        let x_clamped: Array1<f64> = x.iter().map(|&xi| xi.max(ll).min(ul)).collect();
        let mut out = self.evaluate_derivative(&x_clamped, 0);

        for i in 0..n {
            let xi = x[i];
            if xi < ll {
                let dx = xi - ll;
                for j in 0..p {
                    out[[i, j]] = basis_at_bnds[[0, j]] + dx * dbasis_at_bnds[[0, j]];
                }
            } else if xi > ul {
                let dx = xi - ul;
                for j in 0..p {
                    out[[i, j]] = basis_at_bnds[[1, j]] + dx * dbasis_at_bnds[[1, j]];
                }
            }
        }
        Ok(out)
    }

    fn num_basis(&self) -> usize {
        self.num_basis
    }

    fn knots(&self) -> Option<&Array1<f64>> {
        Some(&self.knots)
    }
}

impl BSplineBasis {
    /// 2nd-derivative-squared penalty matrix.
    /// `S[i,j] = ∫ B_i''(x) B_j''(x) dx` over the data range.
    /// Approximated via Gauss-Legendre 5-point quadrature on each interior
    /// knot interval — exact for polynomials up to order 9, more than
    /// enough for the cubic 2nd-derivative-squared product (degree 2 each).
    pub fn second_derivative_penalty(&self) -> Array2<f64> {
        let p = self.num_basis;
        let order = self.order;
        let nk_total = self.knots.len();
        let mut s = Array2::<f64>::zeros((p, p));
        // 5-point Gauss-Legendre on [-1, 1]:
        let gl_x = [
            -0.906_179_845_938_664_0,
            -0.538_469_310_105_683_1,
            0.0,
            0.538_469_310_105_683_1,
            0.906_179_845_938_664_0,
        ];
        let gl_w = [
            0.236_926_885_056_189_1,
            0.478_628_670_499_366_5,
            0.568_888_888_888_888_9,
            0.478_628_670_499_366_5,
            0.236_926_885_056_189_1,
        ];
        // Integrate over interior knot intervals only (boundary knots are
        // outside the data range and contribute zero to the data-domain
        // penalty in mgcv's convention).
        let interior_lo = order - 1;
        let interior_hi = nk_total - order;
        for i in interior_lo..interior_hi {
            let a = self.knots[i];
            let b = self.knots[i + 1];
            if (b - a).abs() < 1e-15 {
                continue;
            }
            let half = 0.5 * (b - a);
            let mid = 0.5 * (a + b);
            for q in 0..5 {
                let xq = mid + half * gl_x[q];
                let wq = gl_w[q] * half;
                let xq_arr = Array1::from_vec(vec![xq]);
                let d2 = self.evaluate_derivative(&xq_arr, 2);
                for r in 0..p {
                    for c in 0..p {
                        s[[r, c]] += wq * d2[[0, r]] * d2[[0, c]];
                    }
                }
            }
        }
        // Symmetrise (cleans tiny floating-point asymmetry).
        for r in 0..p {
            for c in (r + 1)..p {
                let avg = 0.5 * (s[[r, c]] + s[[c, r]]);
                s[[r, c]] = avg;
                s[[c, r]] = avg;
            }
        }
        s
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
                "ThinPlateSpline::evaluate currently only supports 1D".to_string(),
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

/// Random-effect basis (`bs="re"` in mgcv).
///
/// Models a categorical predictor as independent random effects: the design
/// matrix is a one-hot indicator matrix (one column per unique level), and
/// the penalty is the identity matrix I_p. The smoothing parameter λ acts
/// as the inverse variance of the Gaussian prior on each level's coefficient.
///
/// Levels are stored as `Vec<f64>` — cluster IDs in this codebase are
/// integer-valued floats (1.0, 2.0, …). Exact equality (`bits`) is used
/// for the lookup because the values are integer-valued.
///
/// Prediction on unseen levels returns a row of zeros (the smooth
/// contributes nothing for unseen groups — matches mgcv's behavior).
pub struct RandomEffectBasis {
    /// Unique sorted levels from training data.
    pub levels: Vec<f64>,
}

impl RandomEffectBasis {
    /// Build a `RandomEffectBasis` from raw training data.
    ///
    /// Collects unique values, sorts them, and stores them as the level
    /// vocabulary. Duplicate detection uses exact float-bit equality
    /// (appropriate for integer-valued category IDs).
    pub fn from_data(x: &[f64]) -> Self {
        let mut vals: Vec<f64> = x.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| a.to_bits() == b.to_bits());
        Self { levels: vals }
    }

    /// Number of unique levels (= number of basis functions = number of
    /// columns in the design matrix = size of the identity penalty).
    pub fn k(&self) -> usize {
        self.levels.len()
    }

    /// One-hot design matrix: Z[i, j] = 1 iff x[i] == levels[j], else 0.
    /// Unseen levels → row of zeros.
    pub fn design_matrix_from_slice(&self, x: &[f64]) -> Array2<f64> {
        let n = x.len();
        let p = self.levels.len();
        let mut z = Array2::<f64>::zeros((n, p));
        for (i, &xi) in x.iter().enumerate() {
            // Binary search on sorted levels for an exact bit match.
            let pos = self
                .levels
                .partition_point(|&lv| lv.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less) == std::cmp::Ordering::Less);
            if pos < p && self.levels[pos].to_bits() == xi.to_bits() {
                z[[i, pos]] = 1.0;
            }
            // else: unseen level → row stays zero
        }
        z
    }

    /// Identity penalty matrix I_p.
    pub fn penalty_matrix(&self) -> Array2<f64> {
        let p = self.levels.len();
        Array2::<f64>::eye(p)
    }
}

impl BasisFunction for RandomEffectBasis {
    fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        Ok(self.design_matrix_from_slice(x.as_slice().unwrap_or(&x.to_vec())))
    }

    fn evaluate_for_predict(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        // Same as fit-time: one-hot lookup; unseen → zeros.
        self.evaluate(x)
    }

    fn num_basis(&self) -> usize {
        self.levels.len()
    }

    fn knots(&self) -> Option<&Array1<f64>> {
        None // No knots for random effects
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
