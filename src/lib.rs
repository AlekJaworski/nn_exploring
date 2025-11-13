//! mgcv_rust: A Rust implementation of Generalized Additive Models
//!
//! This library implements GAMs with automatic smoothing parameter selection
//! using REML (Restricted Maximum Likelihood) and the PiRLS (Penalized
//! Iteratively Reweighted Least Squares) algorithm, similar to R's mgcv package.

pub mod basis;
pub mod penalty;
pub mod reml;
pub mod pirls;
pub mod smooth;
pub mod gam;
pub mod gam_optimized;
pub mod gam_parallel;
pub mod utils;
pub mod linalg;

pub use gam::{GAM, SmoothTerm};
pub use basis::{BasisFunction, CubicSpline, ThinPlateSpline};
pub use smooth::{SmoothingParameter, OptimizationMethod};
pub use pirls::Family;

use thiserror::Error;

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Error, Debug)]
pub enum GAMError {
    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Singular matrix encountered")]
    SingularMatrix,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Linear algebra error: {0}")]
    LinAlgError(String),
}

pub type Result<T> = std::result::Result<T, GAMError>;

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};

/// Parse formula string like "s(0, k=10) + s(1, k=15)"
/// Returns Vec of (column_index, num_basis)
#[cfg(feature = "python")]
fn parse_formula(formula: &str) -> PyResult<Vec<(usize, usize)>> {
    let mut smooths = Vec::new();

    // Split by '+' to get individual smooth terms
    for term in formula.split('+') {
        let term = term.trim();

        // Check if it starts with 's(' and ends with ')'
        if !term.starts_with("s(") || !term.ends_with(")") {
            return Err(PyValueError::new_err(format!(
                "Invalid smooth term: '{}'. Expected format: s(col, k=value)",
                term
            )));
        }

        // Extract content between s( and )
        let content = &term[2..term.len()-1];
        let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();

        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Invalid smooth term: '{}'. Expected format: s(col, k=value)",
                term
            )));
        }

        // Parse column index
        let col_idx = parts[0].parse::<usize>().map_err(|_| {
            PyValueError::new_err(format!("Invalid column index: '{}'", parts[0]))
        })?;

        // Parse k=value
        if !parts[1].starts_with("k=") && !parts[1].starts_with("k =") {
            return Err(PyValueError::new_err(format!(
                "Invalid k specification: '{}'. Expected format: k=value",
                parts[1]
            )));
        }

        let k_value = parts[1].split('=').nth(1).ok_or_else(|| {
            PyValueError::new_err("Missing value after 'k='")
        })?.trim();

        let num_basis = k_value.parse::<usize>().map_err(|_| {
            PyValueError::new_err(format!("Invalid k value: '{}'", k_value))
        })?;

        smooths.push((col_idx, num_basis));
    }

    if smooths.is_empty() {
        return Err(PyValueError::new_err("No smooth terms found in formula"));
    }

    Ok(smooths)
}

/// Python wrapper for GAM
#[cfg(feature = "python")]
#[pyclass(name = "GAM")]
pub struct PyGAM {
    inner: GAM,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGAM {
    #[new]
    fn new() -> Self {
        PyGAM {
            inner: GAM::new(Family::Gaussian),
        }
    }

    fn add_cubic_spline(
        &mut self,
        var_name: String,
        num_basis: usize,
        x_min: f64,
        x_max: f64,
    ) -> PyResult<()> {
        let smooth = SmoothTerm::cubic_spline(var_name, num_basis, x_min, x_max)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        self.inner.add_smooth(smooth);
        Ok(())
    }

    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        method: &str,
        max_iter: Option<usize>,
    ) -> PyResult<PyObject> {
        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        let opt_method = match method {
            "GCV" => OptimizationMethod::GCV,
            "REML" => OptimizationMethod::REML,
            _ => return Err(PyValueError::new_err("method must be 'GCV' or 'REML'")),
        };

        let max_outer = max_iter.unwrap_or(10);

        self.inner.fit(&x_array, &y_array, opt_method, max_outer, 100, 1e-6)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        let result = pyo3::types::PyDict::new_bound(py);

        if let Some(ref params) = self.inner.smoothing_params {
            result.set_item("lambda", params.lambda[0])?;
            // Also include all lambdas for multi-variable GAMs
            let all_lambdas = PyArray1::from_vec_bound(py, params.lambda.clone());
            result.set_item("all_lambdas", all_lambdas)?;
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        // Return fitted values if available
        if let Some(ref fitted_values) = self.inner.fitted_values {
            let fitted_array = PyArray1::from_vec_bound(py, fitted_values.to_vec());
            result.set_item("fitted_values", fitted_array)?;
        }

        result.set_item("fitted", self.inner.fitted)?;

        Ok(result.into())
    }

    /// Fit GAM with automatic smooth setup from k values
    ///
    /// Args:
    ///     x: Input data (n x d array)
    ///     y: Response variable (n array)
    ///     k: List of basis dimensions for each column (like k in mgcv)
    ///     method: "GCV" or "REML"
    ///     bs: Basis type: "bs" (B-splines) or "cr" (cubic regression splines, mgcv default)
    ///     max_iter: Maximum iterations
    ///
    /// Example:
    ///     gam = GAM()
    ///     result = gam.fit_auto(X, y, k=[10, 15, 20], method='REML', bs='cr')
    fn fit_auto<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        method: &str,
        bs: Option<&str>,
        max_iter: Option<usize>,
    ) -> PyResult<PyObject> {
        let x_array = x.as_array().to_owned();
        let (n, d) = x_array.dim();

        // Check k dimensions
        if k.len() != d {
            return Err(PyValueError::new_err(format!(
                "k length ({}) must match number of columns ({})",
                k.len(), d
            )));
        }

        let basis_type = bs.unwrap_or("bs");  // Default to B-splines for backward compatibility

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths for each column using quantile-based knots (like mgcv)
        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i);
            let col_owned = col.to_owned();

            let smooth = match basis_type {
                "cr" => {
                    SmoothTerm::cr_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                "bs" => {
                    SmoothTerm::cubic_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown basis type '{}'. Use 'bs' or 'cr'.", basis_type
                    )));
                }
            };

            self.inner.add_smooth(smooth);
        }

        // Call regular fit
        self.fit(py, x, y, method, max_iter)
    }

    /// Fit GAM with automatic smooth setup (optimized version with caching)
    ///
    /// Uses caching and improved algorithms for better performance
    #[pyo3(signature = (x, y, k, method, bs=None, max_iter=None))]
    fn fit_auto_optimized<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        method: &str,
        bs: Option<&str>,
        max_iter: Option<usize>,
    ) -> PyResult<PyObject> {
        use crate::gam_optimized::*;

        let x_array = x.as_array().to_owned();
        let (_n, d) = x_array.dim();
        let y_array = y.as_array().to_owned();

        // Check k dimensions
        if k.len() != d {
            return Err(PyValueError::new_err(format!(
                "k length ({}) must match number of columns ({})",
                k.len(), d
            )));
        }

        let basis_type = bs.unwrap_or("bs");

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths for each column
        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i);
            let col_owned = col.to_owned();

            let smooth = match basis_type {
                "cr" => {
                    SmoothTerm::cr_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                "bs" => {
                    SmoothTerm::cubic_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown basis type '{}'. Use 'bs' or 'cr'.", basis_type
                    )));
                }
            };

            self.inner.add_smooth(smooth);
        }

        // Call optimized fit
        let opt_method = match method {
            "GCV" => OptimizationMethod::GCV,
            "REML" => OptimizationMethod::REML,
            _ => return Err(PyValueError::new_err("method must be 'GCV' or 'REML'")),
        };

        let max_outer = max_iter.unwrap_or(10);

        self.inner.fit_optimized(&x_array, &y_array, opt_method, max_outer, 100, 1e-6)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Return results
        let result = pyo3::types::PyDict::new_bound(py);

        if let Some(ref params) = self.inner.smoothing_params {
            result.set_item("lambda", params.lambda[0])?;
            let all_lambdas = PyArray1::from_vec_bound(py, params.lambda.clone());
            result.set_item("all_lambdas", all_lambdas)?;
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        if let Some(ref fitted_values) = self.inner.fitted_values {
            let fitted_array = PyArray1::from_vec_bound(py, fitted_values.to_vec());
            result.set_item("fitted_values", fitted_array)?;
        }

        result.set_item("fitted", self.inner.fitted)?;

        Ok(result.into())
    }

    /// Fit GAM with automatic smooth setup (parallel version with rayon)
    ///
    /// Uses multi-threading for basis evaluation and penalty construction.
    /// Best for multi-dimensional GAMs (d >= 3) where parallelization overhead is justified.
    #[pyo3(signature = (x, y, k, method, bs=None, max_iter=None))]
    fn fit_auto_parallel<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        method: &str,
        bs: Option<&str>,
        max_iter: Option<usize>,
    ) -> PyResult<PyObject> {
        use crate::gam_parallel::*;

        let x_array = x.as_array().to_owned();
        let (_n, d) = x_array.dim();
        let y_array = y.as_array().to_owned();

        // Check k dimensions
        if k.len() != d {
            return Err(PyValueError::new_err(format!(
                "k length ({}) must match number of columns ({})",
                k.len(), d
            )));
        }

        let basis_type = bs.unwrap_or("bs");

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths for each column
        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i);
            let col_owned = col.to_owned();

            let smooth = match basis_type {
                "cr" => {
                    SmoothTerm::cr_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                "bs" => {
                    SmoothTerm::cubic_spline_quantile(
                        format!("x{}", i),
                        num_basis,
                        &col_owned,
                    ).map_err(|e| PyValueError::new_err(format!("{}", e)))?
                },
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown basis type '{}'. Use 'bs' or 'cr'.", basis_type
                    )));
                }
            };

            self.inner.add_smooth(smooth);
        }

        // Call parallel fit
        let opt_method = match method {
            "GCV" => OptimizationMethod::GCV,
            "REML" => OptimizationMethod::REML,
            _ => return Err(PyValueError::new_err("method must be 'GCV' or 'REML'")),
        };

        let max_outer = max_iter.unwrap_or(10);

        self.inner.fit_parallel(&x_array, &y_array, opt_method, max_outer, 100, 1e-6)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Return results
        let result = pyo3::types::PyDict::new_bound(py);

        if let Some(ref params) = self.inner.smoothing_params {
            result.set_item("lambda", params.lambda[0])?;
            let all_lambdas = PyArray1::from_vec_bound(py, params.lambda.clone());
            result.set_item("all_lambdas", all_lambdas)?;
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        if let Some(ref fitted_values) = self.inner.fitted_values {
            let fitted_array = PyArray1::from_vec_bound(py, fitted_values.to_vec());
            result.set_item("fitted_values", fitted_array)?;
        }

        result.set_item("fitted", self.inner.fitted)?;

        Ok(result.into())
    }

    /// Fit GAM with formula-like syntax (mgcv-style)
    ///
    /// Args:
    ///     x: Input data (n x d array)
    ///     y: Response variable (n array)
    ///     formula: Formula string like "s(0, k=10) + s(1, k=15)"
    ///     method: "REML" (default) or "GCV"
    ///     max_iter: Maximum iterations
    ///
    /// Example:
    ///     gam = GAM()
    ///     result = gam.fit_formula(X, y, formula="s(0, k=10) + s(1, k=15)", method='REML')
    fn fit_formula<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        formula: &str,
        method: &str,
        max_iter: Option<usize>,
    ) -> PyResult<PyObject> {
        let x_array = x.as_array().to_owned();

        // Parse formula
        let smooths = parse_formula(formula)?;

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths based on formula using quantile-based knots (like mgcv)
        for (col_idx, num_basis) in smooths {
            let col = x_array.column(col_idx);
            let col_owned = col.to_owned();

            let smooth = SmoothTerm::cubic_spline_quantile(
                format!("x{}", col_idx),
                num_basis,
                &col_owned,
            ).map_err(|e| PyValueError::new_err(format!("{}", e)))?;

            self.inner.add_smooth(smooth);
        }

        // Call regular fit
        self.fit(py, x, y, method, max_iter)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_array = x.as_array().to_owned();

        let predictions = self.inner.predict(&x_array)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(PyArray1::from_vec_bound(py, predictions.to_vec()))
    }

    fn get_lambda(&self) -> PyResult<f64> {
        self.inner.smoothing_params
            .as_ref()
            .map(|p| p.lambda[0])
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))
    }

    /// Get all smoothing parameters (for multi-variable GAMs)
    fn get_all_lambdas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let lambdas = self.inner.smoothing_params
            .as_ref()
            .map(|p| p.lambda.clone())
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec_bound(py, lambdas))
    }

    fn get_fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.inner.fitted_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec_bound(py, fitted.to_vec()))
    }
}

/// Compute penalty matrix for debugging/comparison
/// Returns the raw penalty matrix for a given basis type
#[cfg(feature = "python")]
#[pyfunction]
fn compute_penalty_matrix<'py>(
    py: Python<'py>,
    basis_type: &str,
    num_basis: usize,
    knots: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    use numpy::PyArray2;
    use ndarray::Array1;

    let knots_array = Array1::from_vec(knots.to_vec()?);

    let penalty = penalty::compute_penalty(basis_type, num_basis, Some(&knots_array), 1)
        .map_err(|e| PyValueError::new_err(format!("Failed to compute penalty: {}", e)))?;

    Ok(PyArray2::from_owned_array_bound(py, penalty))
}

#[cfg(feature = "python")]
#[pymodule]
fn mgcv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGAM>()?;
    m.add_function(wrap_pyfunction!(compute_penalty_matrix, m)?)?;
    Ok(())
}
