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
pub mod utils;
pub mod linalg;
#[cfg(feature = "blas")]
pub mod newton_optimizer;
#[cfg(feature = "blas")]
pub mod reml_optimized;
pub mod blockwise_qr;
pub mod chunked_qr;

pub use gam::{GAM, SmoothTerm};
pub use basis::{BasisFunction, CubicSpline, ThinPlateSpline};
pub use smooth::{SmoothingParameter, OptimizationMethod};
#[cfg(feature = "blas")]
use crate::reml::ScaleParameterMethod;
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

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

pub type Result<T> = std::result::Result<T, GAMError>;

// Python bindings
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;

#[cfg(feature = "python")]
use pyo3::types::PyAny;

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
    #[pyo3(signature = (family=None))]
    fn new(family: Option<&str>) -> PyResult<Self> {
        let fam = match family {
            Some("gaussian") | None => Family::Gaussian,
            Some("binomial") => Family::Binomial,
            Some("poisson") => Family::Poisson,
            Some("gamma") => Family::Gamma,
            Some(f) => return Err(PyValueError::new_err(
                format!("Unknown family '{}'. Use 'gaussian', 'binomial', 'poisson', or 'gamma'", f)
            )),
        };
        Ok(PyGAM {
            inner: GAM::new(fam),
        })
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

    /// Fit GAM with automatic smooth setup and all optimizations (recommended)
    ///
    /// This is the main fitting method with sensible defaults and best performance.
    /// It automatically sets up smooths for each column and uses all optimizations.
    ///
    /// Args:
    ///     x: Input data (n x d array)
    ///     y: Response variable (n array)
    ///     k: List of basis dimensions for each column (like k in mgcv)
    ///     method: "REML" (default) or "GCV"
    ///     bs: Basis type: "cr" (cubic regression splines, default) or "bs" (B-splines)
    ///     max_iter: Maximum iterations (default: 10)
    ///     use_edf: Use Effective Degrees of Freedom for scale parameter (default: False)
    ///              When True, matches mgcv exactly but ~35% slower. Use for ill-conditioned problems.
    ///
    /// Example:
    ///     gam = GAM()
    ///     result = gam.fit(X, y, k=[10, 15, 20])
    ///     result = gam.fit(X, y, k=[10, 15, 20], use_edf=True)  # For extreme cases
    #[pyo3(signature = (x, y, k, method="REML", bs=None, max_iter=None, use_edf=None))]
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        method: &str,
        bs: Option<&str>,
        max_iter: Option<usize>,
        use_edf: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Route to the optimized implementation
        self.fit_auto_optimized(py, x, y, k, method, bs, max_iter, use_edf)
    }

    /// Low-level fit method for users who manually configure smooths
    ///
    /// Most users should use `fit()` instead, which provides automatic setup.
    /// This method is for advanced users who want full control over smooth configuration.
    fn fit_manual<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        method: &str,
        max_iter: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
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

        let result = pyo3::types::PyDict::new(py);

        if let Some(ref params) = self.inner.smoothing_params {
            // Return lambda as array (for multi-variable GAMs)
            // For single variable, this will be a 1-element array
            let lambdas = PyArray1::from_vec(py, params.lambda.clone());
            result.set_item("lambda", lambdas)?;
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        // Return fitted values if available
        if let Some(ref fitted_values) = self.inner.fitted_values {
            let fitted_array = PyArray1::from_vec(py, fitted_values.to_vec());
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
    ) -> PyResult<Py<PyAny>> {
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
                    // Use evenly-spaced knots (mgcv default) instead of quantile-based
                    let x_min = col_owned.iter().copied().fold(f64::INFINITY, f64::min);
                    let x_max = col_owned.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    SmoothTerm::cr_spline(
                        format!("x{}", i),
                        num_basis,
                        x_min,
                        x_max,
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

        // Call manual fit with pre-configured smooths
        self.fit_manual(py, x, y, method, max_iter)
    }

    /// Fit GAM with automatic smooth setup (optimized version with caching)
    ///
    /// Uses caching and improved algorithms for better performance
    #[pyo3(signature = (x, y, k, method, bs=None, max_iter=None, use_edf=None))]
    fn fit_auto_optimized<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        method: &str,
        bs: Option<&str>,
        max_iter: Option<usize>,
        use_edf: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
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
                    // Use evenly-spaced knots (mgcv default) instead of quantile-based
                    let x_min = col_owned.iter().copied().fold(f64::INFINITY, f64::min);
                    let x_max = col_owned.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    SmoothTerm::cr_spline(
                        format!("x{}", i),
                        num_basis,
                        x_min,
                        x_max,
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

        // Choose scale method based on use_edf parameter
        let scale_method = if use_edf.unwrap_or(false) {
            ScaleParameterMethod::EDF
        } else {
            ScaleParameterMethod::Rank
        };

        self.inner.fit_optimized_with_scale_method(&x_array, &y_array, opt_method, max_outer, 100, 1e-6, scale_method)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Return results
        let result = pyo3::types::PyDict::new(py);

        if let Some(ref params) = self.inner.smoothing_params {
            // Return all lambdas as array (consistent with fit_auto)
            let all_lambdas = PyArray1::from_vec(py, params.lambda.clone());
            result.set_item("lambda", all_lambdas.clone())?;
            result.set_item("all_lambdas", all_lambdas)?;
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        if let Some(ref fitted_values) = self.inner.fitted_values {
            let fitted_array = PyArray1::from_vec(py, fitted_values.to_vec());
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
    ) -> PyResult<Py<PyAny>> {
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

        // Call manual fit with pre-configured smooths
        self.fit_manual(py, x, y, method, max_iter)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_array = x.as_array().to_owned();

        let predictions = self.inner.predict(&x_array)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(PyArray1::from_vec(py, predictions.to_vec()))
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

        Ok(PyArray1::from_vec(py, lambdas))
    }

    fn get_fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.inner.fitted_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec(py, fitted.to_vec()))
    }

    /// Get the family (distribution) used by this GAM
    fn get_family(&self) -> &str {
        match self.inner.family {
            Family::Gaussian => "gaussian",
            Family::Binomial => "binomial",
            Family::Poisson => "poisson",
            Family::Gamma => "gamma",
        }
    }

    /// Get the fitted coefficients
    fn get_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coefficients = self.inner.coefficients
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec(py, coefficients.to_vec()))
    }

    /// Get the design matrix (predictor matrix)
    fn get_design_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        use numpy::PyArray2;

        let design_matrix = self.inner.design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray2::from_owned_array(py, design_matrix.clone()))
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

    Ok(PyArray2::from_owned_array(py, penalty))
}

/// Evaluate REML gradient at fixed lambda (for testing/comparison)
/// Returns gradient vector
#[cfg(feature = "python")]
#[pyfunction]
fn evaluate_gradient<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    lambdas: Vec<f64>,
    k_values: Vec<usize>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::PyArray1;
    use ndarray::{Array1, Array2};

    let x_array = x.as_array().to_owned();
    let y_array = y.as_array().to_owned();
    let (_n, d) = x_array.dim();

    // Build design matrix and penalties for each smooth
    let mut x_full = Array2::<f64>::zeros((x_array.nrows(), 0));
    let mut penalties_vec = Vec::new();

    for (col_idx, &k_val) in k_values.iter().enumerate() {
        let col = x_array.column(col_idx);
        let x_min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Create CR spline smooth
        let smooth = SmoothTerm::cr_spline(format!("x{}", col_idx), k_val, x_min, x_max)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Evaluate basis
        let basis_vals = smooth.basis.evaluate(&col.to_owned())
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        // Append to design matrix
        let old_cols = x_full.ncols();
        let new_cols = old_cols + basis_vals.ncols();
        let mut x_new = Array2::<f64>::zeros((x_full.nrows(), new_cols));
        for i in 0..x_full.nrows() {
            for j in 0..old_cols {
                x_new[[i, j]] = x_full[[i, j]];
            }
            for j in 0..basis_vals.ncols() {
                x_new[[i, old_cols + j]] = basis_vals[[i, j]];
            }
        }
        x_full = x_new;

        // Get penalty matrix (expand to full size)
        let penalty_small = smooth.penalty.clone();
        let total_cols = x_full.ncols();
        let mut penalty_full = Array2::<f64>::zeros((total_cols, total_cols));
        for i in 0..penalty_small.nrows() {
            for j in 0..penalty_small.ncols() {
                penalty_full[[old_cols + i, old_cols + j]] = penalty_small[[i, j]];
            }
        }
        penalties_vec.push(penalty_full);
    }

    // Compute gradient using QR method
    let w = Array1::from_elem(y_array.len(), 1.0);
    let gradient = reml::reml_gradient_multi_qr(&y_array, &x_full, &w, &lambdas, &penalties_vec)
        .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))?;

    Ok(PyArray1::from_owned_array(py, gradient))
}

#[cfg(feature = "python")]
#[pyfunction]
fn reml_gradient_multi_qr_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    lambdas: Vec<f64>,
    penalties: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::PyArray1;

    let y_array = y.as_array().to_owned();
    let x_array = x.as_array().to_owned();
    let w_array = w.as_array().to_owned();

    let penalties_vec: Vec<_> = penalties.iter()
        .map(|p| p.as_array().to_owned())
        .collect();

    let gradient = reml::reml_gradient_multi_qr(&y_array, &x_array, &w_array, &lambdas, &penalties_vec)
        .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))?;

    Ok(PyArray1::from_owned_array(py, gradient))
}

#[cfg(feature = "python")]
#[pyfunction]
fn reml_hessian_multi_qr_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    lambdas: Vec<f64>,
    penalties: Vec<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    use numpy::PyArray2;

    let y_array = y.as_array().to_owned();
    let x_array = x.as_array().to_owned();
    let w_array = w.as_array().to_owned();

    let penalties_vec: Vec<_> = penalties.iter()
        .map(|p| p.as_array().to_owned())
        .collect();

    let hessian = reml::reml_hessian_multi_qr(&y_array, &x_array, &w_array, &lambdas, &penalties_vec)
        .map_err(|e| PyValueError::new_err(format!("Hessian computation failed: {}", e)))?;

    Ok(PyArray2::from_owned_array(py, hessian))
}

#[cfg(feature = "python")]
#[cfg(feature = "blas")]
#[pyfunction]
fn newton_pirls_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    initial_log_lambda: PyReadonlyArray1<f64>,
    penalties: Vec<PyReadonlyArray2<f64>>,
    max_iter: Option<usize>,
    grad_tol: Option<f64>,
    verbose: Option<bool>,
) -> PyResult<(Bound<'py, numpy::PyArray1<f64>>, Bound<'py, numpy::PyArray1<f64>>, f64, usize, bool, String)> {
    use numpy::PyArray1;
    use newton_optimizer::NewtonPIRLS;

    let y_array = y.as_array().to_owned();
    let x_array = x.as_array().to_owned();
    let w_array = w.as_array().to_owned();
    let initial_log_lambda_array = initial_log_lambda.as_array().to_owned();

    let penalties_vec: Vec<_> = penalties.iter()
        .map(|p| p.as_array().to_owned())
        .collect();

    let mut optimizer = NewtonPIRLS::new();
    if let Some(max_iter) = max_iter {
        optimizer.max_iter = max_iter;
    }
    if let Some(grad_tol) = grad_tol {
        optimizer.grad_tol = grad_tol;
    }
    if let Some(verbose) = verbose {
        optimizer.verbose = verbose;
    }

    let result = optimizer.optimize(&y_array, &x_array, &w_array, &initial_log_lambda_array, &penalties_vec)
        .map_err(|e| PyValueError::new_err(format!("Newton-PIRLS optimization failed: {}", e)))?;

    Ok((
        PyArray1::from_owned_array(py, result.log_lambda),
        PyArray1::from_owned_array(py, result.lambda),
        result.reml_value,
        result.iterations,
        result.converged,
        result.message,
    ))
}

#[cfg(feature = "python")]
#[pymodule]
fn mgcv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGAM>()?;
    m.add_function(wrap_pyfunction!(compute_penalty_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(reml_gradient_multi_qr_py, m)?)?;
    m.add_function(wrap_pyfunction!(reml_hessian_multi_qr_py, m)?)?;
    m.add_function(wrap_pyfunction!(newton_pirls_py, m)?)?;
    Ok(())
}
