//! mgcv_rust: A Rust implementation of Generalized Additive Models
//!
//! This library implements GAMs with automatic smoothing parameter selection
//! using REML (Restricted Maximum Likelihood) and the PiRLS (Penalized
//! Iteratively Reweighted Least Squares) algorithm, similar to R's mgcv package.

pub mod basis;
pub mod block_penalty;
pub mod blockwise_qr;
pub mod chunked_qr;
pub mod discrete;
pub mod gam;
pub mod gam_optimized;
pub mod linalg;
#[cfg(feature = "blas")]
pub mod newton_optimizer;
pub mod penalty;
pub mod pirls;
pub mod reml;
#[cfg(feature = "blas")]
pub mod reml_optimized;
pub mod smooth;
pub mod utils;

#[cfg(feature = "blas")]
pub use crate::reml::ScaleParameterMethod;
pub use basis::{BasisFunction, CubicSpline, ThinPlateSpline};
pub use gam::{SmoothTerm, GAM};
pub use pirls::Family;
pub use smooth::{OptimizationMethod, SmoothingParameter};

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
use pyo3::exceptions::PyValueError;

#[cfg(feature = "python")]
use pyo3::types::PyAny;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

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
        let content = &term[2..term.len() - 1];
        let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();

        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Invalid smooth term: '{}'. Expected format: s(col, k=value)",
                term
            )));
        }

        // Parse column index
        let col_idx = parts[0]
            .parse::<usize>()
            .map_err(|_| PyValueError::new_err(format!("Invalid column index: '{}'", parts[0])))?;

        // Parse k=value
        if !parts[1].starts_with("k=") && !parts[1].starts_with("k =") {
            return Err(PyValueError::new_err(format!(
                "Invalid k specification: '{}'. Expected format: k=value",
                parts[1]
            )));
        }

        let k_value = parts[1]
            .split('=')
            .nth(1)
            .ok_or_else(|| PyValueError::new_err("Missing value after 'k='"))?
            .trim();

        let num_basis = k_value
            .parse::<usize>()
            .map_err(|_| PyValueError::new_err(format!("Invalid k value: '{}'", k_value)))?;

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
    /// Opt-in to a parallel "mgcv-exact" code path that uses mgcv's
    /// basis Z, no penalty normalisation, and mgcv's REML formula
    /// byte-for-byte. Default false — current behaviour preserved.
    /// When true, fits should reproduce mgcv outputs to machine
    /// precision (Stage 4 test). Slower; experimental until each piece
    /// is implemented (basis Z → penalty norm drop → REML formula).
    mgcv_exact: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyGAM {
    #[new]
    #[pyo3(signature = (family=None, mgcv_exact=None, link=None))]
    fn new(
        family: Option<&str>,
        mgcv_exact: Option<bool>,
        link: Option<&str>,
    ) -> PyResult<Self> {
        let fam = match (family, link) {
            (Some("gaussian"), None) | (None, None) => Family::Gaussian,
            (Some("gaussian"), Some("identity")) => Family::Gaussian,
            (Some("binomial"), None) | (Some("binomial"), Some("logit")) => Family::Binomial,
            (Some("poisson"), None) | (Some("poisson"), Some("log")) => Family::Poisson,
            (Some("gamma"), None) | (Some("gamma"), Some("inverse")) => Family::Gamma,
            (Some("gamma"), Some("log")) => Family::GammaLog,
            (Some(f), Some(l)) => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported family/link combination: {}({}). Supported: \
                     gaussian(identity), binomial(logit), poisson(log), \
                     gamma(inverse), gamma(log).",
                    f, l
                )))
            }
            (Some(f), None) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown family '{}'. Use 'gaussian', 'binomial', 'poisson', or 'gamma'",
                    f
                )))
            }
            (None, Some(_)) => Family::Gaussian, // link without family — assume gaussian
        };
        let mut g = GAM::new(fam);
        // Default flipped to mgcv_exact=True after Parity 3j: byte-for-
        // byte mgcv reproduction is the documented intent. Pass
        // `mgcv_exact=False` explicitly for the legacy fast path.
        let exact = mgcv_exact.unwrap_or(true);
        g.mgcv_exact = exact;
        Ok(PyGAM { inner: g, mgcv_exact: exact })
    }

    /// Whether this GAM is using the mgcv-exact code path.
    #[getter]
    fn mgcv_exact(&self) -> bool {
        self.mgcv_exact
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
        self.fit_auto_optimized(py, x, y, k, method, bs, max_iter, use_edf, None)
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

        self.inner
            .fit(&x_array, &y_array, opt_method, max_outer, 100, 1e-6)
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
                k.len(),
                d
            )));
        }

        let basis_type = bs.unwrap_or("bs"); // Default to B-splines for backward compatibility

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths for each column using quantile-based knots (like mgcv)
        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i);
            let col_owned = col.to_owned();

            let smooth = match basis_type {
                "cr" => {
                    // mgcv's cr places knots at quantiles of the
                    // covariate by default — match that, not evenly
                    // spaced. The old comment claiming "mgcv default"
                    // for linspace was wrong.
                    SmoothTerm::cr_spline_quantile(format!("x{}", i), num_basis, &col_owned)
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?
                }
                "bs" => SmoothTerm::cubic_spline_quantile(format!("x{}", i), num_basis, &col_owned)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown basis type '{}'. Use 'bs' or 'cr'.",
                        basis_type
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
    ///
    /// Args:
    ///     algorithm: "newton" (default) or "fellner-schall" (faster, matches bam)
    #[cfg(feature = "blas")]
    #[pyo3(signature = (x, y, k, method, bs=None, max_iter=None, use_edf=None, algorithm=None))]
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
        algorithm: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        use crate::gam_optimized::*;

        let x_array = x.as_array().to_owned();
        let (_n, d) = x_array.dim();
        let y_array = y.as_array().to_owned();

        // Check k dimensions
        if k.len() != d {
            return Err(PyValueError::new_err(format!(
                "k length ({}) must match number of columns ({})",
                k.len(),
                d
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
                    // mgcv's cr places knots at quantiles of the
                    // covariate by default — match that, not evenly
                    // spaced. The old comment claiming "mgcv default"
                    // for linspace was wrong.
                    SmoothTerm::cr_spline_quantile(format!("x{}", i), num_basis, &col_owned)
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?
                }
                "bs" => SmoothTerm::cubic_spline_quantile(format!("x{}", i), num_basis, &col_owned)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown basis type '{}'. Use 'bs' or 'cr'.",
                        basis_type
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

        // Choose REML algorithm
        use crate::smooth::REMLAlgorithm;
        let reml_algo = match algorithm {
            Some("fellner-schall") | Some("fs") | Some("fREML") => {
                Some(REMLAlgorithm::FellnerSchall)
            }
            Some("newton") => Some(REMLAlgorithm::Newton),
            None => None, // Use default (Newton)
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown algorithm '{}'. Use 'newton' or 'fellner-schall'.",
                    other
                )))
            }
        };

        self.inner
            .fit_optimized_full(
                &x_array,
                &y_array,
                opt_method,
                max_outer,
                100,
                1e-6,
                scale_method,
                reml_algo,
            )
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

            let smooth =
                SmoothTerm::cubic_spline_quantile(format!("x{}", col_idx), num_basis, &col_owned)
                    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

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

        let predictions = self
            .inner
            .predict(&x_array)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

        Ok(PyArray1::from_vec(py, predictions.to_vec()))
    }

    fn get_lambda(&self) -> PyResult<f64> {
        self.inner
            .smoothing_params
            .as_ref()
            .map(|p| p.lambda[0])
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))
    }

    /// Get all smoothing parameters (for multi-variable GAMs)
    fn get_all_lambdas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let lambdas = self
            .inner
            .smoothing_params
            .as_ref()
            .map(|p| p.lambda.clone())
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec(py, lambdas))
    }

    fn get_fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .inner
            .fitted_values
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
            Family::GammaLog => "gamma",
        }
    }

    /// Get the link function name used by this GAM
    fn get_link(&self) -> &str {
        match self.inner.family {
            Family::Gaussian => "identity",
            Family::Binomial => "logit",
            Family::Poisson => "log",
            Family::Gamma => "inverse",
            Family::GammaLog => "log",
        }
    }

    /// Get the fitted coefficients
    fn get_coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coefficients = self
            .inner
            .coefficients
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec(py, coefficients.to_vec()))
    }

    /// Get the design matrix (predictor matrix)
    fn get_design_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        use numpy::PyArray2;

        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray2::from_owned_array(py, design_matrix.clone()))
    }

    /// Predictor (smooth-term) names in the order they were added to the
    /// model. The ergonomics wrapper uses this to align user-supplied
    /// predictor lists with the actual fitted smooth order.
    fn get_predictor_names(&self) -> Vec<String> {
        self.inner
            .smooth_terms
            .iter()
            .map(|s| s.name.clone())
            .collect()
    }

    /// Per-smooth column index range in the design matrix.
    /// Returns a list of `(predictor_name, first_index, last_index)`
    /// tuples, with indices INCLUSIVE on both ends and 0-based against
    /// the full design `[1 | s_0 | s_1 | ...]` (so the intercept is at
    /// index 0 and the first smooth starts at index 1).
    ///
    /// Mirrors the schema produced by `extract_term_indices` in
    /// `r_fitting.r_model.py`, which indexes smooth contributions for
    /// marginal predictions and serialization.
    fn get_term_indices(&self) -> Vec<(String, usize, usize)> {
        let mut out = Vec::with_capacity(self.inner.smooth_terms.len());
        let mut col = 1usize; // skip the intercept column
        for sm in &self.inner.smooth_terms {
            let nb = sm.num_basis();
            out.push((sm.name.clone(), col, col + nb - 1));
            col += nb;
        }
        out
    }

    /// Evaluate the design matrix at the given `X` — same `[1 |
    /// smooth_1 | smooth_2 | ...]` layout as `get_design_matrix()` but
    /// at arbitrary x rather than the training set. This is mgcv's
    /// `predict(fit, newdata=X, type="lpmatrix")` and is the building
    /// block for posterior sampling and confidence intervals.
    fn evaluate_lpmatrix<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        use numpy::PyArray2;
        let x_array = x.as_array().to_owned();
        let lp = self
            .inner
            .build_lpmatrix(&x_array)
            .map_err(|e| PyValueError::new_err(format!("lpmatrix evaluation failed: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, lp))
    }

    /// Posterior covariance of β̂. For Gaussian: `σ² · (X'WX + λS)⁻¹`
    /// with σ² = RSS / (n − tr(A⁻¹X'WX)). For non-Gaussian (binomial,
    /// poisson, gamma): `(X'WX + λS)⁻¹` with W from the converged
    /// PIRLS step (no σ² scaling for known-scale families; profiled
    /// scale for gamma/gaussian). Mirrors mgcv's `vcov(fit)`. Used by
    /// the ergonomics wrapper for confidence intervals and posterior
    /// sampling.
    #[cfg(feature = "blas")]
    fn get_vcov<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        use numpy::PyArray2;
        let design = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let weights_arr;
        let weights = match self.inner.weights.as_ref() {
            Some(w) => w,
            None => {
                weights_arr = ndarray::Array1::ones(design.nrows());
                &weights_arr
            }
        };
        let smoothing = self
            .inner
            .smoothing_params
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Smoothing parameters not stored"))?;
        let lambdas = &smoothing.lambda;
        let coefficients = self
            .inner
            .coefficients
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        // Build penalties at the same offsets used at fit time:
        // intercept at column 0 (unpenalised), then per-smooth blocks.
        let total_cols = design.ncols();
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> = Vec::new();
        let mut col_offset = 1usize;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            penalties.push(crate::block_penalty::BlockPenalty::new(
                smooth.penalty.clone(),
                col_offset,
                total_cols,
            ));
            col_offset += nb;
        }

        // A = X'WX + Σλ_j S_j   (no ridge — consistent with mgcv-exact)
        let xtwx = crate::reml::compute_xtwx(design, weights);
        let mut a = xtwx.clone();
        for (lam, pen) in lambdas.iter().zip(penalties.iter()) {
            pen.scaled_add_to(&mut a, *lam);
        }
        // Tiny ridge purely for numerical stability of the inverse;
        // scaled to the matrix's own magnitude so it's invisible at
        // typical β scale.
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let ridge = 1e-12 * max_diag;
        for i in 0..a.nrows() {
            a[[i, i]] += ridge;
        }
        let a_inv = crate::linalg::inverse(&a)
            .map_err(|e| PyValueError::new_err(format!("vcov inverse failed: {}", e)))?;

        // Scale parameter σ²:
        //   - Gaussian / Gamma: profiled  σ² = deviance / (n − tr(A⁻¹X'WX))
        //     For Gaussian, deviance = RSS exactly.
        //     For Gamma, deviance is the GLM deviance (Pearson would be
        //     more standard for σ² estimation; we follow mgcv's choice).
        //   - Binomial / Poisson: known   σ² = 1
        let scale = match self.inner.family {
            Family::Binomial | Family::Poisson => 1.0,
            _ => {
                let dev = self
                    .inner
                    .deviance
                    .ok_or_else(|| PyValueError::new_err("Deviance not stored"))?;
                let n = design.nrows() as f64;
                let tr_a = crate::reml::compute_xtwx(design, weights)
                    .dot(&a_inv)
                    .diag()
                    .sum();
                let dof = (n - tr_a).max(1e-10);
                dev / dof
            }
        };
        let _ = coefficients; // explicit no-op marker if scale path didn't use it

        let vcov: ndarray::Array2<f64> = a_inv * scale;
        Ok(PyArray2::from_owned_array(py, vcov))
    }

    /// Closed-form Hessian of the mgcv-exact REML score at the given
    /// λ. Diagnostic — used by Stage 5 unit tests.
    #[cfg(feature = "blas")]
    fn evaluate_reml_hessian_closed_form<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let weights_arr;
        let weights: &ndarray::Array1<f64> = match self.inner.weights.as_ref() {
            Some(w) => w,
            None => {
                weights_arr = ndarray::Array1::ones(design_matrix.nrows());
                &weights_arr
            }
        };
        let total_cols = design_matrix.ncols();
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> = Vec::new();
        let mut col_offset = 1usize;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            penalties.push(crate::block_penalty::BlockPenalty::new(
                smooth.penalty.clone(),
                col_offset,
                total_cols,
            ));
            col_offset += nb;
        }
        let y_array = y.as_array().to_owned();
        let hess = crate::reml::reml_hessian_mgcv_exact_closed_form(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            self.inner.family,
        )
        .map_err(|e| PyValueError::new_err(format!("hessian failed: {}", e)))?;
        Ok(numpy::PyArray2::from_owned_array(py, hess))
    }

    /// Closed-form gradient of the mgcv-exact REML score at the
    /// given λ vector. Diagnostic — used by Stage 5 unit tests to
    /// verify against finite-difference gradient.
    #[cfg(feature = "blas")]
    fn evaluate_reml_gradient_closed_form<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let weights_arr;
        let weights: &ndarray::Array1<f64> = match self.inner.weights.as_ref() {
            Some(w) => w,
            None => {
                weights_arr = ndarray::Array1::ones(design_matrix.nrows());
                &weights_arr
            }
        };
        // Build penalties at the same offsets as fit_optimized_full
        let total_cols = design_matrix.ncols();
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> = Vec::new();
        let mut col_offset = 1usize;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            penalties.push(crate::block_penalty::BlockPenalty::new(
                smooth.penalty.clone(),
                col_offset,
                total_cols,
            ));
            col_offset += nb;
        }
        let y_array = y.as_array().to_owned();
        let grad = crate::reml::reml_gradient_mgcv_exact_closed_form(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            self.inner.family,
        )
        .map_err(|e| PyValueError::new_err(format!("gradient failed: {}", e)))?;
        Ok(numpy::PyArray1::from_owned_array(py, grad))
    }

    /// Per-smooth centred penalty matrix as stored on SmoothTerm.
    /// Returns a list of ndarrays, one per smooth (each is the
    /// post-centring penalty Z'SZ, after any pre-Z normalisation if
    /// mgcv_exact was used at fit time). Diagnostic only.
    fn get_smooth_penalties<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray2<f64>>>> {
        let mut out = Vec::with_capacity(self.inner.smooth_terms.len());
        for smooth in &self.inner.smooth_terms {
            out.push(numpy::PyArray2::from_owned_array(py, smooth.penalty.clone()));
        }
        Ok(out)
    }

    /// Per-smooth penalty scale factor used by the optimizer
    /// (gam_optimized.rs::FitCache::new computes `ma_xx / inf_norm_s`
    /// for each smooth and uses it as a multiplier on S_j during the
    /// fit). The reported λ values from get_all_lambdas() multiply the
    /// SCALED penalty, not the raw S_j — so to compare with mgcv,
    /// divide by the scale factors here.
    #[cfg(feature = "blas")]
    fn get_penalty_scale_factors<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let mut scales = Vec::with_capacity(self.inner.smooth_terms.len());
        let mut col_offset = 1usize;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            let block = design_matrix.slice(ndarray::s![.., col_offset..col_offset + nb]);
            let inf_norm_x = block
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let ma_xx = inf_norm_x * inf_norm_x;
            let inf_norm_s = (0..nb)
                .map(|i| (0..nb).map(|j| smooth.penalty[[i, j]].abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let sf = if inf_norm_s > 1e-10 { ma_xx / inf_norm_s } else { 1.0 };
            scales.push(sf);
            col_offset += nb;
        }
        Ok(numpy::PyArray1::from_vec(py, scales))
    }

    /// Evaluate this fitted GAM's REML score at an arbitrary lambda
    /// vector, reusing the cached design matrix, IRLS weights, and
    /// per-smooth (centred) penalty matrices. Returns the same score
    /// the Newton optimiser was minimising during fit().
    ///
    /// Evaluate this fitted GAM's REML at λ using mgcv's exact formula
    /// (gam.fit3.r:621). For Gaussian: requires the gam to have been
    /// fitted in mgcv_exact=True mode so the design matrix and penalties
    /// are in the matching parameterisation.
    #[cfg(feature = "blas")]
    fn evaluate_reml_mgcv_formula(
        &self,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
    ) -> PyResult<f64> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let weights_arr;
        let weights: &ndarray::Array1<f64> = match self.inner.weights.as_ref() {
            Some(w) => w,
            None => {
                weights_arr = ndarray::Array1::ones(design_matrix.nrows());
                &weights_arr
            }
        };
        if lambdas.len() != self.inner.smooth_terms.len() {
            return Err(PyValueError::new_err("lambdas length mismatch"));
        }
        // Build raw penalties at intercept-aware offsets.
        let total_cols = design_matrix.ncols();
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> = Vec::new();
        let mut col_offset = 1usize;
        // Mp = 1 (intercept) + Σ (k_j_centred - rank(S_j)) — null space per smooth.
        // Use eigen-based rank (robust to centring; the legacy
        // estimate_rank subtracts 2 which is the RAW penalty's null
        // space, not the centred one's).
        let mut mp: usize = 1;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            let pen_block = crate::block_penalty::BlockPenalty::new(
                smooth.penalty.clone(),
                col_offset,
                total_cols,
            );
            let rank_s = crate::reml::estimate_rank_eigen(&pen_block);
            let null_dim = nb.saturating_sub(rank_s);
            mp += null_dim;
            penalties.push(pen_block);
            col_offset += nb;
        }
        let y_array = y.as_array().to_owned();
        crate::reml::reml_criterion_multi_cached_mgcv_exact(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            mp,
            self.inner.family,
        )
        .map_err(|e| PyValueError::new_err(format!("REML evaluation failed: {}", e)))
    }

    /// `mgcv_coords` controls how the input lambdas are interpreted:
    ///   - `False` (default): lambdas are in our optimizer's coordinate
    ///     system (matching what get_all_lambdas returns). The
    ///     effective penalty is `λ_j * scale_factor_j * S_j`.
    ///   - `True`: lambdas are mgcv's raw values that multiply S_j
    ///     directly. We divide them by our scale_factor before applying,
    ///     so the effective penalty becomes `λ_j * S_j` exactly as
    ///     mgcv would compute. Use this to compare REML at mgcv's
    ///     converged λ in our score.
    ///
    /// Used by the parity test suite to probe whether mgcv_rust and
    /// mgcv are optimising the same objective: compute our REML at
    /// mgcv's converged λ (with mgcv_coords=True) and compare to our
    /// REML at our own converged λ (mgcv_coords=False).
    #[cfg(feature = "blas")]
    fn evaluate_reml_at(
        &self,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
        mgcv_coords: Option<bool>,
    ) -> PyResult<f64> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let weights_arr;
        let weights: &ndarray::Array1<f64> = match self.inner.weights.as_ref() {
            Some(w) => w,
            None => {
                weights_arr = ndarray::Array1::ones(design_matrix.nrows());
                &weights_arr
            }
        };
        if lambdas.len() != self.inner.smooth_terms.len() {
            return Err(PyValueError::new_err(format!(
                "lambdas length ({}) must match number of smooth terms ({})",
                lambdas.len(),
                self.inner.smooth_terms.len()
            )));
        }
        // We always work the REML criterion through with the *raw*
        // (unscaled) penalties S_j. The optimizer's reported λ
        // multiplies a SCALED S_j (= scale_factor_j * S_j), so to get
        // the same effective penalty in raw-S terms we multiply each
        // optimizer-coordinate λ by its scale factor. mgcv-coordinate
        // λ values are already raw multipliers and pass through.
        let total_cols = design_matrix.ncols();
        let use_mgcv_coords = mgcv_coords.unwrap_or(false);
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> =
            Vec::with_capacity(self.inner.smooth_terms.len());
        let mut effective_lambdas: Vec<f64> = Vec::with_capacity(lambdas.len());
        let mut col_offset = 1usize;
        for (i, smooth) in self.inner.smooth_terms.iter().enumerate() {
            let nb = smooth.num_basis();
            // Always pass the raw (unscaled) S_j to the REML criterion.
            penalties.push(crate::block_penalty::BlockPenalty::new(
                smooth.penalty.clone(),
                col_offset,
                total_cols,
            ));
            // Compute scale_factor once, only used to convert
            // optimizer-coordinate λ to a raw multiplier.
            let block = design_matrix.slice(ndarray::s![.., col_offset..col_offset + nb]);
            let inf_norm_x = block
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let ma_xx = inf_norm_x * inf_norm_x;
            let inf_norm_s = (0..nb)
                .map(|i| (0..nb).map(|j| smooth.penalty[[i, j]].abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let scale_factor = if inf_norm_s > 1e-10 { ma_xx / inf_norm_s } else { 1.0 };
            let lam_i = if use_mgcv_coords {
                lambdas[i]
            } else {
                lambdas[i] * scale_factor
            };
            effective_lambdas.push(lam_i);
            col_offset += nb;
        }
        let lambdas_to_use = effective_lambdas;
        let y_array = y.as_array().to_owned();
        crate::reml::reml_criterion_multi_cached(
            &y_array,
            design_matrix,
            weights,
            &lambdas_to_use,
            &penalties,
            None,
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("REML evaluation failed: {}", e)))
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
    use ndarray::Array1;
    use numpy::PyArray2;

    let knots_array = Array1::from_vec(knots.to_vec()?);

    let penalty = penalty::compute_penalty(basis_type, num_basis, Some(&knots_array), 1)
        .map_err(|e| PyValueError::new_err(format!("Failed to compute penalty: {}", e)))?;

    Ok(PyArray2::from_owned_array(py, penalty))
}

/// Evaluate REML gradient at fixed lambda (for testing/comparison)
/// Returns gradient vector
#[cfg(feature = "python")]
#[pyfunction]
#[cfg(feature = "blas")]
fn evaluate_gradient<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    lambdas: Vec<f64>,
    k_values: Vec<usize>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use ndarray::{Array1, Array2};
    use numpy::PyArray1;

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
        let basis_vals = smooth
            .basis
            .evaluate(&col.to_owned())
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
    let penalties_block: Vec<_> = penalties_vec
        .into_iter()
        .map(|p| block_penalty::BlockPenalty::new(p.clone(), 0, p.nrows()))
        .collect();
    let gradient =
        reml::reml_gradient_multi_qr_adaptive(&y_array, &x_full, &w, &lambdas, &penalties_block)
            .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))?;

    Ok(PyArray1::from_owned_array(py, gradient))
}

#[cfg(feature = "python")]
#[cfg(feature = "blas")]
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

    let penalties_vec: Vec<_> = penalties.iter().map(|p| p.as_array().to_owned()).collect();
    let penalties_block: Vec<_> = penalties_vec
        .iter()
        .map(|p| block_penalty::BlockPenalty::new(p.clone(), 0, p.nrows()))
        .collect();

    let gradient = reml::reml_gradient_multi_qr_adaptive(
        &y_array,
        &x_array,
        &w_array,
        &lambdas,
        &penalties_block,
    )
    .map_err(|e| PyValueError::new_err(format!("Gradient computation failed: {}", e)))?;

    Ok(PyArray1::from_owned_array(py, gradient))
}

#[cfg(feature = "python")]
#[cfg(feature = "blas")]
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

    let penalties_vec: Vec<_> = penalties.iter().map(|p| p.as_array().to_owned()).collect();
    let penalties_block: Vec<_> = penalties_vec
        .iter()
        .map(|p| block_penalty::BlockPenalty::new(p.clone(), 0, p.nrows()))
        .collect();

    let hessian =
        reml::reml_hessian_multi_qr(&y_array, &x_array, &w_array, &lambdas, &penalties_block)
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
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    f64,
    usize,
    bool,
    String,
)> {
    use newton_optimizer::NewtonPIRLS;
    use numpy::PyArray1;

    let y_array = y.as_array().to_owned();
    let x_array = x.as_array().to_owned();
    let w_array = w.as_array().to_owned();
    let initial_log_lambda_array = initial_log_lambda.as_array().to_owned();

    let penalties_vec: Vec<_> = penalties.iter().map(|p| p.as_array().to_owned()).collect();

    // Wrap dense penalties in BlockPenalty for the optimizer
    // These are full p×p matrices from Python, so offset=0 and total_size=p
    let block_penalties: Vec<crate::block_penalty::BlockPenalty> = penalties_vec
        .iter()
        .map(|p| crate::block_penalty::BlockPenalty::new(p.clone(), 0, p.nrows()))
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

    let result = optimizer
        .optimize(
            &y_array,
            &x_array,
            &w_array,
            &initial_log_lambda_array,
            &block_penalties,
        )
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
    #[cfg(feature = "blas")]
    m.add_function(wrap_pyfunction!(newton_pirls_py, m)?)?;
    Ok(())
}
