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
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};

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
        }

        if let Some(deviance) = self.inner.deviance {
            result.set_item("deviance", deviance)?;
        }

        result.set_item("fitted", self.inner.fitted)?;

        Ok(result.into())
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

    fn get_fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.inner.fitted_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;

        Ok(PyArray1::from_vec_bound(py, fitted.to_vec()))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn mgcv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGAM>()?;
    Ok(())
}
