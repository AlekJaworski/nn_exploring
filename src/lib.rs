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
