//! # MGCV-RS: Generalized Additive Models in Rust
//!
//! This library provides a Rust implementation of Generalized Additive Models (GAMs)
//! inspired by the R package `mgcv` by Simon Wood.
//!
//! ## Features
//!
//! - Multiple smooth basis functions:
//!   - Cubic regression splines
//!   - Thin plate regression splines
//!   - P-splines (penalized B-splines)
//! - Automatic smoothing parameter selection via:
//!   - REML (Restricted Maximum Likelihood)
//!   - GCV (Generalized Cross-Validation)
//! - Multiple distribution families:
//!   - Gaussian (identity link)
//!   - Binomial (logit link)
//!   - Poisson (log link)
//! - Penalized Iteratively Reweighted Least Squares (PIRLS) fitting
//!
//! ## Example
//!
//! ```rust,no_run
//! use mgcv::prelude::*;
//! use ndarray::Array1;
//!
//! // Create data
//! let x = Array1::linspace(0.0, 10.0, 100);
//! let y = x.mapv(|xi| (xi / 2.0).sin() + 0.1 * rand::random::<f64>());
//!
//! // Create a GAM with a cubic spline
//! let mut gam = GAM::new(
//!     Box::new(Gaussian),
//!     SmoothingMethod::REML,
//! );
//!
//! let basis = CubicSpline::new(&x.view(), 10).unwrap();
//! gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);
//!
//! // Fit the model
//! gam.fit(&[x.view()], &y.view(), 100, 1e-6).unwrap();
//!
//! // Make predictions
//! let x_pred = Array1::linspace(0.0, 10.0, 200);
//! let y_pred = gam.predict(&[x_pred.view()]).unwrap();
//! ```

pub mod errors;
pub mod basis;
pub mod family;
pub mod smooth;
pub mod gam;
pub mod reml;
pub mod gcv;

pub mod prelude {
    pub use crate::errors::{MgcvError, Result};
    pub use crate::basis::{Basis, CubicSpline, ThinPlateSpline, PSpline};
    pub use crate::family::{Family, Gaussian, Binomial, Poisson};
    pub use crate::smooth::Smooth;
    pub use crate::gam::{GAM, SmoothingMethod, GAMSummary};
    pub use crate::reml::REML;
    pub use crate::gcv::GCV;
}
