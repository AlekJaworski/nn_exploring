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
pub mod family_theta;
pub mod gam;
pub mod gam_optimized;
pub mod linalg;
pub mod link;
#[cfg(feature = "blas")]
pub mod newton_optimizer;
pub mod ocat;
pub mod penalty;
pub mod pirls;
pub mod reml;
#[cfg(feature = "blas")]
pub mod reml_optimized;
#[cfg(feature = "blas")]
pub mod reparam;
pub mod smooth;
pub mod utils;

#[cfg(feature = "blas")]
pub use crate::reml::ScaleParameterMethod;
pub use basis::{BasisFunction, CubicSpline, RandomEffectBasis, ThinPlateSpline};
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
    #[pyo3(signature = (family=None, mgcv_exact=None, link=None, df=None, p=None, theta=None, tau=None, sigma=None, co=None, r=None))]
    fn new(
        family: Option<&str>,
        mgcv_exact: Option<bool>,
        link: Option<&str>,
        df: Option<f64>,
        p: Option<f64>,
        theta: Option<f64>,
        tau: Option<f64>,
        sigma: Option<f64>,
        co: Option<f64>,
        r: Option<u8>,
    ) -> PyResult<Self> {
        // Validate df early if provided
        if let Some(df_val) = df {
            if df_val < 2.0 {
                return Err(PyValueError::new_err(format!(
                    "t-dist df must be >= 2.0, got {}",
                    df_val
                )));
            }
            if df_val > 100.0 {
                return Err(PyValueError::new_err(format!(
                    "t-dist df must be <= 100.0, got {}. Use df ∈ [2, 100].",
                    df_val
                )));
            }
        }
        let fam = match (family, link) {
            (Some("gaussian"), None) | (None, None) => Family::Gaussian,
            (Some("gaussian"), Some("identity")) => Family::Gaussian,
            (Some("binomial"), None) | (Some("binomial"), Some("logit")) => Family::Binomial,
            (Some("poisson"), None) | (Some("poisson"), Some("log")) => Family::Poisson,
            (Some("gamma") | Some("Gamma"), None)
            | (Some("gamma") | Some("Gamma"), Some("inverse")) => Family::Gamma,
            (Some("gamma") | Some("Gamma"), Some("log")) => Family::GammaLog,
            // Quasi-Poisson: same as Poisson but with profiled dispersion φ̂.
            (Some("quasipoisson"), None) | (Some("quasipoisson"), Some("log")) => Family::QuasiPoisson,
            // Quasi-Binomial: same as Binomial but with profiled dispersion φ̂.
            (Some("quasibinomial"), None) | (Some("quasibinomial"), Some("logit")) => Family::QuasiBinomial,
            // Scaled t-distribution (mgcv's scat family). Identity link only.
            // df=None ⟹ mgcv scat-style profiling, seeded at ν=5 and driven
            // by the outer gam.fit5-style LAML path. df=Some(_) stays fixed.
            (Some("t-dist") | Some("scat"), None)
            | (Some("t-dist") | Some("scat"), Some("identity")) => {
                Family::TDist { df: df.unwrap_or(5.0), sigma2: 1.0 }
            }
            // Inverse Gaussian with log link.
            (Some("inverse.gaussian") | Some("inverse_gaussian"), Some("log") | None) => {
                Family::InverseGaussian
            }
            // Tweedie with log link (1 < p < 2).
            // p=None → profile-p mode (mgcv's tw()), starting at p=1.5.
            // p=Some(val) → fixed-p mode (mgcv's Tweedie(p=val)).
            (Some("tweedie") | Some("tw") | Some("Tweedie"), None)
            | (Some("tweedie") | Some("tw") | Some("Tweedie"), Some("log")) => {
                let tweedie_p = p.unwrap_or(1.5);
                if tweedie_p <= 1.0 || tweedie_p >= 2.0 {
                    return Err(PyValueError::new_err(format!(
                        "Tweedie p must be in (1, 2), got {}", tweedie_p
                    )));
                }
                Family::Tweedie { p: tweedie_p }
            }
            // Negative Binomial with log link.
            // theta=Some(val) → fixed-θ mode (mgcv's negbin(theta=val)).
            // family="nb" → profile-θ mode (mgcv's nb()), θ optimised jointly with λ.
            (Some("negbin") | Some("negative.binomial"), Some("log") | None) => {
                let theta_val = theta.unwrap_or(2.0);
                if theta_val <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "nb theta must be > 0, got {}", theta_val
                    )));
                }
                Family::NegBin { theta: theta_val }
            }
            (Some("nb"), Some("log") | None) => {
                // profile-θ mode; actual theta set by outer Newton on log(θ)
                Family::NegBin { theta: 2.0 }
            }
            // Quantile (qgam-style): identity link only, ELF-smoothed pinball
            // loss. Requires `tau` ∈ (0, 1); `sigma` is qgam's exp(theta)
            // scale and `co` is qgam's logistic width. Omitting `co` preserves
            // the old one-parameter λ = σ behaviour.
            (Some("quantile"), None) | (Some("quantile"), Some("identity")) => {
                let tau_val = tau.unwrap_or(0.5);
                if tau_val <= 0.0 || tau_val >= 1.0 {
                    return Err(PyValueError::new_err(format!(
                        "quantile tau must be in (0, 1), got {}", tau_val
                    )));
                }
                let sigma_val = sigma.unwrap_or(0.0); // 0.0 = auto-calibrate at fit
                if sigma_val < 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "quantile sigma must be >= 0, got {}", sigma_val
                    )));
                }
                let lambda_val = co.unwrap_or(sigma_val); // omitted co preserves old λ = σ behaviour
                if lambda_val < 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "quantile co must be >= 0, got {}",
                        lambda_val
                    )));
                }
                Family::Quantile { tau: tau_val, sigma: sigma_val, lambda: lambda_val }
            }
            // Ordered categorical (mgcv's ocat(R=K)). Identity link.
            (Some("ocat"), None) | (Some("ocat"), Some("identity")) => {
                let r_val = r.unwrap_or(0);
                if r_val < 3 {
                    return Err(PyValueError::new_err(format!(
                        "ocat requires R >= 3 categories (set via `r=...`), got R={r_val}"
                    )));
                }
                Family::Ocat { r: r_val }
            }
            (Some(f), Some(l)) => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported family/link combination: {}({}). Supported: \
                     gaussian(identity), binomial(logit), poisson(log), \
                     gamma(inverse), gamma(log), quasipoisson(log), quasibinomial(logit), \
                     t-dist(identity), scat(identity), quantile(identity), \
                     tweedie(log), inverse.gaussian(log), \
                     negbin(log), nb(log).",
                    f, l
                )))
            }
            (Some(f), None) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown family '{}'. Use 'gaussian', 'binomial', 'poisson', 'gamma' (or 'Gamma'), \
                     'quasipoisson', 'quasibinomial', 't-dist', 'scat', 'quantile', 'tweedie' (or 'Tweedie'/'tw'), 'inverse.gaussian', \
                     'negbin', 'nb', or 'negative.binomial'",
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
        // Profile-p for Tweedie: enabled when the caller passes p=None
        // (mgcv's tw() family). Fixed-p mode uses p=Some(val).
        let is_tweedie = matches!(fam, Family::Tweedie { .. });
        if is_tweedie && p.is_none() {
            g.tweedie_profile = true;
        }
        // Profile-θ for NegBin: enabled when family="nb" (mgcv's nb() extended family).
        // Fixed-θ mode uses family="negbin" with optional theta kwarg.
        if matches!(family, Some("nb")) {
            g.negbin_profile = true;
        }
        // Profile df/σ² for scat when df was not user-supplied. User-fixed
        // df keeps the historical fixed-df path.
        if matches!(fam, Family::TDist { .. }) && df.is_none() {
            g.tdist_profile = true;
        }
        Ok(PyGAM {
            inner: g,
            mgcv_exact: exact,
        })
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
    #[pyo3(signature = (x, y, k, method="REML", bs=None, max_iter=None, use_edf=None, pc_values=None, bs_list=None, weights=None, discrete=None))]
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
        pc_values: Option<Vec<Option<f64>>>,
        bs_list: Option<Vec<Option<String>>>,
        weights: Option<PyReadonlyArray1<f64>>,
        discrete: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        // Route to the optimized implementation
        self.fit_auto_optimized(
            py, x, y, k, method, bs, max_iter, use_edf, None, pc_values, bs_list, weights, discrete,
        )
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
            "fREML" => OptimizationMethod::FastREML,
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'GCV', 'REML', or 'fREML'",
                ))
            }
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
    #[pyo3(signature = (x, y, k, method, bs=None, max_iter=None, use_edf=None, algorithm=None, pc_values=None, bs_list=None, weights=None, discrete=None))]
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
        pc_values: Option<Vec<Option<f64>>>,
        bs_list: Option<Vec<Option<String>>>,
        weights: Option<PyReadonlyArray1<f64>>,
        discrete: Option<bool>,
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

        let default_basis_type = bs.unwrap_or("bs");

        // Clear any existing smooths
        self.inner.smooth_terms.clear();

        // Add smooths for each column
        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i);
            let col_owned = col.to_owned();

            // Per-term basis type overrides the global `bs` when provided.
            let term_bs: &str = bs_list
                .as_ref()
                .and_then(|v| v.get(i))
                .and_then(|opt| opt.as_deref())
                .unwrap_or(default_basis_type);

            let smooth = match term_bs {
                "re" => {
                    // Random-effect smooth: one-hot design, identity penalty, no centering.
                    SmoothTerm::random_effect(format!("x{}", i), &col_owned)
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?
                }
                "parametric" | "linear" => {
                    // Parametric (linear, unsmoothed) term: one raw column, zero
                    // penalty. The basis ignores k (always 1 column). See
                    // docs/PARAMETRIC_TERMS_DESIGN.md. ``"linear"`` is an alias
                    // for ``"parametric"`` matching mgcv-user mental model
                    // (closes 0.16 customer-feedback discoverability gap).
                    SmoothTerm::parametric(format!("x{}", i), &col_owned)
                        .map_err(|e| PyValueError::new_err(format!("{}", e)))?
                }
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
                        "Unknown basis type '{}'. Use 'bs', 'cr', 're', or \
                         'parametric' (alias: 'linear').",
                        term_bs
                    )));
                }
            };

            self.inner.add_smooth(smooth);
        }

        // Apply per-smooth pc anchoring values if provided.
        // pc_values[i] = Some(v) means smooth i should use pc-anchoring at v.
        if let Some(ref pcs) = pc_values {
            for (i, pc_opt) in pcs.iter().enumerate() {
                if let Some(pc) = pc_opt {
                    if i < self.inner.smooth_terms.len() {
                        self.inner.smooth_terms[i].pc_value = Some(*pc);
                    }
                }
            }
        }

        // Call optimized fit
        let opt_method = match method {
            "GCV" => OptimizationMethod::GCV,
            "REML" => OptimizationMethod::REML,
            "fREML" => OptimizationMethod::FastREML,
            _ => {
                return Err(PyValueError::new_err(
                    "method must be 'GCV', 'REML', or 'fREML'",
                ))
            }
        };

        let max_outer = max_iter.unwrap_or(10);

        // Plumb optional per-row prior weights through to the core via
        // `self.inner.prior_weights`. The fit path reads it as a constant
        // factor that multiplies the IRLS working weights every PIRLS
        // step (see gam_optimized::fit_optimized_full and
        // pirls::fit_pirls_cached). Setting to `None` here also clears
        // any leftover weights from a previous fit on the same instance.
        if let Some(w) = weights.as_ref() {
            let w_array = w.as_array().to_owned();
            let n_rows = x_array.nrows();
            if w_array.len() != n_rows {
                return Err(PyValueError::new_err(format!(
                    "weights length ({}) must match number of rows ({})",
                    w_array.len(),
                    n_rows
                )));
            }
            // Reject non-positive prior weights — they would zero out
            // (or negate) observations in a way mgcv would refuse too.
            if w_array.iter().any(|&v| !(v > 0.0) || !v.is_finite()) {
                return Err(PyValueError::new_err(
                    "weights must be strictly positive and finite",
                ));
            }
            self.inner.prior_weights = Some(w_array);
        } else {
            self.inner.prior_weights = None;
        }

        // Opt-in covariate-binning fast path (mgcv's `discrete=TRUE`). Reset
        // every call so a previous fit can't leak across constructions.
        self.inner.discrete_enabled = discrete.unwrap_or(false);

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

    /// Run ELF PIRLS at caller-supplied fixed smoothing parameters.
    ///
    /// Does NOT run the outer REML/FS optimization loop — sp values are held
    /// fixed. This is the step-6 contract hook: verify PIRLS convergence at
    /// R-derived (co, sigma, sp) before touching lambda optimization.
    ///
    /// Args:
    ///     x: Raw covariates (n × d).
    ///     y: Response (n).
    ///     k: Basis sizes per smooth (length d).
    ///     sp: Fixed smoothing parameters (length d).
    ///     tau: Quantile level.
    ///     sigma: ELF sigma = exp(lsig).
    ///     co: ELF logistic-width = err * sqrt(2π*varHat) / (2*log2).
    ///     bs: Basis type ("cr" default, "bs").
    ///     bs_list: Per-smooth basis overrides.
    ///     max_iter: PIRLS iterations (default 200).
    ///     tol: PIRLS convergence tolerance (default 1e-6).
    ///
    /// Returns dict with: coef, fitted_values, deviance, converged, iterations.
    fn fit_quantile_fixed_sp<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        k: Vec<usize>,
        sp: Vec<f64>,
        tau: f64,
        sigma: f64,
        co: f64,
        bs: Option<&str>,
        bs_list: Option<Vec<Option<String>>>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        use crate::gam_optimized::*;

        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();
        let (_n, d) = x_array.dim();

        if k.len() != d {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k length ({}) must match columns ({})",
                k.len(),
                d
            )));
        }
        if sp.len() != d {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sp length ({}) must match columns ({})",
                sp.len(),
                d
            )));
        }

        let default_bs = bs.unwrap_or("cr");
        self.inner.smooth_terms.clear();

        for (i, &num_basis) in k.iter().enumerate() {
            let col = x_array.column(i).to_owned();
            let term_bs: &str = bs_list
                .as_ref()
                .and_then(|v| v.get(i))
                .and_then(|opt| opt.as_deref())
                .unwrap_or(default_bs);
            let smooth = match term_bs {
                "cr" => SmoothTerm::cr_spline_quantile(format!("x{}", i), num_basis, &col)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?,
                "bs" => SmoothTerm::cubic_spline_quantile(format!("x{}", i), num_basis, &col)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "bs must be 'cr' or 'bs'",
                    ))
                }
            };
            self.inner.add_smooth(smooth);
        }

        let pirls_result = self
            .inner
            .fit_fixed_sp_quantile(
                &x_array,
                &y_array,
                &sp,
                tau,
                sigma,
                co,
                max_iter.unwrap_or(200),
                tol.unwrap_or(1e-6),
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let result = pyo3::types::PyDict::new(py);
        result.set_item(
            "coef",
            PyArray1::from_vec(py, pirls_result.coefficients.to_vec()),
        )?;
        result.set_item(
            "fitted_values",
            PyArray1::from_vec(py, pirls_result.fitted_values.to_vec()),
        )?;
        result.set_item("deviance", pirls_result.deviance)?;
        result.set_item("converged", pirls_result.converged)?;
        result.set_item("iterations", pirls_result.iterations)?;
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

    /// For the ordered-categorical (`ocat`) family: per-row n×R category
    /// probability matrix. Reads the converged θ from the fitted
    /// SmoothingParameter and applies mgcv's
    /// `P(Y=k) = F(α_{k+1} − η) − F(α_k − η)` mapping.
    fn predict_proba_ocat<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let r = match self.inner.family {
            Family::Ocat { r } => r as usize,
            _ => {
                return Err(PyValueError::new_err(
                    "predict_proba_ocat requires family='ocat'",
                ))
            }
        };
        let theta = self
            .inner
            .smoothing_params
            .as_ref()
            .map(|p| p.ocat_theta.clone())
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet (ocat θ unavailable)"))?;
        if theta.len() != r - 2 {
            return Err(PyValueError::new_err(format!(
                "ocat θ length {} doesn't match R-2 = {} — model may not be fitted",
                theta.len(),
                r - 2
            )));
        }
        let x_array = x.as_array().to_owned();
        let eta = self
            .inner
            .predict(&x_array)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let prob = crate::ocat::ocat_prob(&eta, &theta, r);
        Ok(numpy::PyArray2::from_owned_array(py, prob))
    }

    /// Get the ocat θ vector (log-gap parameters, length R-2). Errors if
    /// the family is not ocat or the model isn't fitted.
    fn get_ocat_theta(&self) -> PyResult<Vec<f64>> {
        match self.inner.family {
            Family::Ocat { .. } => self
                .inner
                .smoothing_params
                .as_ref()
                .map(|p| p.ocat_theta.clone())
                .ok_or_else(|| PyValueError::new_err("Model not fitted yet")),
            _ => Err(PyValueError::new_err(
                "get_ocat_theta requires family='ocat'",
            )),
        }
    }

    fn get_lambda(&self) -> PyResult<f64> {
        self.inner
            .smoothing_params
            .as_ref()
            .map(|p| p.lambda[0])
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))
    }

    /// Get the REML / LAML score at the converged fit. Used by wrapper-level
    /// σ profilers (e.g. for the Quantile family) to drive Brent on σ via
    /// the same likelihood criterion mgcv uses.
    fn get_reml_score(&self) -> PyResult<f64> {
        self.inner
            .reml_score
            .ok_or_else(|| PyValueError::new_err(
                "REML score not available — model not fitted, or the optimizer didn't compute it for this family/path"
            ))
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

    /// Shift the final quantile intercept so empirical training coverage matches τ.
    fn calibrate_quantile_intercept(&mut self, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let tau = match self.inner.family {
            Family::Quantile { tau, .. } => tau,
            _ => {
                return Err(PyValueError::new_err(
                    "calibrate_quantile_intercept is only valid for family='quantile'",
                ))
            }
        };

        let y_array = y.as_array();
        let fitted = self
            .inner
            .fitted_values
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        if y_array.len() != fitted.len() {
            return Err(PyValueError::new_err(format!(
                "y length ({}) must match fitted length ({})",
                y_array.len(),
                fitted.len()
            )));
        }

        let mut residuals: Vec<f64> = y_array
            .iter()
            .zip(fitted.iter())
            .map(|(&yi, &fi)| yi - fi)
            .filter(|r| r.is_finite())
            .collect();
        if residuals.is_empty() {
            return Err(PyValueError::new_err(
                "cannot calibrate quantile intercept from non-finite residuals",
            ));
        }
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((residuals.len() as f64 - 1.0) * tau)
            .floor()
            .clamp(0.0, residuals.len() as f64 - 1.0) as usize;
        let shift = residuals[idx];
        if !shift.is_finite() {
            return Err(PyValueError::new_err(
                "non-finite quantile calibration shift",
            ));
        }

        let coefficients = self
            .inner
            .coefficients
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        coefficients[0] += shift;
        fitted.mapv_inplace(|v| v + shift);
        if let Some(linear_predictor) = self.inner.linear_predictor.as_mut() {
            linear_predictor.mapv_inplace(|v| v + shift);
        }
        self.inner.deviance = Some(
            y_array
                .iter()
                .zip(fitted.iter())
                .map(|(&yi, &fi)| (yi - fi).powi(2))
                .sum(),
        );
        Ok(shift)
    }

    /// Get the family (distribution) used by this GAM
    fn get_family(&self) -> &str {
        match self.inner.family {
            Family::Gaussian => "gaussian",
            Family::Binomial => "binomial",
            Family::Poisson => "poisson",
            Family::Gamma => "gamma",
            Family::GammaLog => "gamma",
            Family::QuasiPoisson => "quasipoisson",
            Family::QuasiBinomial => "quasibinomial",
            Family::TDist { .. } => "t-dist",
            Family::Tweedie { .. } => "tweedie",
            Family::InverseGaussian => "inverse.gaussian",
            Family::NegBin { .. } => "negbin",
            Family::Quantile { .. } => "quantile",
            Family::Ocat { .. } => "ocat",
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
            Family::QuasiPoisson => "log",
            Family::QuasiBinomial => "logit",
            Family::TDist { .. } => "identity",
            Family::Tweedie { .. } => "log",
            Family::InverseGaussian => "log",
            Family::NegBin { .. } => "log",
            Family::Quantile { .. } => "identity",
            Family::Ocat { .. } => "identity",
        }
    }

    /// Get the converged family-shape parameters as a dict.
    ///
    /// For families with profile-able shape parameters (TDist, NegBin,
    /// Tweedie, Quantile), returns the *current* values stored on the
    /// family enum — which after `fit()` are the converged values. For
    /// scale-only families (Gaussian, Binomial, Poisson, etc.) returns
    /// an empty dict.
    ///
    /// Keys per family:
    ///   - TDist:    {"df", "sigma2"}
    ///   - NegBin:   {"theta"}
    ///   - Tweedie:  {"p"}
    ///   - Quantile: {"tau", "sigma"}
    ///   - others:   {} (empty)
    fn get_family_params<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = pyo3::types::PyDict::new(py);
        match self.inner.family {
            Family::TDist { df, sigma2 } => {
                dict.set_item("df", df)?;
                dict.set_item("sigma2", sigma2)?;
            }
            Family::NegBin { theta } => {
                dict.set_item("theta", theta)?;
            }
            Family::Tweedie { p } => {
                dict.set_item("p", p)?;
            }
            Family::Quantile { tau, sigma, lambda } => {
                dict.set_item("tau", tau)?;
                dict.set_item("sigma", sigma)?;
                dict.set_item("co", lambda)?;
            }
            _ => {}
        }
        Ok(dict.into())
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
    fn get_vcov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
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
        let xtwx = crate::reml::compute_xtwx_dispatch(None, design, weights);
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
                let tr_a = crate::reml::compute_xtwx_dispatch(None, design, weights)
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
            None,
            self.inner.family,
        )
        .map_err(|e| PyValueError::new_err(format!("gradient failed: {}", e)))?;
        Ok(numpy::PyArray1::from_owned_array(py, grad))
    }

    /// Like `evaluate_reml_gradient_closed_form` but with a caller-pinned σ².
    /// Useful for FD-Hessian verification: fixes σ² at the base-point value so
    /// FD and CF differentiate exactly the same function.
    #[cfg(feature = "blas")]
    fn evaluate_reml_gradient_closed_form_fixed_sigma2<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
        fixed_sigma2: f64,
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
        let grad = crate::reml::reml_gradient_mgcv_exact_closed_form_fixed_sigma2(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            None,
            self.inner.family,
            fixed_sigma2,
        )
        .map_err(|e| PyValueError::new_err(format!("gradient fixed-σ² failed: {}", e)))?;
        Ok(numpy::PyArray1::from_owned_array(py, grad))
    }

    /// Returns the plug-in σ² = RSS/(n-trA) that `evaluate_reml_gradient_closed_form`
    /// would use at these λ values. Call this once at the base λ and pass
    /// the result to `evaluate_reml_gradient_closed_form_fixed_sigma2` for FD
    /// Hessian verification.
    #[cfg(feature = "blas")]
    fn evaluate_scale_at_lambdas(
        &self,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
    ) -> PyResult<f64> {
        use crate::linalg::{inverse, solve};
        use crate::reml::{compute_xtwx_dispatch, compute_xtwy_dispatch};
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
        let n = y_array.len();
        let xtwx = compute_xtwx_dispatch(None, design_matrix, weights);
        let mut a = xtwx.clone();
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            penalty.scaled_add_to(&mut a, *lambda);
        }
        let xtwy = compute_xtwy_dispatch(None, design_matrix, weights, &y_array);
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let mut a_solve = a.clone();
        a_solve
            .diag_mut()
            .iter_mut()
            .for_each(|d| *d += 1e-12 * max_diag);
        let beta = solve(a_solve, xtwy)
            .map_err(|e| PyValueError::new_err(format!("solve failed: {}", e)))?;
        let fitted = design_matrix.dot(&beta);
        let rss: f64 = y_array
            .iter()
            .zip(fitted.iter())
            .zip(weights.iter())
            .map(|((yi, fi), wi)| (yi - fi).powi(2) * wi)
            .sum();
        let a_inv =
            inverse(&a).map_err(|e| PyValueError::new_err(format!("inverse failed: {}", e)))?;
        let tr_a = (xtwx.dot(&a_inv)).diag().sum();
        Ok(rss / ((n as f64) - tr_a).max(1e-10))
    }

    /// IFT-based gradient of the mgcv-exact REML score at the given λ.
    /// Pass `y_original=Some(y_orig)` to enable the GLM-deviance form
    /// (matches mgcv's gdi.c::ift1 + gdi1 path); pass None to fall
    /// back to working-RSS deviance (collapses to envelope at our β).
    #[cfg(feature = "blas")]
    fn evaluate_reml_gradient_ift<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
        y_original: Option<PyReadonlyArray1<f64>>,
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
        let y_orig_array = y_original.as_ref().map(|y| y.as_array().to_owned());
        let grad = crate::reml::reml_gradient_mgcv_exact_ift(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            self.inner.family,
            y_orig_array.as_ref(),
            self.inner.prior_weights.as_ref(),
        )
        .map_err(|e| PyValueError::new_err(format!("ift gradient failed: {}", e)))?;
        Ok(numpy::PyArray1::from_owned_array(py, grad))
    }

    /// IFT-based Hessian of the mgcv-exact REML score at the given λ.
    /// At converged β with working-RSS deviance equals the envelope Hessian.
    #[cfg(feature = "blas")]
    fn evaluate_reml_hessian_ift<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
        y_original: Option<PyReadonlyArray1<f64>>,
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
        let y_orig_array = y_original.as_ref().map(|y| y.as_array().to_owned());
        let hess = crate::reml::reml_hessian_mgcv_exact_ift(
            &y_array,
            design_matrix,
            weights,
            &lambdas,
            &penalties,
            None,
            self.inner.family,
            y_orig_array.as_ref(),
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("ift hessian failed: {}", e)))?;
        Ok(numpy::PyArray2::from_owned_array(py, hess))
    }

    /// Re-runs PIRLS at the given λ (using raw `y` and the fitter's cached
    /// design matrix), then evaluates the mgcv-exact REML score, IFT
    /// gradient, and IFT Hessian at the freshly-converged (β, W, z).
    ///
    /// Diagnostic-only — exposes what the outer Newton would see at λ. The
    /// regular `evaluate_reml_*` entry points reuse stale weights from the
    /// last `fit()` call which is fine for Gaussian (W=I) but produces a
    /// β that is NOT the PIRLS solution at the new λ for GLM families.
    #[cfg(feature = "blas")]
    fn evaluate_reml_at_sp_freshly_fit<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        lambdas: Vec<f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let design_matrix = self
            .inner
            .design_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted yet"))?;
        let total_cols = design_matrix.ncols();
        let mut penalties: Vec<crate::block_penalty::BlockPenalty> = Vec::new();
        let mut col_offset = 1usize;
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
        let y_raw = y.as_array().to_owned();

        let pirls = crate::pirls::fit_pirls(
            &y_raw,
            design_matrix,
            &lambdas,
            &penalties,
            self.inner.family,
            100,
            1e-8,
        )
        .map_err(|e| PyValueError::new_err(format!("fit_pirls failed: {}", e)))?;

        let w_fresh = &pirls.weights;
        let z_fresh = &pirls.working_response;

        let reml = crate::reml::reml_criterion_multi_cached_mgcv_exact(
            z_fresh,
            design_matrix,
            w_fresh,
            &lambdas,
            &penalties,
            None,
            None,
            mp,
            self.inner.family,
            Some(&y_raw),
            self.inner.prior_weights.as_ref(),
        )
        .map_err(|e| PyValueError::new_err(format!("reml score failed: {}", e)))?;

        // For non-canonical-link dispersion-bearing GLMs (e.g. InvGauss+log),
        // mgcv's gdi2 evaluates the IFT gradient with the raw Newton W and the
        // PIRLS-converged β; PIRLS's stored weights are Fisher-fallback so the
        // generic `reml_gradient_mgcv_exact_ift` (which re-solves β from input
        // W) gives the wrong gradient. The Newton-at-β path matches mgcv.
        let use_newton_at_beta = !self.inner.family.is_canonical_link()
            && !matches!(self.inner.family, crate::pirls::Family::Gaussian);
        let grad = if use_newton_at_beta {
            crate::reml::reml_gradient_mgcv_exact_ift_newton_at_beta(
                design_matrix,
                &y_raw,
                &pirls.coefficients,
                &lambdas,
                &penalties,
                self.inner.family,
                mp,
            )
            .map_err(|e| {
                PyValueError::new_err(format!("ift gradient (Newton-at-β) failed: {}", e))
            })?
        } else {
            crate::reml::reml_gradient_mgcv_exact_ift(
                z_fresh,
                design_matrix,
                w_fresh,
                &lambdas,
                &penalties,
                None,
                self.inner.family,
                Some(&y_raw),
                self.inner.prior_weights.as_ref(),
            )
            .map_err(|e| PyValueError::new_err(format!("ift gradient failed: {}", e)))?
        };

        let hess = crate::reml::reml_hessian_mgcv_exact_ift(
            z_fresh,
            design_matrix,
            w_fresh,
            &lambdas,
            &penalties,
            None,
            self.inner.family,
            Some(&y_raw),
            None,
        )
        .map_err(|e| PyValueError::new_err(format!("ift hessian failed: {}", e)))?;

        let result = pyo3::types::PyDict::new(py);
        result.set_item("reml", reml)?;
        result.set_item("grad", numpy::PyArray1::from_owned_array(py, grad))?;
        result.set_item("hess", numpy::PyArray2::from_owned_array(py, hess))?;
        result.set_item(
            "beta",
            numpy::PyArray1::from_owned_array(py, pirls.coefficients),
        )?;
        result.set_item(
            "weights",
            numpy::PyArray1::from_owned_array(py, pirls.weights),
        )?;
        result.set_item(
            "working_response",
            numpy::PyArray1::from_owned_array(py, pirls.working_response),
        )?;
        result.set_item("deviance", pirls.deviance)?;
        result.set_item("iterations", pirls.iterations)?;
        result.set_item("converged", pirls.converged)?;
        Ok(result)
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
            out.push(numpy::PyArray2::from_owned_array(
                py,
                smooth.penalty.clone(),
            ));
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
            let sf = if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0
            };
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
            None,
            mp,
            self.inner.family,
            None,
            self.inner.prior_weights.as_ref(),
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
            let scale_factor = if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0
            };
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

    /// Per-smooth EDF (effective degrees of freedom), matching
    /// `summary.gam()$edf` in mgcv. Intercept is excluded.
    ///
    /// EDF_j = Σ_{i ∈ [first_j, last_j]} (A⁻¹ X'WX)[i, i]
    /// where A = X'WX + Σ λ_k S_k.
    #[cfg(feature = "blas")]
    fn get_edf_per_smooth(&self) -> PyResult<Vec<(String, f64)>> {
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

        let xtwx = crate::reml::compute_xtwx_dispatch(None, design, weights);
        let mut a = xtwx.clone();
        for (lam, pen) in lambdas.iter().zip(penalties.iter()) {
            pen.scaled_add_to(&mut a, *lam);
        }
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let ridge = 1e-12 * max_diag;
        for i in 0..a.nrows() {
            a[[i, i]] += ridge;
        }
        let a_inv = crate::linalg::inverse(&a).map_err(|e| {
            PyValueError::new_err(format!("get_edf_per_smooth inverse failed: {}", e))
        })?;

        // diag(A⁻¹ · X'WX): element [i,i] = row i of a_inv dotted with col i of xtwx
        let a_inv_xtwx_diag: Vec<f64> = (0..total_cols)
            .map(|i| {
                let row = a_inv.row(i);
                let col = xtwx.column(i);
                row.iter().zip(col.iter()).map(|(a, b)| a * b).sum()
            })
            .collect();

        let mut out = Vec::with_capacity(self.inner.smooth_terms.len());
        let mut first = 1usize;
        for smooth in &self.inner.smooth_terms {
            let nb = smooth.num_basis();
            let last = first + nb - 1; // inclusive
            let edf: f64 = a_inv_xtwx_diag[first..=last].iter().sum();
            out.push((smooth.name.clone(), edf));
            first += nb;
        }
        Ok(out)
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

/// Per-obs-σ ELF quantile fit — location-only stage of qgam's
/// (post-1.3) gaulss-then-ELF pipeline for heteroskedastic τ-quantile
/// regression.
///
/// Takes σ_G(x) — the Gaussian conditional SD at each training row,
/// from Python-side gaulss preprocessing. Computes the ELF per-obs σ,
/// the qgam-heuristic σ_global (when `sigma_global` is None), and runs
/// the IRLS internally — Python is just orchestration.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (x_loc, y, s_loc_total, sigma_g_per_obs, tau, sigma_global=None, max_iter=None, tolerance=None))]
fn fit_quantile_lss_raw_py<'py>(
    py: Python<'py>,
    x_loc: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    s_loc_total: PyReadonlyArray2<f64>,
    sigma_g_per_obs: PyReadonlyArray1<f64>,
    tau: f64,
    sigma_global: Option<f64>,
    max_iter: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let x_loc_arr = x_loc.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let s_loc_arr = s_loc_total.as_array().to_owned();
    let sigma_g_arr = sigma_g_per_obs.as_array().to_owned();

    let (res, sigma_global_used) = crate::pirls::fit_pirls_quantile_lss(
        &y_arr,
        &x_loc_arr,
        &s_loc_arr,
        &sigma_g_arr,
        sigma_global,
        tau,
        max_iter.unwrap_or(50),
        tolerance.unwrap_or(1e-6),
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let result = pyo3::types::PyDict::new(py);
    result.set_item(
        "beta_loc",
        PyArray1::from_vec(py, res.coefficients_loc.to_vec()),
    )?;
    result.set_item("eta_loc", PyArray1::from_vec(py, res.eta_loc.to_vec()))?;
    result.set_item("sigma", PyArray1::from_vec(py, res.sigma.to_vec()))?;
    result.set_item("sigma_global", sigma_global_used)?;
    result.set_item("deviance", res.deviance)?;
    result.set_item("iterations", res.iterations)?;
    result.set_item("converged", res.converged)?;
    Ok(result.into())
}

/// Re-tune per-location-smooth λ under the per-obs-σ ELF likelihood via
/// a Fellner-Schall outer loop. `penalty_blocks` is a list of full-design
/// (p × p) penalty matrices, one per location smooth, each with non-zeros
/// only in the smooth's coefficient block. `lambda_init` is the matching
/// per-smooth initial λ vector (typically the lambdas from the location
/// Gaussian GAM).
///
/// Returns a dict matching `fit_quantile_lss_raw_py` plus `lambda_loc`
/// (final tuned λs) and `fs_iterations` (FS sweep count).
#[pyfunction]
#[pyo3(signature = (x_loc, y, penalty_blocks, lambda_init, sigma_g_per_obs, tau, sigma_global=None, max_outer=None, max_inner=None, tolerance=None))]
fn fit_quantile_lss_retune_py<'py>(
    py: Python<'py>,
    x_loc: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    penalty_blocks: Vec<PyReadonlyArray2<f64>>,
    lambda_init: PyReadonlyArray1<f64>,
    sigma_g_per_obs: PyReadonlyArray1<f64>,
    tau: f64,
    sigma_global: Option<f64>,
    max_outer: Option<usize>,
    max_inner: Option<usize>,
    tolerance: Option<f64>,
) -> PyResult<Py<PyAny>> {
    use crate::block_penalty::BlockPenalty;
    let x_loc_arr = x_loc.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let sigma_g_arr = sigma_g_per_obs.as_array().to_owned();
    let lambda_init_vec: Vec<f64> = lambda_init.as_array().iter().copied().collect();
    let p_loc = x_loc_arr.ncols();

    if penalty_blocks.len() != lambda_init_vec.len() {
        return Err(PyValueError::new_err(format!(
            "penalty_blocks has {} entries but lambda_init has {}",
            penalty_blocks.len(),
            lambda_init_vec.len()
        )));
    }

    // Each input penalty is the full p×p matrix with only one smooth's
    // block populated. Collapse it to BlockPenalty by finding the
    // diagonal extent of nonzeros — Python builds these via
    // _build_per_smooth_blocks where the offset is known, but we don't
    // need to trust it: scan for the first/last nonzero row/col.
    let mut block_penalties = Vec::with_capacity(penalty_blocks.len());
    for pen_full in &penalty_blocks {
        let arr = pen_full.as_array();
        if arr.nrows() != p_loc || arr.ncols() != p_loc {
            return Err(PyValueError::new_err(format!(
                "penalty block has shape ({}, {}); expected ({}, {})",
                arr.nrows(),
                arr.ncols(),
                p_loc,
                p_loc
            )));
        }
        // Find the bounding box of nonzeros.
        let mut lo = p_loc;
        let mut hi = 0usize;
        for i in 0..p_loc {
            for j in 0..p_loc {
                if arr[[i, j]].abs() > 0.0 {
                    if i < lo {
                        lo = i;
                    }
                    if i > hi {
                        hi = i;
                    }
                    if j < lo {
                        lo = j;
                    }
                    if j > hi {
                        hi = j;
                    }
                }
            }
        }
        if lo > hi {
            // All-zero penalty (e.g. for a smooth with no penalty). Use a
            // 1×1 zero block at offset 0 so trace_product/quadratic_form
            // contribute nothing.
            block_penalties.push(BlockPenalty::new(
                ndarray::Array2::<f64>::zeros((1, 1)),
                0,
                p_loc,
            ));
        } else {
            let k = hi - lo + 1;
            let block = arr.slice(ndarray::s![lo..=hi, lo..=hi]).to_owned();
            block_penalties.push(BlockPenalty::new(block, lo, p_loc));
        }
    }

    let (res, sigma_global_used) = crate::pirls::fit_pirls_quantile_lss_fs_tune(
        &y_arr,
        &x_loc_arr,
        &block_penalties,
        &lambda_init_vec,
        &sigma_g_arr,
        sigma_global,
        tau,
        max_outer.unwrap_or(20),
        max_inner.unwrap_or(50),
        tolerance.unwrap_or(1e-6),
    )
    .map_err(|e| PyValueError::new_err(format!("{}", e)))?;

    let result = pyo3::types::PyDict::new(py);
    result.set_item(
        "beta_loc",
        PyArray1::from_vec(py, res.coefficients_loc.to_vec()),
    )?;
    result.set_item("eta_loc", PyArray1::from_vec(py, res.eta_loc.to_vec()))?;
    result.set_item("sigma", PyArray1::from_vec(py, res.sigma.to_vec()))?;
    result.set_item("sigma_global", sigma_global_used)?;
    result.set_item("deviance", res.deviance)?;
    result.set_item("iterations", res.iterations)?;
    result.set_item("converged", res.converged)?;
    result.set_item("lambda_loc", PyArray1::from_vec(py, res.lambda_loc))?;
    result.set_item("fs_iterations", res.fs_iterations)?;
    Ok(result.into())
}

/// Diagnostic hook for one Fellner-Schall update. This intentionally does not
/// fit a model; callers pass the fixed β, A⁻¹, λ, ranks, and per-smooth
/// penalties so Rust's update terms can be compared directly with R/qgam.
#[cfg(feature = "python")]
#[pyfunction(name = "fellner_schall_step_terms")]
#[pyo3(signature = (penalty_blocks, penalty_ranks, lambdas, a_inv, beta, phi=1.0, log_step_clamp=3.0, lambda_min=1e-9, lambda_max=1e7))]
fn fellner_schall_step_terms_py<'py>(
    py: Python<'py>,
    penalty_blocks: Vec<PyReadonlyArray2<f64>>,
    penalty_ranks: PyReadonlyArray1<f64>,
    lambdas: PyReadonlyArray1<f64>,
    a_inv: PyReadonlyArray2<f64>,
    beta: PyReadonlyArray1<f64>,
    phi: f64,
    log_step_clamp: f64,
    lambda_min: f64,
    lambda_max: f64,
) -> PyResult<Py<PyAny>> {
    use crate::block_penalty::BlockPenalty;
    use pyo3::types::PyList;

    let a_inv_arr = a_inv.as_array().to_owned();
    let beta_arr = beta.as_array().to_owned();
    let penalty_ranks_vec: Vec<f64> = penalty_ranks.as_array().iter().copied().collect();
    let lambdas_vec: Vec<f64> = lambdas.as_array().iter().copied().collect();
    let p = beta_arr.len();

    if a_inv_arr.nrows() != p || a_inv_arr.ncols() != p {
        return Err(PyValueError::new_err(format!(
            "a_inv has shape ({}, {}); expected ({}, {})",
            a_inv_arr.nrows(),
            a_inv_arr.ncols(),
            p,
            p
        )));
    }
    if penalty_blocks.len() != penalty_ranks_vec.len() || penalty_blocks.len() != lambdas_vec.len()
    {
        return Err(PyValueError::new_err(format!(
            "penalty_blocks ({}) must match penalty_ranks ({}) and lambdas ({})",
            penalty_blocks.len(),
            penalty_ranks_vec.len(),
            lambdas_vec.len()
        )));
    }

    let mut block_penalties = Vec::with_capacity(penalty_blocks.len());
    for pen_full in &penalty_blocks {
        let arr = pen_full.as_array();
        if arr.nrows() != p || arr.ncols() != p {
            return Err(PyValueError::new_err(format!(
                "penalty block has shape ({}, {}); expected ({}, {})",
                arr.nrows(),
                arr.ncols(),
                p,
                p
            )));
        }
        let mut lo = p;
        let mut hi = 0usize;
        for i in 0..p {
            for j in 0..p {
                if arr[[i, j]].abs() > 0.0 {
                    lo = lo.min(i).min(j);
                    hi = hi.max(i).max(j);
                }
            }
        }
        if lo > hi {
            block_penalties.push(BlockPenalty::new(
                ndarray::Array2::<f64>::zeros((1, 1)),
                0,
                p,
            ));
        } else {
            let block = arr.slice(ndarray::s![lo..=hi, lo..=hi]).to_owned();
            block_penalties.push(BlockPenalty::new(block, lo, p));
        }
    }

    let terms = crate::smooth::fellner_schall_step_terms(
        &block_penalties,
        &penalty_ranks_vec,
        &lambdas_vec,
        &a_inv_arr,
        &beta_arr,
        phi,
        log_step_clamp,
        (lambda_min, lambda_max),
    );

    let out = PyList::empty(py);
    for term in terms {
        let item = pyo3::types::PyDict::new(py);
        item.set_item("lambda_old", term.lambda_old)?;
        item.set_item("lambda_new", term.lambda_new)?;
        item.set_item("rank", term.rank)?;
        item.set_item("trace_a_inv_s", term.trace_a_inv_s)?;
        item.set_item("beta_s_beta", term.beta_s_beta)?;
        item.set_item("numerator_raw", term.numerator_raw)?;
        item.set_item("numerator", term.numerator)?;
        item.set_item("log_ratio_raw", term.log_ratio_raw)?;
        item.set_item("log_ratio", term.log_ratio)?;
        out.append(item)?;
    }
    Ok(out.into())
}

#[cfg(feature = "python")]
/// Validation hook for `gam_reparam_core` — direct port of mgcv's
/// `get_stableS` (gdi.c:550-792). Inputs match the R wrapper at
/// `gam.fit3.r:9-63`. Used by
/// `scripts/python/diagnostics/get_stableS_oracle.py` to diff Rust
/// against `mgcv:::gam.reparam` byte-for-byte.
#[cfg(feature = "blas")]
#[pyfunction]
#[pyo3(signature = (rs_flat, rs_ncol, q, log_sp, deriv, fixed_penalty=false))]
fn gam_reparam_core_py<'py>(
    py: Python<'py>,
    rs_flat: PyReadonlyArray1<f64>,
    rs_ncol: Vec<usize>,
    q: usize,
    log_sp: PyReadonlyArray1<f64>,
    deriv: u8,
    fixed_penalty: bool,
) -> PyResult<Py<PyAny>> {
    use ndarray::Array2;
    use numpy::{PyArray1, PyArray2};
    use pyo3::types::PyDict;

    // Unpack the flat (column-major-packed) buffer into per-component rs.
    let flat = rs_flat.as_array();
    let mut rs: Vec<Array2<f64>> = Vec::with_capacity(rs_ncol.len());
    let mut offset = 0usize;
    for &nc in &rs_ncol {
        let mut block = Array2::<f64>::zeros((q, nc));
        for j in 0..nc {
            for i in 0..q {
                block[[i, j]] = flat[offset + i + q * j];
            }
        }
        offset += q * nc;
        rs.push(block);
    }
    let sp: Vec<f64> = log_sp.as_array().iter().map(|x| x.exp()).collect();
    let (d_tol, r_tol) = reparam::default_tolerances();

    let result = reparam::gam_reparam_core(&rs, &sp, deriv, d_tol, r_tol, fixed_penalty)
        .map_err(|e| PyValueError::new_err(format!("gam_reparam_core failed: {}", e)))?;

    let out = PyDict::new(py);
    out.set_item("S", PyArray2::from_owned_array(py, result.s))?;
    out.set_item("Qs", PyArray2::from_owned_array(py, result.qs))?;
    let rs_py: Vec<_> = result
        .rs
        .into_iter()
        .map(|r| PyArray2::from_owned_array(py, r))
        .collect();
    out.set_item("rs", rs_py)?;
    out.set_item("det", result.det)?;
    if let Some(d1) = result.det1 {
        out.set_item("det1", PyArray1::from_owned_array(py, d1))?;
    } else {
        out.set_item("det1", py.None())?;
    }
    if let Some(d2) = result.det2 {
        out.set_item("det2", PyArray2::from_owned_array(py, d2))?;
    } else {
        out.set_item("det2", py.None())?;
    }
    Ok(out.into())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, mu, tau, sigma, co))]
fn quantile_elf_parts_py<'py>(
    py: Python<'py>,
    y: f64,
    mu: f64,
    tau: f64,
    sigma: f64,
    co: f64,
) -> PyResult<Py<PyAny>> {
    let parts = crate::pirls::quantile_elf_parts(y, mu, tau, sigma, co);
    let out = pyo3::types::PyDict::new(py);
    out.set_item("deviance", parts.deviance)?;
    out.set_item("Dmu", 2.0 * parts.dmu)?;
    out.set_item("Dmu2", 2.0 * parts.dmu2)?;
    out.set_item("EDmu2", 2.0 * parts.dmu2)?;
    out.set_item("Dth", parts.dth)?;
    out.set_item("Dmuth", parts.dmuth)?;
    out.set_item("Dmu3", parts.dmu3)?;
    out.set_item("Dmu2th", parts.dmu2th)?;
    out.set_item("Dmu4", parts.dmu4)?;
    out.set_item("Dth2", parts.dth2)?;
    out.set_item("Dmuth2", parts.dmuth2)?;
    out.set_item("Dmu2th2", parts.dmu2th2)?;
    out.set_item("Dmu3th", parts.dmu3th)?;
    out.set_item("nll_dmu", parts.dmu)?;
    out.set_item("nll_dmu2", parts.dmu2)?;
    out.set_item("w", parts.dmu2)?;
    out.set_item("g", -parts.dmu)?;
    out.set_item("z", mu - parts.dmu / parts.dmu2)?;
    out.set_item("rhs_value", parts.dmu2 * mu - parts.dmu)?;
    out.set_item("sigmoid", parts.sigmoid)?;
    Ok(out.into())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (tau, sigma, co, n=1))]
fn quantile_elf_saturated_loglik_py(tau: f64, sigma: f64, co: f64, n: usize) -> PyResult<f64> {
    Ok((n as f64) * crate::pirls::quantile_elf_saturated_loglik_per_obs(tau, sigma, co))
}

#[pymodule]
fn mgcv_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGAM>()?;
    m.add_function(wrap_pyfunction!(compute_penalty_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(reml_gradient_multi_qr_py, m)?)?;
    m.add_function(wrap_pyfunction!(reml_hessian_multi_qr_py, m)?)?;
    #[cfg(feature = "blas")]
    m.add_function(wrap_pyfunction!(newton_pirls_py, m)?)?;
    #[cfg(feature = "blas")]
    m.add_function(wrap_pyfunction!(gam_reparam_core_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_quantile_lss_raw_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_quantile_lss_retune_py, m)?)?;
    m.add_function(wrap_pyfunction!(fellner_schall_step_terms_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_elf_parts_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_elf_saturated_loglik_py, m)?)?;
    Ok(())
}
