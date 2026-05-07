"""mgcv_rust — Rust implementation of mgcv-style Generalized Additive Models.

The package exposes two layers:

1. **Low-level Rust core** (`mgcv_rust.GAM`): direct Python bindings to the
   Rust GAM solver. Numerics-focused, byte-for-byte mgcv parity on Gaussian
   cases. See [[mgcv_rust - Parity Tests]].

2. **High-level ergonomics wrapper** (`mgcv_rust.GAMFitter`): drop-in
   replacement for the rpy2-based `r_fitting.GamFitter` used in the
   neighbourhoods repo. Pandas/Polars DataFrame inputs, named predictors,
   per-term k mapping, family/link selection, posterior sampling,
   confidence intervals, marginal predictions, serialization. See
   `_fitter.py`.

For most users, prefer `GAMFitter`. Drop to the lower-level `GAM` only when
you need direct control over the fit.
"""

from .mgcv_rust import *  # compiled Rust extension — exposes GAM and helpers
from .mgcv_rust import GAM as _NativeGAM
from ._fitter import GAMFitter
from ._quantile import tune_quantile_sigma, fit_quantile, fit_quantile_lss, QuantileLSSFit

# Keep the native GAM accessible as `GAM` so existing scripts that import
# `mgcv_rust.GAM` continue to work unchanged.
GAM = _NativeGAM

__all__ = [
    "GAM", "GAMFitter",
    "tune_quantile_sigma", "fit_quantile",
    "fit_quantile_lss", "QuantileLSSFit",
]
