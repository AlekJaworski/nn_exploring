"""mgcv_rust — Rust implementation of mgcv-style Generalized Additive Models.

The package exposes two layers:

1. **High-level ergonomics wrapper** (`mgcv_rust.Gam`): the user-facing class.
   Pandas/Polars DataFrame inputs, named predictors, per-term k mapping,
   family/link selection, posterior sampling, confidence intervals,
   marginal predictions (subset views via ``gam[name]``), serialization.
   See `_fitter.py`.

   ``GAMFitter`` is a deprecated alias of ``Gam`` (same constructor, same
   methods) and emits :class:`DeprecationWarning` on instantiation. It will
   be removed in a future release; migrate to :class:`Gam`.

2. **Low-level Rust core** (`mgcv_rust.GAM`): direct Python bindings to the
   Rust GAM solver. Numerics-focused, byte-for-byte mgcv parity on Gaussian
   cases. Drop to this only when you need direct control over the fit. See
   `[[mgcv_rust - Parity Tests]]`.

For most users, prefer :class:`Gam`.
"""

from .mgcv_rust import *  # compiled Rust extension — exposes GAM and helpers
from .mgcv_rust import GAM as _NativeGAM
from ._fitter import Gam, GAMFitter, TermContributions
from ._quantile import tune_quantile_sigma, fit_quantile, fit_quantile_lss, QuantileLSSFit

# Keep the native GAM accessible as `GAM` so existing scripts that import
# `mgcv_rust.GAM` continue to work unchanged.
GAM = _NativeGAM

__all__ = [
    "Gam", "GAMFitter", "GAM", "TermContributions",
    "tune_quantile_sigma", "fit_quantile",
    "fit_quantile_lss", "QuantileLSSFit",
]
