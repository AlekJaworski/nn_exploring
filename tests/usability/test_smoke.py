"""End-to-end usability smoke for the installed wheel.

Goal: catch "wheel imports but is silently broken" — wrong BLAS, missing
runtime symbol, abi3 ABI drift, miscompiled OpenBLAS giving NaN, etc. Runs
the same on linux/mac/windows.
"""

import numpy as np
import pytest

import mgcv_rust
from mgcv_rust import GAM, GAMFitter


def _finite_array(a, expected_len):
    assert isinstance(a, np.ndarray), f"expected ndarray, got {type(a).__name__}"
    assert a.shape == (expected_len,), f"shape {a.shape} != ({expected_len},)"
    assert np.all(np.isfinite(a)), "predictions contain NaN/Inf"


# ---------------------------------------------------------------------------
# Layer 1: low-level `GAM` — direct PyO3 bindings
# ---------------------------------------------------------------------------

def test_import_surface():
    """Public symbols are present after wheel install."""
    assert hasattr(mgcv_rust, "GAM")
    assert hasattr(mgcv_rust, "GAMFitter")


def test_gam_low_level_gaussian(smooth_1d):
    x, y = smooth_1d
    gam = GAM()
    result = gam.fit_auto(x, y, k=[10], method="REML", bs="cr", max_iter=50)
    assert "lambda" in result
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))
    # Track signal at all — fitted should beat the null model handily.
    null_rss = float(np.sum((y - y.mean()) ** 2))
    fit_rss = float(np.sum((y - y_pred) ** 2))
    assert fit_rss < 0.5 * null_rss, f"fit_rss {fit_rss:.3f} >= 0.5*null {null_rss:.3f}"


def test_gam_low_level_2d(smooth_2d):
    x, y = smooth_2d
    gam = GAM()
    gam.fit_auto(x, y, k=[8, 8], method="REML", bs="cr", max_iter=50)
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))


def test_gam_determinism(smooth_1d):
    """Same input, same output — catches uninitialized memory regressions."""
    x, y = smooth_1d
    preds = []
    for _ in range(2):
        gam = GAM()
        gam.fit_auto(x, y, k=[10], method="REML", bs="cr", max_iter=50)
        preds.append(gam.predict(x))
    np.testing.assert_array_equal(preds[0], preds[1])


# ---------------------------------------------------------------------------
# Layer 2: high-level `GAMFitter` across GLM families
# ---------------------------------------------------------------------------

def test_fitter_gaussian(smooth_1d):
    x, y = smooth_1d
    gam = GAMFitter(family="gaussian").fit(x, y)
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))


def test_fitter_binomial(binomial_data):
    x, y = binomial_data
    gam = GAMFitter(family="binomial", link="logit").fit(x, y)
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))
    # Binomial predictions on response scale ∈ [0, 1].
    assert (y_pred.min() >= -1e-9) and (y_pred.max() <= 1.0 + 1e-9), (
        f"binomial preds out of [0,1]: [{y_pred.min()}, {y_pred.max()}]"
    )


def test_fitter_poisson(poisson_data):
    x, y = poisson_data
    gam = GAMFitter(family="poisson", link="log").fit(x, y)
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))
    assert y_pred.min() > 0, "Poisson predictions on response scale must be > 0"


def test_fitter_gamma(gamma_data):
    x, y = gamma_data
    gam = GAMFitter(family="gamma", link="log").fit(x, y)
    y_pred = gam.predict(x)
    _finite_array(y_pred, len(y))
    assert y_pred.min() > 0, "Gamma predictions on response scale must be > 0"


# ---------------------------------------------------------------------------
# Sanity: predict on held-out rows shaped correctly
# ---------------------------------------------------------------------------

def test_predict_held_out_shape(smooth_1d):
    x, y = smooth_1d
    gam = GAMFitter(family="gaussian").fit(x, y)
    x_new = np.linspace(0.05, 0.95, 50).reshape(-1, 1)
    y_new = gam.predict(x_new)
    _finite_array(y_new, 50)
