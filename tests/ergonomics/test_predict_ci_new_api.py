"""Ergo-N6: `predict_ci` new (mean, lo, hi) API + scale + deprecated alpha.

Covers:
- New default returns ``(mean, lo, hi)`` with response-scale mean
  matching ``predict(X)`` exactly (no manual "+= intercept" needed).
- ``level=`` controls the two-sided interval.
- ``scale='link'`` skips the inverse link; ``scale='response'`` applies
  it; the two are consistent (response = inv_link(link)).
- ``scale='deviation'`` is subset-only and centered on training data.
- Subset views inherit; ``mean`` matches the subset's ``predict()``.
- Old ``alpha=`` form returns the 2-tuple ``(lo, hi)`` and emits
  ``DeprecationWarning``.
- Invalid ``level`` / ``scale`` raise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(11)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(0, 5, n)
    y = np.sin(x0) + 0.3 * (x1 - 2.5) ** 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x0": x0, "x1": x1}), y


@pytest.fixture(scope="module")
def gaussian_gam(gaussian_data):
    X, y = gaussian_data
    return Gam(family="gaussian").fit(X, y)


@pytest.fixture(scope="module")
def binomial_data():
    rng = np.random.default_rng(13)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-1, 1, n)
    eta = 0.8 * x0 - 1.2 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return pd.DataFrame({"x0": x0, "x1": x1}), y


@pytest.fixture(scope="module")
def binomial_gam(binomial_data):
    X, y = binomial_data
    return Gam(family="binomial").fit(X, y)


# ---------------------------------------------------------------------- #
# New API: returns (mean, lo, hi)                                        #
# ---------------------------------------------------------------------- #


def test_new_api_returns_three_arrays(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    out = gaussian_gam.predict_ci(X[:50], n_samples=300)
    assert len(out) == 3
    mean, lo, hi = out
    assert mean.shape == (50,)
    assert lo.shape == (50,)
    assert hi.shape == (50,)
    assert np.all(hi >= lo)


def test_new_api_mean_matches_predict(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    mean, _, _ = gaussian_gam.predict_ci(X[:50], n_samples=300)
    np.testing.assert_allclose(mean, gaussian_gam.predict(X[:50]), rtol=1e-12)


def test_new_api_mean_matches_predict_binomial(binomial_gam, binomial_data):
    """Regression: matches predict(X) WITHOUT a manual intercept correction.
    This is the structural fix for the TrueTracts c75a517 bug class."""
    X, _ = binomial_data
    mean, lo, hi = binomial_gam.predict_ci(X[:50], n_samples=300)
    np.testing.assert_allclose(mean, binomial_gam.predict(X[:50]), rtol=1e-10)
    # CIs must straddle the mean for a well-formed two-sided interval.
    inside = np.sum((lo <= mean) & (mean <= hi))
    assert inside >= 45, f"mean inside CI for {inside}/50 rows"


def test_level_controls_interval_width(gaussian_gam, gaussian_data):
    """Wider level → wider interval (monotonic in level)."""
    X, _ = gaussian_data
    _, lo_50, hi_50 = gaussian_gam.predict_ci(X[:30], level=0.5, n_samples=500)
    _, lo_95, hi_95 = gaussian_gam.predict_ci(X[:30], level=0.95, n_samples=500)
    width_50 = (hi_50 - lo_50).mean()
    width_95 = (hi_95 - lo_95).mean()
    assert width_95 > width_50, f"width(95%)={width_95:.3e} not > width(50%)={width_50:.3e}"


# ---------------------------------------------------------------------- #
# scale dispatch                                                         #
# ---------------------------------------------------------------------- #


def test_scale_link_no_inv_link(binomial_gam, binomial_data):
    X, _ = binomial_data
    mean_link, lo_link, hi_link = binomial_gam.predict_ci(
        X[:30], scale="link", n_samples=300, seed=7
    )
    mean_resp, lo_resp, hi_resp = binomial_gam.predict_ci(
        X[:30], scale="response", n_samples=300, seed=7
    )
    # Same seed → same eta samples → response endpoints are sigmoid of link endpoints.
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-lo_link)), lo_resp, rtol=1e-10
    )
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-hi_link)), hi_resp, rtol=1e-10
    )
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-mean_link)), mean_resp, rtol=1e-10
    )


def test_invalid_scale_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="scale must be"):
        gaussian_gam.predict_ci(X[:10], scale="bogus")


def test_invalid_level_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="level must be"):
        gaussian_gam.predict_ci(X[:10], level=1.5)
    with pytest.raises(ValueError, match="level must be"):
        gaussian_gam.predict_ci(X[:10], level=0.0)


# ---------------------------------------------------------------------- #
# Subset interactions                                                    #
# ---------------------------------------------------------------------- #


def test_subset_predict_ci_mean_matches_subset_predict(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    sub = gaussian_gam[["x0"]]
    mean, _, _ = sub.predict_ci(X[:50], n_samples=300)
    np.testing.assert_allclose(mean, sub.predict(X[:50]), rtol=1e-10)


def test_deviation_scale_only_on_full_view_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="subset views"):
        gaussian_gam.predict_ci(X[:10], scale="deviation")


def test_subset_deviation_centered_on_train(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    mean, lo, hi = gaussian_gam[["x0"]].predict_ci(
        X, scale="deviation", n_samples=500, seed=21
    )
    assert abs(mean.mean()) < 1e-8
    # lo/hi straddle the centered mean
    assert np.all(lo <= mean)
    assert np.all(mean <= hi)


# ---------------------------------------------------------------------- #
# Deprecated alpha= form                                                 #
# ---------------------------------------------------------------------- #


def test_deprecated_alpha_returns_two_tuple_and_warns(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        out = gaussian_gam.predict_ci(X[:50], alpha=0.05, n_samples=300)
    assert len(out) == 2, "old API must return (lo, hi)"
    assert any(
        issubclass(w.category, DeprecationWarning) and "alpha" in str(w.message)
        for w in caught
    ), "expected DeprecationWarning mentioning alpha"


def test_deprecated_alpha_matches_new_lo_hi(gaussian_gam, gaussian_data):
    """The deprecated 2-tuple must equal the new 3-tuple's (lo, hi) for
    the same alpha=1-level, same seed."""
    X, _ = gaussian_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lo_old, hi_old = gaussian_gam.predict_ci(
            X[:50], alpha=0.05, n_samples=300, seed=99
        )
    _, lo_new, hi_new = gaussian_gam.predict_ci(
        X[:50], level=0.95, n_samples=300, seed=99
    )
    np.testing.assert_allclose(lo_old, lo_new, rtol=1e-12)
    np.testing.assert_allclose(hi_old, hi_new, rtol=1e-12)
