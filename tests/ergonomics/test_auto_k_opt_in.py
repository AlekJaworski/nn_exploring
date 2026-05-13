"""Ergo-N10: auto-k must be opt-in via `Gam(auto_k=True)`.

Default behavior is a single fit with ``k = term_k_mapping[name]`` or
``k_default``. The iterative auto-k growth is reserved for the explicit
opt-in to keep fits deterministic from the constructor args and avoid
hidden multi-fit costs in production code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam


@pytest.fixture
def wiggly_data():
    rng = np.random.default_rng(303)
    n = 600
    X = pd.DataFrame(
        {"age": rng.uniform(0, 1, n), "wiggle": rng.uniform(0, 1, n)}
    )
    y = np.sin(20 * X["age"]) + 0.1 * X["wiggle"] + rng.normal(0, 0.1, n)
    return X, y


def test_default_is_single_fit(wiggly_data):
    """Without auto_k=True, fit runs once at k_default."""
    X, y = wiggly_data
    gam = Gam(family="gaussian").fit(X, y)
    assert gam._auto_k_iterations == 0
    # k frozen at k_default for every predictor (mgcv default is 10).
    assert np.all(gam.k_ == gam.k_default)


def test_default_respects_term_k_mapping(wiggly_data):
    X, y = wiggly_data
    gam = Gam(
        family="gaussian",
        term_k_mapping={"age": 15, "wiggle": 5},
    ).fit(X, y)
    assert gam._auto_k_iterations == 0
    k_by_name = dict(zip(gam.feature_names_in_, gam.k_))
    assert k_by_name["age"] == 15
    assert k_by_name["wiggle"] == 5


def test_opt_in_grows_k(wiggly_data):
    X, y = wiggly_data
    gam = Gam(family="gaussian", auto_k=True).fit(X, y)
    assert gam._auto_k_iterations > 0
    # The wiggly `age` smooth must have grown beyond k_default.
    k_by_name = dict(zip(gam.feature_names_in_, gam.k_))
    assert k_by_name["age"] > gam.k_default


def test_opt_in_with_partial_mapping_only_grows_uncovered(wiggly_data):
    """Predictors in term_k_mapping stay fixed even with auto_k=True;
    only uncovered ones grow."""
    X, y = wiggly_data
    gam = Gam(
        family="gaussian",
        auto_k=True,
        term_k_mapping={"wiggle": 6},
    ).fit(X, y)
    k_by_name = dict(zip(gam.feature_names_in_, gam.k_))
    assert k_by_name["wiggle"] == 6  # fixed
    assert k_by_name["age"] > gam.k_default  # grew


def test_default_predictions_unchanged_for_x0_x1_named_data():
    """For DataFrames with default-named columns (x0, x1), the single
    fit at k_default produces the same predictions whether or not the
    user passes auto_k=False explicitly. (Sanity check that the new
    default isn't disturbing the previously-working path.)"""
    rng = np.random.default_rng(404)
    n = 300
    X = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
    y = np.sin(2 * np.pi * X["x0"]) + 0.5 * X["x1"] + rng.normal(0, 0.1, n)
    a = Gam(family="gaussian").fit(X, y).predict(X)
    b = Gam(family="gaussian", auto_k=False).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-12)
