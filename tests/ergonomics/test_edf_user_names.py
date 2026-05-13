"""Regression test for the user-name vs native-name mismatch in
`get_edf_per_smooth()` lookup.

The Rust core stores smooths as ``x0``, ``x1``, ... internally. The
Python wrapper carries user-supplied predictor names (e.g. ``"age"``,
``"score"``). Earlier code did ``dict(get_edf_per_smooth()).get(user_name)``
which silently returned NaN whenever the user's column names didn't
happen to be ``x0``/``x1``. That broke:

- ``gam.edf_`` (always NaN for non-x0 names)
- ``gam.get_edf_df()`` (NaN edf column)
- ``gam.summary()`` (NaN scale and R² because they depend on edf_total)
- ``_auto_fit_k()`` (never grew k because edf lookup was 0)

This test locks the fix: ``edf_`` / ``get_edf_df`` / summary all work
with DataFrame columns that don't match the internal names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam


@pytest.fixture(scope="module")
def model_with_user_names():
    rng = np.random.default_rng(101)
    n = 300
    X = pd.DataFrame(
        {
            "age": rng.uniform(20, 80, n),
            "score": rng.uniform(0, 100, n),
        }
    )
    y = (
        0.05 * (X["age"] - 50) ** 2
        + 0.001 * (X["score"] - 50) ** 2
        + rng.normal(0, 1.0, n)
    )
    gam = Gam(
        family="gaussian",
        term_k_mapping={"age": 8, "score": 8},  # fixed-k to avoid auto-k path interaction
    ).fit(X, y)
    return gam, X, y


def test_edf_not_nan_with_user_names(model_with_user_names):
    gam, _, _ = model_with_user_names
    assert not np.any(np.isnan(gam.edf_)), f"edf_ should not be NaN, got {gam.edf_}"


def test_edf_positive_and_below_k(model_with_user_names):
    gam, _, _ = model_with_user_names
    # k=8 means k-1=7 basis after sum-to-zero — EDF must be in (0, 7].
    assert np.all(gam.edf_ > 0)
    assert np.all(gam.edf_ <= 7.0 + 1e-6)


def test_get_edf_df_returns_finite_values(model_with_user_names):
    gam, _, _ = model_with_user_names
    df = gam.get_edf_df()
    assert set(df["predictor"]) == {"age", "score"}
    assert df["edf"].notna().all()
    assert (df["edf"] > 0).all()


def test_summary_scale_and_r2_finite_for_gaussian(model_with_user_names):
    gam, _, _ = model_with_user_names
    s = gam.summary()
    assert not np.isnan(s.scale), "scale should be finite for Gaussian"
    assert not np.isnan(s.r_squared), "R² should be finite for Gaussian"
    assert 0.5 < s.r_squared < 1.0


def test_subset_edf_user_names(model_with_user_names):
    gam, _, _ = model_with_user_names
    sub = gam[["age"]]
    np.testing.assert_array_equal(sub.feature_names_in_, ["age"])
    assert not np.isnan(sub.edf_[0])
    assert sub.edf_[0] > 0


def test_auto_k_grows_for_user_named_dataframe():
    """Without the fix, the auto-k loop pulled edf=0 from the lookup
    miss and never grew k. Verify it actually grows here."""
    rng = np.random.default_rng(202)
    n = 800
    X = pd.DataFrame(
        {
            "age": rng.uniform(0, 1, n),
            "wiggle": rng.uniform(0, 1, n),
        }
    )
    # Highly wiggly truth in `age` so the smooth needs many basis funcs.
    y = np.sin(20 * X["age"]) + 0.1 * X["wiggle"] + rng.normal(0, 0.1, n)
    gam = Gam(family="gaussian").fit(X, y)  # no term_k_mapping → auto-k path
    # At least one predictor must have grown beyond the default of 10.
    assert gam.k_.max() > 10, f"auto-k never grew, got k_={gam.k_}"
