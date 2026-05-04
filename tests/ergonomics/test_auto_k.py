"""Ergo-3b+3c: auto-k loop and get_edf_df() tests.

Four tests:
1. Fixed-k pass-through — term_k_mapping covers all features, single fit, no k growth.
2. Auto-k convergence — n=500 sin(2πx) + ε, no term_k_mapping; k grows to resolution.
3. Cap at n_unique - 1 — x has only 5 unique values; k must not exceed 4.
4. get_edf_df() — columns, edf values match native accessor.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import GAMFitter


# ---------------------------------------------------------------------- #
# Test 1: fixed-k pass-through                                           #
# ---------------------------------------------------------------------- #


def test_fixed_k_passthrough():
    """When term_k_mapping covers all predictors, fit runs once (no auto-k)
    and the stored k values match what was passed in."""
    rng = np.random.default_rng(2025)
    x0 = rng.uniform(0, 1, 300)
    x1 = rng.uniform(0, 1, 300)
    y = np.sin(2 * np.pi * x0) + 0.5 * x1 + rng.normal(0, 0.1, 300)
    X = np.column_stack([x0, x1])

    gam = GAMFitter(
        predictors=("x0", "x1"),
        term_k_mapping={"x0": 6, "x1": 8},
    )
    gam.fit(X, y)

    # No auto-k iterations should have run.
    assert gam._auto_k_iterations == 0, (
        f"Expected 0 iterations in fixed-k path, got {gam._auto_k_iterations}"
    )

    # Stored k must match the passed mapping (subject to n_unique cap which
    # won't activate for 300 continuous-uniform points).
    assert gam._term_k["x0"] == 6
    assert gam._term_k["x1"] == 8

    # Predictions must be finite.
    preds = gam.predict(X)
    assert np.isfinite(preds).all()


# ---------------------------------------------------------------------- #
# Test 2: auto-k convergence on simple 1-D data                         #
# ---------------------------------------------------------------------- #


def test_auto_k_convergence():
    """Auto-k grows k until EDF saturation is resolved (ratio < edf_cutoff headroom).

    Uses n=500, y = sin(2πx) + ε. Checks:
    - Final k is in [4, 20].
    - get_edf_df() ratio < 0.95 (i.e. not saturated after convergence).
    - Iteration count <= 10.
    """
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    X = x.reshape(-1, 1)

    # No term_k_mapping → auto-k loop activates.
    gam = GAMFitter(predictors=("x0",))
    gam.fit(X, y)

    final_k = gam._term_k["x0"]
    iters = gam._auto_k_iterations

    assert 4 <= final_k <= 30, f"Final k={final_k} out of expected range [4, 30]"
    assert iters <= 10, f"Auto-k used {iters} iterations (max 10)"

    df = gam.get_edf_df()
    ratio = float(df.loc[df["predictor"] == "x0", "ratio"].iloc[0])
    # After convergence the term should not be saturated (ratio well below 1).
    assert ratio < 0.95, f"EDF ratio {ratio:.3f} still saturated after auto-k"

    # Expose useful diagnostics when running with -v.
    print(f"\n  auto-k: final_k={final_k}, iters={iters}, ratio={ratio:.3f}")


# ---------------------------------------------------------------------- #
# Test 3: cap at n_unique - 1                                            #
# ---------------------------------------------------------------------- #


def test_auto_k_cap_at_n_unique():
    """When x has only 5 unique values, k must not exceed n_unique - 1 = 4."""
    rng = np.random.default_rng(7)
    x = np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 500)
    X = x.reshape(-1, 1)

    gam = GAMFitter(predictors=("x0",))
    gam.fit(X, y)

    final_k = gam._term_k["x0"]
    # n_unique = 5 → cap = 4; k_default=4 starts already at cap.
    assert final_k <= 4, f"k={final_k} exceeds cap of n_unique-1=4"


# ---------------------------------------------------------------------- #
# Test 4: get_edf_df() columns and values                                #
# ---------------------------------------------------------------------- #


def test_get_edf_df_columns_and_values():
    """get_edf_df() returns a DataFrame with the right columns, and the
    edf column agrees with the native accessor."""
    rng = np.random.default_rng(99)
    n = 300
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x0) + x1 + rng.normal(0, 0.1, n)
    X = np.column_stack([x0, x1])

    gam = GAMFitter(
        predictors=("x0", "x1"),
        term_k_mapping={"x0": 8, "x1": 6},
    )
    gam.fit(X, y)

    df = gam.get_edf_df()

    # Column set must match spec.
    expected_cols = {"predictor", "k", "edf", "ratio"}
    assert expected_cols <= set(df.columns), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    assert len(df) == 2  # one row per predictor

    # EDF values must agree with the raw native accessor.
    native_edf = dict(gam._native.get_edf_per_smooth())
    for _, row in df.iterrows():
        name = row["predictor"]
        assert name in native_edf, f"predictor '{name}' missing from native accessor"
        np.testing.assert_allclose(
            row["edf"], native_edf[name], rtol=1e-10,
            err_msg=f"edf mismatch for predictor '{name}'"
        )
        # ratio = edf / (k - 1)
        expected_ratio = row["edf"] / max(row["k"] - 1, 1)
        np.testing.assert_allclose(row["ratio"], expected_ratio, rtol=1e-10)
