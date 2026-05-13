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
    """Auto-k grows k until the per-smooth k-index says no residual
    structure remains along x_j.

    Uses n=500, y = sin(2πx) + ε. Checks:
    - Final k is in [4, 30].
    - Iteration count <= 10.
    - At the last iteration, k_index ≥ 1 − k_index_margin (the stopping
      criterion — residuals look like white noise along x).
    """
    rng = np.random.default_rng(42)
    n = 500
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    X = x.reshape(-1, 1)

    gam = GAMFitter(predictors=("x0",), auto_k=True)
    gam.fit(X, y)

    final_k = gam._term_k["x0"]
    iters = gam._auto_k_iterations

    assert 4 <= final_k <= 30, f"Final k={final_k} out of expected range [4, 30]"
    assert iters <= gam.auto_k_max_iter, (
        f"Auto-k used {iters} iterations (cap {gam.auto_k_max_iter})"
    )

    trace = gam.auto_k_trace_
    last_k_index = float(
        trace.loc[trace["iteration"] == iters - 1, "k_index"].iloc[0]
    )
    threshold = 1.0 - gam.k_index_margin
    assert last_k_index >= threshold, (
        f"k_index={last_k_index:.3f} below stopping threshold {threshold:.3f} "
        f"— auto-k stopped early (would have grown if not for cap / max_iter)"
    )

    print(f"\n  auto-k: final_k={final_k}, iters={iters}, k_index={last_k_index:.3f}")


# ---------------------------------------------------------------------- #
# Test 3: cap at n_unique - 1                                            #
# ---------------------------------------------------------------------- #


def test_auto_k_does_not_blow_up_on_clean_sine():
    """Regression: under the old (k-1)-edf < cutoff rule, high-SNR data
    drove λ→0, edf tracked k−1 indefinitely, and auto-k grew k up to
    n_unique−1 (~270 on n=500 even for a single sine). The k-index rule
    detects "no structure left in residuals" instead and must stop at a
    reasonable k for clean signal.
    """
    rng = np.random.default_rng(0)
    n = 500
    x = np.linspace(0, 2 * np.pi, n)
    for noise in [0.01, 0.05, 0.1]:
        y = np.sin(x) + noise * rng.standard_normal(n)
        gam = GAMFitter(predictors=("x0",), auto_k=True)
        gam.fit(x.reshape(-1, 1), y)
        final_k = gam._term_k["x0"]
        assert final_k <= 30, (
            f"auto-k blew up at noise={noise}: final_k={final_k}"
        )

    # And the absolute ceiling holds even on pathological zero-noise data
    # where residuals are numerical zero.
    y = np.sin(x)
    gam = GAMFitter(predictors=("x0",), auto_k=True, max_k_auto=40)
    gam.fit(x.reshape(-1, 1), y)
    assert gam._term_k["x0"] <= 40, (
        f"max_k_auto cap not honoured: got k={gam._term_k['x0']}"
    )


def test_auto_k_cap_at_n_unique():
    """When x has only 5 unique values, k must not exceed n_unique - 1 = 4."""
    rng = np.random.default_rng(7)
    x = np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 500)
    X = x.reshape(-1, 1)

    gam = GAMFitter(predictors=("x0",), auto_k=True)
    gam.fit(X, y)

    final_k = gam._term_k["x0"]
    # n_unique = 5 → cap = 4; k_default starts already at cap.
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
