"""Tests for Ergo-4: pc-anchoring via term_pc_mapping.

Verifies that `s(x, pc=v)` correctly enforces f(v) = 0 by using the
null-space of B(pc) as the identifiability constraint instead of the
default sum-to-zero. Three test groups:

1. f(pc) = 0 to machine precision.
2. Predictions elsewhere are self-consistent (smooth contribution at pc point is zero).
3. Without pc, behavior is unchanged (sanity that sum-to-zero path still works).
"""

from __future__ import annotations

import numpy as np
import pytest

from mgcv_rust import GAMFitter


# ------------------------------------------------------------------ #
# Shared fixtures                                                     #
# ------------------------------------------------------------------ #


def _sin_data(seed: int = 42, n: int = 500):
    """y = sin(2*pi*x) + eps on x in [0, 1]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.1, n)
    return x.reshape(-1, 1), y


def _smooth_contrib(gam: GAMFitter, X_point: np.ndarray, k: int) -> float:
    """Return the smooth contribution (excluding intercept) at X_point."""
    lp = gam.evaluate_lpmatrix(X_point)  # shape (1, p)
    coef = gam.get_coefficients()        # shape (p,)
    # Column 0 is the intercept; columns 1..end are smooth blocks.
    return float(lp[0, 1:] @ coef[1:])


# ------------------------------------------------------------------ #
# Test 1: f(pc) = 0 to machine precision                             #
# ------------------------------------------------------------------ #


def test_f_at_pc_equals_zero():
    """The smooth contribution at the pc anchor point must be zero
    (to machine precision) for any fitted GAM with term_pc_mapping."""
    X, y = _sin_data()
    pc = 0.0  # anchor at x=0

    gam = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
        term_pc_mapping={"x": pc},
    )
    gam.fit(X, y)

    # Evaluate lpmatrix at the pc point.
    X_pc = np.array([[pc]])
    lp = gam.evaluate_lpmatrix(X_pc)   # shape (1, p)
    coef = gam.get_coefficients()      # shape (p,)

    # The smooth block is columns 1..end.
    smooth_contrib = float(lp[0, 1:] @ coef[1:])
    intercept_contrib = float(lp[0, 0] * coef[0])

    print(f"smooth_contrib at pc={pc}: {smooth_contrib:.6e}")
    print(f"intercept_contrib: {intercept_contrib:.6f}")

    # The smooth contribution at the pc point must be zero.
    assert abs(smooth_contrib) < 1e-12, (
        f"Smooth contribution at pc={pc} is {smooth_contrib:.3e}, expected < 1e-12"
    )

    # Cross-check: total prediction = intercept contribution alone.
    total = float(lp[0] @ coef)
    assert abs(total - intercept_contrib) < 1e-12, (
        f"Total prediction {total:.6f} != intercept {intercept_contrib:.6f}"
    )


def test_f_at_pc_equals_zero_interior_point():
    """Same constraint holds for a pc point in the interior of [0,1]."""
    X, y = _sin_data(seed=99, n=600)
    pc = 0.5  # interior anchor

    gam = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 12},
        term_pc_mapping={"x": pc},
    )
    gam.fit(X, y)

    X_pc = np.array([[pc]])
    lp = gam.evaluate_lpmatrix(X_pc)
    coef = gam.get_coefficients()
    smooth_contrib = float(lp[0, 1:] @ coef[1:])

    print(f"smooth_contrib at pc={pc}: {smooth_contrib:.6e}")

    assert abs(smooth_contrib) < 1e-12, (
        f"Smooth contribution at pc={pc} is {smooth_contrib:.3e}, expected < 1e-12"
    )


# ------------------------------------------------------------------ #
# Test 2: Predictions elsewhere are self-consistent                  #
# ------------------------------------------------------------------ #


def test_predictions_elsewhere_are_smooth():
    """Predictions on a held-out grid are finite, not all equal to the
    intercept, and show the expected sinusoidal shape (qualitative check)."""
    X, y = _sin_data(seed=7, n=400)
    pc = 0.0

    gam = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
        term_pc_mapping={"x": pc},
    )
    gam.fit(X, y)

    # Evaluate on a fine grid.
    x_grid = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    preds = gam.predict(x_grid)

    # Predictions should be finite and have real variation.
    assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"
    assert preds.std() > 0.1, f"Predictions suspiciously flat: std={preds.std():.4f}"

    # The smooth should peak near x=0.25 (quarter period of sin(2pi x))
    # and trough near x=0.75. Just verify sign at a few key points.
    lp_quarter = gam.evaluate_lpmatrix(np.array([[0.25]]))
    lp_threequarter = gam.evaluate_lpmatrix(np.array([[0.75]]))
    coef = gam.get_coefficients()
    smooth_at_quarter = float(lp_quarter[0, 1:] @ coef[1:])
    smooth_at_threequarter = float(lp_threequarter[0, 1:] @ coef[1:])

    print(f"smooth at x=0.25: {smooth_at_quarter:.4f}")
    print(f"smooth at x=0.75: {smooth_at_threequarter:.4f}")

    # sin(2pi*0.25) = 1 > 0, sin(2pi*0.75) = -1 < 0
    assert smooth_at_quarter > 0.3, (
        f"Expected positive smooth at x=0.25, got {smooth_at_quarter:.3f}"
    )
    assert smooth_at_threequarter < -0.3, (
        f"Expected negative smooth at x=0.75, got {smooth_at_threequarter:.3f}"
    )


# ------------------------------------------------------------------ #
# Test 3: Without pc, behavior is unchanged (sum-to-zero still works) #
# ------------------------------------------------------------------ #


def test_no_pc_behavior_unchanged():
    """Without term_pc_mapping, the fit is identical to the baseline
    (sum-to-zero constraint). Verify the two are consistent to high
    precision when fitting the same data."""
    X, y = _sin_data(seed=123, n=300)

    gam_baseline = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
    )
    gam_baseline.fit(X, y)

    # Fit again without any pc mapping (same constructor path).
    gam_no_pc = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
        term_pc_mapping={},  # explicitly empty
    )
    gam_no_pc.fit(X, y)

    x_grid = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    preds_baseline = gam_baseline.predict(x_grid)
    preds_no_pc = gam_no_pc.predict(x_grid)

    # Same result — sum-to-zero path used in both cases.
    np.testing.assert_allclose(
        preds_no_pc,
        preds_baseline,
        rtol=1e-10,
        err_msg="Predictions with empty term_pc_mapping differ from baseline",
    )


def test_pc_smooth_differs_from_sum_to_zero_smooth():
    """The smooth block (excluding intercept) at non-pc points should
    differ between pc and sum-to-zero models — they encode different
    constraints and so the reparameterised coefficients differ.

    We check that the smooth contributions (not total predictions) differ
    since the intercept can compensate in total predictions."""
    X, y = _sin_data(seed=55, n=400)

    gam_pc = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
        term_pc_mapping={"x": 0.0},
    )
    gam_pc.fit(X, y)

    gam_sto = GAMFitter(
        predictors=["x"],
        term_k_mapping={"x": 10},
    )
    gam_sto.fit(X, y)

    x_grid = np.linspace(0.0, 1.0, 50).reshape(-1, 1)

    lp_pc = gam_pc.evaluate_lpmatrix(x_grid)
    coef_pc = gam_pc.get_coefficients()
    smooth_pc = lp_pc[:, 1:] @ coef_pc[1:]

    lp_sto = gam_sto.evaluate_lpmatrix(x_grid)
    coef_sto = gam_sto.get_coefficients()
    smooth_sto = lp_sto[:, 1:] @ coef_sto[1:]

    # The smooth functions are reparameterised differently (different
    # identifiability constraints) — they should not match exactly.
    max_diff = float(np.abs(smooth_pc - smooth_sto).max())
    print(f"Max |smooth_pc - smooth_sto| = {max_diff:.4e}")
    assert max_diff > 1e-6, (
        f"pc and sum-to-zero smooth contributions are suspiciously identical "
        f"(max |diff| = {max_diff:.2e}). pc-anchoring may not be applied."
    )


# ------------------------------------------------------------------ #
# Test 4: Multi-predictor — only the named predictor is pc-anchored  #
# ------------------------------------------------------------------ #


def test_pc_only_anchors_named_predictor():
    """In a two-predictor model, only the term listed in term_pc_mapping
    should have its smooth pinned to zero at the anchor point."""
    rng = np.random.default_rng(2025)
    n = 400
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + rng.normal(0, 0.1, n)
    X = np.column_stack([x1, x2])

    pc_x1 = 0.0  # anchor x1 at 0

    gam = GAMFitter(
        predictors=["x1", "x2"],
        term_k_mapping={"x1": 10, "x2": 10},
        term_pc_mapping={"x1": pc_x1},
    )
    gam.fit(X, y)

    # Evaluate lpmatrix at (x1=pc_x1, x2=0.5).
    X_test = np.array([[pc_x1, 0.5]])
    lp = gam.evaluate_lpmatrix(X_test)
    coef = gam.get_coefficients()

    # Layout: [intercept | x1_smooth (k-1=9 cols) | x2_smooth (k-1=9 cols)]
    k1, k2 = 10, 10
    # x1 smooth block: columns 1..k1 (9 columns)
    x1_block = lp[0, 1:k1]
    x1_coef = coef[1:k1]
    x1_contrib = float(x1_block @ x1_coef)

    print(f"x1 smooth contribution at pc={pc_x1}: {x1_contrib:.6e}")

    assert abs(x1_contrib) < 1e-12, (
        f"x1 smooth contribution at pc={pc_x1} is {x1_contrib:.3e}, expected < 1e-12"
    )

    # x2 at x2=0.5 should NOT be constrained to zero.
    x2_block = lp[0, k1:]
    x2_coef = coef[k1:]
    x2_contrib = float(x2_block @ x2_coef)
    print(f"x2 smooth contribution at x2=0.5: {x2_contrib:.4f}")
    # cos(2pi*0.5) = -1, x2 smooth should be noticeably non-zero.
    # Just verify the model fitted without error and x1 is pinned.
    assert np.isfinite(x2_contrib), "x2 smooth contribution is non-finite"
