"""Ergo-5: subset prediction via __getitem__ tests.

Four tests:
1. All-smooths subset + intercept matches full prediction.
2. __constant__ toggle: difference between gam[["__constant__", "x0"]]
   and gam[["x0"]] is approximately constant (pure intercept offset).
3. Independence: subset view does not alias the original.
4. Unknown name raises KeyError.

Fixture: 2d_gaussian_additive_n500_k10_cr (two smooths x0 and x1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "parity" / "fixtures"
_PARITY_SCHEMA = Path(__file__).resolve().parents[1] / "parity"
if str(_PARITY_SCHEMA) not in sys.path:
    sys.path.insert(0, str(_PARITY_SCHEMA))

from schema import Fixture  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture loading + fitting                                               #
# ---------------------------------------------------------------------- #

_FIX_NAME = "2d_gaussian_additive_n500_k10_cr"


def _load_and_fit():
    """Load the 2d Gaussian fixture and fit a GAMFitter. Returns (gam, X, y)."""
    fix = Fixture.load(_FIXTURES_DIR / f"{_FIX_NAME}.json")
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)

    gam = mgcv_rust.GAMFitter(
        predictors=["x0", "x1"],
        term_k_mapping={"x0": inp.k[0], "x1": inp.k[1]},
        family="gaussian",
    )
    gam.fit(x, y)
    return gam, x, y


# ---------------------------------------------------------------------- #
# Test 1: all smooths + intercept == full predict                        #
# ---------------------------------------------------------------------- #


def test_all_smooths_plus_intercept_equals_full():
    """gam[["__constant__", "x0", "x1"]].predict(X) == gam.predict(X)."""
    gam, X, _ = _load_and_fit()

    full_pred = gam.predict(X)
    subset_pred = gam[["__constant__", "x0", "x1"]].predict(X)

    np.testing.assert_allclose(
        subset_pred,
        full_pred,
        rtol=1e-10,
        err_msg="Subset with all terms + intercept should equal full prediction",
    )


def test_constant_only_is_scalar():
    """gam[["__constant__"]].predict(X) returns a nearly-constant array
    (the intercept repeated n times)."""
    gam, X, _ = _load_and_fit()

    const_pred = gam[["__constant__"]].predict(X)

    # The intercept coefficient is the same for all rows; the lpmatrix
    # intercept column is all-ones, so the result must be a constant.
    coef = gam.get_coefficients()
    expected_intercept = coef[0]

    np.testing.assert_allclose(
        const_pred,
        expected_intercept * np.ones(len(X)),
        rtol=1e-10,
        err_msg="gam[['__constant__']].predict should equal intercept * ones(n)",
    )


# ---------------------------------------------------------------------- #
# Test 2: __constant__ toggle gives a constant offset                    #
# ---------------------------------------------------------------------- #


def test_constant_toggle_is_constant_offset():
    """gam[["__constant__", "x0"]].predict(X) - gam[["x0"]].predict(X)
    is approximately constant (just the intercept)."""
    gam, X, _ = _load_and_fit()

    with_intercept = gam[["__constant__", "x0"]].predict(X)
    without_intercept = gam[["x0"]].predict(X)
    diff = with_intercept - without_intercept

    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Relative variation must be essentially zero.
    assert abs(mean_diff) > 0, "Intercept should be non-zero for this fixture"
    rel_variation = std_diff / abs(mean_diff)
    assert rel_variation < 1e-10, (
        f"Difference between with/without intercept is not constant: "
        f"mean={mean_diff:.6f}, std={std_diff:.2e}, rel={rel_variation:.2e}"
    )

    # Also verify the offset equals the fitted intercept coefficient.
    coef = gam.get_coefficients()
    np.testing.assert_allclose(
        diff,
        coef[0] * np.ones(len(X)),
        rtol=1e-10,
        err_msg="Intercept offset should equal coef[0]",
    )


# ---------------------------------------------------------------------- #
# Test 3: subset view is independent of original                         #
# ---------------------------------------------------------------------- #


def test_subset_view_does_not_alias_original():
    """Mutating the subset view must not change the original's predictions."""
    gam, X, _ = _load_and_fit()

    full_pred_before = gam.predict(X).copy()

    sub = gam[["x0"]]

    # Calling predict on the view should not alter gam's state.
    _ = sub.predict(X)

    full_pred_after = gam.predict(X)

    np.testing.assert_array_equal(
        full_pred_before,
        full_pred_after,
        err_msg="gam.predict changed after calling sub.predict — aliasing bug",
    )

    # Setting an attribute on the view must not propagate to the original.
    sub._subset_mask = {"x0", "x1"}
    assert "_subset_mask" not in vars(gam) or gam._subset_mask != sub._subset_mask, (
        "_subset_mask on view propagated back to original — copy.copy aliased it"
    )


# ---------------------------------------------------------------------- #
# Test 4: unknown name raises KeyError                                   #
# ---------------------------------------------------------------------- #


def test_unknown_name_raises_key_error():
    """gam[["does_not_exist"]] must raise KeyError."""
    gam, _, _ = _load_and_fit()

    with pytest.raises(KeyError):
        gam[["does_not_exist"]]

    # Also test a single-string unknown.
    with pytest.raises(KeyError):
        gam["also_not_a_predictor"]
