"""Smoke tests for `mgcv_rust.GAMFitter` — the ergonomics wrapper.

Designed to lock in the *interface* used by the neighbourhoods repo:
constructor signature, fit/predict on numpy + pandas + polars,
confidence intervals, posterior samples, vcov, lpmatrix, serialize.
Numerical accuracy is covered by `tests/parity` against the Rust core
directly; these tests just verify the wrapper plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import GAM, GAMFitter


def _make_data(seed: int = 2025, n: int = 400):
    rng = np.random.default_rng(seed)
    days = rng.uniform(0, 1825, n)
    quality = rng.uniform(1, 10, n)
    y = -0.0001 * days + 0.04 * quality**2 - 0.08 * quality + rng.normal(0, 0.1, n)
    return days, quality, y


# ---------------------------------------------------------------------- #
# Constructor + basic fit/predict                                        #
# ---------------------------------------------------------------------- #


def test_constructor_signature_matches_neighbourhoods():
    """Replicates the call site in `gam_logic.fit_gam_model` —
    constructor must accept all those kwargs without complaint."""
    gam = GAMFitter(
        predictors=("days_ago", "quality", "condition"),
        target="y",
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12, "condition": 12},
        term_pc_mapping={"concessions_dollars": 0},
        family="gaussian",
        link="identity",
        method="fREML",  # neighbourhoods uses fREML; we accept and route to REML
    )
    assert list(gam.predictors) == ["days_ago", "quality", "condition"]
    assert gam.k_default == 6
    assert gam.term_k_mapping["days_ago"] == 25
    assert gam.family == "gaussian"


def test_fit_predict_numpy():
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(
        predictors=("days_ago", "quality"),
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam.fit(X, y)
    preds = gam.predict(X)
    assert preds.shape == (X.shape[0],)
    assert np.isfinite(preds).all()


def test_fit_predict_pandas():
    """DataFrame input — column order matched against `predictors`."""
    days, quality, y = _make_data()
    df = pd.DataFrame({"days_ago": days, "quality": quality})
    gam = GAMFitter(
        predictors=("days_ago", "quality"),
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam.fit(df, y)
    preds = gam.predict(df)
    assert preds.shape == (df.shape[0],)


def test_dataframe_columns_reordered_to_match_predictors():
    """If the DataFrame has columns in a different order from
    `predictors`, they should be reordered before fit."""
    days, quality, y = _make_data()
    # Reverse the DataFrame column order — predictors arg sets the canonical order
    df = pd.DataFrame({"quality": quality, "days_ago": days})
    gam_a = GAMFitter(
        predictors=("days_ago", "quality"),
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam_a.fit(df, y)
    # Compare to fit on numpy with explicit order
    X = np.column_stack([days, quality])
    gam_b = GAMFitter(
        predictors=("days_ago", "quality"),
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam_b.fit(X, y)
    np.testing.assert_allclose(gam_a.predict(df), gam_b.predict(X), rtol=1e-10, atol=1e-10)


def test_predict_dataframe_after_numpy_fit():
    """A model fit on a numpy array must accept a DataFrame at predict
    time as long as the columns match (in any order)."""
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"), k_default=6)
    gam.fit(X, y)
    df_predict = pd.DataFrame({"days_ago": days, "quality": quality})
    preds_df = gam.predict(df_predict)
    preds_np = gam.predict(X)
    np.testing.assert_allclose(preds_df, preds_np, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------- #
# Diagnostics (vcov, lpmatrix)                                           #
# ---------------------------------------------------------------------- #


def test_vcov_shape_and_symmetry():
    days, quality, y = _make_data()
    gam = GAMFitter(
        predictors=("days_ago", "quality"),
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam.fit(np.column_stack([days, quality]), y)
    v = gam.get_vcov()
    coef = gam.get_coefficients()
    assert v.shape == (coef.size, coef.size)
    np.testing.assert_allclose(v, v.T, atol=1e-10)
    # vcov should be PSD — diagonal entries non-negative
    assert np.all(np.diag(v) >= -1e-12)


def test_lpmatrix_intercept_column_is_one():
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"))
    gam.fit(X, y)
    lp = gam.evaluate_lpmatrix(X[:5])
    assert lp.shape[0] == 5
    np.testing.assert_allclose(lp[:, 0], 1.0, atol=1e-12)


def test_lpmatrix_dot_coef_equals_predict():
    """The lpmatrix at X dotted with the coefficients should reproduce
    `predict(X)` for identity link."""
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"), family="gaussian", link="identity")
    gam.fit(X, y)
    lp = gam.evaluate_lpmatrix(X)
    coef = gam.get_coefficients()
    preds_via_lp = lp @ coef
    preds_direct = gam.predict(X)
    np.testing.assert_allclose(preds_via_lp, preds_direct, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------- #
# Confidence intervals + posterior samples                               #
# ---------------------------------------------------------------------- #


def test_predict_ci_returns_valid_intervals():
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"))
    gam.fit(X, y)
    lo, hi = gam.predict_ci(X[:50], alpha=0.05, n_samples=500)
    assert lo.shape == (50,)
    assert hi.shape == (50,)
    assert np.all(hi >= lo), "upper CI must be >= lower CI"
    # CI should bracket the point predictions on average
    point_pred = gam.predict(X[:50])
    inside = np.sum((lo <= point_pred) & (point_pred <= hi))
    # Most points should be inside the CI (we used the same coef as the mean)
    assert inside >= 40, f"expected most points inside CI, got {inside}/50"


def test_posterior_samples_shape():
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"))
    gam.fit(X, y)
    post = gam.get_posterior_samples(X[:10], n_samples=100, seed=42)
    assert post.shape == (10, 100)


def test_posterior_samples_seeded_reproducibility():
    days, quality, y = _make_data()
    X = np.column_stack([days, quality])
    gam = GAMFitter(predictors=("days_ago", "quality"))
    gam.fit(X, y)
    s1 = gam.get_posterior_samples(X[:5], n_samples=20, seed=42)
    s2 = gam.get_posterior_samples(X[:5], n_samples=20, seed=42)
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------- #
# Serialization                                                          #
# ---------------------------------------------------------------------- #


def test_serialize_schema_matches_neighbourhoods():
    """The serialized dict must have exactly the keys consumed by
    `GamPredictor` in the neighbourhoods repo."""
    days, quality, y = _make_data()
    gam = GAMFitter(
        predictors=("days_ago", "quality"),
        target="price",
        k_default=6,
        term_k_mapping={"days_ago": 25, "quality": 12},
    )
    gam.fit(np.column_stack([days, quality]), y)
    s = gam.serialize()

    # Keys checked against r_fitting.r_model.GamFitter.serialize:
    expected_keys = {
        "predictors_info",
        "lp_matrix",
        "coefficients",
        "cov_matrix",
        "lp_feature_values",
        "predictors",
        "family",
        "link",
        "pc_map",
    }
    assert set(s.keys()) >= expected_keys, f"missing: {expected_keys - set(s.keys())}"

    # predictors_info has 'constant' + each user-supplied predictor name
    assert "constant" in s["predictors_info"]
    assert "days_ago" in s["predictors_info"]
    assert "quality" in s["predictors_info"]
    # 'constant' first_index/last_index = 0/0 (single intercept column)
    assert s["predictors_info"]["constant"] == {"first_index": 0, "last_index": 0}

    # Per-smooth indices must point inside the lp_matrix column range
    n_cols = s["lp_matrix"].shape[1]
    for name, info in s["predictors_info"].items():
        if name == "constant":
            continue
        assert 0 < info["first_index"] <= info["last_index"] < n_cols, (
            f"{name} indices {info} fall outside lp_matrix [0, {n_cols})"
        )

    # lp_feature_values shape: same n_rows as lp_matrix, n_cols = 1 + d
    assert s["lp_feature_values"].shape[0] == s["lp_matrix"].shape[0]
    assert s["lp_feature_values"].shape[1] == 1 + len(gam._effective_predictors)


def test_serialize_with_explicit_prediction_range():
    days, quality, y = _make_data()
    gam = GAMFitter(predictors=("days_ago", "quality"))
    gam.fit(np.column_stack([days, quality]), y)
    custom_range = {
        "days_ago": {"min": 100.0, "max": 1000.0},
        "quality": {"min": 2.0, "max": 8.0},
    }
    s = gam.serialize(prediction_range=custom_range, n_points=50)
    # 50 grid points + 2 extrap tail rows
    assert s["lp_matrix"].shape[0] == 52


# ---------------------------------------------------------------------- #
# Native GAM still importable alongside the wrapper                      #
# ---------------------------------------------------------------------- #


def test_native_gam_still_works():
    """The lower-level Rust class is still re-exported as `GAM`."""
    days, quality, y = _make_data()
    g = GAM()
    g.fit(np.column_stack([days, quality]), y, k=[10, 8], bs="cr")
    assert g.get_coefficients().shape[0] > 0
