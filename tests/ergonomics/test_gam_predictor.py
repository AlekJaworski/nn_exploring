"""Ergo-N8: `GamPredictor` — frozen inference-only view of a fitted Gam.

Covers:
- Construction requires a fitted Gam.
- ``predict`` / ``predict_ci`` / ``predict_diff`` delegate and match Gam.
- ``__getitem__`` returns a subset GamPredictor.
- ``check_against`` round-trip passes on equal fits and raises on divergent.
- Column drift is caught:
  - Missing column → ValueError.
  - Reordered columns → silently re-aligned (sklearn convention).
- ``__slots__`` prevents attribute leaks.
- sklearn-style attrs are exposed: ``feature_names_in_``, ``coef_``, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam, GamPredictor, TermContributions


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(23)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(0, 5, n)
    y = np.sin(x0) + 0.3 * (x1 - 2.5) ** 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x0": x0, "x1": x1}), y


@pytest.fixture(scope="module")
def gaussian_gam(gaussian_data):
    X, y = gaussian_data
    return Gam(family="gaussian").fit(X, y)


# ---------------------------------------------------------------------- #
# Construction                                                           #
# ---------------------------------------------------------------------- #


def test_construct_from_unfitted_raises():
    gam = Gam(family="gaussian")
    with pytest.raises(RuntimeError, match="fitted"):
        GamPredictor(gam)


def test_construct_rejects_non_gam():
    with pytest.raises(TypeError, match="Gam"):
        GamPredictor("not a gam")  # type: ignore[arg-type]


def test_construct_from_fitted_succeeds(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    assert isinstance(predictor, GamPredictor)


def test_predictor_frozen_slots(gaussian_gam):
    """__slots__ blocks attribute creep — important for production
    safety (you can't accidentally mutate a deployed predictor)."""
    predictor = GamPredictor(gaussian_gam)
    with pytest.raises(AttributeError):
        predictor.new_attr = 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- #
# Delegation matches Gam                                                  #
# ---------------------------------------------------------------------- #


def test_predict_matches_gam(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    np.testing.assert_allclose(predictor.predict(X), gaussian_gam.predict(X), rtol=1e-12)


def test_predict_type_terms_matches_gam(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    p_terms = predictor.predict(X, type="terms")
    g_terms = gaussian_gam.predict(X, type="terms")
    assert isinstance(p_terms, TermContributions)
    np.testing.assert_allclose(p_terms.total, g_terms.total, rtol=1e-12)
    pd.testing.assert_frame_equal(p_terms.contributions, g_terms.contributions)


def test_predict_ci_matches_gam(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    p_mean, p_lo, p_hi = predictor.predict_ci(X[:30], n_samples=300, seed=5)
    g_mean, g_lo, g_hi = gaussian_gam.predict_ci(X[:30], n_samples=300, seed=5)
    np.testing.assert_allclose(p_mean, g_mean, rtol=1e-12)
    np.testing.assert_allclose(p_lo, g_lo, rtol=1e-12)
    np.testing.assert_allclose(p_hi, g_hi, rtol=1e-12)


def test_predict_diff_matches_gam(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    np.testing.assert_allclose(
        predictor.predict_diff(X[:20], X[20:40]),
        gaussian_gam.predict_diff(X[:20], X[20:40]),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------- #
# Subset views                                                            #
# ---------------------------------------------------------------------- #


def test_subset_view_predicts(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    sub_pred = predictor[["x0"]]
    sub_gam = gaussian_gam[["x0"]]
    np.testing.assert_allclose(sub_pred.predict(X), sub_gam.predict(X), rtol=1e-12)


def test_subset_view_is_predictor_type(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    assert isinstance(predictor[["x0"]], GamPredictor)


def test_subset_unknown_name_raises(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    with pytest.raises(KeyError):
        predictor[["does_not_exist"]]


# ---------------------------------------------------------------------- #
# check_against                                                          #
# ---------------------------------------------------------------------- #


def test_check_against_passes_on_same_gam(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    predictor.check_against(gaussian_gam, X[:50])


def test_check_against_raises_on_divergent_gam(gaussian_data):
    X, y = gaussian_data
    gam_a = Gam(family="gaussian", term_k_mapping={"x0": 5, "x1": 5}).fit(X, y)
    gam_b = Gam(family="gaussian", term_k_mapping={"x0": 12, "x1": 12}).fit(X, y)
    predictor = GamPredictor(gam_a)
    with pytest.raises(AssertionError, match="diverge"):
        predictor.check_against(gam_b, X[:50])


# ---------------------------------------------------------------------- #
# Column drift                                                            #
# ---------------------------------------------------------------------- #


def test_missing_column_raises(gaussian_gam, gaussian_data):
    """The fa6d420 bug class: predicting with a DataFrame missing a
    fitted column must raise, not silently use wrong data."""
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    X_missing = X.drop(columns=["x1"])
    with pytest.raises(ValueError, match="missing expected columns"):
        predictor.predict(X_missing)


def test_reordered_columns_are_realigned(gaussian_gam, gaussian_data):
    """Reordering columns must still produce the same predictions —
    feature_names_in_ defines the canonical order and the wrapper
    projects to it."""
    X, _ = gaussian_data
    predictor = GamPredictor(gaussian_gam)
    X_reordered = X[["x1", "x0"]]
    np.testing.assert_allclose(
        predictor.predict(X_reordered), predictor.predict(X), rtol=1e-12
    )


# ---------------------------------------------------------------------- #
# sklearn-style attrs delegate                                            #
# ---------------------------------------------------------------------- #


def test_feature_names_in_delegates(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    np.testing.assert_array_equal(
        predictor.feature_names_in_, gaussian_gam.feature_names_in_
    )


def test_coef_and_intercept_delegate(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    np.testing.assert_allclose(predictor.coef_, gaussian_gam.coef_, rtol=1e-12)
    assert predictor.intercept_ == pytest.approx(gaussian_gam.intercept_)


def test_subset_attrs_filter(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    sub = predictor[["x0"]]
    np.testing.assert_array_equal(sub.feature_names_in_, ["x0"])
    assert sub.n_features_in_ == 1


def test_repr_mentions_family_and_features(gaussian_gam):
    predictor = GamPredictor(gaussian_gam)
    r = repr(predictor)
    assert "gaussian" in r
    assert "x0" in r and "x1" in r
