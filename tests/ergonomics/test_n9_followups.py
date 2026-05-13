"""Ergo-N9: follow-up sklearn-style methods on Gam.

Covers ``partial_effect``, ``plot``, ``summary``, ``predict_proba``,
``score``. Each method is independent; tests are grouped by concern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam, GamSummary


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(29)
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
    rng = np.random.default_rng(31)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-1, 1, n)
    eta = 1.0 * x0 - 1.5 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return pd.DataFrame({"x0": x0, "x1": x1}), y


@pytest.fixture(scope="module")
def binomial_gam(binomial_data):
    X, y = binomial_data
    return Gam(family="binomial").fit(X, y)


# ---------------------------------------------------------------------- #
# partial_effect                                                          #
# ---------------------------------------------------------------------- #


def test_partial_effect_returns_dataframe(gaussian_gam):
    df = gaussian_gam.partial_effect("x0", grid_n=50)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (50, 4)
    assert list(df.columns) == ["x0", "effect", "lo", "hi"]


def test_partial_effect_no_level_drops_ci(gaussian_gam):
    df = gaussian_gam.partial_effect("x0", grid_n=30, level=None)
    assert list(df.columns) == ["x0", "effect"]


def test_partial_effect_grid_spans_training_range(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    df = gaussian_gam.partial_effect("x0", grid_n=80)
    assert df["x0"].min() == pytest.approx(X["x0"].min())
    assert df["x0"].max() == pytest.approx(X["x0"].max())


def test_partial_effect_unknown_predictor_raises(gaussian_gam):
    with pytest.raises(KeyError):
        gaussian_gam.partial_effect("does_not_exist")


def test_partial_effect_ci_brackets_effect(gaussian_gam):
    df = gaussian_gam.partial_effect("x1", grid_n=50, n_samples=300)
    assert np.all(df["lo"] <= df["effect"])
    assert np.all(df["effect"] <= df["hi"])


# ---------------------------------------------------------------------- #
# plot (basic smoke; we don't render visually)                            #
# ---------------------------------------------------------------------- #


def test_plot_single_returns_axes(gaussian_gam):
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = gaussian_gam.plot("x0", grid_n=30)
    assert ax is not None
    plt.close("all")


def test_plot_all_returns_figure(gaussian_gam):
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = gaussian_gam.plot(grid_n=30)
    assert len(fig.axes) == 2  # one per smooth
    plt.close("all")


def test_plot_unknown_predictor_raises(gaussian_gam):
    pytest.importorskip("matplotlib")
    with pytest.raises(KeyError):
        gaussian_gam.plot("does_not_exist")


# ---------------------------------------------------------------------- #
# summary                                                                #
# ---------------------------------------------------------------------- #


def test_summary_returns_dataclass(gaussian_gam):
    s = gaussian_gam.summary()
    assert isinstance(s, GamSummary)
    assert s.family == "gaussian"
    assert s.link == "identity"
    assert s.n_obs == 400


def test_summary_smooths_table_columns(gaussian_gam):
    s = gaussian_gam.summary()
    assert list(s.smooths.columns) == ["predictor", "k", "edf", "lambda"]
    assert set(s.smooths["predictor"]) == {"x0", "x1"}


def test_summary_repr_includes_key_fields(gaussian_gam):
    r = repr(gaussian_gam.summary())
    assert "family=gaussian" in r
    assert "link=identity" in r
    assert "x0" in r
    assert "x1" in r
    assert "edf" in r


def test_summary_r_squared_in_reasonable_range(gaussian_gam):
    """Synthetic data has noise σ=0.1; R² should be high."""
    s = gaussian_gam.summary()
    assert 0.8 < s.r_squared < 1.0, f"R²={s.r_squared}"


def test_summary_for_binomial_skips_r_squared(binomial_gam):
    s = binomial_gam.summary()
    assert np.isnan(s.r_squared)
    assert s.family == "binomial"
    assert s.link == "logit"


# ---------------------------------------------------------------------- #
# predict_proba (binomial only)                                          #
# ---------------------------------------------------------------------- #


def test_predict_proba_shape_and_sum(binomial_gam, binomial_data):
    X, _ = binomial_data
    proba = binomial_gam.predict_proba(X[:50])
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all(proba >= 0)
    assert np.all(proba <= 1)


def test_predict_proba_class1_equals_predict(binomial_gam, binomial_data):
    """Column 1 must be predict(X); column 0 is 1 - that."""
    X, _ = binomial_data
    proba = binomial_gam.predict_proba(X[:50])
    np.testing.assert_allclose(proba[:, 1], binomial_gam.predict(X[:50]), rtol=1e-10)


def test_predict_proba_non_binomial_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(NotImplementedError, match="binomial"):
        gaussian_gam.predict_proba(X[:10])


# ---------------------------------------------------------------------- #
# score                                                                  #
# ---------------------------------------------------------------------- #


def test_gaussian_score_is_r_squared(gaussian_gam, gaussian_data):
    X, y = gaussian_data
    s = gaussian_gam.score(X, y)
    assert 0.8 < s < 1.0


def test_score_perfect_when_train_equals_eval(gaussian_gam, gaussian_data):
    """On train data, R² should equal what the manual computation gives."""
    X, y = gaussian_data
    preds = gaussian_gam.predict(X)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    expected = 1.0 - ss_res / ss_tot
    assert gaussian_gam.score(X, y) == pytest.approx(expected, rel=1e-10)


def test_binomial_score_is_accuracy(binomial_gam, binomial_data):
    """For binomial, score returns 0/1 accuracy at threshold 0.5."""
    X, y = binomial_data
    s = binomial_gam.score(X, y)
    assert 0.0 <= s <= 1.0
    # Manual accuracy
    preds = binomial_gam.predict(X)
    expected = float(np.mean(((preds > 0.5).astype(float) == y).astype(float)))
    assert s == pytest.approx(expected)
