"""Ergo-N5: `predict(scale=)` + `predict(type='terms')` tests.

Covers:
- ``scale='response'`` equals the default ``predict(X)``.
- ``scale='link'`` equals response for identity link, and equals the
  logit of the response for binomial.
- ``type='terms'`` returns a :class:`TermContributions` with
  link-scale, sum-to-zero (on train) contributions and a response-scale
  ``total`` matching ``predict(X)``.
- Subset views (``gam[name]``) filter the contributions DataFrame.
- ``scale='deviation'`` is only valid on subset views and is centered
  on training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam, TermContributions


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(2026)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(0, 5, n)
    y = np.sin(x0) + 0.3 * (x1 - 2.5) ** 2 + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"x0": x0, "x1": x1})
    return X, y


@pytest.fixture(scope="module")
def gaussian_gam(gaussian_data):
    X, y = gaussian_data
    return Gam(family="gaussian").fit(X, y)


@pytest.fixture(scope="module")
def binomial_data():
    rng = np.random.default_rng(7)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-1, 1, n)
    eta = 0.8 * x0 - 1.2 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(float)
    X = pd.DataFrame({"x0": x0, "x1": x1})
    return X, y


@pytest.fixture(scope="module")
def binomial_gam(binomial_data):
    X, y = binomial_data
    return Gam(family="binomial").fit(X, y)


# ---------------------------------------------------------------------- #
# scale=                                                                 #
# ---------------------------------------------------------------------- #


def test_predict_scale_response_equals_default(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    np.testing.assert_allclose(
        gaussian_gam.predict(X),
        gaussian_gam.predict(X, scale="response"),
        rtol=1e-12,
    )


def test_predict_scale_link_for_identity_equals_response(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    np.testing.assert_allclose(
        gaussian_gam.predict(X, scale="link"),
        gaussian_gam.predict(X, scale="response"),
        rtol=1e-12,
    )


def test_predict_scale_link_for_binomial_is_logit(binomial_gam, binomial_data):
    X, _ = binomial_data
    link_preds = binomial_gam.predict(X, scale="link")
    response_preds = binomial_gam.predict(X, scale="response")
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-link_preds)),
        response_preds,
        rtol=1e-10,
    )


def test_predict_invalid_scale_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="scale must be"):
        gaussian_gam.predict(X, scale="bogus")


def test_predict_invalid_type_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="type must be"):
        gaussian_gam.predict(X, type="bogus")


# ---------------------------------------------------------------------- #
# type='terms'                                                           #
# ---------------------------------------------------------------------- #


def test_predict_terms_returns_dataclass(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    result = gaussian_gam.predict(X, type="terms")
    assert isinstance(result, TermContributions)
    assert result.intercept == pytest.approx(gaussian_gam.intercept_)
    assert list(result.contributions.columns) == list(gaussian_gam.feature_names_in_)
    assert result.contributions.shape == (len(X), gaussian_gam.n_features_in_)
    assert result.total.shape == (len(X),)


def test_predict_terms_total_matches_predict(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    result = gaussian_gam.predict(X, type="terms")
    np.testing.assert_allclose(result.total, gaussian_gam.predict(X), rtol=1e-10)


def test_predict_terms_binomial_total_matches_predict(binomial_gam, binomial_data):
    X, _ = binomial_data
    result = binomial_gam.predict(X, type="terms")
    np.testing.assert_allclose(result.total, binomial_gam.predict(X), rtol=1e-10)


def test_predict_terms_contributions_centered_on_train(gaussian_gam, gaussian_data):
    """Sum-to-zero invariant: each smooth's contribution averages to 0 on training X."""
    X, _ = gaussian_data
    result = gaussian_gam.predict(X, type="terms")
    for col in result.contributions.columns:
        assert abs(result.contributions[col].mean()) < 1e-8, (
            f"Contribution of {col!r} not centered on train: "
            f"mean={result.contributions[col].mean():.3e}"
        )


def test_predict_terms_intercept_plus_sum_equals_link(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    result = gaussian_gam.predict(X, type="terms")
    link = gaussian_gam.predict(X, scale="link")
    np.testing.assert_allclose(
        result.intercept + result.contributions.sum(axis=1).to_numpy(),
        link,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------- #
# Subset interactions                                                    #
# ---------------------------------------------------------------------- #


def test_subset_predict_terms_columns(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    sub_terms = gaussian_gam[["x0"]].predict(X, type="terms")
    assert list(sub_terms.contributions.columns) == ["x0"]


def test_subset_predict_response_applies_inv_link(binomial_gam, binomial_data):
    """Regression: subset views used to return link-scale silently.

    After N5, subset ``predict()`` default is response scale (inv-linked)
    so it matches the full view's contract.
    """
    X, _ = binomial_data
    sub = binomial_gam[["x0", "x1", "__constant__"]]
    np.testing.assert_allclose(
        sub.predict(X),
        binomial_gam.predict(X),
        rtol=1e-10,
    )


def test_subset_predict_link_matches_lp_at_coef(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    sub = gaussian_gam[["x0"]]
    # Manual: lp with non-x0 columns + intercept zeroed → @ coef.
    lp = gaussian_gam.evaluate_lpmatrix(X)
    coef = gaussian_gam.coef_
    term_indices = gaussian_gam._native.get_term_indices()
    # Find x0's block.
    x0_block = next((f, l) for (n, f, l) in term_indices if n == "x0")
    eta_manual = lp[:, x0_block[0] : x0_block[1] + 1] @ coef[x0_block[0] : x0_block[1] + 1]
    np.testing.assert_allclose(sub.predict(X, scale="link"), eta_manual, rtol=1e-12)


def test_deviation_scale_only_on_subset(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="subset views"):
        gaussian_gam.predict(X, scale="deviation")


def test_subset_deviation_centered_on_train(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    dev = gaussian_gam[["x0"]].predict(X, scale="deviation")
    assert abs(dev.mean()) < 1e-8, f"deviation not centered: mean={dev.mean():.3e}"


def test_subset_deviation_zeroes_intercept(gaussian_gam, gaussian_data):
    """scale='deviation' on a subset that *includes* the intercept must
    still zero the intercept column — deviation = smooths only."""
    X, _ = gaussian_data
    dev_with_int = gaussian_gam[["__constant__", "x0"]].predict(X, scale="deviation")
    dev_without_int = gaussian_gam[["x0"]].predict(X, scale="deviation")
    np.testing.assert_allclose(dev_with_int, dev_without_int, rtol=1e-12)
