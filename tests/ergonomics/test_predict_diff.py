"""Ergo-N7: `predict_diff(from_X, to_X)` with paired posterior CI.

Covers:
- Diff equals ``predict(to_X) - predict(from_X)`` for identity-link.
- Optional ``level`` returns ``(diff, lo, hi)`` with the diff inside
  ``[lo, hi]``.
- ``broadcast='from'`` / ``'to'`` work and broadcast as documented.
- Paired CI is **narrower** than the naive difference of two
  independent ``predict_ci`` calls (correlated coef cancellation).
- Non-identity link raises ``NotImplementedError`` with a workaround
  hint.
- Subset views inherit (only the selected smooths contribute to the
  diff).
- Bad broadcast / row-count combos raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mgcv_rust import Gam


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(17)
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
    rng = np.random.default_rng(19)
    n = 400
    x0 = rng.uniform(-2, 2, n)
    x1 = rng.uniform(-1, 1, n)
    eta = 0.5 * x0 - 0.8 * x1
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(float)
    return pd.DataFrame({"x0": x0, "x1": x1}), y


@pytest.fixture(scope="module")
def binomial_gam(binomial_data):
    X, y = binomial_data
    return Gam(family="binomial").fit(X, y)


# ---------------------------------------------------------------------- #
# Basic diff                                                             #
# ---------------------------------------------------------------------- #


def test_diff_no_ci_returns_array(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    out = gaussian_gam.predict_diff(X[:50], X[50:100])
    assert isinstance(out, np.ndarray)
    assert out.shape == (50,)


def test_diff_equals_predict_minus_predict(gaussian_gam, gaussian_data):
    """For identity link, the diff is closed-form: it must equal
    predict(to_X) - predict(from_X)."""
    X, _ = gaussian_data
    diff = gaussian_gam.predict_diff(X[:50], X[50:100])
    expected = gaussian_gam.predict(X[50:100]) - gaussian_gam.predict(X[:50])
    np.testing.assert_allclose(diff, expected, rtol=1e-10)


def test_diff_with_ci_returns_three_tuple(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    out = gaussian_gam.predict_diff(X[:30], X[30:60], level=0.95, n_samples=400)
    assert len(out) == 3
    diff, lo, hi = out
    assert diff.shape == (30,)
    assert lo.shape == (30,)
    assert hi.shape == (30,)
    assert np.all(lo <= diff)
    assert np.all(diff <= hi)


# ---------------------------------------------------------------------- #
# Broadcast                                                              #
# ---------------------------------------------------------------------- #


def test_broadcast_from(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    baseline = X.iloc[[0]]  # single row
    targets = X.iloc[1:11]
    diff = gaussian_gam.predict_diff(baseline, targets, broadcast="from")
    expected = gaussian_gam.predict(targets) - gaussian_gam.predict(baseline)
    np.testing.assert_allclose(diff, expected, rtol=1e-10)


def test_broadcast_to(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    target = X.iloc[[0]]  # single row
    sources = X.iloc[1:11]
    diff = gaussian_gam.predict_diff(sources, target, broadcast="to")
    expected = gaussian_gam.predict(target) - gaussian_gam.predict(sources)
    np.testing.assert_allclose(diff, expected, rtol=1e-10)


def test_broadcast_none_mismatched_rows_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="equal row counts"):
        gaussian_gam.predict_diff(X[:5], X[:10])


def test_broadcast_from_multi_row_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="exactly 1 row"):
        gaussian_gam.predict_diff(X[:3], X[:10], broadcast="from")


def test_broadcast_to_multi_row_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="exactly 1 row"):
        gaussian_gam.predict_diff(X[:10], X[:3], broadcast="to")


def test_invalid_broadcast_raises(gaussian_gam, gaussian_data):
    X, _ = gaussian_data
    with pytest.raises(ValueError, match="broadcast must be"):
        gaussian_gam.predict_diff(X[:5], X[:5], broadcast="bogus")


# ---------------------------------------------------------------------- #
# Paired CI is narrower than naive                                       #
# ---------------------------------------------------------------------- #


def test_paired_ci_narrower_than_naive(gaussian_gam, gaussian_data):
    """The paired posterior CI exploits coef-uncertainty correlation
    between from_X and to_X. Two nearby X rows should have a
    near-zero diff with a tight CI — much tighter than naive
    independent sampling, which adds the from_X and to_X CI widths."""
    X, _ = gaussian_data
    # Pick two very close rows to maximise correlation.
    from_X = X.iloc[[0]]
    to_X = from_X.copy()
    to_X.iloc[0, 0] += 1e-4

    _, lo_paired, hi_paired = gaussian_gam.predict_diff(
        from_X, to_X, broadcast="from", level=0.95, n_samples=500, seed=3
    )
    width_paired = float(hi_paired[0] - lo_paired[0])

    _, lo_a, hi_a = gaussian_gam.predict_ci(from_X, level=0.95, n_samples=500, seed=3)
    _, lo_b, hi_b = gaussian_gam.predict_ci(to_X, level=0.95, n_samples=500, seed=3)
    width_naive = float((hi_a[0] - lo_a[0]) + (hi_b[0] - lo_b[0]))

    assert width_paired < width_naive * 0.1, (
        f"paired width {width_paired:.3e} not << naive sum {width_naive:.3e}"
    )


# ---------------------------------------------------------------------- #
# Non-identity link raises                                               #
# ---------------------------------------------------------------------- #


def test_non_identity_link_raises(binomial_gam, binomial_data):
    X, _ = binomial_data
    with pytest.raises(NotImplementedError, match="identity-link"):
        binomial_gam.predict_diff(X[:5], X[5:10])


# ---------------------------------------------------------------------- #
# Subset views inherit                                                   #
# ---------------------------------------------------------------------- #


def test_subset_view_diff_filters(gaussian_gam, gaussian_data):
    """On gam[['x0']], the diff must equal subset.predict(to) - subset.predict(from)."""
    X, _ = gaussian_data
    sub = gaussian_gam[["x0"]]
    diff = sub.predict_diff(X[:20], X[20:40])
    expected = sub.predict(X[20:40]) - sub.predict(X[:20])
    np.testing.assert_allclose(diff, expected, rtol=1e-10)


def test_subset_intercept_drops_out_of_diff(gaussian_gam, gaussian_data):
    """gam[['x0', '__constant__']].predict_diff equals gam[['x0']].predict_diff
    (the intercept column subtracts itself out)."""
    X, _ = gaussian_data
    diff_with = gaussian_gam[["x0", "__constant__"]].predict_diff(X[:20], X[20:40])
    diff_without = gaussian_gam[["x0"]].predict_diff(X[:20], X[20:40])
    np.testing.assert_allclose(diff_with, diff_without, rtol=1e-12)
