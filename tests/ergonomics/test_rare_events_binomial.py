"""Warn when binomial data is too rare-events for the requested smooth.

The textbook rule is ≥ 5 minority-class events per smooth parameter.
Below that, REML's penalty selection can pick λ ≈ 0 and the smooth
saturates near the few events (verified empirically — k=10 with one
positive in 701 gives a 38% probability spike). We warn at fit time so
the failure mode is at least visible to the caller.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mgcv_rust import Gam


def _rare_data(n_pos: int, n_total: int = 701, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n_total)
    y = np.zeros(n_total)
    y[:n_pos] = 1.0
    return x.reshape(-1, 1), y


@pytest.mark.parametrize("n_pos", [1, 2, 3, 5])
def test_warns_when_events_below_5_per_param(n_pos: int):
    X, y = _rare_data(n_pos)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Gam(family="binomial").fit(X, y)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("events per parameter" in m for m in msgs), (
        f"expected rare-events warning at n_pos={n_pos}; got {msgs}"
    )


def test_silent_when_events_above_threshold():
    """50 events with k=10 → 50/9 = 5.55 EPP → no warning."""
    X, y = _rare_data(50)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Gam(family="binomial").fit(X, y)
    msgs = [str(w.message) for w in caught if "events per parameter" in str(w.message)]
    assert msgs == [], f"unexpected rare-events warning: {msgs}"


def test_gaussian_never_warns():
    """Guard is binomial-only; Gaussian fits are unaffected."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, 50).reshape(-1, 1)
    y = rng.standard_normal(50)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Gam(family="gaussian").fit(X, y)
    msgs = [str(w.message) for w in caught if "events per parameter" in str(w.message)]
    assert msgs == []


def test_lower_k_clears_warning():
    """Same data, smaller k_default → more events per parameter → no warning."""
    X, y = _rare_data(10)  # 10 events; warns at default k=10 (10/9 ≈ 1.1 EPP)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # k_default=3 → 2 params per smooth → 10/2 = 5 EPP, threshold met
        Gam(family="binomial", k_default=3).fit(X, y)
    msgs = [str(w.message) for w in caught if "events per parameter" in str(w.message)]
    assert msgs == [], f"lowering k should have cleared the warning; got {msgs}"
