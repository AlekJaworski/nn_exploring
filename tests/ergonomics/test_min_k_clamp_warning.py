"""`Gam(k_default=2)` previously got silently bumped to k=3 via min_k floor.

mgcv has no such floor — `k=2` there yields a genuinely linear smooth (1
effective dim after sum-to-zero centering). Our `min_k=3` default produces
a 2-effective-dim smooth that can fit a parabola, which surprised users.

We can't lower the default `min_k` to 2 because the Rust core panics on
k=2 (pre-existing crash, separate issue). The next-best fix is to make
the silent clamp visible — emit a `UserWarning` whenever `_resolve_ks`
nudges the user's request.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mgcv_rust import Gam


def _data():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)
    y = 0.5 + 1.3 * x + rng.normal(0, 0.05, 100)
    return x.reshape(-1, 1), y


def test_warns_when_min_k_bumps_k_default():
    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = Gam(k_default=2).fit(X, y)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("bumped from 2 to 3" in m for m in msgs), (
        f"expected min_k bump warning; got {msgs}"
    )
    assert gam.k_[0] == 3  # silently clamped value


def test_warns_when_min_k_bumps_term_k_mapping():
    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gam = Gam(predictors=("x0",), term_k_mapping={"x0": 2}).fit(X, y)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("bumped from 2 to 3" in m for m in msgs), (
        f"expected per-term bump warning; got {msgs}"
    )


def test_silent_when_no_clamp():
    X, y = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Gam(k_default=10).fit(X, y)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, UserWarning) and "bumped from" in str(w.message)]
    assert msgs == [], f"unexpected bump warning at k_default=10: {msgs}"


def test_warns_when_cap_at_n_unique():
    """When n_unique − 1 < requested k, the cap clamps downward; should warn."""
    rng = np.random.default_rng(0)
    # Only 5 distinct x values → n_unique - 1 = 4, so k=20 gets capped to 4.
    x = np.repeat([0.0, 0.25, 0.5, 0.75, 1.0], 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 250)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Gam(k_default=20).fit(x.reshape(-1, 1), y)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("capped from 20 to 4" in m for m in msgs), (
        f"expected n_unique cap warning; got {msgs}"
    )
