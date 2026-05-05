"""Shared fixtures for cross-platform usability smoke.

These tests verify the *installed wheel* works end-to-end on every CI
platform — not parity with R, not numerical correctness to N digits, just
"does fit/predict run, return finite numbers of the right shape, and track
the signal at all." Anything tighter belongs in tests/parity/.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260505)


@pytest.fixture(scope="session")
def smooth_1d(rng):
    """y = sin(2πx) + N(0, 0.2), n=400. Used by Gaussian/t-dist."""
    x = rng.uniform(0.0, 1.0, 400).reshape(-1, 1)
    y = np.sin(2.0 * np.pi * x.ravel()) + rng.normal(0.0, 0.2, 400)
    return x, y


@pytest.fixture(scope="session")
def smooth_2d(rng):
    """y = sin(2πx1)·cos(πx2) + N(0, 0.2), n=600. Multi-dim sanity."""
    n = 600
    x = rng.uniform(0.0, 1.0, (n, 2))
    y = np.sin(2.0 * np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) + rng.normal(0.0, 0.2, n)
    return x, y


@pytest.fixture(scope="session")
def binomial_data(rng):
    """Bernoulli with η = 2·sin(2πx); n=500."""
    x = rng.uniform(0.0, 1.0, 500).reshape(-1, 1)
    eta = 2.0 * np.sin(2.0 * np.pi * x.ravel())
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(0.0, 1.0, 500) < p).astype(np.float64)
    return x, y


@pytest.fixture(scope="session")
def poisson_data(rng):
    """Poisson with log μ = 1.5 + sin(2πx); n=500."""
    x = rng.uniform(0.0, 1.0, 500).reshape(-1, 1)
    mu = np.exp(1.5 + np.sin(2.0 * np.pi * x.ravel()))
    y = rng.poisson(mu).astype(np.float64)
    return x, y


@pytest.fixture(scope="session")
def gamma_data(rng):
    """Gamma(shape=2) with log μ = 1.0 + 0.6·sin(2πx); n=500."""
    x = rng.uniform(0.0, 1.0, 500).reshape(-1, 1)
    mu = np.exp(1.0 + 0.6 * np.sin(2.0 * np.pi * x.ravel()))
    shape = 2.0
    y = rng.gamma(shape, mu / shape)
    return x, y
