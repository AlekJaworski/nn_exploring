"""
Canonical battery of parity cases.

Each case is a recipe: seed, sample size, predictor count, basis dims,
family, link, method, plus the data-generating function. The R fixture
generator and the pytest harness both consume this list, so adding a
case is a single edit here.

The data-generating functions are pure Python/numpy. We dump the
realized x_train / y_train into the fixture so R sees identical bytes —
no R-side rng involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------- #
# Data-generating functions                                             #
# --------------------------------------------------------------------- #

def _gen_1d_smooth(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.2, n)
    return x, y


def _gen_1d_wiggly(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1)
    y = (np.sin(4 * np.pi * x.ravel())
         + 0.5 * np.sin(10 * np.pi * x.ravel())
         + rng.normal(0, 0.15, n))
    return x, y


def _gen_1d_near_linear(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1)
    y = 0.5 + 1.3 * x.ravel() + rng.normal(0, 0.1, n)
    return x, y


def _gen_2d_additive(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, (n, 2))
    y = (np.sin(2 * np.pi * x[:, 0])
         + np.cos(2 * np.pi * x[:, 1])
         + rng.normal(0, 0.2, n))
    return x, y


def _gen_4d_mixed(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The 4D case currently failing Bar C. Two wiggly + one quadratic + one near-linear."""
    x = rng.uniform(0.0, 1.0, (n, 4))
    y = (np.sin(2 * np.pi * x[:, 0])
         + 0.5 * np.cos(3 * np.pi * x[:, 1])
         + 0.3 * (x[:, 2] ** 2)
         + 0.2 * x[:, 3]
         + rng.normal(0, 0.2, n))
    return x, y


def _gen_binomial_logit(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, (n, 2))
    eta = 2.0 * np.sin(2 * np.pi * x[:, 0]) + 1.5 * (x[:, 1] - 0.5)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p).astype(float)
    return x, y


def _gen_poisson_log(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, (n, 2))
    eta = 1.5 + 0.8 * np.sin(2 * np.pi * x[:, 0]) + 0.5 * x[:, 1]
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return x, y


def _gen_gamma_log(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0.0, 1.0, (n, 2))
    eta = 1.0 + 0.5 * np.sin(2 * np.pi * x[:, 0]) + 0.3 * x[:, 1]
    mu = np.exp(eta)
    shape = 4.0  # gamma shape parameter
    y = rng.gamma(shape, mu / shape, n)
    return x, y


# --------------------------------------------------------------------- #
# Case definition                                                        #
# --------------------------------------------------------------------- #

@dataclass
class Case:
    name: str
    description: str
    seed: int
    n: int
    d: int
    k: list[int]
    bs: list[str]
    family: str
    link: str
    method: str
    generator: Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]
    n_test: int = 100      # held-out, in-range
    n_extrap: int = 20     # extrapolation, just outside [x_min, x_max]
    weights: Optional[np.ndarray] = None    # currently unused; placeholder

    def realize(self) -> dict:
        """Materialize x_train, y_train, x_test, x_extrap. Deterministic via seed."""
        rng = np.random.default_rng(self.seed)
        x_train, y_train = self.generator(rng, self.n)

        # Held-out: same distribution as training
        x_test, _ = self.generator(rng, self.n_test)

        # Extrapolation: just outside the training x range, per dimension
        x_min = x_train.min(axis=0)
        x_max = x_train.max(axis=0)
        margin = 0.1 * (x_max - x_min)
        # Half on the low side, half on the high side, evenly per dim
        n_per_side = self.n_extrap // 2
        low = rng.uniform(x_min - margin, x_min - 1e-3, (n_per_side, self.d))
        high = rng.uniform(x_max + 1e-3, x_max + margin, (self.n_extrap - n_per_side, self.d))
        x_extrap = np.vstack([low, high])

        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "n": self.n,
            "d": self.d,
            "k": self.k,
            "bs": self.bs,
            "family": self.family,
            "link": self.link,
            "method": self.method,
            "weights": None if self.weights is None else self.weights.tolist(),
            "x_train": x_train.tolist(),
            "y_train": y_train.tolist(),
            "x_test": x_test.tolist(),
            "x_extrap": x_extrap.tolist(),
        }


# --------------------------------------------------------------------- #
# The battery                                                            #
# --------------------------------------------------------------------- #

CASES: list[Case] = [
    # ---- 1D Gaussian ---------------------------------------------------
    Case(
        name="1d_gaussian_smooth_n500_k10_cr",
        description="Smooth 1D sin curve, Gaussian noise, k=10 cr-spline",
        seed=42, n=500, d=1, k=[10], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_smooth,
    ),
    Case(
        name="1d_gaussian_smooth_n500_k20_bs",
        description="Same data, k=20 B-spline (overparameterized)",
        seed=42, n=500, d=1, k=[20], bs=["bs"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_smooth,
    ),
    Case(
        name="1d_gaussian_wiggly_n500_k20_cr",
        description="Higher-frequency 1D, k=20 cr",
        seed=43, n=500, d=1, k=[20], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_wiggly,
    ),
    Case(
        name="1d_gaussian_near_linear_n500_k10_cr",
        description="Linear-with-noise; tests very-high-lambda regime",
        seed=44, n=500, d=1, k=[10], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_near_linear,
    ),
    Case(
        name="1d_gaussian_smooth_n100_k10_cr",
        description="Small-n version of smooth 1D",
        seed=42, n=100, d=1, k=[10], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_smooth,
    ),

    # ---- 2D additive Gaussian -----------------------------------------
    Case(
        name="2d_gaussian_additive_n500_k10_cr",
        description="2D additive sin/cos, k=10 cr per smooth",
        seed=42, n=500, d=2, k=[10, 10], bs=["cr", "cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_2d_additive,
    ),
    Case(
        name="2d_gaussian_additive_n2000_k15_cr",
        description="Larger n, finer basis",
        seed=42, n=2000, d=2, k=[15, 15], bs=["cr", "cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_2d_additive,
    ),

    # ---- 4D mixed Gaussian (the historically-failing Bar C case) -------
    Case(
        name="4d_gaussian_mixed_n1000_k10_cr",
        description="4D additive mixing wiggly, smooth, quadratic, near-linear",
        seed=42, n=1000, d=4, k=[10, 10, 10, 10], bs=["cr"] * 4,
        family="gaussian", link="identity", method="REML",
        generator=_gen_4d_mixed,
    ),

    # ---- Non-Gaussian families ----------------------------------------
    Case(
        name="2d_binomial_logit_n1000_k10_cr",
        description="Binomial logit, 2D additive",
        seed=42, n=1000, d=2, k=[10, 10], bs=["cr", "cr"],
        family="binomial", link="logit", method="REML",
        generator=_gen_binomial_logit,
    ),
    Case(
        name="2d_poisson_log_n1000_k10_cr",
        description="Poisson log, 2D additive",
        seed=42, n=1000, d=2, k=[10, 10], bs=["cr", "cr"],
        family="poisson", link="log", method="REML",
        generator=_gen_poisson_log,
    ),
    Case(
        name="2d_gamma_log_n1000_k10_cr",
        description="Gamma log, 2D additive",
        seed=42, n=1000, d=2, k=[10, 10], bs=["cr", "cr"],
        family="Gamma", link="log", method="REML",
        generator=_gen_gamma_log,
    ),
]


CASES_BY_NAME: dict[str, Case] = {c.name: c for c in CASES}


def all_cases() -> list[Case]:
    return list(CASES)
