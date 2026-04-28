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


# --- Coverage-extending generators (per user ask: multi-d, in/out range) ----


def _gen_1d_sparse_edges(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """1D with most data in [0.3, 0.7] and only a handful of points
    near the boundaries — stresses extrap behaviour just outside the
    sparsely-covered boundary regions."""
    n_dense = int(0.85 * n)
    n_sparse = n - n_dense
    x_dense = rng.uniform(0.3, 0.7, n_dense)
    x_sparse_lo = rng.uniform(0.0, 0.05, n_sparse // 2)
    x_sparse_hi = rng.uniform(0.95, 1.0, n_sparse - n_sparse // 2)
    x = np.concatenate([x_dense, x_sparse_lo, x_sparse_hi]).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.15, n)
    return x, y


def _gen_1d_sigmoid(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """1D logistic shape — flat then sharp transition then flat again.
    Common real-world pattern for dose-response / saturation data."""
    x = rng.uniform(0.0, 1.0, n).reshape(-1, 1)
    z = (x.ravel() - 0.5) * 12
    y = 1.0 / (1.0 + np.exp(-z)) + rng.normal(0, 0.08, n)
    return x, y


def _gen_3d_mixed(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """3D additive: sin / quadratic / near-linear. Subset of the
    historically-failing 4D case to test whether the 4D pathology
    persists at d=3."""
    x = rng.uniform(0.0, 1.0, (n, 3))
    y = (
        np.sin(2 * np.pi * x[:, 0])
        + 0.4 * (x[:, 1] ** 2)
        + 0.25 * x[:, 2]
        + rng.normal(0, 0.2, n)
    )
    return x, y


def _gen_5d_mixed(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """5D: two wiggly + quadratic + near-linear + null (no signal).
    The null dim should converge to high λ — useful for testing
    infinite-λ detection."""
    x = rng.uniform(0.0, 1.0, (n, 5))
    y = (
        np.sin(2 * np.pi * x[:, 0])
        + 0.6 * np.cos(3 * np.pi * x[:, 1])
        + 0.35 * (x[:, 2] ** 2)
        + 0.2 * x[:, 3]
        + 0.0 * x[:, 4]  # null dimension — no signal
        + rng.normal(0, 0.2, n)
    )
    return x, y


def _gen_6d_heatmap_pricing(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror neighbourhoods/heatmap_logic.py::build_pricing_model — the
    location-based pricing model that powers neighborhood heatmaps.

    Production uses `family="t-dist"` (robust regression) which
    mgcv_rust doesn't support yet; we use Gaussian as a proxy. The
    feature shape (lat/lon + days_ago + sqft + beds + year_built) and
    k_default=6 settings are production-exact.

    Features:
      x0  lat (normalised)            ~ N(0, 0.05)   — small geographic spread
      x1  lon (normalised)            ~ N(0, 0.08)
      x2  days_ago_contract_date      ~ U[0, 365]    — ~1-year window
      x3  sqft (normalised log)       ~ N(0, 1)
      x4  beds                        ~ U[1, 6]
      x5  year_built (normalised)     ~ Beta(5, 2)·1 (skewed toward newer)

    Target: log-price-like Gaussian; combines a 2D lat-lon spatial
    surface with monotone size/age effects and date depreciation.
    """
    x0 = rng.normal(0.0, 0.05, n)              # lat offset
    x1 = rng.normal(0.0, 0.08, n)              # lon offset
    x2 = rng.uniform(0.0, 365.0, n)            # days_ago
    x3 = rng.normal(0.0, 1.0, n)               # log sqft
    x4 = rng.uniform(1.0, 6.0, n)              # beds
    x5 = rng.beta(5.0, 2.0, n)                 # year_built normalised

    # Spatial pricing: 2D smooth surface (additive proxy)
    lat_eff = 0.6 * np.sin(20 * x0)
    lon_eff = 0.5 * np.cos(15 * x1)
    days_eff = -0.0008 * x2                    # depreciation
    sqft_eff = 0.4 * x3
    beds_eff = 0.05 * x4
    year_eff = 0.5 * x5

    x = np.column_stack([x0, x1, x2, x3, x4, x5])
    y = (
        12.5  # base log-price
        + lat_eff
        + lon_eff
        + days_eff
        + sqft_eff
        + beds_eff
        + year_eff
        + rng.normal(0, 0.1, n)
    )
    return x, y


def _gen_4d_small_neighbourhood(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Production edge case: small neighborhood with just-enough data.
    Production guards against `n < MIN_PROPERTIES` (~100) but a real
    sparse area can have ~300 properties. Tests robustness when the
    optimizer has to choose between under- and over-smoothing.
    Features: days_ago + sqft + beds + condition (typical minimal set).
    """
    x0 = rng.uniform(0.0, 1825.0, n)           # days
    x1 = rng.normal(0.0, 1.0, n)               # log sqft
    x2 = rng.uniform(1.0, 6.0, n)              # beds
    x3 = rng.uniform(1.0, 10.0, n)             # condition

    days_eff = -0.0001 * x0
    sqft_eff = 0.3 * x1
    beds_eff = 0.05 * x2
    cond_eff = 0.04 * (x3 - 5)

    x = np.column_stack([x0, x1, x2, x3])
    y = (
        days_eff
        + sqft_eff
        + beds_eff
        + cond_eff
        + rng.normal(0, 0.15, n)               # higher noise (small-sample)
    )
    return x, y


def _gen_5d_skewed_features(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Production-realistic feature distributions: log-normal sqft,
    log-normal lot_sqft (long right tail), skewed bedrooms (mode ~3),
    sigmoid-shaped year_built. Tests basis behaviour on skewed inputs
    that real housing data has.
    """
    sqft = rng.lognormal(7.5, 0.4, n)          # ~1800 sqft median, long tail
    lot_sqft = rng.lognormal(9.0, 0.7, n)      # ~8000 sqft median, very long tail
    # Beds: mostly 3, some 2/4, few 1/5/6
    beds = rng.choice(
        [1, 2, 3, 4, 5, 6], size=n,
        p=[0.02, 0.18, 0.45, 0.25, 0.08, 0.02]
    ).astype(float)
    year = rng.uniform(0.0, 1.0, n)            # year_built normalised, uniform
    days = rng.uniform(0.0, 730.0, n)          # 2-year window

    sqft_eff = 0.3 * np.log(sqft / 1800)
    lot_eff = 0.1 * np.log(lot_sqft / 8000)
    beds_eff = 0.06 * beds
    year_eff = 0.4 * year ** 2
    days_eff = -0.0002 * days

    x = np.column_stack([sqft, lot_sqft, beds, year, days])
    y = (
        sqft_eff
        + lot_eff
        + beds_eff
        + year_eff
        + days_eff
        + rng.normal(0, 0.1, n)
    )
    return x, y


def _gen_8d_neighbourhoods_like(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the production GAM call from neighbourhoods/ml_service:
    8 features, ~15k rows, mix of date / quality / condition / size /
    binary-like / monotone / near-linear / mostly-zero. Targets are
    real-estate-like sale-price proxies on the response (Gaussian)
    scale.

    Feature design (matches gam_logic.py:21-50):
      x0  days_ago_for_adjustments    ~ U[0, 1825]   (5-year window)
      x1  quality                     ~ U[1, 10]
      x2  condition                   ~ U[1, 10]
      x3  sqft (normalised)           ~ Beta(2,5)·1
      x4  bedrooms (continuous proxy) ~ U[1, 6]
      x5  bathrooms                   ~ U[1, 5]
      x6  year_built (normalised)     ~ U[0, 1]
      x7  concessions                 ~ 90% zero, 10% U[0, 0.05]

    The target uses depreciation (date), bumpy quality non-linearity,
    monotone size effect, etc. — patterns we know cause distinct λ
    behaviour (some saturating, some not).
    """
    x0 = rng.uniform(0.0, 1825.0, n)            # days
    x1 = rng.uniform(1.0, 10.0, n)              # quality
    x2 = rng.uniform(1.0, 10.0, n)              # condition
    x3 = rng.beta(2.0, 5.0, n)                  # sqft proxy
    x4 = rng.uniform(1.0, 6.0, n)               # beds
    x5 = rng.uniform(1.0, 5.0, n)               # baths
    x6 = rng.uniform(0.0, 1.0, n)               # year_built normalised
    # Concessions: production-realistic — 90% zero, 10% small positive.
    # Our quantile-knot cr-spline handles this correctly after the 4f
    # dedup fix (knots = quantiles of unique values, matching
    # mgcv's smooth.construct.cr.smooth.spec).
    x7 = np.where(rng.uniform(0, 1, n) < 0.9, 0.0, rng.uniform(0.0, 0.05, n))

    # Synthetic price-adjustment-like target
    days_decay = -0.0001 * x0                    # near-linear depreciation
    quality_eff = 0.04 * (x1 ** 2) - 0.08 * x1   # convex
    condition_eff = 0.06 * np.tanh((x2 - 5.0))   # sigmoid-ish
    sqft_eff = 0.4 * x3                          # monotone
    beds_eff = 0.05 * x4                         # mostly linear
    baths_eff = 0.04 * x5
    year_eff = 0.3 * (x6 - 0.5) ** 2             # quadratic dip
    concessions_eff = 2.0 * x7                   # linear, sparse

    x = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7])
    y = (
        days_decay
        + quality_eff
        + condition_eff
        + sqft_eff
        + beds_eff
        + baths_eff
        + year_eff
        + concessions_eff
        + rng.normal(0, 0.1, n)
    )
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

    # ---- Coverage extension: in/out of range, 3D, 5D, edge cases ------
    Case(
        name="1d_gaussian_sparse_edges_n400_k10_cr",
        description="Sparse data near x=0 and x=1 — stresses extrap",
        seed=51, n=400, d=1, k=[10], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_sparse_edges,
    ),
    Case(
        name="1d_gaussian_sigmoid_n300_k10_cr",
        description="Logistic shape — flat-sharp-flat dose-response curve",
        seed=52, n=300, d=1, k=[10], bs=["cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_1d_sigmoid,
    ),
    Case(
        name="3d_gaussian_mixed_n800_k10_cr",
        description="3D additive: sinusoidal + quadratic + near-linear",
        seed=53, n=800, d=3, k=[10, 10, 10], bs=["cr", "cr", "cr"],
        family="gaussian", link="identity", method="REML",
        generator=_gen_3d_mixed,
    ),
    Case(
        name="5d_gaussian_mixed_n1500_k8_cr",
        description="5D mixed with one null dimension — infinite-λ stress",
        seed=54, n=1500, d=5, k=[8, 8, 8, 8, 8], bs=["cr"] * 5,
        family="gaussian", link="identity", method="REML",
        generator=_gen_5d_mixed,
    ),

    # Production-shape big case: mirrors neighbourhoods/ml_service feature
    # adjuster (gam_logic.py:21-77) — 8 features, k mapping matches
    # term_k_mapping for date-like / quality / condition / concessions.
    Case(
        name="8d_neighbourhoods_like_n15000",
        description="Production GAM call: 8 features, 15k rows, mixed k",
        seed=2025,
        n=15000,
        d=8,
        # Order: days, quality, condition, sqft, beds, baths, year, concessions.
        # k mapping matches neighbourhoods/ml_service/gam_logic.py:21-50
        # exactly (term_k_mapping plus k_default=6).
        k=[25, 12, 12, 6, 6, 6, 6, 3],
        bs=["cr"] * 8,
        family="gaussian", link="identity", method="REML",
        generator=_gen_8d_neighbourhoods_like,
    ),
    Case(
        name="6d_heatmap_pricing_n8000",
        description="Heatmap pricing model proxy (lat/lon + features)",
        seed=2026,
        n=8000,
        d=6,
        # k mapping mirrors heatmap_logic.py::build_pricing_model:
        # k_default=6 with k=25 for date (consistent with FeatureAdjuster).
        # lat/lon get k=10 each (typical for spatial smooths).
        k=[10, 10, 25, 6, 6, 6],
        bs=["cr"] * 6,
        family="gaussian", link="identity", method="REML",
        generator=_gen_6d_heatmap_pricing,
    ),
    Case(
        name="4d_small_neighbourhood_n300",
        description="Edge case: sparse neighborhood (n just above MIN_PROPERTIES)",
        seed=2027,
        n=300,
        d=4,
        # Conservative k_default=6 — production uses min_k=2 to fall
        # back when data is too sparse, but n=300 supports k=6 for all.
        k=[12, 6, 6, 6],
        bs=["cr"] * 4,
        family="gaussian", link="identity", method="REML",
        generator=_gen_4d_small_neighbourhood,
    ),
    Case(
        name="5d_skewed_features_n5000",
        description="Production-realistic skewed feature distributions",
        seed=2028,
        n=5000,
        d=5,
        k=[6, 6, 6, 6, 25],  # last is days_ago, gets k=25
        bs=["cr"] * 5,
        family="gaussian", link="identity", method="REML",
        generator=_gen_5d_skewed_features,
    ),
]


CASES_BY_NAME: dict[str, Case] = {c.name: c for c in CASES}


def all_cases() -> list[Case]:
    return list(CASES)
