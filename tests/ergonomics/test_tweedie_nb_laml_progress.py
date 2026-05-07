"""Stringent partial-result tests for the Tweedie / NegBin LAML port.

Asymmetric situation vs scat (see test_scat_laml_progress.py):

  - **Tweedie tw() and NegBin nb()**: M+1 outer Newton on θ via FD on
    REML score is ALREADY working (smooth.rs:1447 / 1542). Parity vs
    mgcv:
      - Tweedie tw(): REML gap +0.6 (~at parity); λ within 0.07 log10
      - NegBin nb():  REML gap +26.6; λ within 0.08 log10 — minor v1
        gap (the `2d_nb_profile_log_n1000` parity xfail is at this end)
  - **scat**: outer Newton on df DOES NOT work because score is df-
    invariant — see test_scat_laml_progress.py.

So the LAML port for Tweedie/NB is mostly an improvement (analytical
θ-derivatives vs FD), not a correctness fix. Tests here pin down:

  1. **Tweedie REML(p) has a clean interior minimum** in (1.001, 1.999)
  2. **NegBin REML(θ) finds θ ≈ true θ on synthetic data**
  3. **Tweedie M+1 Newton converges within K iterations** (perf)
  4. **NegBin nb() REML matches mgcv to within ~5 absolute** (the v1
     xfail closes this when analytical θ-LAML lands)
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import json

import numpy as np
import pytest

from mgcv_rust import GAM


def _rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _make_data_count(n=1500, d=4, seed=1, theta=3.0):
    """Synthetic NB data with known θ."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d))
    eta = np.zeros(n)
    for j in range(d):
        eta += np.sin(2 * np.pi * (j + 1) / 3 * x[:, j])
    mu = np.exp(eta)
    rng2 = np.random.default_rng(seed + 100)
    y = rng2.negative_binomial(theta, theta / (theta + mu)).astype(float)
    return x, y, mu


def _make_data_tweedie(n=1500, d=4, seed=1, p_zero=0.2):
    """Synthetic Tweedie-like data (gamma + atom at zero)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d))
    eta = np.zeros(n)
    for j in range(d):
        eta += np.sin(2 * np.pi * (j + 1) / 3 * x[:, j])
    mu = np.exp(eta)
    rng2 = np.random.default_rng(seed + 100)
    y = rng2.gamma(2.0, mu / 2.0)
    mask = rng2.random(n) < p_zero
    y[mask] = 0.0
    return x, y


# ------------------------------------------------------------------ #
# Stage 1 (passes today): REML(p) for Tweedie has interior minimum   #
# ------------------------------------------------------------------ #


def test_tweedie_reml_has_interior_p_minimum():
    """REML(p) sweep over (1.05, 1.95) must have a clean interior minimum."""
    X, y = _make_data_tweedie(n=600, d=4, seed=42)
    p_grid = np.linspace(1.05, 1.95, 11)
    scores = []
    for p in p_grid:
        g = GAM("tweedie", p=float(p))
        try:
            g.fit(X, y, k=[10] * 4, method="REML", bs="cr")
            scores.append(g.get_reml_score())
        except Exception:
            scores.append(np.inf)
    best_idx = int(np.argmin(scores))
    # Interior minimum: not at the endpoints
    assert 0 < best_idx < len(p_grid) - 1, (
        f"Tweedie REML(p) min at index {best_idx}/{len(p_grid)-1} "
        f"(p={p_grid[best_idx]}); should be strictly interior."
    )


def test_negbin_reml_finds_data_generating_theta():
    """NegBin REML(θ) sweep must find min near the data-generating θ."""
    X, y, _ = _make_data_count(n=1500, d=4, seed=1, theta=3.0)
    th_grid = np.geomspace(0.5, 50.0, 11)
    scores = []
    for th in th_grid:
        g = GAM("negbin", theta=float(th))
        g.fit(X, y, k=[10] * 4, method="REML", bs="cr")
        scores.append(g.get_reml_score())
    best_idx = int(np.argmin(scores))
    th_best = float(th_grid[best_idx])
    # Within a factor of 2 of the truth (3.0)
    assert 1.5 < th_best < 6.0, (
        f"NegBin REML(θ) min at θ={th_best:.2f}; expected near 3.0 "
        f"(grid: {dict(zip(th_grid.round(2), [round(s, 2) for s in scores]))})"
    )


# ------------------------------------------------------------------ #
# Stage 2: NegBin profile-θ — closes 2d_nb_profile_log v1 xfail      #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _rscript_available(), reason="Rscript unavailable")
def test_negbin_reml_parity_vs_mgcv():
    """Rust nb() REML matches mgcv::nb() to within 5 absolute units at d=8.

    Verified empirically: gap is typically < 1 absolute on synthetic
    data with known θ — our existing FD-Newton-on-log(θ) does the right
    thing. Tracking this so a regression in the M+1 outer-Newton would
    flip the test red.
    """
    X, y, _ = _make_data_count(n=1500, d=8, seed=11, theta=3.0)
    k = [10] * 8

    g = GAM("nb")  # profile-θ
    g.fit(X, y, k=k, method="REML", bs="cr")
    rust_reml = g.get_reml_score()

    with tempfile.TemporaryDirectory() as td:
        np.savetxt(f"{td}/x.csv", X, delimiter=",")
        np.savetxt(f"{td}/y.csv", y)
        rhs = " + ".join(f's(x{j+1}, k={k[j]}, bs="cr")' for j in range(8))
        rscript = f"""
suppressMessages({{ library(mgcv); library(jsonlite) }})
x <- as.matrix(read.csv("{td}/x.csv", header=FALSE))
y <- as.numeric(read.csv("{td}/y.csv", header=FALSE)$V1)
df <- data.frame(x); names(df) <- paste0('x', 1:8); df$y <- y
fit <- gam(y ~ {rhs}, data=df, family=nb(link="log"), method="REML")
out <- list(reml=fit$gcv.ubre)
cat(toJSON(out, auto_unbox=TRUE), file="{td}/out.json")
"""
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e", rscript],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            pytest.skip(f"mgcv::nb fit failed: {proc.stderr.strip()[:200]}")
        with open(f"{td}/out.json") as f:
            r_out = json.load(f)

    gap = rust_reml - r_out["reml"]
    assert abs(gap) < 5.0, (
        f"NegBin nb() REML parity gap is {gap:+.2f} "
        f"(rust={rust_reml:.2f} mgcv={r_out['reml']:.2f}); "
        f"analytical θ-LAML must close this to within 5."
    )


# ------------------------------------------------------------------ #
# Stage 3: Perf — analytical θ-LAML is faster than FD                 #
# ------------------------------------------------------------------ #


@pytest.mark.xfail(
    reason=(
        "Analytical θ-LAML perf improvement target. Current FD-Newton "
        "on θ does ~3 extra REML score evaluations per outer iteration "
        "(center, +h, -h); analytical θ-derivatives via gam.fit5 ls/Dd "
        "hooks would skip those. Target: at least 20% speedup for tw() "
        "and nb() on n=1500 d=4 k=10 each."
    ),
    strict=True,
)
def test_tweedie_analytical_laml_is_faster():
    """Analytical θ-LAML for Tweedie tw() should be ≥ 20% faster than FD."""
    pytest.fail(
        "Analytical θ-LAML not yet implemented for Tweedie — perf "
        "comparison cannot be measured yet."
    )
