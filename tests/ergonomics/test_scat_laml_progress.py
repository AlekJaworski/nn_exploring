"""Stringent partial-result tests for the scat (TDist) LAML port.

These tests pin down what each stage of the LAML implementation must
achieve — written xfail-by-default so they fail loudly when each piece
is genuinely landed (vs being a no-op masquerading as progress).

Stages covered:

  1. **df-dependent REML score**: REML(df) must have a non-zero gradient.
     Currently the Gaussian-proxy ls/deviance pair gives dREML/d(log df)
     = 0 structurally — the FD Newton step trivially can't move.

  2. **Heavy-tail preference**: with t(4) noise, REML must prefer a
     finite df rather than monotonically pushing toward Gaussian.

  3. **Joint (df, σ²) Newton convergence**: rust's converged (df, σ²)
     must match mgcv::scat to within ~10%.

  4. **REML parity**: rust's REML score at convergence must match
     mgcv::scat to within a small absolute tolerance.

  5. **λ parity**: log10(λ_rust / λ_mgcv) within 0.2 (factor of ~1.6×)
     per smooth.

The current production path uses a Gaussian-proxy REML (saturated_log_lik
~ -n/2·log(2πσ²) and deviance ~ (y-μ)²) which gives correct PREDICTIONS
(corr > 0.9999 vs mgcv) and reasonable σ² via method-of-moments, but
misses by ~0.78 log10 on λ and +75 on REML score. Properly closing the
gap is the gam.fit5-style LAML port.

Tests reference data: n=1500, d=8, k=10 each, τ-quantile-style true
signal `Σ_j sin(2π(j+1)/3 · x_j)` with t(4) heavy-tail noise scale 0.3.
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


def _make_data(n=1500, d=8, seed=1, df_true=4.0, noise_scale=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, d))
    eta = np.zeros(n)
    for j in range(d):
        eta += np.sin(2 * np.pi * (j + 1) / 3 * x[:, j])
    rng2 = np.random.default_rng(seed + 100)
    y = eta + rng2.standard_t(df=df_true, size=n) * noise_scale
    return x, y, eta


def _mgcv_scat(X, y, k):
    """Run mgcv::scat, return dict with reml/theta/lambdas/predictions."""
    if not _rscript_available():
        return None
    d = X.shape[1]
    with tempfile.TemporaryDirectory() as td:
        np.savetxt(f"{td}/x.csv", X, delimiter=",")
        np.savetxt(f"{td}/y.csv", y)
        rhs = " + ".join(f's(x{j+1}, k={k[j]}, bs="cr")' for j in range(d))
        rscript = f"""
suppressMessages({{ library(mgcv); library(jsonlite) }})
x <- as.matrix(read.csv("{td}/x.csv", header=FALSE))
y <- as.numeric(read.csv("{td}/y.csv", header=FALSE)$V1)
df <- data.frame(x); names(df) <- paste0('x', 1:{d}); df$y <- y
fit <- gam(y ~ {rhs}, data=df, family=scat(link="identity"), method="REML")
out <- list(
  pred=as.numeric(predict(fit, type='response')),
  lambdas=as.numeric(fit$sp),
  theta=as.numeric(fit$family$getTheta(TRUE)),
  reml=fit$gcv.ubre
)
cat(toJSON(out, auto_unbox=TRUE), file="{td}/out.json")
"""
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e", rscript],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return None
        with open(f"{td}/out.json") as f:
            return json.load(f)


# ------------------------------------------------------------------ #
# Stage 1: REML must have a real gradient w.r.t. df                   #
# ------------------------------------------------------------------ #


def test_scat_reml_has_df_gradient():
    """REML(df) at fixed (β, λ, σ²) must have a non-zero finite-difference gradient.

    Closed by phase 1+2 of the gam.fit5 LAML port: estimate_phi_mgcv now
    returns the enum-stored σ² (synced from PIRLS), and σ² is MLE inside
    fit_pirls_tdist. Since the t-IRLS weight depends on df, σ² depends
    on df, and the `-Mp/2·log(2πφ)` term in the REML score picks up a
    real df gradient — even with Gaussian-proxy ls/deviance still in place.
    Tighter parity (Stages 2-5) needs the full t-form ls/deviance + outer
    Newton on log(df), task #15 territory.
    """
    X, y, _ = _make_data(n=600, d=4, seed=42)
    k = [10] * 4
    grid = []
    for df_v in [3.0, 5.0, 8.0, 12.0]:
        g = GAM("t-dist", df=float(df_v))
        g.fit(X, y, k=k, method="REML", bs="cr")
        grid.append((df_v, g.get_reml_score()))

    diffs = [grid[i + 1][1] - grid[i][1] for i in range(len(grid) - 1)]
    # All differences must be non-trivial — at least one above 1.0
    max_step = max(abs(d) for d in diffs)
    assert max_step > 1.0, (
        f"REML score barely changes with df ({diffs}); the Gaussian-proxy "
        f"formulation makes the FD gradient ~0."
    )


# ------------------------------------------------------------------ #
# Stage 2: REML must prefer finite df on heavy-tailed data            #
# ------------------------------------------------------------------ #


@pytest.mark.xfail(
    reason=(
        "Stage 2 of LAML port. Even with df-dependent REML (Stage 1), "
        "without proper joint optimisation REML prefers df→∞ on heavy-"
        "tailed data — verified in the LAML probe with t(4) noise. "
        "mgcv lands at df≈4.5; a working LAML must also land in "
        "the heavy-tail neighbourhood (df < 20)."
    ),
    strict=True,
)
def test_scat_reml_prefers_finite_df_on_heavy_tail():
    """On t(4) data, the REML-minimising df must be finite (not Gaussian)."""
    X, y, _ = _make_data(n=1500, d=4, seed=11, df_true=4.0, noise_scale=0.3)
    k = [10] * 4
    df_grid = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 40.0]
    scores = []
    for df_v in df_grid:
        g = GAM("t-dist", df=float(df_v))
        g.fit(X, y, k=k, method="REML", bs="cr")
        scores.append(g.get_reml_score())
    best_idx = int(np.argmin(scores))
    assert df_grid[best_idx] < 20.0, (
        f"REML-min df is {df_grid[best_idx]} — should land at the heavy-tail "
        f"end (<20) on t(4) data. Scores: {dict(zip(df_grid, scores))}"
    )


# ------------------------------------------------------------------ #
# Stage 3: profile-df Newton must converge near mgcv's df             #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _rscript_available(), reason="Rscript unavailable")
@pytest.mark.xfail(
    reason=(
        "Stage 3 of LAML port. Even with Stages 1+2 in place, the M+1 "
        "outer Newton on log df also needs σ² profiled jointly (mgcv "
        "tracks θ=[df, log σ]). Without joint (df, σ²) Newton, the "
        "df estimate drifts. Acceptance: rust's converged df within "
        "20% of mgcv's."
    ),
    strict=True,
)
def test_scat_profile_df_matches_mgcv():
    """Rust's converged df (when profiled) must agree with mgcv::scat to ~20%."""
    X, y, _ = _make_data(n=1500, d=4, seed=11)
    k = [10] * 4
    g = GAM("t-dist")  # auto-profile df
    g.fit(X, y, k=k, method="REML", bs="cr")

    # Need a way to read rust's converged df — currently not exposed
    # via Python. The test will need a `g.get_family_params()` accessor
    # added when LAML ships. For now, mark the gap by xfailing.
    pytest.fail("rust does not expose converged df via Python API yet")


# ------------------------------------------------------------------ #
# Stage 4: REML score parity with mgcv                                #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _rscript_available(), reason="Rscript unavailable")
@pytest.mark.xfail(
    reason=(
        "Stage 4 of LAML port — final REML parity. Currently rust's "
        "REML at convergence is +75 above mgcv's on the n=1500 "
        "benchmark (rust 958, mgcv 883). A working LAML must close "
        "this absolute gap to ~5 points."
    ),
    strict=True,
)
def test_scat_reml_parity_vs_mgcv():
    """Rust REML must be within 5 absolute units of mgcv's at the same data."""
    X, y, _ = _make_data(n=1500, d=4, seed=11)
    k = [10] * 4

    g = GAM("t-dist")
    g.fit(X, y, k=k, method="REML", bs="cr")
    rust_reml = g.get_reml_score()

    r_out = _mgcv_scat(X, y, k)
    if r_out is None:
        pytest.skip("mgcv::scat fit failed")
    gap = rust_reml - r_out["reml"]
    assert abs(gap) < 5.0, (
        f"REML parity gap is {gap:+.2f} (rust={rust_reml:.2f} "
        f"mgcv={r_out['reml']:.2f}); LAML must close to within 5."
    )


# ------------------------------------------------------------------ #
# Stage 5: λ parity                                                   #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _rscript_available(), reason="Rscript unavailable")
@pytest.mark.xfail(
    reason=(
        "Stage 5 of LAML port — λ parity. Currently rust's λ values "
        "are systematically ~6× smaller than mgcv's (~0.78 log10) on "
        "the n=1500 benchmark — a sign of σ² estimation disagreement. "
        "After joint (df, σ²) Newton, log10(λ_rust/λ_mgcv) must be "
        "within 0.2 (factor 1.6×) per smooth."
    ),
    strict=True,
)
def test_scat_lambda_parity_vs_mgcv():
    """log10(λ_rust / λ_mgcv) within 0.2 per smooth."""
    X, y, _ = _make_data(n=1500, d=4, seed=11)
    k = [10] * 4

    g = GAM("t-dist")
    g.fit(X, y, k=k, method="REML", bs="cr")
    lam_rust = g.get_all_lambdas()

    r_out = _mgcv_scat(X, y, k)
    if r_out is None:
        pytest.skip("mgcv::scat fit failed")
    lam_mgcv = np.asarray(r_out["lambdas"])
    log_diff = np.log10(np.maximum(lam_rust, 1e-12)) - np.log10(np.maximum(lam_mgcv, 1e-12))
    max_log_diff = float(np.max(np.abs(log_diff)))
    assert max_log_diff < 0.2, (
        f"max |log10(λ_rust/λ_mgcv)| = {max_log_diff:.3f}; "
        f"per-smooth diffs: {log_diff.tolist()}"
    )


# ------------------------------------------------------------------ #
# Sanity: the current proxy at least gets predictions right           #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _rscript_available(), reason="Rscript unavailable")
def test_scat_predictions_already_match_mgcv():
    """Current proxy formulation: predictions match mgcv to corr > 0.999.

    This passes today — it's the baseline showing that the LAML gap is
    structural to the score formula, not a fitness-of-β issue.
    """
    X, y, _ = _make_data(n=1500, d=4, seed=11)
    k = [10] * 4

    g = GAM("t-dist")
    g.fit(X, y, k=k, method="REML", bs="cr")
    pred_rust = g.predict(X)

    r_out = _mgcv_scat(X, y, k)
    if r_out is None:
        pytest.skip("mgcv::scat fit failed")
    pred_mgcv = np.asarray(r_out["pred"])

    if pred_rust.std() > 1e-9 and pred_mgcv.std() > 1e-9:
        corr = float(np.corrcoef(pred_rust, pred_mgcv)[0, 1])
        assert corr > 0.999, (
            f"Even the proxy should give corr > 0.999; got {corr:.6f}. "
            f"Something more fundamental is broken."
        )
