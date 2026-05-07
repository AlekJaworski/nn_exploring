"""Parity test for `fit_quantile_lss` against R's qgam ≥ 1.3 location-scale path.

qgam ≥ 1.3, when given a formula list (location + scale), uses the same
"parametric location-scale model" we implement: a Gaussian-init + per-obs-σ
ELF on the location predictor. The σ_global is tuned via `tuneLearnFast`
cross-validation on qgam's side; we use the qgam heuristic by default and
optionally pinball-CV with `calibrate=True`.

This test verifies that on the canonical heteroskedastic scenario,
`fit_quantile_lss` and `qgam(list(loc_form, scale_form), qu=tau)` agree
on:
    1. Empirical coverage (frac y < ŷ) — both within 0.05 of τ, and within
       0.05 of each other.
    2. Prediction correlation across observations — both recover the same
       smooth τ-quantile surface. corr > 0.97 at τ=0.5; corr > 0.90 at
       extreme τ (the gap at extreme τ traces to qgam's σ_global tuning
       via tuneLearnFast vs our heuristic — calibrate=True tightens it
       to ~0.96 but is asserted separately).
    3. σ_global / log learning rate ballpark — within ~3 orders of magnitude
       (we use a heuristic; qgam tunes via cv, sharper σ ⟹ different λ).

Skipped if Rscript or qgam is not available.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from mgcv_rust import fit_quantile_lss


def _qgam_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("qgam", quietly=TRUE))'],
            capture_output=True, text=True, timeout=15,
        )
        return out.returncode == 0 and "TRUE" in out.stdout
    except Exception:
        return False


@pytest.fixture(scope="module")
def hetero_data():
    """σ(x) = 0.1 + 0.4|x_0|, μ(x) = sin(2 x_0) + 0.5 x_1, n=800.

    Mirrors the seed/shape used in tests/test_quantile_lss_rust.rs and
    tests/ergonomics/test_quantile_lss.py so cal_err numbers are
    comparable across the suite.
    """
    rng = np.random.default_rng(7)
    n = 800
    X = rng.uniform(-1, 1, size=(n, 2))
    sigma = 0.1 + 0.4 * np.abs(X[:, 0])
    mu = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y = mu + sigma * rng.standard_normal(n)
    return X, y, sigma


def _fit_qgam_lss(X: np.ndarray, y: np.ndarray, tau: float) -> dict:
    """Call qgam(list(loc, scale), qu=tau) via Rscript; return dict.

    Returned keys: pred (length n), lsig (log learning rate, scalar).
    """
    with tempfile.TemporaryDirectory() as td:
        x_path = f"{td}/x.csv"
        y_path = f"{td}/y.csv"
        out_path = f"{td}/out.json"
        np.savetxt(x_path, X)
        np.savetxt(y_path, y)
        rscript = f"""
        suppressMessages({{ library(qgam) }})
        x <- as.matrix(read.csv("{x_path}", header=FALSE, sep=" "))
        y <- as.numeric(read.csv("{y_path}", header=FALSE)$V1)
        df <- data.frame(x0=x[,1], x1=x[,2], y=y)
        fit <- qgam(
            list(y ~ s(x0, k=10) + s(x1, k=10), ~ s(x0, k=5) + s(x1, k=5)),
            data=df, qu={tau}
        )
        pred <- as.numeric(predict(fit))
        lsig <- if (!is.null(fit$calibr)) fit$calibr$lsig else NA
        out <- list(pred=pred, lsig=lsig)
        cat(jsonlite::toJSON(out, auto_unbox=TRUE), file="{out_path}")
        """
        proc = subprocess.run(
            ["Rscript", "--vanilla", "-e", rscript],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            pytest.skip(f"qgam LSS fit failed: {proc.stderr.strip()[:300]}")
        with open(out_path) as f:
            return json.load(f)


@pytest.mark.skipif(not _qgam_available(), reason="qgam not installed")
@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_lss_parity_vs_qgam(hetero_data, tau):
    """fit_quantile_lss vs qgam(list(...)): coverage, prediction corr, σ ballpark."""
    X, y, _sigma_true = hetero_data

    # Rust LSS — default qgam heuristic σ_global.
    fit, info = fit_quantile_lss(X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5])
    yhat_rust = fit.predict_loc(X)
    sigma_global_rust = float(info["sigma_global"])

    r_out = _fit_qgam_lss(X, y, tau)
    yhat_qgam = np.asarray(r_out["pred"])
    lsig_qgam = float(r_out["lsig"])  # log learning rate; σ_global ≈ exp(lsig) · scale

    # 1. Coverage — both within 0.05 of τ on this heteroskedastic case.
    cov_rust = float((y < yhat_rust).mean())
    cov_qgam = float((y < yhat_qgam).mean())
    cal_rust = abs(cov_rust - tau)
    cal_qgam = abs(cov_qgam - tau)
    assert cal_rust < 0.05, f"τ={tau}: rust cal_err={cal_rust:.4f} > 0.05"
    assert cal_qgam < 0.05, f"τ={tau}: qgam cal_err={cal_qgam:.4f} > 0.05"
    assert abs(cov_rust - cov_qgam) < 0.05, (
        f"τ={tau}: cov diverges — rust={cov_rust:.4f} qgam={cov_qgam:.4f}"
    )

    # 2. Prediction correlation across the n=800 observations. Both should
    # recover the same smooth τ-quantile surface up to noise in σ̂(x).
    # At τ=0.5 the surfaces match tightly (~0.99); at extreme τ the heuristic
    # σ_global drifts from qgam's cv-tuned lsig and the curves diverge in
    # local wiggle (~0.91-0.93). calibrate=True closes that gap to ~0.96.
    corr_min = 0.97 if tau == 0.5 else 0.90
    if yhat_rust.std() > 1e-6 and yhat_qgam.std() > 1e-6:
        corr = float(np.corrcoef(yhat_rust, yhat_qgam)[0, 1])
        assert corr > corr_min, (
            f"τ={tau}: rust/qgam prediction corr={corr:.4f} < {corr_min}"
        )

    # 3. σ_global ballpark. qgam tunes lsig via cross-validation, we use the
    # err·sqrt(2π·varHat)/(2·log 2)·tail_scale heuristic; expect agreement
    # to ~3 orders of magnitude (matches the scalar parity test's bar).
    if np.isfinite(lsig_qgam):
        log_diff = np.log10(max(sigma_global_rust, 1e-12)) - lsig_qgam / np.log(10)
        assert abs(log_diff) < 3.0, (
            f"τ={tau}: log10(σ_global) diff={log_diff:.2f} "
            f"(rust={sigma_global_rust:.3e}, qgam_lsig={lsig_qgam:.3f})"
        )


@pytest.mark.skipif(not _qgam_available(), reason="qgam not installed")
@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_lss_retuned_parity_vs_qgam(hetero_data, tau):
    """With `retune_lambda=True`, the FS outer loop re-fits λ_loc under
    the per-obs-σ ELF likelihood. The predictions should match qgam
    extremely tightly (corr > 0.99) at every τ — this is the gap the
    default heuristic leaves at extreme τ (corr ~0.91-0.93), closed by
    re-tuning λ instead of inheriting it from the Gaussian-init GAM.
    """
    X, y, _ = hetero_data
    fit, info = fit_quantile_lss(
        X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5],
        retune_lambda=True, fs_max_outer=20,
    )
    yhat_rust = fit.predict_loc(X)

    r_out = _fit_qgam_lss(X, y, tau)
    yhat_qgam = np.asarray(r_out["pred"])

    if yhat_rust.std() > 1e-6 and yhat_qgam.std() > 1e-6:
        corr = float(np.corrcoef(yhat_rust, yhat_qgam)[0, 1])
        assert corr > 0.99, f"retuned τ={tau}: corr={corr:.4f} < 0.99"

    # Coverage stays within 0.05 of τ.
    cov_rust = float((y < yhat_rust).mean())
    assert abs(cov_rust - tau) < 0.05, (
        f"retuned τ={tau}: cal_err={abs(cov_rust-tau):.4f} > 0.05"
    )


@pytest.mark.skipif(not _qgam_available(), reason="qgam not installed")
@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_lss_calibrated_parity_vs_qgam(hetero_data, tau):
    """With calibrate=True (pinball-CV), the τ-quantile surface should
    match qgam tightly (corr > 0.95) at every τ — qgam also calibrates
    σ_global via tuneLearnFast, so this is the apples-to-apples comparison.

    Coverage may diverge slightly more than the heuristic variant at
    extreme τ because pinball-CV and tuneLearnFast minimise different
    objectives, but the curve *shapes* line up.
    """
    X, y, _ = hetero_data
    fit, _ = fit_quantile_lss(
        X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5],
        calibrate=True, n_folds=3, seed=0,
    )
    yhat_rust = fit.predict_loc(X)

    r_out = _fit_qgam_lss(X, y, tau)
    yhat_qgam = np.asarray(r_out["pred"])

    if yhat_rust.std() > 1e-6 and yhat_qgam.std() > 1e-6:
        corr = float(np.corrcoef(yhat_rust, yhat_qgam)[0, 1])
        assert corr > 0.95, f"calibrated τ={tau}: corr={corr:.4f} < 0.95"
