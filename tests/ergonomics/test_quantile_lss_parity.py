"""Parity test for `fit_quantile_lss` against R's qgam ≥ 1.3 location-scale path.

qgam ≥ 1.3, given a formula list (location + scale), runs a "parametric
location-scale model": Gaussian preprocessing for σ̂(x), per-obs-σ ELF for
the location, plus internal λ_loc retuning under the ELF likelihood via
mgcv's `gam.fit5` LAML. We mirror the same pipeline; the default path
runs Fellner-Schall λ_loc retuning (the FS analogue of qgam's LAML step).

The two methods still differ on **σ_global tuning**: qgam runs the
cross-validated `tuneLearnFast`; we use either the analytical heuristic
`err·sqrt(2π·varHat)/(2·log 2)·tail_scale` (default) or pinball-CV
(`calibrate=True`).

This test verifies that on the canonical heteroskedastic scenario,
`fit_quantile_lss` and `qgam(list(loc_form, scale_form), qu=tau)` agree
on:
    1. Empirical coverage (frac y < ŷ) — both within 0.05 of τ, and within
       0.05 of each other.
    2. Prediction correlation across observations — corr > 0.99 at every
       τ with the default (FS-retuned) path. Without retuning the corr
       drops to ~0.91-0.93 at extreme τ (asserted separately in the
       no-retune test).
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
    """Default `fit_quantile_lss` (FS-retuned λ) vs qgam: coverage, prediction
    corr > 0.99 at every τ, σ_global ballpark.
    """
    X, y, _sigma_true = hetero_data

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

    # 2. Prediction correlation. With FS λ retuning (default), corr is
    # essentially 1 at every τ — the curves match qgam's gam.fit5 LAML output.
    if yhat_rust.std() > 1e-6 and yhat_qgam.std() > 1e-6:
        corr = float(np.corrcoef(yhat_rust, yhat_qgam)[0, 1])
        assert corr > 0.99, f"τ={tau}: rust/qgam prediction corr={corr:.4f} < 0.99"

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
@pytest.mark.parametrize("tau", [0.1, 0.9])
def test_lss_no_retune_documents_heuristic_gap(hetero_data, tau):
    """Pinned regression: with `retune_lambda=False`, the heuristic-λ path
    leaves a corr-with-qgam gap at extreme τ (~0.91-0.93). Coverage still
    matches within 0.05, but the curve shape diverges locally because
    λ_loc is the Gaussian-fit auto-tune, wrong for the ELF likelihood.
    Documenting the cost of opting out of the FS retune.
    """
    X, y, _ = hetero_data
    fit, _ = fit_quantile_lss(
        X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5],
        retune_lambda=False,
    )
    yhat_rust = fit.predict_loc(X)
    r_out = _fit_qgam_lss(X, y, tau)
    yhat_qgam = np.asarray(r_out["pred"])

    cov_rust = float((y < yhat_rust).mean())
    assert abs(cov_rust - tau) < 0.05, (
        f"no-retune τ={tau}: cal_err={abs(cov_rust-tau):.4f} > 0.05"
    )
    # Wide envelope; the test exists to flag if heuristic-λ behaviour
    # changes drastically (e.g. drops below 0.85 — would be a regression).
    if yhat_rust.std() > 1e-6 and yhat_qgam.std() > 1e-6:
        corr = float(np.corrcoef(yhat_rust, yhat_qgam)[0, 1])
        assert 0.85 < corr < 0.999, (
            f"no-retune τ={tau}: corr={corr:.4f} outside [0.85, 0.999); "
            "if corr ≥ 0.999, retune is being applied accidentally"
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
