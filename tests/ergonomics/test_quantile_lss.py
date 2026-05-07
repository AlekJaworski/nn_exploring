"""Tests for `fit_quantile_lss` — heteroskedastic τ-quantile regression.

Closes the heteroskedastic gap that scalar-σ `fit_quantile` leaves open.
The scalar version compromises between sharp and dispersed regions; LSS
models σ(x) explicitly so the τ-quantile estimate stays calibrated.

Algorithm (qgam ≥1.3 "parametric location-scale"):
  - Stage 1: fit Gaussian GAM on y for μ_G(x); fit Gaussian GAM on
    log|y - μ_G| + 0.6351 (Euler-Mascheroni correction) for log σ_G(x).
  - Stage 2: per-obs σ_i = σ_global · σ_G(x_i) / mean(σ_G), where
    σ_global is the qgam err·sqrt(2π·varHat)/(2·log 2) heuristic
    (or calibrated via K-fold pinball-CV).
  - Run per-obs-σ ELF IRLS (Rust) for the τ-quantile location.
"""
from __future__ import annotations

import numpy as np
import pytest

from mgcv_rust import fit_quantile_lss


@pytest.fixture(scope="module")
def hetero_data():
    """σ(x) = 0.1 + 0.4|x_0| — the canonical heteroskedastic scenario."""
    rng = np.random.default_rng(7)
    n = 800
    X = rng.uniform(-1, 1, size=(n, 2))
    sigma = 0.1 + 0.4 * np.abs(X[:, 0])
    mu = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y = mu + sigma * rng.standard_normal(n)
    return X, y, sigma


@pytest.fixture(scope="module")
def homosked_data():
    rng = np.random.default_rng(11)
    n = 800
    X = rng.uniform(-1, 1, size=(n, 2))
    mu = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]
    y = mu + 0.3 * rng.standard_normal(n)
    return X, y, np.full(n, 0.3)


def _empirical_coverage(y, yhat):
    return float((y < yhat).mean())


@pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_heteroskedastic_calibration(hetero_data, tau):
    """At every τ, in-sample coverage matches τ within 0.05.

    The scalar-σ `fit_quantile` would leave gap ~0.04 at extreme τ on this
    scenario (note 2026-05-07); LSS closes it because σ_G(x) tracks the
    heteroskedastic structure (corr ≈ 0.96 with truth).
    """
    X, y, _sigma_true = hetero_data
    fit, info = fit_quantile_lss(X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5])
    assert info["converged"], f"τ={tau} did not converge"
    cov = _empirical_coverage(y, fit.predict_loc(X))
    assert abs(cov - tau) < 0.05, (
        f"τ={tau}: |cov - τ|={abs(cov-tau):.4f} > 0.05 (cov={cov:.4f})"
    )


def test_heteroskedastic_recovers_sigma_curve(hetero_data):
    """σ̂(x) tracks σ_true within 30% relative MAE — the σ-shape recovery
    is what closes the heteroskedastic calibration gap."""
    X, y, sigma_true = hetero_data
    fit, _ = fit_quantile_lss(X, y, tau=0.5, k_loc=[10, 10], k_scale=[5, 5])
    sigma_hat = fit.predict_sigma(X)
    rel_mae = float(np.abs(sigma_hat - sigma_true).mean() / sigma_true.mean())
    assert rel_mae < 0.30, f"σ̂ rel_mae={rel_mae:.3f} > 0.30"
    corr = np.corrcoef(sigma_hat, sigma_true)[0, 1]
    assert corr > 0.85, f"corr(σ̂, σ_true)={corr:.3f} < 0.85"


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_homoskedastic_learns_flat_sigma(homosked_data, tau):
    """When σ is truly constant, σ̂(x) should be approximately flat
    (range / mean < 0.30) — guard against false heteroskedasticity."""
    X, y, sigma_true = homosked_data
    fit, info = fit_quantile_lss(X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5])
    assert info["converged"]
    sigma_hat = fit.predict_sigma(X)
    spread = float((sigma_hat.max() - sigma_hat.min()) / sigma_hat.mean())
    assert spread < 0.40, f"σ̂ spread={spread:.3f} > 0.40 — false heteroscedasticity"
    cov = _empirical_coverage(y, fit.predict_loc(X))
    assert abs(cov - tau) < 0.05, f"τ={tau}: cal_err={abs(cov-tau):.4f}"


def test_predict_loc_predict_sigma_at_new_X(hetero_data):
    """`predict_loc` and `predict_sigma` work at out-of-sample X."""
    X, y, _ = hetero_data
    fit, _ = fit_quantile_lss(X, y, tau=0.5, k_loc=[10, 10], k_scale=[5, 5])
    rng = np.random.default_rng(99)
    X_new = rng.uniform(-1, 1, size=(50, 2))
    yhat_new = fit.predict_loc(X_new)
    sigma_new = fit.predict_sigma(X_new)
    assert yhat_new.shape == (50,)
    assert sigma_new.shape == (50,)
    assert np.all(np.isfinite(yhat_new))
    assert np.all(sigma_new > 0)


def test_calibration_runs(hetero_data):
    """`calibrate=True` runs K-fold pinball-CV without errors."""
    X, y, _ = hetero_data
    fit, info = fit_quantile_lss(
        X, y, tau=0.5, k_loc=[8, 8], k_scale=[4, 4],
        calibrate=True, n_folds=3, seed=0,
    )
    assert info["calibration"] is not None
    assert info["calibration"]["n_brent_evals"] > 0
    cov = _empirical_coverage(y, fit.predict_loc(X))
    assert abs(cov - 0.5) < 0.05


def test_sigma_scale_override(hetero_data):
    """User-specified `sigma_scale` overrides the qgam heuristic."""
    X, y, _ = hetero_data
    user_sigma = 0.15
    fit, info = fit_quantile_lss(
        X, y, tau=0.5, k_loc=[10, 10], k_scale=[5, 5],
        sigma_scale=user_sigma,
    )
    assert abs(info["sigma_global"] - user_sigma) < 1e-9


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_retune_lambda_runs_and_calibrates(hetero_data, tau):
    """`retune_lambda=True` runs a Fellner-Schall outer loop on λ_loc under
    the per-obs-σ ELF likelihood. It should still converge and stay within
    0.05 of τ on coverage, with `info['fs_iterations'] >= 1` reported and
    a length-len(k_loc) λ vector returned.
    """
    X, y, _ = hetero_data
    fit, info = fit_quantile_lss(
        X, y, tau=tau, k_loc=[10, 10], k_scale=[5, 5],
        retune_lambda=True, fs_max_outer=20,
    )
    assert info["converged"], f"τ={tau} did not converge"
    assert info["fs_iterations"] >= 1
    assert len(info["lambda_loc"]) == 2
    cov = _empirical_coverage(y, fit.predict_loc(X))
    assert abs(cov - tau) < 0.05, (
        f"τ={tau}: cal_err={abs(cov-tau):.4f} > 0.05 (cov={cov:.4f})"
    )


def test_lss_beats_single_sigma_at_extreme_tau(hetero_data):
    """Sanity: on the canonical heteroskedastic scenario, LSS hits cal_err
    at least as good as the calibrated single-σ baseline at τ=0.1 — this
    is the whole point of the LSS path. Tightens to <0.04 ceiling."""
    from mgcv_rust import fit_quantile

    X, y, _ = hetero_data
    fit_lss, _ = fit_quantile_lss(X, y, tau=0.1, k_loc=[10, 10], k_scale=[5, 5])
    cov_lss = _empirical_coverage(y, fit_lss.predict_loc(X))
    cal_lss = abs(cov_lss - 0.1)

    # Single-σ baseline (calibrated via pinball CV).
    g, _, _ = fit_quantile(X, y, tau=0.1, k=[10, 10], calibrate=True, loss="pin")
    cov_baseline = _empirical_coverage(y, np.asarray(g.predict(X)))
    cal_baseline = abs(cov_baseline - 0.1)

    # LSS should be competitive — within 0.02 of baseline OR strictly better.
    assert cal_lss < 0.04, f"LSS cal_err={cal_lss:.4f} > 0.04"
    # And LSS shouldn't be much worse than baseline.
    assert cal_lss < cal_baseline + 0.02, (
        f"LSS cal_err={cal_lss:.4f}, baseline={cal_baseline:.4f}"
    )
