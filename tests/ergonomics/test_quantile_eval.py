"""Two-track evaluation framework for qgam quality assessment.

Truth track: synthetic data with known true τ-quantile → RMSE and coverage.
Parity track: production fixture holdout → pinball loss vs qgam reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mgcv_rust._quantile import fit_quantile

# ── constants ────────────────────────────────────────────────────────────────

HANDOFF = Path.home() / "vibe_coding" / "mgcv_rust_parity_handoff"
ROOT = Path(__file__).resolve().parents[2]
SALE_FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
HOLDOUT_PINBALL_FIXTURE = ROOT / "test_data" / "qgam_holdout_pinball_contract.json"

QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]


# ── pinball loss helper ───────────────────────────────────────────────────────

def _pinball(y: np.ndarray, pred: np.ndarray, tau: float) -> float:
    r = y - pred
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))


# ── synthetic fixtures ────────────────────────────────────────────────────────

def _make_synthetic(tau: float = 0.9):
    """Return (X_train, y_train, X_test, y_test, true_quantile_test)."""
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(42)
    n_train, n_test = 800, 400
    X_all = rng.uniform(0, 1, (n_train + n_test, 2))
    f = np.sin(2 * np.pi * X_all[:, 0]) + 0.5 * np.cos(2 * np.pi * X_all[:, 1])
    sigma_noise = 0.3
    y_all = f + sigma_noise * rng.standard_normal(n_train + n_test)
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    f_test = f[n_train:]
    true_q_test = f_test + sigma_noise * stats.norm.ppf(tau)
    return X_train, y_train, X_test, y_test, true_q_test


# ── tests ─────────────────────────────────────────────────────────────────────

def test_synthetic_truth_track() -> None:
    """RMSE of predicted τ-quantile vs true quantile at test points < 0.15."""
    pytest.importorskip("scipy")
    tau = 0.9
    X_train, y_train, X_test, y_test, true_q_test = _make_synthetic(tau)

    gam, _sigma, _info = fit_quantile(X_train, y_train, tau=tau, k=[10, 10], bs="cr")
    pred_test = np.asarray(gam.predict(X_test), dtype=float)

    rmse = float(np.sqrt(np.mean((pred_test - true_q_test) ** 2)))
    max_abs = float(np.max(np.abs(pred_test - true_q_test)))

    print(f"\n  RMSE vs true quantile: {rmse:.4f}  (threshold 0.15)")
    print(f"  max_abs vs true quantile: {max_abs:.4f}  (threshold 0.5)")

    assert rmse < 0.15, f"RMSE {rmse:.4f} >= 0.15 — quantile fit is not tracking the true τ-quantile"
    assert max_abs < 0.5, f"max_abs {max_abs:.4f} >= 0.5 — sanity check failed"


def test_empirical_coverage() -> None:
    """Empirical coverage at τ=0.9 should be within 0.06 of 0.9 on the test set."""
    pytest.importorskip("scipy")
    tau = 0.9
    X_train, y_train, X_test, y_test, _true_q_test = _make_synthetic(tau)

    gam, _sigma, _info = fit_quantile(X_train, y_train, tau=tau, k=[10, 10], bs="cr")
    pred_test = np.asarray(gam.predict(X_test), dtype=float)

    coverage = float(np.mean(y_test < pred_test))
    print(f"\n  empirical coverage: {coverage:.4f}  (target {tau}, tolerance ±0.06)")

    assert abs(coverage - tau) < 0.06, (
        f"coverage {coverage:.4f} deviates from τ={tau} by "
        f"{abs(coverage - tau):.4f} (threshold 0.06)"
    )


def test_holdout_pinball_vs_qgam() -> None:
    """Rust qgam pinball loss on holdout <= 1.10 × qgam OOS pinball loss.

    Both Rust and qgam are fit on the first 80% of rows, evaluated on the last
    20% — a fair apples-to-apples comparison. qgam OOS predictions generated
    by scripts/r/tests/extract_qgam_holdout_pinball_contract.R.
    """
    import json

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    train_path = SALE_FIXTURE_DIR / "entire_dataset_train.parquet"
    mean_path = SALE_FIXTURE_DIR / "mgcv_rust_parity" / "mean_gam_entire_dataset.parquet"

    for p in (train_path, mean_path, HOLDOUT_PINBALL_FIXTURE):
        if not p.exists():
            pytest.skip(f"missing fixture: {p}")

    contract = json.loads(HOLDOUT_PINBALL_FIXTURE.read_text())
    tau = contract["tau"]
    split = contract["split_index"]
    qgam_pred_test = np.asarray(contract["qgam_pred_test"], dtype=float)
    qgam_pinball = contract["qgam_oos_pinball"]

    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = (mean_ref["y"].to_numpy(dtype=float)
             - mean_ref["pred_prod"].to_numpy(dtype=float))

    X = df[QGAM_SMOOTHS].to_numpy(dtype=float)
    X_train, X_test = X[:split], X[split:]
    resid_train, resid_test = resid[:split], resid[split:]

    gam, _sigma, _info = fit_quantile(
        X_train, resid_train, tau=tau, k=[5, 5, 5, 5], bs="cr", method="REML",
        calibrate=True, loss="pin", n_folds=3, coverage_calibrate=True,
    )
    rust_pred_test = np.asarray(gam.predict(X_test), dtype=float)
    rust_pinball = _pinball(resid_test, rust_pred_test, tau)

    ratio = rust_pinball / qgam_pinball
    print(
        f"\n  rust pinball: {rust_pinball:.6f}"
        f"\n  qgam pinball: {qgam_pinball:.6f}  (OOS, same 80/20 split)"
        f"\n  ratio: {ratio:.4f}  (threshold 1.00; calibrated fast path)"
    )

    # Pinball-CV σ plus coverage calibration should beat qgam's OOS pinball
    # while staying below qgam's production fit time on this fixture.
    assert rust_pinball <= qgam_pinball, (
        f"rust pinball {rust_pinball:.6f} > qgam OOS pinball "
        f"{qgam_pinball:.6f} (ratio {ratio:.4f})"
    )


def _load_holdout_contract() -> tuple:
    """Return (contract, X_train, X_test, resid_train, resid_test) or skip."""
    import json

    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    train_path = SALE_FIXTURE_DIR / "entire_dataset_train.parquet"
    mean_path = SALE_FIXTURE_DIR / "mgcv_rust_parity" / "mean_gam_entire_dataset.parquet"
    for p in (train_path, mean_path, HOLDOUT_PINBALL_FIXTURE):
        if not p.exists():
            pytest.skip(f"missing fixture: {p}")

    contract = json.loads(HOLDOUT_PINBALL_FIXTURE.read_text())
    split = contract["split_index"]

    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = (mean_ref["y"].to_numpy(dtype=float)
             - mean_ref["pred_prod"].to_numpy(dtype=float))

    X = df[QGAM_SMOOTHS].to_numpy(dtype=float)
    return (
        contract,
        X[:split], X[split:],
        resid[:split], resid[split:],
    )


def test_holdout_coverage_vs_qgam() -> None:
    """Side-by-side empirical coverage at τ=0.95: Rust vs qgam, both OOS."""
    contract, X_train, X_test, resid_train, resid_test = _load_holdout_contract()

    tau = contract["tau"]
    qgam_pred_test = np.asarray(contract["qgam_pred_test"], dtype=float)
    qgam_coverage = contract["qgam_oos_coverage"]

    gam, _sigma, _info = fit_quantile(
        X_train, resid_train, tau=tau, k=[5, 5, 5, 5], bs="cr", method="REML",
        calibrate=True, loss="pin", n_folds=3, coverage_calibrate=True,
    )
    rust_pred_test = np.asarray(gam.predict(X_test), dtype=float)
    rust_coverage = float(np.mean(resid_test < rust_pred_test))

    rust_pinball = _pinball(resid_test, rust_pred_test, tau)
    qgam_pinball = _pinball(resid_test, qgam_pred_test, tau)

    print(
        f"\n  {'':12s}  {'pinball':>10s}  {'coverage':>10s}  {'cov-target':>10s}"
        f"\n  {'Rust':12s}  {rust_pinball:10.6f}  {rust_coverage:10.4f}  {rust_coverage - tau:+10.4f}"
        f"\n  {'qgam':12s}  {qgam_pinball:10.6f}  {qgam_coverage:10.4f}  {qgam_coverage - tau:+10.4f}"
        f"\n  {'target':12s}  {'(lower)':>10s}  {tau:10.4f}  {'0':>10s}"
    )

    # Rust coverage should be in a sensible range — not wildly under-predicting.
    # Current baseline (FS heuristic σ): Rust ~0.993, qgam ~0.973, target 0.95.
    # Both are over-conservative. Upper bound is 1.0; lower bound catches
    # under-coverage regressions (predictions collapsing toward the mean).
    assert rust_coverage >= 0.88, (
        f"rust coverage {rust_coverage:.4f} < 0.88 — severe under-coverage"
    )


def _make_preset_equivalence_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.uniform(0, 1, (120, 2))
    signal = np.sin(2 * np.pi * X[:, 0]) + 0.25 * X[:, 1]
    y = signal + 0.2 * rng.standard_normal(len(X))
    return X, y


def test_fast_oos_preset_equals_explicit_options() -> None:
    X, y = _make_preset_equivalence_data()

    g_preset, sigma_preset, info_preset = fit_quantile(
        X, y, tau=0.9, k=[6, 6], bs="cr", preset="fast_oos",
    )
    g_explicit, sigma_explicit, info_explicit = fit_quantile(
        X, y, tau=0.9, k=[6, 6], bs="cr", coverage_calibrate=True,
    )

    assert sigma_preset == sigma_explicit == 0.0
    assert info_preset is None
    assert info_explicit is None
    np.testing.assert_allclose(
        g_preset.predict(X), g_explicit.predict(X), atol=1e-12,
    )


def test_quality_oos_preset_equals_explicit_options() -> None:
    pytest.importorskip("scipy")
    X, y = _make_preset_equivalence_data()

    g_preset, sigma_preset, info_preset = fit_quantile(
        X, y, tau=0.9, k=[6, 6], bs="cr", preset="quality_oos",
        n_folds=2, seed=7,
    )
    g_explicit, sigma_explicit, info_explicit = fit_quantile(
        X, y, tau=0.9, k=[6, 6], bs="cr",
        calibrate=True, loss="pin", coverage_calibrate=True,
        n_folds=2, seed=7,
    )

    assert info_preset is not None
    assert info_explicit is not None
    assert sigma_preset == sigma_explicit
    assert (
        info_preset["coverage_calibration_shift"]
        == info_explicit["coverage_calibration_shift"]
    )
    np.testing.assert_allclose(
        g_preset.predict(X), g_explicit.predict(X), atol=1e-12,
    )


def test_unknown_quantile_preset_rejected() -> None:
    X, y = _make_preset_equivalence_data()

    with pytest.raises(ValueError, match="unknown quantile preset"):
        fit_quantile(X, y, tau=0.9, k=[6, 6], preset="not_a_preset")


def test_final_algorithm_warns_diagnostic() -> None:
    X, y = _make_preset_equivalence_data()

    with pytest.warns(RuntimeWarning, match="diagnostic for quantile fits"):
        fit_quantile(
            X, y, tau=0.9, k=[6, 6], bs="cr",
            final_algorithm="fellner-schall",
        )
