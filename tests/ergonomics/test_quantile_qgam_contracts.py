"""Replay tiny qgam::elf numeric contracts against Rust ELF subcomponents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mgcv_rust
from mgcv_rust import quantile_elf_parts_py, quantile_elf_saturated_loglik_py
from mgcv_rust._shash import (
    shash_loglik,
    shash_loglik_grad,
    shash_qf,
    shash_cdf,
    shash_mode,
    fit_shash,
    compute_err_param,
    _dm_at_point,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test_data" / "qgam_elf_contracts.json"
GAUSSIAN_INIT_FIXTURE = ROOT / "test_data" / "qgam_gaussian_init_contract.json"
ERR_CO_FIXTURE = ROOT / "test_data" / "qgam_err_co_contract.json"
SALE_FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]


def _fixture() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing qgam ELF fixture: {FIXTURE}")
    return json.loads(FIXTURE.read_text())


def _gaussian_init_fixture() -> dict:
    if not GAUSSIAN_INIT_FIXTURE.exists():
        pytest.skip(f"missing qgam Gaussian-init fixture: {GAUSSIAN_INIT_FIXTURE}")
    return json.loads(GAUSSIAN_INIT_FIXTURE.read_text())


def test_quantile_elf_parts_match_qgam_scalar_contract() -> None:
    payload = _fixture()
    p = payload["params"]

    for row in payload["rows"]:
        rust = quantile_elf_parts_py(row["y"], row["mu"], p["tau"], p["sigma"], p["co"])
        assert rust["deviance"] == pytest.approx(row["dev_resids"], abs=1e-10)
        assert rust["Dmu"] == pytest.approx(row["Dmu"], abs=1e-10)
        assert rust["Dmu2"] == pytest.approx(row["Dmu2"], rel=1e-10, abs=1e-10)
        assert rust["EDmu2"] == pytest.approx(row["EDmu2"], rel=1e-10, abs=1e-10)


def test_quantile_elf_higher_derivatives_match_qgam_scalar_contract() -> None:
    payload = _fixture()
    p = payload["params"]
    fields = [
        "Dth",
        "Dmuth",
        "Dmu3",
        "Dmu2th",
        "Dmu4",
        "Dth2",
        "Dmuth2",
        "Dmu2th2",
        "Dmu3th",
    ]

    for row in payload["rows"]:
        rust = quantile_elf_parts_py(row["y"], row["mu"], p["tau"], p["sigma"], p["co"])
        for field in fields:
            assert rust[field] == pytest.approx(row[field], rel=1e-10, abs=1e-10), (
                row["name"],
                field,
            )


def test_quantile_elf_working_quantities_follow_qgam_dd_contract() -> None:
    payload = _fixture()
    p = payload["params"]

    for row in payload["rows"]:
        rust = quantile_elf_parts_py(row["y"], row["mu"], p["tau"], p["sigma"], p["co"])
        expected_w = row["Dmu2"] / 2.0
        expected_g = -row["Dmu"] / 2.0
        expected_z = row["mu"] - row["Dmu"] / row["Dmu2"]
        expected_rhs_value = expected_w * row["mu"] + expected_g
        assert rust["w"] == pytest.approx(expected_w, rel=1e-10, abs=1e-10)
        assert rust["g"] == pytest.approx(expected_g, rel=1e-10, abs=1e-10)
        # In saturated tails, Dmu2 is tiny and z = mu - Dmu/Dmu2 amplifies
        # fixture JSON rounding. w/g above remain the tight qgam contract.
        assert rust["z"] == pytest.approx(expected_z, rel=1e-6, abs=1e-10)
        assert rust["rhs_value"] == pytest.approx(expected_rhs_value, rel=1e-6, abs=1e-10)


def test_quantile_elf_working_system_assembly_from_qgam_dd() -> None:
    payload = _fixture()
    p = payload["params"]
    rows = payload["rows"]
    # Tiny deterministic two-column design: intercept plus a stable covariate.
    x_rows = [[1.0, -0.5], [1.0, -0.25], [1.0, 0.0], [1.0, 0.25], [1.0, 0.5]]

    xtwx = [[0.0, 0.0], [0.0, 0.0]]
    rhs = [0.0, 0.0]
    xtwz = [0.0, 0.0]
    for x, row in zip(x_rows, rows):
        rust = quantile_elf_parts_py(row["y"], row["mu"], p["tau"], p["sigma"], p["co"])
        w = row["Dmu2"] / 2.0
        g = -row["Dmu"] / 2.0
        z = row["mu"] - row["Dmu"] / row["Dmu2"]
        assert rust["w"] == pytest.approx(w, rel=1e-10, abs=1e-10)
        for a in range(2):
            rhs[a] += x[a] * (w * row["mu"] + g)
            xtwz[a] += x[a] * w * z
            for b in range(2):
                xtwx[a][b] += x[a] * w * x[b]

    assert rhs == pytest.approx(xtwz, rel=1e-10, abs=1e-10)
    assert xtwx[0][1] == pytest.approx(xtwx[1][0], rel=1e-12, abs=1e-12)


def test_quantile_elf_saturated_loglik_matches_qgam_ls_value() -> None:
    payload = _fixture()
    p = payload["params"]
    n = len(payload["rows"])

    rust_ls = quantile_elf_saturated_loglik_py(p["tau"], p["sigma"], p["co"], n=n)
    assert rust_ls == pytest.approx(payload["ls"]["value"], abs=1e-10)


def test_qgam_gaussian_init_stage_matches_rust_gaussian_reml() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    payload = _gaussian_init_fixture()
    train_path = SALE_FIXTURE_DIR / "entire_dataset_train.parquet"
    mean_path = SALE_FIXTURE_DIR / "mgcv_rust_parity" / "mean_gam_entire_dataset.parquet"
    if not train_path.exists() or not mean_path.exists():
        pytest.skip("missing sale-price fixture data for qgam Gaussian-init contract")

    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = mean_ref["y"].to_numpy(dtype=float) - mean_ref["pred_prod"].to_numpy(dtype=float)
    k_map = payload["k"]

    rust_g = mgcv_rust.Gam(
        predictors=QGAM_SMOOTHS,
        family="gaussian",
        method="REML",
        term_k_mapping=k_map,
        predictor_basis_map={c: "cr" for c in QGAM_SMOOTHS},
    )
    rust_g.fit(df[QGAM_SMOOTHS], resid)
    fitted = np.asarray(rust_g.predict(df[QGAM_SMOOTHS]), dtype=float)
    summary = rust_g.summary()
    fitted_summary = np.quantile(fitted, [0.0, 0.25, 0.5, 0.75, 1.0])

    assert summary.scale == pytest.approx(payload["varHat"], rel=1e-6)
    assert rust_g.get_lambdas() == pytest.approx(payload["gaussian_sp"], rel=6e-3)
    assert rust_g.get_coefficients() == pytest.approx(payload["gaussian_coef"], abs=5e-6)
    assert fitted_summary == pytest.approx(payload["gaussian_fitted_summary"], abs=6e-6)
    assert payload["rust_gaussian"]["max_abs_fitted_vs_qgam_gaussian"] < 1e-5
    assert payload["intercept_shift"] == pytest.approx(0.07386184188909634, abs=1e-12)


# ── err/co contract tests ────────────────────────────────────────────────────

def _err_co_fixture() -> dict:
    if not ERR_CO_FIXTURE.exists():
        pytest.skip(f"missing qgam err/co fixture: {ERR_CO_FIXTURE}")
    return json.loads(ERR_CO_FIXTURE.read_text())


def test_shash_loglik_at_fixture_parSH() -> None:
    """Verify SHASH log-density at qhat matches R's .llkShash(deriv=0)$l0."""
    payload = _err_co_fixture()
    par_sh = payload["parSH"]
    for row in payload["qu_contracts"]:
        qhat = row["qhat"]
        lf0_r = row["lf0"]
        lf0_py = shash_loglik(np.array([qhat]), *par_sh)
        assert lf0_py == pytest.approx(lf0_r, abs=1e-12)


def test_shash_dm_at_fixture_parSH() -> None:
    """Verify -Dm at qhat matches R's lf1_raw = -.llkShash(qhat)$l1[1]."""
    payload = _err_co_fixture()
    par_sh = payload["parSH"]
    for row in payload["qu_contracts"]:
        qhat = row["qhat"]
        lf1_raw_r = row["lf1_raw"]
        lf1_raw_py = -_dm_at_point(qhat, *par_sh)
        assert lf1_raw_py == pytest.approx(lf1_raw_r, rel=1e-12, abs=1e-15)


def test_shash_bandwidth_formula_at_fixture_parSH() -> None:
    """Verify h and err computed from R's parSH match the fixture exactly."""
    payload = _err_co_fixture()
    par_sh = payload["parSH"]
    n = payload["n"]
    d = payload["d"]
    for row in payload["qu_contracts"]:
        qhat = row["qhat"]
        lf0 = shash_loglik(np.array([qhat]), *par_sh)
        lf1_raw = -_dm_at_point(qhat, *par_sh)
        lf1 = float(np.log(abs(lf1_raw)) + lf0)
        h = (d * 9.0 / (n * np.pi ** 4)) ** (1.0 / 3.0) * np.exp(lf0 / 3.0 - 2.0 * lf1 / 3.0)
        err = min(h * 2.0 * np.log(2.0) / np.sqrt(2.0 * np.pi), 1.0)
        assert lf1 == pytest.approx(row["lf1"], abs=1e-12)
        assert h == pytest.approx(row["h"], rel=1e-12)
        assert err == pytest.approx(row["err"], rel=1e-12)


def test_shash_qf_cdf_round_trip_at_fixture_parSH() -> None:
    """shash_qf and shash_cdf are mutual inverses at fixture params."""
    payload = _err_co_fixture()
    par_sh = np.array(payload["parSH"])
    for p in [0.05, 0.25, 0.50, 0.75, 0.95]:
        q = shash_qf(p, par_sh)
        assert shash_cdf(q, par_sh) == pytest.approx(p, abs=1e-10)


def test_shash_mode_at_fixture_parSH() -> None:
    """shash_mode returns the CDF-at-mode that matches R's pmode.

    scipy minimize_scalar may find a slightly different optimum than R's
    optimize(), so we allow 1e-4 on pmode_x.  pmode (the CDF value) is what
    drives the err-computation branch, and is tested at 1e-5.
    """
    payload = _err_co_fixture()
    par_sh = np.array(payload["parSH"])
    pmode_x_py = shash_mode(par_sh)
    pmode_py = shash_cdf(pmode_x_py, par_sh)
    assert pmode_x_py == pytest.approx(payload["pmode_x"], abs=1e-4)
    assert pmode_py == pytest.approx(payload["pmode"], abs=1e-5)


def test_shash_fit_recovers_known_params() -> None:
    """fit_shash on data drawn from R's parSH recovers similar params.

    Generates synthetic data from the fixture's SHASH params, then fits.
    Checks that the recovered params reproduce the same quantile structure
    (rather than testing exact param values, which depend on BFGS convergence).
    """
    payload = _err_co_fixture()
    par_r = np.array(payload["parSH"])
    # Draw from the SHASH with R's params
    rng = np.random.default_rng(42)
    p_draw = rng.uniform(0.001, 0.999, size=2000)
    x_draw = np.array([shash_qf(float(p), par_r) for p in p_draw])
    par_py = fit_shash(x_draw)
    # Verify CDF at key quantiles matches between R params and Python-fit params
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        q_r = shash_qf(p, par_r)
        assert shash_cdf(q_r, par_py) == pytest.approx(p, abs=0.03)


def test_compute_err_param_with_rust_gaussian_fit() -> None:
    """Full compute_err_param pipeline using Rust Gaussian REML fit.

    Uses mgcv_rust.Gam to reproduce the Gaussian init, computes standardized
    residuals, then calls compute_err_param.  Requires sale-price fixture data.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    payload = _err_co_fixture()
    train_path = SALE_FIXTURE_DIR / "entire_dataset_train.parquet"
    mean_path = SALE_FIXTURE_DIR / "mgcv_rust_parity" / "mean_gam_entire_dataset.parquet"
    if not train_path.exists() or not mean_path.exists():
        pytest.skip("missing sale-price fixture data for qgam err/co contract")

    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = mean_ref["y"].to_numpy(dtype=float) - mean_ref["pred_prod"].to_numpy(dtype=float)

    k_map = {s: 5 for s in QGAM_SMOOTHS}
    rust_g = mgcv_rust.Gam(
        predictors=QGAM_SMOOTHS,
        family="gaussian",
        method="REML",
        term_k_mapping=k_map,
        predictor_basis_map={c: "cr" for c in QGAM_SMOOTHS},
    )
    rust_g.fit(df[QGAM_SMOOTHS], resid)
    fitted = np.asarray(rust_g.predict(df[QGAM_SMOOTHS]), dtype=float)
    var_hat = float(rust_g.summary().scale)
    r = (resid - fitted) / np.sqrt(var_hat)

    # d = 1 (intercept) + sum of smooth EDFs
    d = 1.0 + float(rust_g.edf_.sum())

    err_r = payload["err_official"]  # 0.038489...
    err_py = compute_err_param(r, d, [0.95])
    # Require err within 5% of R's value; BFGS optima may differ slightly.
    assert err_py[0] == pytest.approx(err_r, rel=0.05)
