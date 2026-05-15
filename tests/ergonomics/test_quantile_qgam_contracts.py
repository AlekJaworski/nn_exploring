"""Replay tiny qgam::elf numeric contracts against Rust ELF subcomponents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import mgcv_rust
from mgcv_rust import quantile_elf_parts_py, quantile_elf_saturated_loglik_py


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test_data" / "qgam_elf_contracts.json"
GAUSSIAN_INIT_FIXTURE = ROOT / "test_data" / "qgam_gaussian_init_contract.json"
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
