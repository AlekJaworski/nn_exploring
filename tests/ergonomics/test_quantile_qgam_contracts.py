"""Replay tiny qgam::elf numeric contracts against Rust ELF subcomponents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mgcv_rust import quantile_elf_parts_py, quantile_elf_saturated_loglik_py


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "test_data" / "qgam_elf_contracts.json"


def _fixture() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"missing qgam ELF fixture: {FIXTURE}")
    return json.loads(FIXTURE.read_text())


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
