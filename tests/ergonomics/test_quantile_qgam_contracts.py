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


def test_quantile_elf_saturated_loglik_matches_qgam_ls_value() -> None:
    payload = _fixture()
    p = payload["params"]
    n = len(payload["rows"])

    rust_ls = quantile_elf_saturated_loglik_py(p["tau"], p["sigma"], p["co"], n=n)
    assert rust_ls == pytest.approx(payload["ls"]["value"], abs=1e-10)
