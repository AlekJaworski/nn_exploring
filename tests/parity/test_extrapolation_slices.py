"""Focused extrapolation parity tests for correlated-feature 1D slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mgcv_rust

from schema import Fixture


HERE = Path(__file__).resolve().parent
FIXTURE_NAME = "4d_gaussian_correlated_extrap_slices_n800_k10_cr"
FIXTURE_PATH = HERE / "fixtures" / f"{FIXTURE_NAME}.json"


def _fit_fixture(fix: Fixture):
    g = mgcv_rust.GAM(fix.inputs.family.lower(), link=fix.inputs.link)
    g.fit(
        np.asarray(fix.inputs.x_train, dtype=float),
        np.asarray(fix.inputs.y_train, dtype=float),
        k=fix.inputs.k,
        bs_list=fix.inputs.bs,
        method=fix.inputs.method,
    )
    return g


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="missing correlated extrapolation fixture")
def test_correlated_one_feature_extrapolation_slices_match_mgcv_by_block() -> None:
    """Compare R/mgcv vs Rust per one-feature slice block.

    Fixture block layout is dim-major, then low / in-domain / high blocks, with
    nine rows per block. This catches extrapolation-tail drift that aggregate
    train/test metrics can hide when features are correlated.
    """
    fix = Fixture.load(FIXTURE_PATH)
    gam = _fit_fixture(fix)
    x_extrap = np.asarray(fix.inputs.x_extrap, dtype=float)
    rust = np.asarray(gam.predict(x_extrap), dtype=float)
    mgcv = np.asarray(fix.mgcv_output.predictions_extrap, dtype=float)

    block_n = 9
    zones = ("low", "in", "high")
    failures: list[str] = []
    for dim in range(fix.inputs.d):
        for zone_idx, zone in enumerate(zones):
            start = (dim * len(zones) + zone_idx) * block_n
            stop = start + block_n
            diff = np.abs(rust[start:stop] - mgcv[start:stop])
            max_abs = float(diff.max())
            # In-domain slices should be tight. Out-of-domain CR tails are more
            # sensitive to lambda/centering differences, but should still stay
            # well below the historical 4e-1 extrapolation gap called out in the
            # parity backlog.
            tol = 2.0e-3 if zone == "in" else 2.0e-2
            if max_abs > tol:
                failures.append(f"dim={dim} zone={zone} max_abs={max_abs:.6g} tol={tol:.6g}")

    assert not failures, "correlated extrapolation slice parity failures:\n" + "\n".join(failures)
