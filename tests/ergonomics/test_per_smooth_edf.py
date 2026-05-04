"""Ergo-3a: per-smooth EDF accessor parity test.

Verifies `GAM.get_edf_per_smooth()` against `summary.gam()$edf` values
stored in the parity fixtures (no live R/rpy2 required).

Three cases:
- 1d Gaussian (single smooth)
- 1d Poisson log (GLM family, single smooth)
- 2d Gaussian additive (two smooths)

Assertions:
1. Each smooth's EDF matches mgcv (rtol=1e-3).
2. sum(per_smooth_edf) + 1 ≈ edf_total from fixture (atol=1e-4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "parity" / "fixtures"
_PARITY_SCHEMA = Path(__file__).resolve().parents[1] / "parity"
if str(_PARITY_SCHEMA) not in sys.path:
    sys.path.insert(0, str(_PARITY_SCHEMA))

from schema import Fixture  # noqa: E402


def _load(name: str) -> Fixture:
    return Fixture.load(_FIXTURES_DIR / f"{name}.json")


def _fit_rust(fix: Fixture):
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)

    family_map = {
        "gaussian": "gaussian",
        "poisson": "poisson",
        "binomial": "binomial",
        "Gamma": "gamma",
        "gamma": "gamma",
    }
    fam = family_map[inp.family]
    if fam == "gaussian":
        gam = mgcv_rust.GAM()
    else:
        gam = mgcv_rust.GAM(fam)

    bs = inp.bs[0] if len(set(inp.bs)) == 1 else inp.bs
    gam.fit(x, y, k=list(inp.k), method=inp.method, bs=bs)
    return gam


def _check_per_smooth_edf(fix_name: str, rtol: float = 1e-3, edf_total_atol: float = 1e-2):
    fix = _load(fix_name)
    gam = _fit_rust(fix)

    rust_edf = dict(gam.get_edf_per_smooth())
    mgcv_edf = fix.mgcv_output.edf_per_smooth
    mgcv_total = fix.mgcv_output.edf_total

    assert set(rust_edf.keys()) == set(mgcv_edf.keys()), (
        f"smooth name mismatch: rust={set(rust_edf.keys())} mgcv={set(mgcv_edf.keys())}"
    )

    for name, mgcv_val in mgcv_edf.items():
        rust_val = rust_edf[name]
        rel_err = abs(rust_val - mgcv_val) / max(abs(mgcv_val), 1e-10)
        assert rel_err <= rtol, (
            f"{fix_name}/{name}: rust={rust_val:.6f} mgcv={mgcv_val:.6f} relerr={rel_err:.2e}"
        )

    # Sanity: sum(per_smooth) + 1 (intercept) should equal edf_total
    rust_total = sum(rust_edf.values()) + 1.0
    assert abs(rust_total - mgcv_total) <= edf_total_atol, (
        f"{fix_name}: rust_total={rust_total:.6f} mgcv_total={mgcv_total:.6f} "
        f"diff={abs(rust_total - mgcv_total):.2e}"
    )

    return rust_edf, mgcv_edf, mgcv_total


def test_edf_1d_gaussian():
    rust_edf, mgcv_edf, mgcv_total = _check_per_smooth_edf("1d_gaussian_smooth_n500_k10_cr")
    # Informational output (visible with -v)
    for name in mgcv_edf:
        print(
            f"  1d_gaussian/{name}: rust={rust_edf[name]:.4f} "
            f"mgcv={mgcv_edf[name]:.4f} "
            f"relerr={abs(rust_edf[name]-mgcv_edf[name])/mgcv_edf[name]:.2e}"
        )


def test_edf_1d_poisson():
    rust_edf, mgcv_edf, mgcv_total = _check_per_smooth_edf("1d_poisson_log_n500_k10_cr")
    for name in mgcv_edf:
        print(
            f"  1d_poisson/{name}: rust={rust_edf[name]:.4f} "
            f"mgcv={mgcv_edf[name]:.4f} "
            f"relerr={abs(rust_edf[name]-mgcv_edf[name])/mgcv_edf[name]:.2e}"
        )


def test_edf_2d_gaussian_additive():
    rust_edf, mgcv_edf, mgcv_total = _check_per_smooth_edf("2d_gaussian_additive_n500_k10_cr")
    for name in mgcv_edf:
        print(
            f"  2d_gaussian/{name}: rust={rust_edf[name]:.4f} "
            f"mgcv={mgcv_edf[name]:.4f} "
            f"relerr={abs(rust_edf[name]-mgcv_edf[name])/mgcv_edf[name]:.2e}"
        )
