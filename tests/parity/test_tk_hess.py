"""Parity smoke for the default-on Tk·KK' analytical Hessian.

`tk_kkt_hessian_analytical` in `src/reml.rs` adds mgcv's full `det2`
W-dependent Tk·KK' Hessian (gdi.c:919-932, pieces P1+P2+P4+P5) to the
IFT Hessian whenever the Tk·KK' gradient term is active — default-on
for InverseGaussian / Binomial / QuasiBinomial, env-opt-in via
`MGCV_TK_GRAD=1` for other families. Combined with mgcv-aligned Newton
score weights in the REML `log|H|` term (reml.rs:541), this is the
algorithmic path mgcv runs for these families.

Run:
    pytest tests/parity/test_tk_hess.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schema import Fixture  # noqa: E402


FIXTURES_DIR = HERE / "fixtures"


@pytest.mark.parametrize(
    "fixture_name,family",
    [
        ("2d_invgauss_log_n800_k10_cr.json", "inverse.gaussian"),
        ("2d_binomial_logit_n1000_k10_cr.json", "binomial"),
    ],
)
def test_tk_hess_default_on_parity(fixture_name, family):
    fix_path = FIXTURES_DIR / fixture_name
    if not fix_path.exists():
        pytest.skip(f"fixture not found: {fix_path}")
    fix = Fixture.load(fix_path)
    inp = fix.inputs

    gam = mgcv_rust.GAM(family)
    gam.fit(
        np.asarray(inp.x_train, dtype=float),
        np.asarray(inp.y_train, dtype=float),
        k=list(inp.k),
        method=inp.method,
        bs=inp.bs[0],
    )

    pred_train = np.asarray(
        gam.predict(np.asarray(inp.x_train, dtype=float)), dtype=float
    )
    pred_test = np.asarray(
        gam.predict(np.asarray(inp.x_test, dtype=float)), dtype=float
    )

    np.testing.assert_allclose(
        pred_train,
        np.asarray(fix.mgcv_output.predictions_train, dtype=float),
        rtol=1e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        pred_test,
        np.asarray(fix.mgcv_output.predictions_test, dtype=float),
        rtol=1e-3,
        atol=2e-3,
    )
