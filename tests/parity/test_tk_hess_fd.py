"""Smoke tests for `MGCV_TK_HESS_FD=1`.

The new helper `tk_kkt_hessian_fd` in `src/reml.rs` adds the FD-of-Tk
contribution to the IFT Hessian when `MGCV_TK_HESS_FD=1` is set. This file
exercises that path on the same Bucket-D fixtures the Tk gradient term
already converges on (inverse-Gaussian + a representative Binomial). The
goal is parity: with `MGCV_TK_HESS_FD=1` enabled, the same fits should
remain green.

Run with the regular suite *and* in the env-controlled run:
    pytest tests/parity/test_tk_hess_fd.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schema import Fixture  # noqa: E402


FIXTURES_DIR = HERE / "fixtures"


@pytest.fixture(scope="module")
def _tk_hess_env():
    """Set MGCV_TK_HESS_FD=1 for the duration of this module, restoring
    whatever the caller had set before."""
    saved = os.environ.get("MGCV_TK_HESS_FD")
    os.environ["MGCV_TK_HESS_FD"] = "1"
    yield
    if saved is None:
        os.environ.pop("MGCV_TK_HESS_FD", None)
    else:
        os.environ["MGCV_TK_HESS_FD"] = saved


@pytest.mark.parametrize(
    "fixture_name,family",
    [
        ("2d_invgauss_log_n800_k10_cr.json", "inverse.gaussian"),
        ("2d_binomial_logit_n1000_k10_cr.json", "binomial"),
    ],
)
def test_tk_hess_fd_keeps_parity(_tk_hess_env, fixture_name, family):
    """With MGCV_TK_HESS_FD=1 the affected default-on families (InvGauss /
    Binomial / QuasiBinomial) should continue to match mgcv predictions.

    This is a *non-regression* gate: we are not asserting the Hessian fix
    closes any new gap, only that turning it on does not break existing
    parity.
    """
    fix_path = FIXTURES_DIR / fixture_name
    if not fix_path.exists():
        pytest.skip(f"fixture not found: {fix_path}")
    fix = Fixture.load(fix_path)
    inp = fix.inputs

    assert os.environ.get("MGCV_TK_HESS_FD") == "1"

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

    # Same tolerances as the standing invgauss regression test.
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
