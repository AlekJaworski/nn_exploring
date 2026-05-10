from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

from schema import Fixture  # noqa: E402


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "2d_invgauss_log_n800_k10_cr.json"


def test_invgauss_log_reml_matches_mgcv_lambda_and_predictions() -> None:
    """Regression for inverse-Gaussian REML using mgcv's Tk determinant term."""
    fix = Fixture.load(FIXTURE)
    inp = fix.inputs

    gam = mgcv_rust.GAM("inverse.gaussian")
    gam.fit(
        np.asarray(inp.x_train, dtype=float),
        np.asarray(inp.y_train, dtype=float),
        k=list(inp.k),
        method=inp.method,
        bs=inp.bs[0],
    )

    pred_train = np.asarray(gam.predict(np.asarray(inp.x_train, dtype=float)), dtype=float)
    pred_test = np.asarray(gam.predict(np.asarray(inp.x_test, dtype=float)), dtype=float)
    pred_extrap = np.asarray(gam.predict(np.asarray(inp.x_extrap, dtype=float)), dtype=float)

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
    np.testing.assert_allclose(
        pred_extrap,
        np.asarray(fix.mgcv_output.predictions_extrap, dtype=float),
        rtol=5e-2,
        atol=2e-3,
    )

    actual_lam = np.asarray(gam.get_all_lambdas(), dtype=float)
    expected_lam = np.asarray(fix.mgcv_output.lambda_, dtype=float)
    lambda_relerr = np.abs(actual_lam - expected_lam) / np.maximum(np.abs(expected_lam), 1e-12)
    assert float(lambda_relerr.max()) < 1e-2, (
        f"inverse-Gaussian lambda drifted from mgcv: actual={actual_lam.tolist()}, "
        f"expected={expected_lam.tolist()}, relerr={lambda_relerr.tolist()}"
    )
