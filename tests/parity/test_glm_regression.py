from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

from schema import Fixture  # noqa: E402


HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    ("fixture_name", "family", "pred_atol", "lambda0_relerr"),
    [
        ("2d_binomial_logit_n200_k10_cr", "binomial", 2e-4, 1e-2),
        ("2d_gamma_inverse_n1000_k10_cr", "gamma", 5e-3, 1e-2),
    ],
)
def test_glm_reml_edge_cases_match_mgcv_predictions_and_primary_lambda(
    fixture_name: str,
    family: str,
    pred_atol: float,
    lambda0_relerr: float,
) -> None:
    """Regression coverage for GLM REML cases sensitive to IFT policy gates."""
    fix = Fixture.load(HERE / "fixtures" / f"{fixture_name}.json")
    inp = fix.inputs

    gam = mgcv_rust.GAM(family)
    gam.fit(
        np.asarray(inp.x_train, dtype=float),
        np.asarray(inp.y_train, dtype=float),
        k=list(inp.k),
        method=inp.method,
        bs=inp.bs[0],
    )

    for x, expected in (
        (inp.x_train, fix.mgcv_output.predictions_train),
        (inp.x_test, fix.mgcv_output.predictions_test),
        (inp.x_extrap, fix.mgcv_output.predictions_extrap),
    ):
        actual = np.asarray(gam.predict(np.asarray(x, dtype=float)), dtype=float)
        np.testing.assert_allclose(actual, np.asarray(expected, dtype=float), atol=pred_atol)

    actual_lambda0 = float(np.asarray(gam.get_all_lambdas(), dtype=float)[0])
    expected_lambda0 = float(np.asarray(fix.mgcv_output.lambda_, dtype=float)[0])
    relerr = abs(actual_lambda0 - expected_lambda0) / max(abs(expected_lambda0), 1e-12)
    assert relerr < lambda0_relerr
