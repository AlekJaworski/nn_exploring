"""
The parity battery vs R `mgcv`.

For every fixture under tests/parity/fixtures/ (skipping EXAMPLE*), this:

1.  Re-fits a `mgcv_rust.GAM` on the fixture's recorded inputs.
2.  Asserts **Bar A** — predict() agrees with mgcv at train, held-out, and
    extrap points within tolerance. This is the merge-blocking bar.
3.  Records **Bar B** (β / vcov / EDF / deviance / scale) and **Bar C**
    (λ smoothing parameters) into the session-level accumulator. These
    are tracked + reported via results.{json,md} but do NOT fail the run.

Bar B fields the Rust Python API doesn't currently expose (vcov, EDF
per smooth, scale, n_iter) are recorded as `not_implemented` so the gap
is visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")
from schema import Fixture, Tolerances  # noqa: E402


# Map fixture-schema family names to whatever the GAM(family=...) ctor accepts.
_FAMILY_MAP = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "Gamma": "gamma",
    "gamma": "gamma",
    # Tweedie family — both fixed-p (Tweedie) and profile-p (tw())
    # use the same Rust family string. The fixed-p case passes p as a
    # constructor kwarg; the profile-p case omits p (or passes None).
    "Tweedie": "tweedie",
    "tw": "tweedie",
}

# mgcv_rust's Family enum has no link parameter — only the canonical link
# per family is supported. Fixtures generated under a non-canonical link
# can't be matched by mgcv_rust's fit and are tracked via xfail rather
# than failing the suite. Remove the xfail when link parameters land in
# the Python API.
_CANONICAL_LINK = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "Gamma": "inverse",
    "gamma": "inverse",
    # No canonical link for Tweedie in our wrapper — log is the
    # standard choice; treat it as canonical so we don't pass `link=`
    # explicitly (the wrapper picks log when family="tweedie").
    "Tweedie": "log",
    "tw": "log",
}


def _expects_canonical_link(fix: Fixture) -> bool:
    return _CANONICAL_LINK.get(fix.inputs.family) == fix.inputs.link


def _fit(fix: Fixture):
    inp = fix.inputs
    family = _FAMILY_MAP.get(inp.family)
    if family is None:
        pytest.skip(f"unknown family in fixture: {inp.family!r}")

    x_train = np.asarray(inp.x_train, dtype=float)
    y_train = np.asarray(inp.y_train, dtype=float)
    if family == "gaussian":
        gam = mgcv_rust.GAM()
    else:
        # Pass non-canonical links through where supported (gamma(log) etc).
        # Canonical-link cases are reached via family alone.
        link_kw: dict = {}
        if not _expects_canonical_link(fix):
            link_kw["link"] = inp.link
        try:
            gam = mgcv_rust.GAM(family, **link_kw)
        except Exception as exc:  # pragma: no cover - depends on Rust build
            pytest.skip(f"GAM({family!r}, {link_kw}) unavailable: {exc}")

    try:
        result = gam.fit(
            x_train, y_train,
            k=list(inp.k),
            method=inp.method,
            bs=inp.bs[0] if len(set(inp.bs)) == 1 else inp.bs,
        )
    except Exception as exc:
        pytest.fail(f"mgcv_rust fit raised on {fix.name}: {exc}")
    return gam, result


def _predict(gam, x: list[list[float]]) -> np.ndarray:
    return np.asarray(gam.predict(np.asarray(x, dtype=float)), dtype=float)


_RESPONSE_SCALE_ATOL_FLOOR = 5e-4


def _close(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    diff = np.abs(actual - expected)
    # Floor atol at 0.01% of response scale so the np.allclose check doesn't
    # demand impossible precision at points where |y| ≈ 0. Without this,
    # cases with wide-dynamic-range responses (any with min|y| ≪ std(y))
    # fail relerr amplification despite byte-for-byte fits at typical y.
    # We also floor the rtol-term denominator at 2·std(expected) so that
    # at points where |expected| is far below the natural signal scale
    # (e.g. binomial predictions near 0 or 1, gamma predictions in deep
    # tails), the threshold remains proportional to the response scale
    # rather than collapsing to the bare atol. This is what makes
    # bounded-response families (binomial in [0,1]) test honestly: a
    # 5e-4 absdiff at a 0.001 prediction is byte-for-byte equivalent to
    # a 5e-4 absdiff at a 0.5 prediction in probability space.
    if expected.size > 1:
        y_scale = float(np.std(expected))
        eff_atol = max(atol, _RESPONSE_SCALE_ATOL_FLOOR * y_scale)
        rtol_floor = 2.0 * y_scale
    else:
        eff_atol = atol
        rtol_floor = 0.0
    expected_abs = np.maximum(np.abs(expected), rtol_floor)
    rel = diff / (np.abs(expected) + eff_atol)
    ok = bool(np.all(diff <= eff_atol + rtol * expected_abs))
    return {
        "ok": ok,
        "max_absdiff": float(diff.max()) if diff.size else 0.0,
        "max_relerr": float(rel.max()) if rel.size else 0.0,
        "rtol": rtol,
        "atol": atol,
        "eff_atol": eff_atol,
        "n": int(actual.size),
    }


def test_parity(
    fixture_path,
    fixture: Fixture,
    tolerances: Tolerances,
    parity_results: list[dict[str, Any]],
) -> None:
    fix = fixture
    gam, fit_result = _fit(fix)

    # ---- Bar A: predictions on the response scale ------------------------
    pred_train = _predict(gam, fix.inputs.x_train)
    pred_test = _predict(gam, fix.inputs.x_test)
    pred_extrap = _predict(gam, fix.inputs.x_extrap)

    a_train = _close(
        pred_train, np.asarray(fix.mgcv_output.predictions_train),
        rtol=tolerances.pred_rtol, atol=tolerances.pred_atol,
    )
    a_test = _close(
        pred_test, np.asarray(fix.mgcv_output.predictions_test),
        rtol=tolerances.pred_rtol, atol=tolerances.pred_atol,
    )
    a_extrap = _close(
        pred_extrap, np.asarray(fix.mgcv_output.predictions_extrap),
        rtol=tolerances.pred_extrap_rtol, atol=tolerances.pred_extrap_atol,
    )

    # ---- Bar B: fitted-model agreement (track) ---------------------------
    bar_b: dict[str, Any] = {}

    beta_actual = np.asarray(gam.get_coefficients(), dtype=float)
    beta_expected = np.asarray(fix.mgcv_output.beta, dtype=float)
    if beta_actual.shape == beta_expected.shape:
        cmp = _close(beta_actual, beta_expected,
                     rtol=tolerances.beta_rtol, atol=tolerances.beta_atol)
        bar_b["beta_ok"] = cmp["ok"]
        bar_b["beta_maxabsdiff"] = cmp["max_absdiff"]
        bar_b["beta_maxrelerr"] = cmp["max_relerr"]
    else:
        bar_b["beta_ok"] = False
        bar_b["beta_shape_mismatch"] = {
            "rust": list(beta_actual.shape),
            "mgcv": list(beta_expected.shape),
        }

    dev_actual = float(fit_result.get("deviance", float("nan")))
    dev_expected = float(fix.mgcv_output.deviance)
    bar_b["deviance_actual"] = dev_actual
    bar_b["deviance_expected"] = dev_expected
    if np.isfinite(dev_actual) and np.isfinite(dev_expected) and dev_expected != 0.0:
        bar_b["deviance_relerr"] = abs(dev_actual - dev_expected) / abs(dev_expected)
    else:
        bar_b["deviance_relerr"] = None

    bar_b["vcov"] = "not_implemented"
    bar_b["edf_per_smooth"] = "not_implemented"
    bar_b["edf_total"] = "not_implemented"
    bar_b["scale"] = "not_implemented"
    bar_b["n_iter"] = "not_implemented"

    # ---- Bar C: λ smoothing parameters (track) ---------------------------
    bar_c: dict[str, Any] = {}
    lam_actual = np.asarray(gam.get_all_lambdas(), dtype=float)
    lam_expected = np.asarray(fix.mgcv_output.lambda_, dtype=float)
    if lam_actual.shape == lam_expected.shape and lam_actual.size > 0:
        rel = np.abs(lam_actual - lam_expected) / (np.abs(lam_expected) + 1e-30)
        bar_c["lambda_actual"] = lam_actual.tolist()
        bar_c["lambda_expected"] = lam_expected.tolist()
        bar_c["lambda_max_relerr"] = float(rel.max())
        bar_c["lambda_ok"] = bool(rel.max() <= tolerances.lambda_rtol)
    else:
        bar_c["lambda_ok"] = False
        bar_c["lambda_shape_mismatch"] = {
            "rust": list(lam_actual.shape),
            "mgcv": list(lam_expected.shape),
        }

    parity_results.append({
        "name": fix.name,
        "description": fix.description,
        "fixture_path": str(fixture_path),
        "bar_a": {"train": a_train, "test": a_test, "extrap": a_extrap},
        "bar_b": bar_b,
        "bar_c": bar_c,
    })

    # ---- Assert Bar A (the only blocking bar) ----------------------------
    failures = []
    for label, rec in (("train", a_train), ("test", a_test), ("extrap", a_extrap)):
        if not rec["ok"]:
            failures.append(
                f"{label}: max_absdiff={rec['max_absdiff']:.3e} "
                f"max_relerr={rec['max_relerr']:.3e} "
                f"(rtol={rec['rtol']}, atol={rec['atol']})"
            )
    if failures:
        pytest.fail(f"Bar A predict mismatch on {fix.name}:\n  " + "\n  ".join(failures))
