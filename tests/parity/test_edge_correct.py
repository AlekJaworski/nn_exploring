"""
Default-mode parity on the two log-link non-canonical-dispersion fixtures
that historically motivated the (now-removed) in-loop MGCV_EDGE_CORRECT
mechanism.

History (2026-05-14): an earlier iteration of this file tested an
in-loop step-clamp gated behind `MGCV_EDGE_CORRECT=1`. That Rust-original
mechanism was removed after the methodical bisection in Critical
Finding 2026-05-14 confirmed it had no mgcv counterpart (mgcv's actual
edge.correct only writes variance-reporting attrs on `hess` AFTER outer
Newton converges — verified by direct trace, mgcv's outer Newton
trajectory is bit-identical with edge.correct={FALSE,TRUE}).

What this test now asserts: **default-mode parity** on the two log-link
fixtures (Gamma(log) and nb-profile(log)) lands well under the 1e-3
|Δη|∞ bar against mgcv. This captures the actual guarantee — there is
no env-var gymnastics, no Rust-custom mechanism being exercised.

If a future workstream ports mgcv's *real* edge.correct (the post-
Newton variance walk at `gam.fit3.r:1661-1705`) for `Vc` / `edf2`
parity, that's a different test surface (asserts attrs, not
predictions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mgcv_rust = pytest.importorskip("mgcv_rust")

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"

import sys
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schema import Fixture  # noqa: E402


_LOG_LINK_FIXTURES = [
    "2d_gamma_log_n200_k10_cr.json",
    "2d_nb_profile_log_n1000_k10_cr.json",
]


_FAMILY_MAP = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "Gamma": "gamma",
    "gamma": "gamma",
    "Tweedie": "tweedie",
    "tw": "tweedie",
    "quasipoisson": "quasipoisson",
    "quasibinomial": "quasibinomial",
    "inverse.gaussian": "inverse.gaussian",
    "negative.binomial": "negbin",
    "nb": "nb",
}

_CANONICAL_LINK = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "Gamma": "inverse",
    "gamma": "inverse",
    "Tweedie": "log",
    "tw": "log",
    "quasipoisson": "log",
    "quasibinomial": "logit",
    "inverse.gaussian": "log",
    "negative.binomial": "log",
    "nb": "log",
}


def _fit_default(fix: Fixture):
    """Fit the fixture in default mode (no env vars). Mirrors test_parity's
    construction logic."""
    inp = fix.inputs
    family = _FAMILY_MAP.get(inp.family)
    if family is None:
        pytest.skip(f"unknown family in fixture: {inp.family!r}")

    x_train = np.asarray(inp.x_train, dtype=float)
    y_train = np.asarray(inp.y_train, dtype=float)
    if family == "gaussian":
        gam = mgcv_rust.GAM()
    else:
        link_kw: dict = {}
        if _CANONICAL_LINK.get(inp.family) != inp.link:
            link_kw["link"] = inp.link
        if inp.family == "Tweedie":
            link_kw["p"] = 1.5
        if inp.family == "negative.binomial":
            link_kw["theta"] = 2.0
        try:
            gam = mgcv_rust.GAM(family, **link_kw)
        except Exception as exc:
            pytest.skip(f"GAM({family!r}, {link_kw}) unavailable: {exc}")

    gam.fit(
        x_train, y_train,
        k=list(inp.k),
        method=inp.method,
        bs=inp.bs[0] if len(set(inp.bs)) == 1 else inp.bs,
    )
    return gam


def _eta_max_absdiff(rust_pred: np.ndarray, mgcv_pred: np.ndarray, link: str) -> float:
    rust_pred = np.asarray(rust_pred, dtype=float)
    mgcv_pred = np.asarray(mgcv_pred, dtype=float)
    if link == "log":
        floor = max(1e-12, 1e-12 * float(np.abs(mgcv_pred).max()))
        rust_eta = np.log(np.maximum(rust_pred, floor))
        mgcv_eta = np.log(np.maximum(mgcv_pred, floor))
    elif link == "identity":
        rust_eta = rust_pred
        mgcv_eta = mgcv_pred
    elif link == "inverse":
        rust_eta = 1.0 / rust_pred
        mgcv_eta = 1.0 / mgcv_pred
    elif link == "logit":
        eps = 1e-12
        rp = np.clip(rust_pred, eps, 1 - eps)
        mp = np.clip(mgcv_pred, eps, 1 - eps)
        rust_eta = np.log(rp / (1 - rp))
        mgcv_eta = np.log(mp / (1 - mp))
    else:
        rust_eta = rust_pred
        mgcv_eta = mgcv_pred
    return float(np.max(np.abs(rust_eta - mgcv_eta)))


@pytest.mark.parametrize("fixture_name", _LOG_LINK_FIXTURES)
def test_default_mode_parity_under_1e3(fixture_name):
    """Default mode (no env vars) on log-link non-canonical-dispersion
    fixtures lands well under the 1e-3 |Δη|∞ bar. Captures the actual
    parity guarantee for these fixtures.
    """
    fix_path = FIXTURES_DIR / fixture_name
    if not fix_path.exists():
        pytest.skip(f"fixture not found: {fix_path}")
    fix = Fixture.load(fix_path)

    gam = _fit_default(fix)

    inp = fix.inputs
    mgcv_train = np.asarray(fix.mgcv_output.predictions_train, dtype=float)
    rust_train = np.asarray(gam.predict(np.asarray(inp.x_train, dtype=float)), dtype=float)

    delta_eta = _eta_max_absdiff(rust_train, mgcv_train, inp.link)

    assert delta_eta < 1.0e-3, (
        f"{fixture_name}: |Δη|∞ = {delta_eta:.3e} >= 1e-3 in default mode. "
        f"This used to pass at 3.5e-5 (gamma_log) / 4.4e-4 (nb_profile) — "
        f"a regression here indicates the underlying outer-Newton / "
        f"PIRLS / gradient code path changed in a way that breaks these "
        f"two cases. Bisect against the pre-regression HEAD."
    )
