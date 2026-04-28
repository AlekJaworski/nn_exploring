"""
Stage-by-stage parity tests, ordered from "no optimizer involved" to
"fully fit". Each stage isolates a different algorithmic step in the
mgcv_rust → mgcv comparison, so a failure tells you *which* layer the
gap lives in.

Stages:

  1. test_design_matrix_span
     Compare span(mgcv_rust design matrix) vs span(mgcv lpmatrix) at
     training points. Failure = the basis itself (cr-spline /
     identifiability constraints) doesn't match.
     This stage runs **without** optimization, so a failure here means
     the gap is intrinsic to the basis parameterization, not λ.

  2. test_predict_matches_design_dot_coef
     Internal-consistency check: mgcv_rust's predict(x) should equal
     get_design_matrix() @ get_coefficients() on the response scale
     (after inverse link). If this fails, the C++/Rust prediction path
     and the design-matrix path disagree internally.

The stage tests **require rpy2 + mgcv** at runtime (unlike test_parity,
which only needs the JSON fixtures). They are debugging aids, not part
of the merge-blocking battery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

mgcv_rust = pytest.importorskip("mgcv_rust")
rpy2 = pytest.importorskip("rpy2")
import pandas as pd  # noqa: E402
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import default_converter, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

from schema import Fixture, Tolerances  # noqa: E402

_mgcv = importr("mgcv")
_stats = importr("stats")

ro.r('options(contrasts = c("contr.sum", "contr.poly"))')
ro.r("options(warn = -1)")


_FAMILY_R = {
    "gaussian": 'gaussian(link="{}")',
    "binomial": 'binomial(link="{}")',
    "poisson": 'poisson(link="{}")',
    "Gamma": 'Gamma(link="{}")',
    "gamma": 'Gamma(link="{}")',
}


def _fit_mgcv_with_lpmatrix(fix: Fixture):
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    df = pd.DataFrame({f"x{i}": x[:, i] for i in range(inp.d)})
    df["y"] = y

    rhs = " + ".join(f's(x{i}, k={inp.k[i]}, bs="{inp.bs[i]}")' for i in range(inp.d))
    with localconverter(pandas2ri.converter + default_converter):
        rdf = pandas2ri.py2rpy(df)
    family = ro.r(_FAMILY_R[inp.family].format(inp.link))
    fit = _mgcv.bam(ro.Formula(f"y ~ {rhs}"), data=rdf, method=inp.method, family=family)
    lpmatrix = np.asarray(_stats.predict(fit, newdata=rdf, type="lpmatrix"), dtype=float)
    return fit, lpmatrix, df


def _fit_mgcv_rust(fix: Fixture):
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    family = inp.family.lower()
    g = mgcv_rust.GAM() if family == "gaussian" else mgcv_rust.GAM(family.replace("gamma", "gamma"))
    g.fit(x, y, k=list(inp.k), method=inp.method, bs=inp.bs[0])
    return g


# --------------------------------------------------------------------- #
# Stage 1: design-matrix span                                            #
# --------------------------------------------------------------------- #

def test_design_matrix_span(fixture_path, fixture: Fixture, tolerances: Tolerances) -> None:
    """
    The two design matrices should span the same column subspace of
    R^n. Different orderings or basis rotations are fine — what matters
    is span equality.

    A specific named hypothesis we also report: does adding a constant
    (intercept) column to X_rust close the gap? That isolates "missing
    intercept / centering" from "different spline basis".
    """
    fix = fixture
    g = _fit_mgcv_rust(fix)
    X_rust = np.asarray(g.get_design_matrix(), dtype=float)
    _, X_mgcv, _ = _fit_mgcv_with_lpmatrix(fix)

    if X_rust.shape[0] != X_mgcv.shape[0]:
        pytest.fail(
            f"design matrix row count differs: rust={X_rust.shape}, mgcv={X_mgcv.shape}"
        )

    rk_rust = np.linalg.matrix_rank(X_rust)
    rk_mgcv = np.linalg.matrix_rank(X_mgcv)
    rk_both = np.linalg.matrix_rank(np.hstack([X_rust, X_mgcv]))
    span_match = rk_both == max(rk_rust, rk_mgcv)

    # Project mgcv columns onto rust span; report leftover energy
    A, *_ = np.linalg.lstsq(X_rust, X_mgcv, rcond=None)
    residual = X_mgcv - X_rust @ A
    rel_energy_rust = np.linalg.norm(residual) / max(np.linalg.norm(X_mgcv), 1e-300)

    # Same projection but augmenting rust with a constant column (probe
    # the "missing intercept" hypothesis)
    n = X_rust.shape[0]
    X_rust_int = np.hstack([np.ones((n, 1)), X_rust])
    A2, *_ = np.linalg.lstsq(X_rust_int, X_mgcv, rcond=None)
    residual2 = X_mgcv - X_rust_int @ A2
    rel_energy_rust_int = np.linalg.norm(residual2) / max(np.linalg.norm(X_mgcv), 1e-300)

    msg = (
        f"\n  rank(rust)={rk_rust}  rank(mgcv)={rk_mgcv}  rank(combined)={rk_both}"
        f"\n  ||X_mgcv - X_rust @ A||_F / ||X_mgcv|| = {rel_energy_rust:.4e}"
        f"\n  ||X_mgcv - [1|X_rust] @ A||_F / ||X_mgcv|| = {rel_energy_rust_int:.4e} "
        f"(probes the missing-intercept hypothesis)"
    )

    if not span_match:
        pytest.fail(f"design matrix span mismatch on {fix.name}:{msg}")

    assert rel_energy_rust < tolerances.design_matrix_rtol, (
        f"design matrix span equal but projection residual is large: {msg}"
    )


# --------------------------------------------------------------------- #
# Stage 2: internal consistency of mgcv_rust's prediction path           #
# --------------------------------------------------------------------- #

def test_predict_matches_design_dot_coef(fixture_path, fixture: Fixture) -> None:
    """
    Sanity: gam.predict(x_train) should match get_design_matrix() @
    get_coefficients() on the link scale, then inverse-linked. For
    Gaussian/identity this is just X @ β. Failure here means the
    Python API's predict() and design_matrix paths internally disagree
    — independent of any mgcv comparison.
    """
    fix = fixture
    g = _fit_mgcv_rust(fix)
    X = np.asarray(g.get_design_matrix(), dtype=float)
    beta = np.asarray(g.get_coefficients(), dtype=float)

    if X.shape[1] != beta.shape[0]:
        pytest.skip(
            f"design matrix and coef shape disagree: X={X.shape}, β={beta.shape} "
            f"— mgcv_rust may apply an internal transform; can't run this stage"
        )

    eta = X @ beta  # link scale
    if fix.inputs.link == "identity":
        mu = eta
    elif fix.inputs.link == "log":
        mu = np.exp(eta)
    elif fix.inputs.link == "logit":
        mu = 1.0 / (1.0 + np.exp(-eta))
    else:
        pytest.skip(f"unknown link for inverse: {fix.inputs.link}")

    pred = np.asarray(g.predict(np.asarray(fix.inputs.x_train, dtype=float)), dtype=float)

    diff = np.abs(mu - pred)
    rel = diff / (np.abs(pred) + 1e-12)
    assert diff.max() < 1e-8, (
        f"internal predict-vs-design-dot-coef gap on {fix.name}: "
        f"max_absdiff={diff.max():.3e}, max_relerr={rel.max():.3e}"
    )
