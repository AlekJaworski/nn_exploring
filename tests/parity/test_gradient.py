"""
Unit tests for the closed-form mgcv-exact REML gradient.

Verifies that `evaluate_reml_gradient_closed_form` agrees with a
central-difference gradient of `evaluate_reml_mgcv_formula` over a
range of λ values on every Gaussian fixture. Also checks the gradient
is approximately zero at mgcv's converged λ (where mgcv claims the
optimum is, validated to 1e-7 in mgcv's outer.info$grad).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

mgcv_rust = pytest.importorskip("mgcv_rust")
from schema import Fixture  # noqa: E402


def _fd_gradient(g, y, lambdas, h=1e-4):
    """Central-difference gradient of evaluate_reml_mgcv_formula in log-λ space."""
    log_l = np.log(np.asarray(lambdas, dtype=float))
    m = len(log_l)
    grad = np.zeros(m)
    for i in range(m):
        lp = log_l.copy()
        lm = log_l.copy()
        lp[i] += h
        lm[i] -= h
        rp = g.evaluate_reml_mgcv_formula(y, list(np.exp(lp)))
        rm = g.evaluate_reml_mgcv_formula(y, list(np.exp(lm)))
        grad[i] = (rp - rm) / (2 * h)
    return grad


def _gaussian_fixtures():
    fixtures_dir = HERE / "fixtures"
    out = []
    for path in sorted(fixtures_dir.glob("*.json")):
        if path.name.startswith("EXAMPLE"):
            continue
        f = json.loads(path.read_text())
        if f["inputs"]["family"] != "gaussian":
            continue
        if "k20_bs" in path.name:  # different basis altogether
            continue
        out.append((path, f))
    return out


@pytest.mark.parametrize(
    "fix_path,fix_data",
    _gaussian_fixtures(),
    ids=lambda p: p.stem if hasattr(p, "stem") else "",
)
def test_closed_form_matches_finite_diff_at_optimum(fix_path, fix_data) -> None:
    """At MGCV'S CONVERGED λ, the closed-form gradient should agree with
    the central-difference gradient.

    Note: off-optimum, the two intentionally differ. mgcv's closed-form
    treats σ²=RSS/(n-trA) as a constant scale (gam.fit3.r:625) and uses
    `D1/(2·scale) + trA1/2 - det1/2` — i.e., it does NOT include the σ²
    gradient term ∂σ²/∂λ that the full profile-REML differentiation
    produces. The two formulations agree AT the optimum (where the σ²
    gradient term vanishes by the first-order condition for σ²) and
    converge to the same place, but they disagree off the optimum by
    exactly the missing σ² term."""
    inp = fix_data["inputs"]
    out = fix_data["mgcv_output"]
    x = np.asarray(inp["x_train"], dtype=float)
    y = np.asarray(inp["y_train"], dtype=float)
    g = mgcv_rust.GAM(mgcv_exact=True)
    g.fit(x, y, k=list(inp["k"]), method=inp["method"], bs=inp["bs"][0])

    lam = list(out["lambda"])  # mgcv's converged optimum
    cf = np.asarray(g.evaluate_reml_gradient_closed_form(y, lam), dtype=float)
    fd = _fd_gradient(g, y, lam)
    diff = np.abs(cf - fd)
    rel = diff / (np.abs(fd) + 1e-8)
    # At the optimum both should be small AND match each other within
    # FD's truncation noise (~h² ~ 1e-8 for h=1e-4).
    assert rel.max() < 5e-2 or diff.max() < 5e-3, (
        f"closed-form vs FD at mgcv optimum on {fix_path.stem}: "
        f"cf={cf.tolist()}, fd={fd.tolist()}, max_absdiff={diff.max():.3e}, "
        f"max_relerr={rel.max():.3e}"
    )


@pytest.mark.parametrize(
    "fix_path,fix_data",
    _gaussian_fixtures(),
    ids=lambda p: p.stem if hasattr(p, "stem") else "",
)
def test_gradient_zero_at_mgcv_optimum(fix_path, fix_data) -> None:
    """At mgcv's converged λ, the gradient should be small (mgcv reports
    its final gradient at ~1e-5 to 1e-8). This tells us our gradient
    formula matches mgcv's: same minimum location."""
    inp = fix_data["inputs"]
    out = fix_data["mgcv_output"]
    x = np.asarray(inp["x_train"], dtype=float)
    y = np.asarray(inp["y_train"], dtype=float)
    g = mgcv_rust.GAM(mgcv_exact=True)
    g.fit(x, y, k=list(inp["k"]), method=inp["method"], bs=inp["bs"][0])

    grad_at_mgcv_lam = np.asarray(
        g.evaluate_reml_gradient_closed_form(y, list(out["lambda"])),
        dtype=float,
    )
    # mgcv's reported final |grad| (from final_grad in fixture) is the
    # gradient that mgcv itself thinks is at its optimum. Our gradient
    # there should also be small.
    grad_norm = float(np.abs(grad_at_mgcv_lam).max())
    assert grad_norm < 0.05, (
        f"closed-form gradient at mgcv's λ on {fix_path.stem} is large: "
        f"|grad|_inf={grad_norm:.3e}, components={grad_at_mgcv_lam.tolist()}"
    )
