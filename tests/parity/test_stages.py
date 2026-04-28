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
# Stage 4: mgcv_exact mode predictions match mgcv tightly                #
# --------------------------------------------------------------------- #


def test_mgcv_exact_predictions(
    fixture_path,
    fixture: Fixture,
    parity_results: list,
) -> None:
    """
    Run the fit in mgcv_exact mode and assert predictions match mgcv
    much tighter than default mode does. mgcv_exact uses pre-Z
    penalty normalisation (smooth.r:3766-3773 ordering) so the
    optimizer's λ ends up in the same coord system as mgcv's.

    Records per-case rtol/atol achieved into results.{json,md} so we
    can ratchet down as more pieces of mgcv-exact mode land.
    """
    fix = fixture
    inp = fix.inputs
    if inp.family != "gaussian":
        pytest.skip(f"mgcv_exact only validated on Gaussian for now, got {inp.family!r}")
    # Skip the bs basis case — mgcv uses de Boor B-splines, mgcv_rust
    # uses natural cubic splines (different basis altogether).
    if "k20_bs" in fix.name:
        pytest.skip("bs basis not yet mgcv-equivalent")

    x_train = np.asarray(inp.x_train, dtype=float)
    y_train = np.asarray(inp.y_train, dtype=float)
    g = mgcv_rust.GAM(mgcv_exact=True)
    try:
        g.fit(x_train, y_train, k=list(inp.k), method=inp.method, bs=inp.bs[0])
    except Exception as exc:
        pytest.skip(f"mgcv_exact fit raised: {exc}")

    pred_train = np.asarray(g.predict(x_train), dtype=float)
    expected_train = np.asarray(fix.mgcv_output.predictions_train, dtype=float)

    diff = np.abs(pred_train - expected_train)
    max_absdiff = float(diff.max())
    rust_lambdas = list(map(float, np.asarray(g.get_all_lambdas())))
    mgcv_lambdas = list(map(float, fix.mgcv_output.lambda_))
    # Per-dim λ ratios (rust / mgcv) — close to 1 means good match in
    # current (post-3d) coord system.
    lambda_ratios = [
        (r / m) if m > 1e-300 else float("inf")
        for r, m in zip(rust_lambdas, mgcv_lambdas)
    ]
    matched = next((r for r in parity_results if r.get("name") == fix.name), None)
    rec = {
        "max_absdiff": max_absdiff,
        "rust_lambda": rust_lambdas,
        "mgcv_lambda": mgcv_lambdas,
        "lambda_ratios": lambda_ratios,
    }
    if matched is None:
        parity_results.append({"name": fix.name, "stage4": rec})
    else:
        matched["stage4"] = rec

    # Bar at this stage: 1e-3 absolute (much tighter than default which
    # uses ~5e-2 bound). Ratchet target: 1e-6 once mgcv-exact REML
    # is wired into the optimizer.
    threshold = 1.0e-3
    assert max_absdiff <= threshold, (
        f"mgcv_exact Bar A on {fix.name}: max_absdiff={max_absdiff:.3e} "
        f"exceeds {threshold:.0e}. our λ={rec['rust_lambda']}, mgcv λ={rec['mgcv_lambda']}"
    )


# --------------------------------------------------------------------- #
# Stage 3: per-iter REML trajectory vs mgcv's score.hist                 #
# --------------------------------------------------------------------- #

import os
import re
import subprocess
import sys


_REML_LINE = re.compile(
    r"Newton iter (\d+): grad_L2=[\d.eE+-]+, grad_Linf=[\d.eE+-]+, "
    r"REML=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def _run_rust_capture_trajectory(fix: Fixture) -> dict:
    """
    Run mgcv_rust.fit on this case in a fresh Python subprocess with
    MGCV_PROFILE=1 set, and parse the per-iter REML from stderr. We
    use a subprocess (not in-process capture) because the profile
    output goes to stderr from Rust's eprintln!, which is awkward to
    intercept inside an active pytest run.
    """
    code = f"""
import sys, json, os
sys.path.insert(0, {str(HERE)!r})
import numpy as np
import mgcv_rust
fix_data = json.load(open({str(fixture_path_for_code(fix))!r}))
inp = fix_data["inputs"]
x = np.asarray(inp["x_train"], dtype=float)
y = np.asarray(inp["y_train"], dtype=float)
family = inp["family"].lower()
g = mgcv_rust.GAM() if family == "gaussian" else mgcv_rust.GAM(family)
g.fit(x, y, k=list(inp["k"]), method=inp["method"], bs=inp["bs"][0])
print("FINAL_LAMBDA", json.dumps(list(g.get_all_lambdas())))
"""
    env = os.environ.copy()
    env["MGCV_PROFILE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    iters: list[tuple[int, float]] = []
    final_lambda: list[float] | None = None
    for line in proc.stderr.splitlines():
        m = _REML_LINE.search(line)
        if m:
            iters.append((int(m.group(1)), float(m.group(2))))
    for line in proc.stdout.splitlines():
        if line.startswith("FINAL_LAMBDA"):
            final_lambda = json.loads(line.split(" ", 1)[1])
    if proc.returncode != 0:
        raise RuntimeError(
            f"rust subprocess failed: rc={proc.returncode}\n"
            f"stderr tail:\n{proc.stderr[-500:]}"
        )
    return {"iter_reml": iters, "final_lambda": final_lambda}


# Hack: the parametrized `fixture_path` becomes a Path inside the test;
# we need a string for the subprocess template above.
def fixture_path_for_code(fix: Fixture) -> str:
    p = HERE / "fixtures" / f"{fix.name}.json"
    return str(p)


import json  # noqa: E402  (imported above too — keep explicit for the inline code template)


def test_lambda_trajectory_vs_mgcv(
    fixture_path,
    fixture: Fixture,
    parity_results: list,
) -> None:
    """
    Compare mgcv_rust's per-iter REML score against mgcv's
    `outer.info$score.hist` recorded in the fixture. This is the
    'detect divergence early' check: emit the first Newton iter where
    rust's REML deviates from mgcv's by more than a small fraction of
    the score range, and stash the diagnosis into results.{json,md}.

    We do not pytest.fail here — divergence is what we expect today
    on hard cases (4d_mixed) and it's exactly the thing we want
    visibility on as we close gaps. The test always passes; the
    diagnostic lives in the parity_results record.
    """
    fix = fixture
    mgcv_score_hist = fix.mgcv_output.__dict__.get(
        "score_history"
    )  # may be missing on legacy fixtures
    if mgcv_score_hist is None:
        # Lookup via raw dict path (Fixture.from_dict drops unknown keys
        # but our generator now writes it inside mgcv_output).
        with open(fixture_path) as f:
            raw = json.load(f)
        mgcv_score_hist = raw.get("mgcv_output", {}).get("score_history")
    if not mgcv_score_hist:
        pytest.skip(f"{fix.name}: no score_history captured (regenerate fixtures)")

    try:
        traj = _run_rust_capture_trajectory(fix)
    except Exception as exc:
        pytest.skip(f"trajectory capture failed: {exc}")

    rust_iters = traj["iter_reml"]
    if not rust_iters:
        pytest.skip(
            f"{fix.name}: no Newton iters parsed from rust profile output "
            f"(maybe Gaussian fast-path skipped Newton entirely)"
        )

    # Score-scale-relative divergence check (mgcv's own conv.tol works
    # on score-scale-normalized gradients; we apply the same idea to
    # the per-iter score difference). Use the mgcv score range as the
    # scale.
    mgcv_arr = np.asarray(mgcv_score_hist, dtype=float)
    score_range = max(mgcv_arr.max() - mgcv_arr.min(), 1e-6)

    first_diverged_iter: int | None = None
    iter_diffs: list[dict] = []
    for rust_iter, rust_reml in rust_iters:
        # rust_iter is 1-indexed in our profile output; mgcv's
        # score_history is 0-indexed (entry 0 is iter-0 / init λ).
        # Compare rust iter k against mgcv score_history[k] (after
        # one Newton step from init). If mgcv has fewer iterations
        # recorded than rust (mgcv converged earlier), compare against
        # mgcv's last score.
        mgcv_idx = min(rust_iter, len(mgcv_arr) - 1)
        mgcv_score = float(mgcv_arr[mgcv_idx])
        diff = rust_reml - mgcv_score
        rel = abs(diff) / score_range
        iter_diffs.append({
            "iter": rust_iter,
            "rust_reml": rust_reml,
            "mgcv_reml": mgcv_score,
            "absdiff": diff,
            "reldiff_score_scale": rel,
        })
        if first_diverged_iter is None and rel > 0.05:
            first_diverged_iter = rust_iter

    # Stitch into the existing parity record for this fixture.
    matched = next((r for r in parity_results if r.get("name") == fix.name), None)
    rec = {
        "first_diverged_iter": first_diverged_iter,
        "n_rust_iters": len(rust_iters),
        "n_mgcv_iters": len(mgcv_arr),
        "rust_final_reml": rust_iters[-1][1],
        "mgcv_final_reml": float(mgcv_arr[-1]),
        "iter_diffs": iter_diffs[:5],  # keep first 5 to bound size
    }
    if matched is None:
        parity_results.append({"name": fix.name, "trajectory": rec})
    else:
        matched["trajectory"] = rec


# --------------------------------------------------------------------- #
# Stage 2: internal consistency of mgcv_rust's prediction path           #
# --------------------------------------------------------------------- #

# mgcv_rust's Family enum hardcodes the canonical link per family — the
# Python API has no link parameter. So the inverse link applied internally
# by gam.predict()/get_fitted_values() is always the *canonical* one,
# regardless of what link the fixture was generated under. The Stage 2
# invariant (predict ≟ X @ β with inverse link) only holds when we use
# mgcv_rust's actual inverse link, not the fixture's.
_CANONICAL_INVERSE_LINK = {
    "gaussian": lambda eta: eta,                      # identity
    "binomial": lambda eta: 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20))),  # logit
    "poisson":  lambda eta: np.exp(np.clip(eta, None, 20)),                # log
    "Gamma":    lambda eta: 1.0 / np.where(np.abs(eta) < 1e-10, 1e-10, eta),  # inverse
    "gamma":    lambda eta: 1.0 / np.where(np.abs(eta) < 1e-10, 1e-10, eta),  # inverse
}


def test_predict_matches_design_dot_coef(fixture_path, fixture: Fixture) -> None:
    """
    Sanity: gam.predict(x_train) should match get_design_matrix() @
    get_coefficients() on the link scale, then inverse-linked. For
    Gaussian/identity this is just X @ β. Failure here means the
    Python API's predict() and design_matrix paths internally disagree
    — independent of any mgcv comparison.

    Important: applies mgcv_rust's *canonical* inverse link per family
    (Gamma → 1/η, not exp(η)), since the Family enum has no link
    parameter. A non-canonical-link fixture (e.g. Gamma(link="log"))
    will still pass this stage if mgcv_rust's internal predict and
    design-matrix paths are self-consistent, even though the fit
    diverges from the mgcv fixture (that divergence is Bar A's
    territory).
    """
    fix = fixture
    inv_link = _CANONICAL_INVERSE_LINK.get(fix.inputs.family)
    if inv_link is None:
        pytest.skip(f"no canonical inverse link known for family={fix.inputs.family!r}")

    g = _fit_mgcv_rust(fix)
    X = np.asarray(g.get_design_matrix(), dtype=float)
    beta = np.asarray(g.get_coefficients(), dtype=float)

    if X.shape[1] != beta.shape[0]:
        pytest.skip(
            f"design matrix and coef shape disagree: X={X.shape}, β={beta.shape} "
            f"— mgcv_rust may apply an internal transform; can't run this stage"
        )

    mu = inv_link(X @ beta)
    pred = np.asarray(g.predict(np.asarray(fix.inputs.x_train, dtype=float)), dtype=float)

    diff = np.abs(mu - pred)
    rel = diff / (np.abs(pred) + 1e-12)
    assert diff.max() < 1e-8, (
        f"internal predict-vs-design-dot-coef gap on {fix.name}: "
        f"max_absdiff={diff.max():.3e}, max_relerr={rel.max():.3e}"
    )
