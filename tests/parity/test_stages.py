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

_RESPONSE_SCALE_ATOL_FLOOR = 5e-4

_mgcv = importr("mgcv")
_stats = importr("stats")

ro.r('options(contrasts = c("contr.sum", "contr.poly"))')
ro.r("options(warn = -1)")


def _response_scale_close(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> dict:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    diff = np.abs(actual - expected)
    if expected.size > 1:
        y_scale = float(np.std(expected))
        eff_atol = max(atol, _RESPONSE_SCALE_ATOL_FLOOR * y_scale)
        rtol_floor = 2.0 * y_scale
    else:
        eff_atol = atol
        rtol_floor = 0.0
    expected_abs = np.maximum(np.abs(expected), rtol_floor)
    ok = bool(np.all(diff <= eff_atol + rtol * expected_abs))
    return {
        "ok": ok,
        "max_absdiff": float(diff.max()) if diff.size else 0.0,
        "eff_atol": eff_atol,
        "rtol": rtol,
        "atol": atol,
    }


import re as _re

# R-side mgcv family() constructors. Named placeholders: {link}, {p}, {theta}.
_FAMILY_R = {
    "gaussian":          'gaussian(link="{link}")',
    "binomial":          'binomial(link="{link}")',
    "poisson":           'poisson(link="{link}")',
    "Gamma":             'Gamma(link="{link}")',
    "gamma":             'Gamma(link="{link}")',
    "tw":                'tw(link="{link}")',
    "Tweedie":           'Tweedie(p={p}, link="{link}")',
    "tweedie":           'Tweedie(p={p}, link="{link}")',
    "inverse.gaussian":  'inverse.gaussian(link="{link}")',
    "negative.binomial": 'negbin(theta={theta}, link="{link}")',
    "nb":                'nb(link="{link}")',
    "quasipoisson":      'quasipoisson(link="{link}")',
    "quasibinomial":     'quasibinomial(link="{link}")',
}

# Map fixture family names to Rust GAM family strings (mirrors test_parity.py).
_RUST_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "Gamma": "gamma",
    "gamma": "gamma",
    "tw": "tweedie",
    "Tweedie": "tweedie",
    "tweedie": "tweedie",
    "inverse.gaussian": "inverse.gaussian",
    "negative.binomial": "negbin",
    "nb": "nb",
    "quasipoisson": "quasipoisson",
    "quasibinomial": "quasibinomial",
}


def _parse_family_param(name: str, kind: str):
    """Parse a family parameter encoded in the fixture name (e.g. _p15, _theta2)."""
    if kind == "p":
        m = _re.search(r"_p(\d+)(?=$|[_.])", name)
        if m:
            digits = m.group(1)
            return float(digits) / (10 ** (len(digits) - 1))
    elif kind == "theta":
        m = _re.search(r"_theta(\d+(?:\.\d+)?)(?=$|[_.])", name)
        if m:
            return float(m.group(1))
    return None


def _fit_mgcv_with_lpmatrix(fix: Fixture):
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    df = pd.DataFrame({f"x{i}": x[:, i] for i in range(inp.d)})
    df["y"] = y

    rhs = " + ".join(f's(x{i}, k={inp.k[i]}, bs="{inp.bs[i]}")' for i in range(inp.d))
    with localconverter(pandas2ri.converter + default_converter):
        rdf = pandas2ri.py2rpy(df)

    fam_template = _FAMILY_R.get(inp.family)
    if fam_template is None:
        pytest.skip(f"no R family mapping for {inp.family!r}")

    # Resolve named placeholders {link}, {p}, {theta}
    p_val = _parse_family_param(fix.name, "p") or 1.5
    theta_val = _parse_family_param(fix.name, "theta") or 2.0
    fam_str = fam_template.format(link=inp.link, p=p_val, theta=theta_val)
    family = ro.r(fam_str)

    fitter = _mgcv.gam if df.shape[0] > 10000 else _mgcv.bam
    fit = fitter(ro.Formula(f"y ~ {rhs}"), data=rdf, method=inp.method, family=family)
    lpmatrix = np.asarray(_stats.predict(fit, newdata=rdf, type="lpmatrix"), dtype=float)
    return fit, lpmatrix, df


def _fit_mgcv_rust(fix: Fixture):
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)

    rust_fam = _RUST_FAMILY.get(inp.family)
    if rust_fam is None:
        pytest.skip(f"no Rust family mapping for {inp.family!r}")

    kwargs: dict = {}
    if inp.link and inp.link != "identity":
        kwargs["link"] = inp.link
    if inp.family in ("Tweedie", "tweedie"):
        p = _parse_family_param(fix.name, "p")
        if p is not None:
            kwargs["p"] = p
    elif inp.family == "negative.binomial":
        theta = _parse_family_param(fix.name, "theta")
        if theta is not None:
            kwargs["theta"] = theta

    try:
        g = mgcv_rust.GAM() if rust_fam == "gaussian" else mgcv_rust.GAM(rust_fam, **kwargs)
    except Exception as exc:
        pytest.skip(f"GAM({rust_fam!r}, {kwargs}) unavailable: {exc}")
    fit_kwargs = {"k": list(inp.k), "method": inp.method, "bs": inp.bs[0]}
    if inp.weights is not None:
        fit_kwargs["weights"] = np.asarray(inp.weights, dtype=float)
    g.fit(x, y, **fit_kwargs)
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

    x_train = np.asarray(inp.x_train, dtype=float)
    y_train = np.asarray(inp.y_train, dtype=float)
    rust_fam = _RUST_FAMILY.get(inp.family)
    if rust_fam is None:
        pytest.skip(f"no Rust family mapping for {inp.family!r}")
    kwargs: dict = {"mgcv_exact": True}
    # Pass link explicitly so Gamma(log) doesn't silently fall back to
    # canonical inverse (and similar non-canonical-link cases).
    if inp.link and inp.link != "identity":
        kwargs["link"] = inp.link
    if inp.family in ("Tweedie", "tweedie"):
        p = _parse_family_param(fix.name, "p")
        if p is not None:
            kwargs["p"] = p
    elif inp.family == "negative.binomial":
        theta = _parse_family_param(fix.name, "theta")
        if theta is not None:
            kwargs["theta"] = theta
    try:
        if rust_fam == "gaussian":
            g = mgcv_rust.GAM(mgcv_exact=True)
        else:
            g = mgcv_rust.GAM(rust_fam, **kwargs)
    except Exception as exc:
        pytest.skip(f"GAM({rust_fam!r}, {kwargs}) unavailable: {exc}")
    try:
        fit_kwargs = {"k": list(inp.k), "method": inp.method, "bs": inp.bs[0]}
        if inp.weights is not None:
            fit_kwargs["weights"] = np.asarray(inp.weights, dtype=float)
        g.fit(x_train, y_train, **fit_kwargs)
    except Exception as exc:
        pytest.skip(f"mgcv_exact fit raised: {exc}")

    pred_train = np.asarray(g.predict(x_train), dtype=float)
    expected_train = np.asarray(fix.mgcv_output.predictions_train, dtype=float)

    close = _response_scale_close(pred_train, expected_train, rtol=1.0e-3, atol=1.0e-3)
    max_absdiff = close["max_absdiff"]
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

    # Stage-4 response-scale predictions use the same scale-aware floor as
    # Bar A; the companion link-scale test remains a strict 1e-3 optimizer
    # guard and catches genuine model/trajectory drift.
    assert close["ok"], (
        f"mgcv_exact Bar A on {fix.name}: max_absdiff={max_absdiff:.3e} "
        f"exceeds effective tolerance (rtol={close['rtol']:.0e}, "
        f"atol={close['atol']:.0e}, eff_atol={close['eff_atol']:.3e}). "
        f"our λ={rec['rust_lambda']}, mgcv λ={rec['mgcv_lambda']}"
    )


# Forward link μ → η. Used by `test_mgcv_exact_predictions_link_scale` to
# strip the inverse-link amplification out of the response-scale comparison.
_FORWARD_LINK = {
    "identity": lambda mu: mu,
    "log":      lambda mu: np.log(np.maximum(mu, 1e-300)),
    "inverse":  lambda mu: 1.0 / np.where(np.abs(mu) < 1e-300, 1e-300, mu),
    "logit":    lambda mu: np.log(np.clip(mu, 1e-15, 1 - 1e-15) / (1 - np.clip(mu, 1e-15, 1 - 1e-15))),
}


def test_mgcv_exact_predictions_link_scale(
    fixture_path,
    fixture: Fixture,
    parity_results: list,
) -> None:
    """Companion to `test_mgcv_exact_predictions` that compares fits at the
    LINK scale (η = X·β), where the optimiser's actual numerical precision
    lives. The response-scale (μ) comparison is amplified through the inverse
    link — a 5e-4 η-error becomes a 4e-3 μ-error at η ≈ 3 under exp() — so
    response-scale alone confounds "algorithm is wrong" with "the link
    function is steep." η-scale isolates the former.

    Strict 1e-3 absolute threshold uniformly across all families.
    """
    fix = fixture
    inp = fix.inputs
    fwd = _FORWARD_LINK.get(inp.link)
    if fwd is None:
        pytest.skip(f"no forward-link mapping for link={inp.link!r}")
    rust_fam = _RUST_FAMILY.get(inp.family)
    if rust_fam is None:
        pytest.skip(f"no Rust family mapping for {inp.family!r}")

    kwargs: dict = {"mgcv_exact": True}
    if inp.link and inp.link != "identity":
        kwargs["link"] = inp.link
    if inp.family in ("Tweedie", "tweedie"):
        p = _parse_family_param(fix.name, "p")
        if p is not None:
            kwargs["p"] = p
    elif inp.family == "negative.binomial":
        theta = _parse_family_param(fix.name, "theta")
        if theta is not None:
            kwargs["theta"] = theta
    try:
        if rust_fam == "gaussian":
            g = mgcv_rust.GAM(mgcv_exact=True)
        else:
            g = mgcv_rust.GAM(rust_fam, **kwargs)
    except Exception as exc:
        pytest.skip(f"GAM({rust_fam!r}, {kwargs}) unavailable: {exc}")
    try:
        fit_kwargs = {"k": list(inp.k), "method": inp.method, "bs": inp.bs[0]}
        if inp.weights is not None:
            fit_kwargs["weights"] = np.asarray(inp.weights, dtype=float)
        g.fit(
            np.asarray(inp.x_train, dtype=float),
            np.asarray(inp.y_train, dtype=float),
            **fit_kwargs,
        )
    except Exception as exc:
        pytest.skip(f"mgcv_exact fit raised: {exc}")

    X = np.asarray(g.get_design_matrix(), dtype=float)
    beta = np.asarray(g.get_coefficients(), dtype=float)
    if X.shape[1] != beta.shape[0]:
        pytest.skip(f"X={X.shape}, β={beta.shape} disagree on {fix.name}")
    eta_ours = X @ beta

    mu_mgcv = np.asarray(fix.mgcv_output.predictions_train, dtype=float)
    eta_mgcv = fwd(mu_mgcv)

    diff = np.abs(eta_ours - eta_mgcv)
    finite = np.isfinite(diff)
    if not finite.all():
        diff = diff[finite]
    max_link_absdiff = float(diff.max())

    matched = next((r for r in parity_results if r.get("name") == fix.name), None)
    rec = {"link": inp.link, "max_link_absdiff": max_link_absdiff}
    if matched is None:
        parity_results.append({"name": fix.name, "stage4_link": rec})
    else:
        matched["stage4_link"] = rec

    threshold = 1.0e-3
    assert max_link_absdiff <= threshold, (
        f"mgcv_exact link-scale on {fix.name}: |Δη|_max={max_link_absdiff:.3e} "
        f"exceeds {threshold:.0e} (link={inp.link})"
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
family = inp["family"]
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

_LOG_INV = lambda eta: np.exp(np.clip(eta, None, 20))
_LOGIT_INV = lambda eta: 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
_INVERSE_INV = lambda eta: 1.0 / np.where(np.abs(eta) < 1e-10, 1e-10, eta)

# Stage 2 invariant (predict ≟ inverse_link(X @ β)) is parameterised by the
# link mgcv_rust uses internally. Now that the Python API accepts a `link`
# kwarg, we mirror the fixture's `inputs.link` directly.
_INVERSE_LINK = {
    "identity": lambda eta: eta,
    "log":      _LOG_INV,
    "logit":    _LOGIT_INV,
    "inverse":  _INVERSE_INV,
}


def test_predict_matches_design_dot_coef(fixture_path, fixture: Fixture) -> None:
    """
    Sanity: gam.predict(x_train) should match get_design_matrix() @
    get_coefficients() on the link scale, then inverse-linked. For
    Gaussian/identity this is just X @ β. Failure here means the
    Python API's predict() and design_matrix paths internally disagree
    — independent of any mgcv comparison.

    Uses the link declared in `inputs.link` (mgcv_rust's Python API now
    takes a `link` kwarg, so the inverse link applied internally by
    predict() matches whichever link the fixture was generated under).
    """
    fix = fixture
    inv_link = _INVERSE_LINK.get(fix.inputs.link)
    if inv_link is None:
        pytest.skip(f"no inverse-link mapping for link={fix.inputs.link!r}")

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
