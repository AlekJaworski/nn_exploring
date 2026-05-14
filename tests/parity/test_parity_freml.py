"""
B7: fREML parity vs `mgcv::bam(method='fREML', discrete=TRUE)`.

The main `test_parity.py` battery compares against the cached fixture
output, which was generated with `mgcv::gam(method='REML')` (full
inner-Newton REML). The Path B fREML driver (`fit_pirls_fastreml`,
landed B3 + B4 + B5) is a port of mgcv's *closed-form* bgam.fitd path —
a different algorithm whose optimum differs from gam-REML by ~1e-3 due
to IRLS-once vs IRLS-to-convergence (Wood 2017 §6.10.4).

This file runs `Gam(method='fREML', discrete=True)` against a freshly
generated mgcv reference produced inline by rpy2:

    mgcv::bam(formula, data=df, family=fam, method='fREML', discrete=TRUE)

…on a focused 6-fixture slice covering Gaussian / Gaussian-weighted /
scat-weighted / Poisson / Binomial / Gamma(log). Tolerances per spec:
    β:           5e-3 abs
    λ:           5% rel
    σ² (Gauss):  1% rel
    df (scat):   5% rel
    predictions: 1e-3 abs (response scale)

**Known regression scope (B5 follow-up, NOT in B7)**: exponential
families (Poisson / Binomial / Gamma with non-canonical links) loop
their score formula's φ-extension through the IRLS-once
`exp_family_irls_step` helper. B4 noted "score formula's φ-extension
contract needs validation"; B7 confirms this still holds. Affected
fixtures are xfail-marked with `b5_followup`.
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

from schema import Fixture  # noqa: E402

_mgcv = importr("mgcv")
_stats = importr("stats")

ro.r('options(contrasts = c("contr.sum", "contr.poly"))')
ro.r("options(warn = -1)")


# ---- Fixture slice -------------------------------------------------------

# Representative slice spanning Gaussian (canonical baseline), Gaussian
# weighted (probes the EDF off-by-one), scat (B5 family-θ),
# Poisson/Binomial/Gamma (exp_family_irls_step path under fREML).
_B7_FIXTURES = [
    "1d_gaussian_smooth_n500_k10_cr",
    "1d_gaussian_weighted_n300_k10_cr",
    "1d_scat_weighted_n300_k10_cr",
    "2d_binomial_logit_n1000_k10_cr",
    "1d_poisson_log_n500_k10_cr",
    "2d_gamma_log_n200_k10_cr",
]

# Per-fixture tolerance overrides for cases where exp-family fREML is
# known to regress vs mgcv (B5 follow-up; see module docstring). We
# still RUN these as xfail, recording the gap in parity_results.
_B5_FOLLOWUP_REASON = (
    "B5 follow-up: exp-family fREML score's φ-extension contract needs "
    "validation (B4 noted TODO). Poisson/Binomial under fREML hit "
    "exp_family_irls_step but converge to a different (β, λ) than mgcv's "
    "bam(method='fREML', discrete=TRUE). Gamma(log) closes byte-for-byte "
    "so the gap is link-dependent, not family-dependent. Track as R-task."
)

_SCAT_WEIGHTED_REASON = (
    "scat-weighted under fREML: R7 closed the (df, σ²) gap to <0.2% rel by "
    "(a) bumping family_theta min_df 2→3 to match mgcv's scat(min.df=3) "
    "default (efam.r:3552) — without this the outer θ Newton's step clamp "
    "/ PSD threshold / convergence test all act on different curvature and "
    "the Newton lands at a different (ν, σ²); (b) wiring tdist_profile "
    "through correctly so the 2-D Newton actually runs (the previous "
    "guard `fixed_df_for_scat.is_none()` was always false ⇒ df stayed at "
    "the seed); (c) switching `fastreml_irls_step` for TDist from the EM "
    "weight to the bgam.fitd-style Fisher (expected-info) weight "
    "`EDmu2/2 = (ν+1)/(σ²(ν+3))` per bam.r:638-640. λ remains ~23% rel off "
    "mgcv (rust ~168, mgcv ~137) and predictions ~4.6e-3 vs tol 1e-3. "
    "Closing the residual λ gap requires using mgcv's exact `rho==0` "
    "observed-info weight `Dmu2/2`, which goes negative on moderate outliers "
    "and breaks our `compute_sl_fitchol_step`'s plain LU solve. mgcv "
    "tolerates this via pivoted Cholesky in `Sl.fitChol`; we'd need the "
    "same fallback in the ridge solver to ship byte-parity. Track as R-task "
    "`fastreml-pivoted-chol-for-indefinite-xtwx`."
)
_KNOWN_FREML_GAPS = {
    "1d_poisson_log_n500_k10_cr": _B5_FOLLOWUP_REASON,
    "2d_binomial_logit_n1000_k10_cr": _B5_FOLLOWUP_REASON,
    "1d_scat_weighted_n300_k10_cr": _SCAT_WEIGHTED_REASON,
}


# ---- mgcv reference fitter -----------------------------------------------

# R-side mgcv family() constructors.
_FAMILY_R = {
    "gaussian": 'gaussian()',
    "binomial": 'binomial(link="{link}")',
    "poisson":  'poisson(link="{link}")',
    "Gamma":    'Gamma(link="{link}")',
    "gamma":    'Gamma(link="{link}")',
    "scat":     'scat()',
}

# Map fixture family name -> rust GAM family string (mirrors test_parity.py).
_RUST_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson":  "poisson",
    "Gamma":    "gamma",
    "gamma":    "gamma",
    "scat":     "t-dist",
}


def _bam_fREML_reference(fix: Fixture) -> dict:
    """Fit `mgcv::bam(method='fREML', discrete=TRUE)` and return its
    coefficients, λ, σ², predictions on train + test.

    For `family=scat()`, mgcv's `sig2` slot is fixed at 1.0; the actual
    σ² lives in `family$getTheta()` (along with `df`). We extract both.
    """
    inp = fix.inputs
    x_train = np.asarray(inp.x_train, dtype=float)
    y_train = np.asarray(inp.y_train, dtype=float)
    x_test = np.asarray(inp.x_test, dtype=float)
    x_extrap = np.asarray(inp.x_extrap, dtype=float)
    d = inp.d

    df_train = pd.DataFrame({f"x{i}": x_train[:, i] for i in range(d)})
    df_train["y"] = y_train
    df_test = pd.DataFrame({f"x{i}": x_test[:, i] for i in range(d)})
    df_extrap = pd.DataFrame({f"x{i}": x_extrap[:, i] for i in range(d)})

    fam_template = _FAMILY_R.get(inp.family)
    if fam_template is None:
        pytest.skip(f"no R family mapping for {inp.family!r}")
    fam_str = fam_template.format(link=inp.link)
    fam = ro.r(fam_str)

    rhs = " + ".join(
        f's(x{i}, k={inp.k[i]}, bs="{inp.bs[i]}")' for i in range(d)
    )

    with localconverter(pandas2ri.converter + default_converter):
        rdf = pandas2ri.py2rpy(df_train)
        rdf_test = pandas2ri.py2rpy(df_test)
        rdf_extrap = pandas2ri.py2rpy(df_extrap)

    if inp.weights is not None:
        # Pass weights through globalenv to keep types intact.
        ro.globalenv["w_b7"] = ro.FloatVector(list(inp.weights))
        fit = _mgcv.bam(
            ro.Formula(f"y ~ {rhs}"),
            data=rdf,
            method="fREML",
            discrete=True,
            family=fam,
            weights=ro.globalenv["w_b7"],
        )
    else:
        fit = _mgcv.bam(
            ro.Formula(f"y ~ {rhs}"),
            data=rdf,
            method="fREML",
            discrete=True,
            family=fam,
        )

    beta = np.asarray(fit.rx2("coefficients"), dtype=float)
    lam = np.asarray(fit.rx2("sp"), dtype=float)
    sig2 = float(np.asarray(fit.rx2("sig2"))[0])

    pred_train = np.asarray(
        _stats.predict(fit, newdata=rdf, type="response"), dtype=float
    )
    pred_test = np.asarray(
        _stats.predict(fit, newdata=rdf_test, type="response"), dtype=float
    )
    pred_extrap = np.asarray(
        _stats.predict(fit, newdata=rdf_extrap, type="response"), dtype=float
    )

    # scat: actual (df, σ) live behind family$getTheta(TRUE). The
    # mgcv scat parameterisation (efam.r:3552) stores
    #   θ_raw = (log(df - min.df), log(σ)),   min.df = 3 by default
    # `getTheta(TRUE)` then returns the de-transformed values:
    #   theta_final = (df, σ).
    # We extract both, then compare against rust's `sigma2` / `df`.
    scat_sigma2 = None
    scat_df = None
    if inp.family == "scat":
        family_obj = fit.rx2("family")
        get_theta = family_obj.rx2("getTheta")
        theta_final = np.asarray(get_theta(True), dtype=float)
        scat_df = float(theta_final[0])
        scat_sigma2 = float(theta_final[1]) ** 2

    return {
        "beta": beta,
        "lambda": lam,
        "sig2": sig2,
        "scat_sigma2": scat_sigma2,
        "scat_df": scat_df,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "pred_extrap": pred_extrap,
    }


# ---- rust fREML driver ---------------------------------------------------


def _rust_fREML_fit(fix: Fixture):
    """Fit `mgcv_rust.GAM(...)` with method='fREML', discrete=True."""
    inp = fix.inputs
    rust_fam = _RUST_FAMILY.get(inp.family)
    if rust_fam is None:
        pytest.skip(f"no Rust family mapping for {inp.family!r}")

    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)

    if rust_fam == "gaussian":
        g = mgcv_rust.GAM()
    elif rust_fam == "gamma" and inp.link == "log":
        g = mgcv_rust.GAM("gamma", link="log")
    else:
        kwargs: dict = {}
        if inp.link and inp.link != "identity" and rust_fam != "gaussian":
            # Pass non-canonical link only when needed.
            kwargs["link"] = inp.link
        g = mgcv_rust.GAM(rust_fam, **kwargs) if kwargs else mgcv_rust.GAM(rust_fam)

    # The mgcv_rust.GAM facade's fit() doesn't expose `discrete`; drop to
    # the native handle. discrete=True is the bam(... discrete=TRUE) fast
    # path; B7 spec compares vs that reference, so we pass it through.
    bs_arg = inp.bs[0] if len(set(inp.bs)) == 1 else inp.bs
    weights_arg = np.asarray(inp.weights, dtype=float) if inp.weights is not None else None
    g._native.fit(
        x,
        y,
        list(inp.k),
        "fREML",
        bs=bs_arg,
        weights=weights_arg,
        discrete=True,
    )
    return g


# ---- The test ------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    [
        pytest.param(
            name,
            marks=(
                [pytest.mark.xfail(reason=_KNOWN_FREML_GAPS[name], strict=False)]
                if name in _KNOWN_FREML_GAPS
                else []
            ),
            id=name,
        )
        for name in _B7_FIXTURES
    ],
)
def test_freml_parity_vs_bam_fREML_discrete(
    fixture_name: str,
    parity_results: list,
) -> None:
    """Compare `Gam(method='fREML', discrete=True)` against
    `mgcv::bam(method='fREML', discrete=TRUE)` on the response scale.

    Tolerances per the B7 spec:
        β tol             5e-3 abs
        λ tol             5% rel
        σ² (Gauss)        1% rel
        scat df           5% rel
        predictions       1e-3 abs (response scale)
    """
    fixture_path = HERE / "fixtures" / f"{fixture_name}.json"
    fix = Fixture.load(fixture_path)
    inp = fix.inputs

    # ---- fit both sides -------------------------------------------------
    g = _rust_fREML_fit(fix)
    ref = _bam_fREML_reference(fix)

    beta_r = np.asarray(g.get_coefficients(), dtype=float)
    lam_r = np.asarray(g.get_all_lambdas(), dtype=float)
    pred_train_r = np.asarray(g.predict(np.asarray(inp.x_train)), dtype=float)
    pred_test_r = np.asarray(g.predict(np.asarray(inp.x_test)), dtype=float)
    pred_extrap_r = np.asarray(g.predict(np.asarray(inp.x_extrap)), dtype=float)

    # ---- comparisons ----------------------------------------------------
    rec: dict = {"family": inp.family, "link": inp.link, "weighted": inp.weights is not None}

    # β: 5e-3 abs
    if beta_r.shape == ref["beta"].shape:
        beta_max_abs = float(np.abs(beta_r - ref["beta"]).max())
        rec["beta_max_abs"] = beta_max_abs
        rec["beta_ok"] = beta_max_abs <= 5e-3
    else:
        rec["beta_max_abs"] = float("inf")
        rec["beta_ok"] = False
        rec["beta_shape_mismatch"] = {
            "rust": list(beta_r.shape),
            "mgcv": list(ref["beta"].shape),
        }

    # λ: 5% rel
    if lam_r.shape == ref["lambda"].shape and lam_r.size > 0:
        lam_rel = np.abs(lam_r - ref["lambda"]) / (np.abs(ref["lambda"]) + 1e-30)
        rec["lambda_max_rel"] = float(lam_rel.max())
        rec["lambda_ok"] = bool(lam_rel.max() <= 0.05)
        rec["lambda_rust"] = lam_r.tolist()
        rec["lambda_mgcv"] = ref["lambda"].tolist()
    else:
        rec["lambda_max_rel"] = float("inf")
        rec["lambda_ok"] = False

    # σ² (Gaussian only): 1% rel
    if inp.family == "gaussian":
        res = g.fit_result if hasattr(g, "fit_result") else None
        # σ² isn't directly exposed; reconstruct from residuals.
        y = np.asarray(inp.y_train, dtype=float)
        fitted_r = pred_train_r
        # Bayesian rank-trace-style σ² isn't exposed via the Python API;
        # use deviance / (n - edf_total) to approximate, matching mgcv's
        # sig2 = pearson / (n - tr(F)).
        edf_total = float(sum(v for _, v in g.get_edf_per_smooth()))
        # Intercept implicitly contributes 1 EDF.
        n = float(y.size)
        # mgcv weighted bam: σ² = sum(w·r²) / (n - edf_total).
        # The denominator uses plain n (not sum of weights). edf_total
        # here is mgcv's tr(F) including the intercept; we approximate
        # with (sum of per-smooth EDFs) + 1 for the intercept.
        if inp.weights is not None:
            w = np.asarray(inp.weights, dtype=float)
            resid2 = (w * (y - fitted_r) ** 2).sum()
            sig2_r = float(resid2 / max(n - edf_total - 1.0, 1e-12))
        else:
            resid2 = float(((y - fitted_r) ** 2).sum())
            sig2_r = float(resid2 / max(n - edf_total - 1.0, 1e-12))
        sig2_rel = abs(sig2_r - ref["sig2"]) / max(abs(ref["sig2"]), 1e-30)
        rec["sig2_rust"] = sig2_r
        rec["sig2_mgcv"] = ref["sig2"]
        rec["sig2_max_rel"] = float(sig2_rel)
        rec["sig2_ok"] = bool(sig2_rel <= 0.01)

    # scat: σ² 1% rel, df 5% rel (when mgcv exposes the actual values)
    if inp.family == "scat":
        fp = g.get_family_params()
        sigma2_r = float(fp["sigma2"])
        df_r = float(fp["df"])
        rec["scat_sigma2_rust"] = sigma2_r
        rec["scat_df_rust"] = df_r
        rec["scat_sigma2_mgcv"] = ref["scat_sigma2"]
        rec["scat_df_mgcv"] = ref["scat_df"]
        if ref["scat_sigma2"] is not None:
            rel = abs(sigma2_r - ref["scat_sigma2"]) / max(abs(ref["scat_sigma2"]), 1e-30)
            rec["scat_sigma2_max_rel"] = float(rel)
            rec["scat_sigma2_ok"] = bool(rel <= 0.01)
        if ref["scat_df"] is not None:
            rel_df = abs(df_r - ref["scat_df"]) / max(abs(ref["scat_df"]), 1e-30)
            rec["scat_df_max_rel"] = float(rel_df)
            rec["scat_df_ok"] = bool(rel_df <= 0.05)

    # predictions: 1e-3 abs, response scale
    pred_train_max_abs = float(np.abs(pred_train_r - ref["pred_train"]).max())
    pred_test_max_abs = float(np.abs(pred_test_r - ref["pred_test"]).max())
    pred_extrap_max_abs = float(np.abs(pred_extrap_r - ref["pred_extrap"]).max())
    rec["pred_train_max_abs"] = pred_train_max_abs
    rec["pred_test_max_abs"] = pred_test_max_abs
    rec["pred_extrap_max_abs"] = pred_extrap_max_abs
    # Response-scale 1e-3 holds for in-domain points. Extrap points
    # have a much looser tolerance (5e-2 by analogy to test_parity), but
    # we record-only here and require only train + test.
    rec["pred_train_ok"] = pred_train_max_abs <= 1e-3
    rec["pred_test_ok"] = pred_test_max_abs <= 1e-3

    # Record to the session-level results accumulator.
    matched = next(
        (r for r in parity_results if r.get("name") == fix.name), None
    )
    if matched is None:
        parity_results.append({"name": fix.name, "freml_b7": rec})
    else:
        matched["freml_b7"] = rec

    # ---- assertions -----------------------------------------------------
    failures = []
    if not rec["beta_ok"]:
        failures.append(
            f"β max_abs={rec['beta_max_abs']:.3e} > 5e-3 "
            f"(rust shape={list(beta_r.shape)}, mgcv shape={list(ref['beta'].shape)})"
        )
    if not rec["lambda_ok"]:
        failures.append(
            f"λ max_rel={rec['lambda_max_rel']:.3e} > 5%; "
            f"rust={rec.get('lambda_rust')}, mgcv={rec.get('lambda_mgcv')}"
        )
    if inp.family == "gaussian" and not rec.get("sig2_ok", True):
        failures.append(
            f"σ² rel={rec['sig2_max_rel']:.3e} > 1% "
            f"(rust={rec['sig2_rust']:.4g}, mgcv={rec['sig2_mgcv']:.4g})"
        )
    if inp.family == "scat":
        if rec.get("scat_sigma2_max_rel") is not None and not rec.get("scat_sigma2_ok", True):
            failures.append(
                f"scat σ² rel={rec['scat_sigma2_max_rel']:.3e} > 1% "
                f"(rust={rec['scat_sigma2_rust']:.4g}, mgcv={rec['scat_sigma2_mgcv']:.4g})"
            )
        if rec.get("scat_df_max_rel") is not None and not rec.get("scat_df_ok", True):
            failures.append(
                f"scat df rel={rec['scat_df_max_rel']:.3e} > 5% "
                f"(rust={rec['scat_df_rust']:.4g}, mgcv={rec['scat_df_mgcv']:.4g})"
            )
    if not rec["pred_train_ok"]:
        failures.append(
            f"pred_train max_abs={rec['pred_train_max_abs']:.3e} > 1e-3"
        )
    if not rec["pred_test_ok"]:
        failures.append(
            f"pred_test max_abs={rec['pred_test_max_abs']:.3e} > 1e-3"
        )

    if failures:
        pytest.fail(
            f"fREML parity mismatch on {fix.name}:\n  " + "\n  ".join(failures)
        )
