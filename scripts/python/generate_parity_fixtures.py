"""
Generate parity fixtures by calling mgcv via rpy2.

Mirrors the slim pattern in libraries/r_fitting/src/r_fitting/r_model.py
(neighbourhoods repo) — `mgcv.bam(formula, data, method, family)` plus
predictions on the response scale with SE. No k-tuning loop, no other
production extras.

Reads the canonical case battery from tests/parity/cases.py and writes
one fixture JSON per case under tests/parity/fixtures/<case>.json,
conforming to schema_version 1 (tests/parity/README.md).

Run from repo root:

    python scripts/python/generate_parity_fixtures.py

Run a single case:

    python scripts/python/generate_parity_fixtures.py 1d_gaussian_smooth_n500_k10_cr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TESTS_PARITY = REPO_ROOT / "tests" / "parity"
FIXTURES_DIR = TESTS_PARITY / "fixtures"
sys.path.insert(0, str(TESTS_PARITY))

from cases import CASES_BY_NAME, all_cases  # noqa: E402

# rpy2 imports get noisy on first load; quiet them
import rpy2.rinterface_lib.callbacks as _cb  # noqa: E402
from rpy2.rinterface_lib.callbacks import logger as _rpy2_logger  # noqa: E402

_rpy2_logger.setLevel(logging.ERROR)

import rpy2.robjects as ro  # noqa: E402
from rpy2.rinterface_lib import openrlib  # noqa: E402
from rpy2.robjects import default_converter, pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

mgcv = importr("mgcv")
stats = importr("stats")


def _r_str(expr: str) -> str:
    return str(ro.r(expr)[0])

# Sum-to-zero contrasts to match neighbourhoods r_model.py exactly.
ro.r('options(contrasts = c("contr.sum", "contr.poly"))')
ro.r("options(warn = -1)")


# Family builder — same shape as r_model.py::r_family_builders, trimmed
# to the families our cases.py uses. mgcv accepts the bare strings for
# gaussian/binomial/poisson; Gamma needs the function form.
_FAMILY_BUILDERS = {
    "gaussian": lambda link: ro.r(f'gaussian(link="{link}")'),
    "binomial": lambda link: ro.r(f'binomial(link="{link}")'),
    "poisson": lambda link: ro.r(f'poisson(link="{link}")'),
    "Gamma": lambda link: ro.r(f'Gamma(link="{link}")'),
    "gamma": lambda link: ro.r(f'Gamma(link="{link}")'),
    # Tweedie with fixed p — currently hard-codes p=1.5 to match
    # `_gen_tweedie_log`'s data-generating p. If we add cases at other
    # p values, generalize via a per-case `tweedie_p` field on Case.
    "Tweedie": lambda link: ro.r(f'Tweedie(p=1.5, link="{link}")'),
    # Profile p — mgcv's tw() over default range a=1.001..b=1.999.
    "tw": lambda link: ro.r(f'tw(link="{link}")'),
}


def _build_formula(d: int, k: list[int], bs: list[str]) -> ro.Formula:
    rhs = " + ".join(
        f's(x{i}, k={k[i]}, bs="{bs[i]}")' for i in range(d)
    )
    return ro.Formula(f"y ~ {rhs}")


def _df_to_r(df: pd.DataFrame):
    with localconverter(pandas2ri.converter + default_converter):
        return pandas2ri.py2rpy(df)


def _to_np(r_obj) -> np.ndarray:
    return np.asarray(r_obj, dtype=float)


def _build_train_df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    cols["y"] = y
    return pd.DataFrame(cols)


def _build_x_df(x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({f"x{i}": x[:, i] for i in range(x.shape[1])})


def _edf_per_smooth(fit) -> dict[str, float]:
    """
    Map mgcv smooth labels (s(x0)) -> bare term name (x0) -> sum of EDFs
    in that smooth's coefficient span. Mirrors the R script's edf_key().
    """
    out: dict[str, float] = {}
    smooths = fit.rx2("smooth")
    edf_vec = np.asarray(fit.rx2("edf"), dtype=float)
    for sm in smooths:
        term = str(sm.rx2("term")[0])
        first = int(sm.rx2("first.para")[0]) - 1
        last = int(sm.rx2("last.para")[0]) - 1
        out[term] = float(edf_vec[first : last + 1].sum())
    return out


def _scalar(fit, name: str) -> float | None:
    val = fit.rx2(name)
    if val is ro.NULL or val is None:
        return None
    arr = np.asarray(val).ravel()
    if arr.size == 0:
        return None
    return float(arr[0])


def fit_one(case) -> dict[str, Any]:
    realized = case.realize()
    # case.realize() bundles name/description with the inputs; the v1
    # schema puts those on the Fixture, not on Inputs.
    inputs = {k: v for k, v in realized.items() if k not in ("name", "description")}
    family_builder = _FAMILY_BUILDERS.get(inputs["family"])
    if family_builder is None:
        raise ValueError(f"unknown family: {inputs['family']!r}")
    family = family_builder(inputs["link"])

    x_train = np.asarray(inputs["x_train"], dtype=float)
    y_train = np.asarray(inputs["y_train"], dtype=float)
    x_test = np.asarray(inputs["x_test"], dtype=float)
    x_extrap = np.asarray(inputs["x_extrap"], dtype=float)

    df_train = _build_train_df(x_train, y_train)
    df_test = _build_x_df(x_test)
    df_extrap = _build_x_df(x_extrap)

    formula = _build_formula(case.d, case.k, case.bs)

    with openrlib.rlock:
        r_train = _df_to_r(df_train)
        r_test = _df_to_r(df_test)
        r_extrap = _df_to_r(df_extrap)

        # Always use `gam`. Our Rust core matches mgcv's `gam.fit3.r`
        # outer-Newton path exactly (Newton-B1..B5 + Newton-SF + N12).
        # `bam` uses a BFGS-style smoothing-parameter optimizer that
        # converges to slightly different ρ in flat-score regimes (e.g.
        # gamma+log small-n), so bam-generated fixtures don't match our
        # Rust even when our Rust is gam-correct. `gam` is slower but
        # the fixture build is offline and one-shot.
        fitter = mgcv.gam
        # mgcv calls the GCV method "GCV.Cp" (its full name); our Rust
        # core uses the shorter "GCV". Translate so cases.py can stay
        # neutral (the case carries the Rust spelling).
        r_method = "GCV.Cp" if case.method == "GCV" else case.method
        fit = fitter(
            formula,
            data=r_train,
            method=r_method,
            family=family,
        )

        # Model summaries
        beta = _to_np(stats.coef(fit))
        vcov = _to_np(stats.vcov(fit))
        lam = _to_np(fit.rx2("sp"))
        deviance = float(stats.deviance(fit)[0])
        scale = _scalar(fit, "scale")
        if scale is None:
            scale = _scalar(fit, "sig2")
        if scale is None:
            scale = float("nan")

        # n_iter: bam reports outer.info$iter when available, else $iter
        outer_info = fit.rx2("outer.info")
        n_iter = None
        score_history: list[float] | None = None
        final_grad: list[float] | None = None
        if outer_info is not ro.NULL and outer_info is not None:
            try:
                n_iter = int(np.asarray(outer_info.rx2("iter")).ravel()[0])
            except Exception:
                n_iter = None
            try:
                # outer.info$score.hist is a 200-slot vector with NaN
                # padding past the actual iteration count. Strip the
                # NaN tail so consumers see only iter 0 .. n_iter.
                sh_raw = np.asarray(outer_info.rx2("score.hist"), dtype=float)
                sh_clean = sh_raw[~np.isnan(sh_raw)]
                score_history = sh_clean.tolist()
            except Exception:
                score_history = None
            try:
                final_grad = _to_np(outer_info.rx2("grad")).tolist()
            except Exception:
                final_grad = None
        if n_iter is None:
            try:
                n_iter = int(np.asarray(fit.rx2("iter")).ravel()[0])
            except Exception:
                n_iter = -1

        edf_total = float(np.asarray(fit.rx2("edf"), dtype=float).sum())
        edf_per = _edf_per_smooth(fit)

        # Predictions on the response scale (with SE for train/test)
        pred_train = _to_np(stats.predict(fit, newdata=r_train, type="response"))
        pred_test = _to_np(stats.predict(fit, newdata=r_test, type="response"))
        pred_extrap = _to_np(stats.predict(fit, newdata=r_extrap, type="response"))

        pred_train_full = stats.predict(fit, newdata=r_train, type="response", **{"se.fit": True})
        pred_test_full = stats.predict(fit, newdata=r_test, type="response", **{"se.fit": True})
        pred_train_se = _to_np(pred_train_full.rx2("se.fit"))
        pred_test_se = _to_np(pred_test_full.rx2("se.fit"))

    out: dict[str, Any] = {
        "schema_version": 1,
        "name": case.name,
        "description": case.description,
        "metadata": {
            "mgcv_version": _r_str('as.character(packageVersion("mgcv"))'),
            "r_version": _r_str("R.version.string"),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "inputs": inputs,
        "mgcv_output": {
            "beta": beta.tolist(),
            "vcov": vcov.tolist(),
            "lambda": lam.tolist(),
            "edf_per_smooth": edf_per,
            "edf_total": edf_total,
            "deviance": deviance,
            "scale": scale,
            "n_iter": n_iter,
            "predictions_train": pred_train.tolist(),
            "predictions_test": pred_test.tolist(),
            "predictions_extrap": pred_extrap.tolist(),
            "predictions_train_se": pred_train_se.tolist(),
            "predictions_test_se": pred_test_se.tolist(),
        },
    }
    # Optional trajectory fields (mgcv stores these on outer.info; they
    # are the canonical "where did Newton go each step" record). We treat
    # them as additive to v1: missing on legacy fixtures, present on new.
    if score_history is not None:
        out["mgcv_output"]["score_history"] = score_history
    if final_grad is not None:
        out["mgcv_output"]["final_grad"] = final_grad
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "names", nargs="*",
        help="Restrict to these case names. Default = all.",
    )
    args = parser.parse_args(argv)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    cases = (
        [CASES_BY_NAME[n] for n in args.names]
        if args.names
        else all_cases()
    )

    print(f"Generating {len(cases)} parity fixtures via rpy2 + mgcv::gam\n")
    successes = 0
    failures: list[str] = []
    for case in cases:
        t0 = time.perf_counter()
        print(f"  {case.name:55s} ", end="", flush=True)
        try:
            fixture = fit_one(case)
            out = FIXTURES_DIR / f"{case.name}.json"
            with open(out, "w") as f:
                json.dump(fixture, f, indent=2)
            dt = time.perf_counter() - t0
            print(f"OK   ({dt:5.2f}s)")
            successes += 1
        except Exception as exc:
            dt = time.perf_counter() - t0
            print(f"FAIL ({dt:5.2f}s): {exc}")
            failures.append(case.name)

    print(f"\n{successes}/{len(cases)} fixtures written.")
    if failures:
        print("Failed:")
        for n in failures:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
