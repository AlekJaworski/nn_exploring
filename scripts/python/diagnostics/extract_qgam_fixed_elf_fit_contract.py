"""Extract qgam fixed-sp ELF fit contract (ladder step 6).

Runs R's mgcv::gam with a fixed ELF family (known co, theta/sigma, sp) on the
production entire_dataset fixture.  Captures coef, fitted summary, and deviance
so Rust's fit_quantile_fixed_sp can be compared before touching the outer REML
optimizer or tuneLearnFast.

The co/lsig/sp values come from a real qgam() call on the same data; we pass
them explicitly to reproduce R's fixed final ELF fit deterministically.

Writes to: test_data/qgam_fixed_elf_fit_contract.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from mgcv_rust import GAM

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
REFERENCE_DIR = FIXTURE_DIR / "mgcv_rust_parity"
OUT = ROOT / "test_data" / "qgam_fixed_elf_fit_contract.json"

TARGET = "sale_to_list_price_ratio"
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]
TAU = 0.95
K = 5

# Known prod stage targets (from /tmp/opencode/qgam_r_stage_probe.log and our
# previously extracted contracts).
LSIG = -4.9598429613200636       # learned log-sigma from tuneLearnFast
ERR = 0.038489416415544007        # from .getErrParam (our contract fixture)
VAR_HAT = 0.0020164420863363727   # from Gaussian init contract


def _co_from_err(err: float, var_hat: float) -> float:
    """Convert err (bandwidth) to ELF co.  Mirrors qgam.R: co <- err * sqrt(2*pi*varHat) / (2*log(2))"""
    return err * math.sqrt(2 * math.pi * var_hat) / (2 * math.log(2))


def _summary(x: np.ndarray) -> list[float]:
    qs = np.quantile(np.asarray(x, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return [float(v) for v in qs]


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main() -> None:
    importr("mgcv")
    importr("qgam")

    df = pd.read_parquet(FIXTURE_DIR / "entire_dataset_train.parquet")
    mean_ref = pd.read_parquet(REFERENCE_DIR / "mean_gam_entire_dataset.parquet")
    df = df.copy()
    df["resid"] = df[TARGET].to_numpy() - mean_ref["pred_prod"].to_numpy()

    co = _co_from_err(ERR, VAR_HAT)
    sigma = math.exp(LSIG)
    print(f"co = {co:.17g}")
    print(f"sigma = exp(lsig) = {sigma:.17g}")

    formula_str = "resid ~ " + " + ".join(
        f"s({c}, k={K}, bs='cr')" for c in QGAM_SMOOTHS
    )
    print(f"formula: {formula_str}")

    # ── R: gam() with fixed sp and ELF family ────────────────────────────────
    # Use the final sp values from the prod stage probe log.
    # These are the smoothing parameters that qgam's tuneLearnFast converged to.
    final_sp_r = [
        1004340.6643248051,
        607622.6891643470,
        421.87326626306725,
        737690.6701503786,
    ]
    print(f"final_sp_r = {final_sp_r}")

    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)

    # Run mgcv::gam with the fixed ELF family and fixed sp
    r_code = f"""
    library(qgam)
    library(mgcv)
    fam <- elf(qu={TAU}, co={co:.17g}, theta={LSIG:.17g}, link="identity")
    fit <- gam(
        as.formula("{formula_str}"),
        family=fam,
        sp=c({",".join(str(v) for v in final_sp_r)}),
        data=r_df,
        method="REML"
    )
    list(
        coef=as.numeric(coef(fit)),
        fitted=as.numeric(fitted(fit)),
        deviance=as.numeric(deviance(fit)),
        sp=as.numeric(fit$sp),
        iter=as.numeric(fit$iter)
    )
    """
    ro.globalenv["r_df"] = r_df
    result_r = ro.r(r_code)

    r_coef = np.asarray(result_r.rx2("coef"), dtype=float)
    r_fitted = np.asarray(result_r.rx2("fitted"), dtype=float)
    r_deviance = float(result_r.rx2("deviance")[0])
    r_sp = np.asarray(result_r.rx2("sp"), dtype=float)
    r_iter = int(result_r.rx2("iter")[0])
    print(f"R deviance (fixed sp) = {r_deviance:.10g}")
    print(f"R iter = {r_iter}")
    print(f"R coef[:4] = {r_coef[:4]}")

    # ── Rust: fit_quantile_fixed_sp ───────────────────────────────────────────
    X = df[QGAM_SMOOTHS].to_numpy(dtype=float)
    y = df["resid"].to_numpy(dtype=float)

    g = GAM("quantile", tau=TAU, sigma=sigma, co=co)
    rust_result = g.fit_quantile_fixed_sp(
        X, y,
        k=[K] * len(QGAM_SMOOTHS),
        sp=final_sp_r,
        tau=TAU,
        sigma=sigma,
        co=co,
        bs="cr",
        max_iter=200,
        tol=1e-6,
    )
    rust_coef = np.asarray(rust_result["coef"], dtype=float)
    rust_fitted = np.asarray(rust_result["fitted_values"], dtype=float)
    rust_deviance = float(rust_result["deviance"])
    print(f"Rust deviance (fixed sp) = {rust_deviance:.10g}")
    print(f"Rust converged = {rust_result['converged']}, iter = {rust_result['iterations']}")
    print(f"Rust coef[:4] = {rust_coef[:4]}")
    print(f"max_abs coef = {_max_abs(rust_coef, r_coef):.3e}")
    print(f"max_abs fitted = {_max_abs(rust_fitted, r_fitted):.3e}")
    print(f"deviance reldiff = {abs(rust_deviance - r_deviance) / max(abs(r_deviance), 1e-10):.3e}")

    payload = {
        "source": "qgam fixed-sp ELF fit production entire_dataset (ladder step 6)",
        "formula": formula_str,
        "n": int(len(df)),
        "tau": TAU,
        "lsig": LSIG,
        "sigma": sigma,
        "err": ERR,
        "co": co,
        "var_hat": VAR_HAT,
        "final_sp": final_sp_r,
        "r": {
            "coef": r_coef.tolist(),
            "fitted_summary": _summary(r_fitted),
            "deviance": r_deviance,
            "sp": r_sp.tolist(),
            "iter": r_iter,
        },
        "rust": {
            "coef": rust_coef.tolist(),
            "fitted_summary": _summary(rust_fitted),
            "deviance": rust_deviance,
            "converged": bool(rust_result["converged"]),
            "iterations": int(rust_result["iterations"]),
            "max_abs_coef_vs_r": _max_abs(rust_coef, r_coef),
            "max_abs_fitted_vs_r": _max_abs(rust_fitted, r_fitted),
            "deviance_reldiff": abs(rust_deviance - r_deviance) / max(abs(r_deviance), 1e-10),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
