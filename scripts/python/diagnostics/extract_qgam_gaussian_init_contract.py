"""Extract qgam's production Gaussian-init stage contract.

This is a diagnostic fixture generator, not a broad parity test. It captures
the `.init_gauss_fit` stage used by qgam before ELF fitting so Rust can compare
stage numbers before changing fit-level behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
REFERENCE_DIR = FIXTURE_DIR / "mgcv_rust_parity"
OUT = ROOT / "test_data" / "qgam_gaussian_init_contract.json"

TARGET = "sale_to_list_price_ratio"
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]


def _qgam_k(df: pd.DataFrame, k_default: int = 5) -> dict[str, int]:
    out: dict[str, int] = {}
    for col in QGAM_SMOOTHS:
        nunique = int(df[col].nunique(dropna=True))
        out[col] = max(3, min(k_default, nunique))
    return out


def _summary(x: np.ndarray) -> list[float]:
    qs = np.quantile(np.asarray(x, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return [float(v) for v in qs]


def main() -> None:
    mgcv = importr("mgcv")
    qgam = importr("qgam")
    base = importr("base")

    df = pd.read_parquet(FIXTURE_DIR / "entire_dataset_train.parquet")
    mean_ref = pd.read_parquet(REFERENCE_DIR / "expected" / "mean_gam_entire_dataset.parquet")
    qdf = df.copy()
    qdf["resid"] = df[TARGET].to_numpy() - mean_ref["pred_prod"].to_numpy()

    k_q = _qgam_k(qdf)
    formula = "resid ~ " + " + ".join(
        f"s({c}, k={k_q[c]}, bs='cr')" for c in QGAM_SMOOTHS
    )

    with (ro.default_converter + pandas2ri.converter).context():
        r_qdf = ro.conversion.get_conversion().py2rpy(qdf)

    init_gauss = ro.r("qgam:::.init_gauss_fit")
    ctrl = qgam.qgam_control()
    arg_gam = ro.ListVector({"method": "REML", "nthreads": 1})
    init = init_gauss(ro.Formula(formula), r_qdf, ctrl, arg_gam, ro.FloatVector([0.95]), False)

    gaus_fit = init.rx2("gausFit")
    init_m = init.rx2("initM").rx2(1)
    fitted = np.asarray(gaus_fit.rx2("fitted.values"), dtype=float)
    coef = np.asarray(ro.r("coef")(gaus_fit), dtype=float)
    mustart = np.asarray(init_m.rx2("mustart"), dtype=float)
    coefstart = np.asarray(init_m.rx2("coefstart"), dtype=float)
    sp = np.asarray(gaus_fit.rx2("sp"), dtype=float)
    var_hat = float(init.rx2("varHat")[0])
    resid = qdf["resid"].to_numpy(dtype=float)

    payload = {
        "source": "qgam:::.init_gauss_fit production entire_dataset",
        "versions": {
            "qgam": str(base.packageVersion("qgam")),
            "mgcv": str(base.packageVersion("mgcv")),
            "R": str(ro.r("getRversion()")),
        },
        "formula": formula,
        "n": int(len(qdf)),
        "k": k_q,
        "varHat": var_hat,
        "gaussian_sp": [float(v) for v in sp],
        "gaussian_coef": [float(v) for v in coef],
        "init_coefstart": [float(v) for v in coefstart],
        "resid_summary": _summary(resid),
        "gaussian_fitted_summary": _summary(fitted),
        "init_mustart_summary": _summary(mustart),
        "intercept_shift": float(coefstart[0] - coef[0]),
        "mean_mustart_minus_fitted": float(np.mean(mustart - fitted)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
