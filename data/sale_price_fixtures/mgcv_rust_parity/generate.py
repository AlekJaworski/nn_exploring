"""
Generate the mgcv-baseline reference fixtures consumed by the mgcv_rust parity check.

Inputs: 6 parquet files in the parent directory (5 CV folds + 1 entire_dataset).
Outputs (written next to this script):

  mean_gam_<fold>.parquet
      For each fixture, predictions on its training X from the production mean-GAM:
          mgcv.bam(family=scat(link='identity'), method='fREML',
                   discrete=TRUE, nthreads=1, weights=weight)
      formula:
          sale_to_list_price_ratio ~ at_least_1_price_drop + at_least_2_price_drops +
                                     at_least_3_price_drops +
                                     s(current_list_price, k, bs='cr') +
                                     s(price_change_pct_from_original, k, bs='cr') +
                                     s(cum_dom_before_current_regime, k, bs='cr') +
                                     s(days_in_current_price_regime, k, bs='cr') +
                                     s(monthly_index, k, bs='cr')
      k_default=7; monthly_index k clamped to <=5; shrink k_default if total > n.
      Columns:
          row_idx        : 0..n-1 alignment index into the source fixture
          listing_number : carried from source
          y              : sale_to_list_price_ratio (target)
          weight         : training weight = 1 / n_obs_per_listing
          pred_prod      : mean-GAM in-sample prediction with parametric indicators
          pred_no_param  : same fit minus the 3 at_least_*_price_drop indicators

  qgam_q95_entire_dataset.parquet
      For entire_dataset only (the actual production qgam input):
          residual = y - pred_prod
          qgam.qgam(qu=0.95, argGam=list(method='REML', nthreads=1))
          formula:
            resid ~ s(days_in_current_price_regime, k, bs='cr') +
                    s(cum_dom_before_current_regime, k, bs='cr') +
                    s(price_change_pct_from_original, k, bs='cr') +
                    s(monthly_index, k, bs='cr')
          k_default=5
      Columns:
          row_idx, listing_number, residual, qgam_q95_pred

  subject_predictions.json
      Single-row "subject" predictions at the per-fold median of the training X
      (parametric indicators set to 0; weight=1). Useful as a tight parity target.

  metadata.json
      Versions, fit times, hashes.

Run with the same env as data_explorations/bench_*.py:
    PYTHONPATH=/home/alex/vibe_coding/nn_exploring/python \\
        /home/alex/vibe_coding/nn_exploring/.venv/bin/python generate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_LOCAL = Path("/home/alex/vibe_coding/nn_exploring/python")
if _LOCAL.exists() and str(_LOCAL) not in sys.path:
    sys.path.insert(0, str(_LOCAL))

import rpy2  # noqa: E402
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import pandas2ri  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

mgcv = importr("mgcv")
qgam_r = importr("qgam")
stats = importr("stats")
base = importr("base")

HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE.parent
FIXTURES = [
    "split_0_train.parquet",
    "split_1_train.parquet",
    "split_2_train.parquet",
    "split_3_train.parquet",
    "split_4_train.parquet",
    "entire_dataset_train.parquet",
]
TARGET = "sale_to_list_price_ratio"
MEAN_SMOOTHS = [
    "current_list_price",
    "price_change_pct_from_original",
    "cum_dom_before_current_regime",
    "days_in_current_price_regime",
    "monthly_index",
]
PARAMETRIC = ["at_least_1_price_drop", "at_least_2_price_drops", "at_least_3_price_drops"]
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]


def _mean_k(df: pd.DataFrame, k_default: int = 7) -> dict[str, int]:
    while True:
        k = {c: min(k_default, df[c].nunique()) for c in MEAN_SMOOTHS}
        k["monthly_index"] = min(min(k_default, 5), df["monthly_index"].nunique())
        if sum(k.values()) + len(PARAMETRIC) < len(df):
            return k
        k_default -= 1


def _qgam_k(df: pd.DataFrame, k_default: int = 5) -> dict[str, int]:
    while True:
        k = {c: min(k_default, df[c].nunique()) for c in QGAM_SMOOTHS}
        if sum(k.values()) < len(df):
            return k
        k_default -= 1


def _mean_formula(k: dict[str, int], include_parametric: bool) -> str:
    smooth = " + ".join(f"s({c}, k={k[c]}, bs='cr')" for c in MEAN_SMOOTHS)
    parts = (" + ".join(PARAMETRIC) + " + " if include_parametric else "") + smooth
    return f"{TARGET} ~ {parts}"


def _fit_mean(df: pd.DataFrame, k: dict[str, int], include_parametric: bool):
    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
    r_weights = ro.FloatVector(df["weight"].to_numpy())
    t0 = time.perf_counter()
    model = mgcv.bam(
        ro.Formula(_mean_formula(k, include_parametric)),
        data=r_df,
        family=mgcv.scat(link="identity"),
        method="fREML",
        weights=r_weights,
        discrete=True,
        nthreads=1,
    )
    fit_time = time.perf_counter() - t0
    pred = np.array(stats.predict(model, newdata=r_df, type="response"), dtype=float, copy=True)
    return model, pred, fit_time


def _subject_pred(model, k: dict[str, int], subject: pd.DataFrame) -> float:
    with (ro.default_converter + pandas2ri.converter).context():
        r_sub = ro.conversion.get_conversion().py2rpy(subject)
    return float(np.array(stats.predict(model, newdata=r_sub, type="response"), dtype=float, copy=True)[0])


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    def _pkg_version(name: str) -> str:
        try:
            return str(ro.r(f"as.character(packageVersion('{name}'))")[0])
        except Exception:
            return "unknown"

    meta = {
        "rpy2_version": getattr(rpy2, "__version__", "unknown"),
        "R_version": str(ro.r("R.version.string")[0]),
        "mgcv_version": _pkg_version("mgcv"),
        "qgam_version": _pkg_version("qgam"),
        "fixture_dir": str(FIXTURE_DIR),
        "fixtures": {},
    }
    subject_preds: dict = {}

    for name in FIXTURES:
        src = FIXTURE_DIR / name
        df = pd.read_parquet(src).reset_index(drop=True)
        fold = name.replace("_train.parquet", "")
        print(f"\n=== {fold}  n={len(df)}")

        k_mean = _mean_k(df)
        subject = pd.DataFrame([{
            **{c: df[c].median() for c in MEAN_SMOOTHS},
            **{c: 0 for c in PARAMETRIC},
            "weight": 1.0,
        }])

        m_prod, pred_prod, t_prod = _fit_mean(df, k_mean, include_parametric=True)
        print(f"  mgcv mean GAM (prod)        : fit={t_prod:.3f}s")
        m_comp, pred_comp, t_comp = _fit_mean(df, k_mean, include_parametric=False)
        print(f"  mgcv mean GAM (no param)    : fit={t_comp:.3f}s")

        sp_prod = _subject_pred(m_prod, k_mean, subject)
        sp_comp = _subject_pred(m_comp, k_mean, subject[MEAN_SMOOTHS + ["weight"]])
        subject_preds[fold] = {
            "subject_features": {c: float(subject[c].iloc[0]) for c in MEAN_SMOOTHS + PARAMETRIC},
            "mean_gam_prod": sp_prod,
            "mean_gam_no_param": sp_comp,
        }

        out = pd.DataFrame({
            "row_idx": np.arange(len(df), dtype=np.int64),
            "listing_number": df["listing_number"].to_numpy(),
            "y": df[TARGET].to_numpy(),
            "weight": df["weight"].to_numpy(),
            "pred_prod": pred_prod,
            "pred_no_param": pred_comp,
        })
        out_path = HERE / f"mean_gam_{fold}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  wrote {out_path.name}  ({len(out)} rows, sha256[:16]={_file_hash(out_path)})")
        meta["fixtures"][fold] = {
            "source": name,
            "n_rows": len(df),
            "k_mean_gam": k_mean,
            "fit_time_prod_s": t_prod,
            "fit_time_no_param_s": t_comp,
            "subject_pred_prod": sp_prod,
            "subject_pred_no_param": sp_comp,
            "mean_gam_output_sha256_16": _file_hash(out_path),
        }

        if fold == "entire_dataset":
            resid = df[TARGET].to_numpy() - pred_prod
            print(f"  residuals (entire_dataset)  : mean={resid.mean():+.4f} std={resid.std():.4f}")

            qgam_df = df.copy()
            qgam_df["resid"] = resid
            k_q = _qgam_k(qgam_df)
            formula_q = "resid ~ " + " + ".join(f"s({c}, k={k_q[c]}, bs='cr')" for c in QGAM_SMOOTHS)
            arg_gam = ro.ListVector({"method": "REML", "nthreads": 1})
            with (ro.default_converter + pandas2ri.converter).context():
                r_qdf = ro.conversion.get_conversion().py2rpy(qgam_df)
            t0 = time.perf_counter()
            qm = qgam_r.qgam(ro.Formula(formula_q), data=r_qdf, qu=0.95, argGam=arg_gam)
            t_q = time.perf_counter() - t0
            q_pred = np.array(stats.predict(qm, newdata=r_qdf, type="response"), dtype=float, copy=True)

            q_subject = pd.DataFrame([{c: df[c].median() for c in QGAM_SMOOTHS}])
            with (ro.default_converter + pandas2ri.converter).context():
                r_qsub = ro.conversion.get_conversion().py2rpy(q_subject)
            sp_q = float(np.asarray(stats.predict(qm, newdata=r_qsub, type="response"))[0])

            print(f"  qgam (qu=0.95)              : fit={t_q:.3f}s  subject_q95={sp_q:+.5f}")
            qout = pd.DataFrame({
                "row_idx": np.arange(len(df), dtype=np.int64),
                "listing_number": df["listing_number"].to_numpy(),
                "residual": resid,
                "qgam_q95_pred": q_pred,
            })
            qout_path = HERE / "qgam_q95_entire_dataset.parquet"
            qout.to_parquet(qout_path, index=False)
            print(f"  wrote {qout_path.name}  ({len(qout)} rows, sha256[:16]={_file_hash(qout_path)})")
            meta["fixtures"]["qgam_q95"] = {
                "source": name,
                "n_rows": len(df),
                "k_qgam": k_q,
                "fit_time_s": t_q,
                "subject_q95_pred": sp_q,
                "output_sha256_16": _file_hash(qout_path),
            }
            subject_preds["qgam_q95_entire_dataset"] = {
                "subject_features": {c: float(q_subject[c].iloc[0]) for c in QGAM_SMOOTHS},
                "qgam_q95_pred": sp_q,
            }

    (HERE / "subject_predictions.json").write_text(json.dumps(subject_preds, indent=2, default=float))
    (HERE / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nwrote subject_predictions.json + metadata.json")


if __name__ == "__main__":
    main()
