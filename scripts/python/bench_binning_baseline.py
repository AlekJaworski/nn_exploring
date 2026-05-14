"""Discrete-binning perf baseline (before/after).

Times `mgcv_rust.Gam(family='t-dist')` on the two largest production
fixtures from `data/sale_price_fixtures/`. Used to capture BEFORE
numbers prior to the discrete-binning workstream (D3..D11 in
`docs/DISCRETE_BINNING_DESIGN.md`) and AFTER numbers as it lands.

Usage:
    python scripts/python/bench_binning_baseline.py
    python scripts/python/bench_binning_baseline.py --binning auto
    python scripts/python/bench_binning_baseline.py --binning auto --method fREML

`--method` forwards the smoothing-parameter selection method to ``Gam``
(``REML`` for the inner-Newton REML path, ``fREML`` for the closed-form
Path B driver).

Min-of-N timing matches `/tmp/bench_scat_production.py` from prior
sessions; Linux jitter is one-sided so min ≈ true cost. Median is also
reported for stability.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mgcv_rust import Gam

SMOOTH_COLS = [
    "current_list_price",
    "price_change_pct_from_original",
    "cum_dom_before_current_regime",
    "days_in_current_price_regime",
    "monthly_index",
]
PARAM_COLS = [
    "at_least_1_price_drop",
    "at_least_2_price_drops",
    "at_least_3_price_drops",
]
TARGET = "sale_to_list_price_ratio"

FIXTURES = [
    "entire_dataset_train.parquet",  # n=6400, biggest
    "split_0_train.parquet",         # n=5157, parity-design fixture
]


def time_stats(fn, n_runs: int) -> tuple[float, float, list[float]]:
    """Return (median_s, min_s, all_times) over n_runs."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), times


def bench_one(parquet_path: Path, binning: str | None, method: str, n_runs: int) -> dict:
    df = pd.read_parquet(parquet_path)
    X = df[SMOOTH_COLS + PARAM_COLS]
    y = df[TARGET].values

    kwargs = {
        "family": "t-dist",
        "predictors": SMOOTH_COLS + PARAM_COLS,
        "k_default": 10,
        "predictor_basis_map": {c: "parametric" for c in PARAM_COLS},
        "method": method,
    }
    if binning == "auto":
        kwargs["discrete"] = True

    def fit():
        return Gam(**kwargs).fit(X, y)

    median_s, min_s, times = time_stats(fit, n_runs=n_runs)

    # Sanity: fit once more for predictions
    last = Gam(**kwargs).fit(X, y)
    preds = last.predict(X)
    pred_finite = bool(np.all(np.isfinite(preds)))
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))

    return {
        "fixture": parquet_path.name,
        "n": len(df),
        "median_ms": median_s * 1000,
        "min_ms": min_s * 1000,
        "all_ms": [t * 1000 for t in times],
        "pred_finite": pred_finite,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binning",
        choices=["auto", "off"],
        default=None,
        help="Forward `binning=` to Gam (when supported). Default: omit "
        "the kwarg entirely (matches BEFORE-state).",
    )
    parser.add_argument(
        "--method",
        default="REML",
        help="Smoothing-parameter selection method (REML, fREML, GCV). Default: REML.",
    )
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    fixture_dir = Path("data/sale_price_fixtures")
    if not fixture_dir.exists():
        raise SystemExit(
            f"fixture dir {fixture_dir} not found — drop production "
            f"parquet files there first"
        )

    print(f"binning kwarg: {args.binning or '(omitted)'}    method: {args.method}    runs: {args.runs}")
    print(f"{'fixture':<40} {'n':>5} {'median (ms)':>14} {'min (ms)':>12}  pred")
    print("-" * 90)
    for name in FIXTURES:
        path = fixture_dir / name
        if not path.exists():
            print(f"{name:<40}  MISSING")
            continue
        try:
            r = bench_one(path, args.binning, args.method, args.runs)
            pred_status = "ok" if r["pred_finite"] else "NaN"
            print(
                f"{r['fixture']:<40} {r['n']:>5} {r['median_ms']:>11.0f} ms "
                f"{r['min_ms']:>9.0f} ms  {pred_status} "
                f"(mean={r['pred_mean']:.4f}, std={r['pred_std']:.4f})"
            )
        except Exception as exc:
            print(f"{name:<40} {'FAILED':>20}  ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
