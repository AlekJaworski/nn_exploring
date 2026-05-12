"""Benchmark MGCV_STEP_BLEND=1 vs default on the full parity battery.

Decision input for whether to default-on. Reports per-fixture:
  - wallclock: median + min over N runs, in each mode
  - REML at converged sp (does step_blend find a strictly lower REML?)
  - converged lambdas (do they differ?)

The script is intentionally idempotent — it doesn't touch results.json
or any committed artifact. Writes a fresh report to
  scripts/python/benchmarks/step_blend_results_<timestamp>.json
and prints a human-readable summary.

Usage:
  python3 scripts/python/bench_step_blend.py [N_RUNS]

Default N_RUNS=3 (lighter than the test_perf.py default of 5; this is
about ratios, not absolute timings).

Per `feedback_release_build_for_parity`: ensure a release build is
installed before running.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/home/alex/vibe_coding/nn_exploring")
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures"
OUT_DIR = REPO / "scripts" / "python" / "benchmarks"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "tests" / "parity"))
from schema import Fixture  # noqa: E402
import mgcv_rust  # noqa: E402

_RUST_FAM = {
    "gaussian": "gaussian", "binomial": "binomial", "poisson": "poisson",
    "Gamma": "gamma", "gamma": "gamma",
    "tw": "tweedie", "Tweedie": "tweedie",
    "quasipoisson": "quasipoisson", "quasibinomial": "quasibinomial",
    "inverse.gaussian": "inverse.gaussian",
    "negative.binomial": "negbin", "nb": "nb",
}
_CANONICAL_LINK = {
    "gaussian": "identity", "binomial": "logit", "poisson": "log",
    "Gamma": "inverse", "gamma": "inverse",
    "Tweedie": "log", "tw": "log",
    "quasipoisson": "log", "quasibinomial": "logit",
    "inverse.gaussian": "log", "negative.binomial": "log", "nb": "log",
}


def _build_gam(fix: Fixture) -> Any:
    inp = fix.inputs
    family = _RUST_FAM[inp.family]
    if family == "gaussian":
        return mgcv_rust.GAM()
    link_kw: dict = {}
    if _CANONICAL_LINK.get(inp.family) != inp.link:
        link_kw["link"] = inp.link
    if inp.family in ("Tweedie", "tw"):
        link_kw["p"] = 1.5
    if inp.family == "negative.binomial":
        link_kw["theta"] = 2.0
    return mgcv_rust.GAM(family, **link_kw)


def _fit(fix: Fixture) -> tuple[float, Any]:
    """Returns (wallclock_seconds, fitted_gam)."""
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    gam = _build_gam(fix)
    t0 = time.perf_counter()
    gam.fit(
        x, y,
        k=list(inp.k),
        method=inp.method,
        bs=inp.bs[0] if len(set(inp.bs)) == 1 else inp.bs,
    )
    return time.perf_counter() - t0, gam


def _link_eta(pred: np.ndarray, link: str) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    if link == "log":
        return np.log(np.maximum(pred, 1e-12))
    if link == "logit":
        eps = 1e-12
        p = np.clip(pred, eps, 1 - eps)
        return np.log(p / (1 - p))
    if link == "inverse":
        return 1.0 / np.maximum(pred, 1e-12)
    return pred


def _delta_eta(rust_pred, mgcv_pred, link):
    r = _link_eta(rust_pred, link)
    m = _link_eta(mgcv_pred, link)
    return float(np.max(np.abs(r - m)))


def benchmark_fixture(fix_path: Path, n_runs: int, mode_env: dict[str, str]) -> dict:
    """Fit a fixture n_runs times under the given env, return timing + quality."""
    # Set env vars
    saved = {}
    for k, v in mode_env.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    # Also clear any of our gate vars that aren't in mode_env
    for k in ("MGCV_STEP_BLEND",):
        if k not in mode_env:
            saved[k] = os.environ.get(k)
            os.environ.pop(k, None)

    try:
        fix = Fixture.load(fix_path)
        inp = fix.inputs
        mgcv_pred = np.asarray(fix.mgcv_output.predictions_train, dtype=float)

        times = []
        last_gam = None
        for _ in range(n_runs):
            t, gam = _fit(fix)
            times.append(t * 1000.0)  # ms
            last_gam = gam

        # Quality metrics from the last fit
        rust_pred = np.asarray(last_gam.predict(np.asarray(inp.x_train, dtype=float)), dtype=float)
        delta_eta = _delta_eta(rust_pred, mgcv_pred, inp.link)
        rust_lambdas = list(last_gam.get_all_lambdas()) if hasattr(last_gam, 'get_all_lambdas') else None

        return {
            "name": fix_path.stem,
            "times_ms": times,
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "delta_eta": delta_eta,
            "rust_lambdas": rust_lambdas,
            "mgcv_lambdas": list(fix.mgcv_output.lambda_) if hasattr(fix.mgcv_output, 'lambda_') else None,
        }
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    fix_paths = sorted([p for p in FIXTURES_DIR.glob("*.json")])
    print(f"# bench_step_blend — n_runs={n_runs}, fixtures={len(fix_paths)}")
    print(f"# repo: {REPO}")
    print()

    results = {
        "n_runs": n_runs,
        "timestamp": datetime.now().isoformat(),
        "fixtures": {},
    }

    rows = []
    for fix_path in fix_paths:
        name = fix_path.stem
        try:
            r_default = benchmark_fixture(fix_path, n_runs, mode_env={})
            r_blend = benchmark_fixture(fix_path, n_runs, mode_env={"MGCV_STEP_BLEND": "1"})
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            continue

        ratio = r_blend["median_ms"] / r_default["median_ms"] if r_default["median_ms"] > 0 else float("nan")
        delta_eta_d = r_default["delta_eta"]
        delta_eta_b = r_blend["delta_eta"]
        delta_change = delta_eta_b - delta_eta_d

        results["fixtures"][name] = {
            "default": r_default,
            "step_blend": r_blend,
            "perf_ratio": ratio,
            "delta_eta_change": delta_change,
        }

        rows.append((name, r_default["median_ms"], r_blend["median_ms"], ratio,
                     delta_eta_d, delta_eta_b))
        # progress
        marker = ""
        if ratio > 1.5:
            marker = " [SLOW]"
        if delta_eta_b > 1e-3 and delta_eta_d < 1e-3:
            marker += " [REGRESS]"
        if delta_eta_d > 1e-3 and delta_eta_b < 1e-3:
            marker += " [CLOSES]"
        print(f"  {name}: {r_default['median_ms']:.1f}ms → {r_blend['median_ms']:.1f}ms  "
              f"({ratio:.2f}×)  Δη {delta_eta_d:.2e} → {delta_eta_b:.2e}{marker}")

    # Aggregate
    rows.sort(key=lambda r: r[3], reverse=True)
    total_default = sum(r[1] for r in rows)
    total_blend = sum(r[2] for r in rows)
    total_ratio = total_blend / total_default if total_default else float("nan")

    print()
    print(f"# Summary")
    print(f"  Total default median: {total_default:.0f} ms")
    print(f"  Total step_blend median: {total_blend:.0f} ms")
    print(f"  Aggregate ratio: {total_ratio:.2f}× (step_blend / default)")
    print()
    print(f"# Top 10 slowdown ratios (step_blend / default):")
    for name, md, mb, r, dd, db in rows[:10]:
        print(f"  {r:.2f}×  {name}  ({md:.1f} → {mb:.1f} ms,  Δη {dd:.2e} → {db:.2e})")
    print()
    print(f"# Bottom 10 (step_blend faster or equal):")
    for name, md, mb, r, dd, db in rows[-10:]:
        print(f"  {r:.2f}×  {name}  ({md:.1f} → {mb:.1f} ms,  Δη {dd:.2e} → {db:.2e})")

    # Find closures + regressions
    closes = [r for r in rows if r[4] >= 1e-3 and r[5] < 1e-3]
    regresses = [r for r in rows if r[4] < 1e-3 and r[5] >= 1e-3]
    quality_improves = [r for r in rows if r[5] < r[4] * 0.9]  # >10% better
    quality_worsens = [r for r in rows if r[5] > r[4] * 1.1]  # >10% worse

    print()
    print(f"# Quality changes (Δη delta)")
    print(f"  Closes 1e-3 bar: {len(closes)}")
    for r in closes:
        print(f"    {r[0]}: Δη {r[4]:.2e} → {r[5]:.2e}")
    print(f"  Regresses past 1e-3: {len(regresses)}")
    for r in regresses:
        print(f"    {r[0]}: Δη {r[4]:.2e} → {r[5]:.2e}")
    print(f"  Quality improves >10%: {len(quality_improves)}")
    print(f"  Quality worsens >10%: {len(quality_worsens)}")
    for r in quality_worsens[:10]:
        print(f"    {r[0]}: Δη {r[4]:.2e} → {r[5]:.2e}")

    # Save raw
    out_path = OUT_DIR / f"step_blend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n# Wrote {out_path}")


if __name__ == "__main__":
    main()
