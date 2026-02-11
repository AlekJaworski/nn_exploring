#!/usr/bin/env python3
"""
Performance regression test for mgcv_rust.

Primary benchmark: n=6000, d=10, k=12 (p=120 total basis functions)
This exercises the BLAS-critical path: 6120x120 QR, 120x120 Cholesky,
6000x120 GEMM for X'WX, batch triangular solves across 10 smooth terms.

Usage:
    python validate_performance.py                  # Run and compare to baseline
    python validate_performance.py --establish      # Establish new baseline
    python validate_performance.py --threshold 0.3  # Custom regression threshold (30%)
    python validate_performance.py --ci             # CI mode: relaxed threshold (50%)
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# Must be importable
try:
    import mgcv_rust
except ImportError:
    print("ERROR: mgcv_rust not installed. Run: maturin develop --features python,blas,blas-system --release")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASELINE_PATH = os.path.join(PROJECT_DIR, "test_data", "performance_baseline_blas.json")

# ---- Benchmark configurations ----
# Primary: n=6000, d=10, k=12 (the must-pass case)
# Secondary: validates scaling behavior at other dimensions
CONFIGS = [
    {"name": "primary",   "n": 6000, "d": 10, "k": 12, "description": "Primary target (n=6000, d=10, k=12, p=120)"},
    {"name": "medium",    "n": 2000, "d": 8,  "k": 8,  "description": "Medium (n=2000, d=8, k=8, p=64)"},
    {"name": "large_n",   "n": 5000, "d": 8,  "k": 8,  "description": "Large n (n=5000, d=8, k=8, p=64)"},
    {"name": "small",     "n": 1000, "d": 4,  "k": 10, "description": "Small (n=1000, d=4, k=10, p=40)"},
]


def generate_data(n, d, seed=42):
    """Generate synthetic multi-dimensional data matching benchmark_multidim.py"""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, d))
    y = np.zeros(n)

    if d >= 1:
        y += np.sin(2 * np.pi * X[:, 0])
    if d >= 2:
        y += 0.5 * np.cos(3 * np.pi * X[:, 1])
    if d >= 3:
        y += 0.3 * (X[:, 2] ** 2)
    if d >= 4:
        y += 0.2 * np.exp(-5 * (X[:, 3] - 0.5) ** 2)
    for i in range(4, d):
        y += 0.1 * np.sin(np.pi * X[:, i])

    y += np.random.normal(0, 0.2, n)
    return X, y


def run_benchmark(config, n_warmup=1, n_runs=5):
    """Run a single benchmark configuration. Returns median time in seconds."""
    n, d, k = config["n"], config["d"], config["k"]
    X, y = generate_data(n, d)

    # Warmup runs (JIT, cache warm)
    for _ in range(n_warmup):
        gam = mgcv_rust.GAM()
        try:
            gam.fit_auto(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
        except Exception as e:
            print(f"  WARNING: warmup failed: {e}")
            return None

    # Timed runs
    times = []
    results = []
    for run in range(n_runs):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        try:
            result = gam.fit_auto(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
        except Exception as e:
            print(f"  WARNING: run {run+1} failed: {e}")
            return None
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        results.append(result)

    median_time = float(np.median(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))
    std_time = float(np.std(times))

    # Extract lambda and iteration info from last result
    lambdas = results[-1].get('lambda', [])
    iterations = results[-1].get('iterations', None)

    return {
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "times": times,
        "lambdas": [float(l) for l in lambdas] if lambdas is not None else [],
        "iterations": int(iterations) if iterations is not None else None,
        "n": n,
        "d": d,
        "k": k,
        "p": d * k,
    }


def establish_baseline():
    """Run all benchmarks and save as baseline."""
    print("=" * 70)
    print("  ESTABLISHING PERFORMANCE BASELINE")
    print("=" * 70)

    baseline = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": sys.platform,
        "python_version": sys.version,
        "configs": {},
    }

    for config in CONFIGS:
        name = config["name"]
        print(f"\n  {config['description']}...")
        result = run_benchmark(config, n_warmup=2, n_runs=7)
        if result is None:
            print(f"  FAILED: {name}")
            continue

        baseline["configs"][name] = result
        print(f"    Median: {result['median_time']*1000:.1f}ms")
        print(f"    Min:    {result['min_time']*1000:.1f}ms")
        print(f"    Max:    {result['max_time']*1000:.1f}ms")
        print(f"    Std:    {result['std_time']*1000:.1f}ms")
        if result['iterations'] is not None:
            print(f"    Iters:  {result['iterations']}")

    # Save
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"\n  Baseline saved to: {BASELINE_PATH}")
    return baseline


def run_regression_test(threshold=0.20, ci_mode=False):
    """Run benchmarks and compare against baseline. Returns True if passed."""
    if ci_mode:
        threshold = 0.50  # 50% relaxed threshold for CI runners

    # Load baseline
    if not os.path.exists(BASELINE_PATH):
        print(f"  No baseline found at {BASELINE_PATH}")
        print(f"  Run with --establish to create one.")
        print(f"  Running benchmarks without comparison...")
        establish_baseline()
        return True

    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)

    print("=" * 70)
    print("  PERFORMANCE REGRESSION TEST")
    print(f"  Threshold: {threshold*100:.0f}% regression allowed")
    print(f"  Baseline from: {baseline.get('created', 'unknown')}")
    print("=" * 70)

    all_passed = True

    for config in CONFIGS:
        name = config["name"]
        if name not in baseline.get("configs", {}):
            print(f"\n  {config['description']}: SKIPPED (no baseline)")
            continue

        base = baseline["configs"][name]
        print(f"\n  {config['description']}...")

        result = run_benchmark(config, n_warmup=1, n_runs=5)
        if result is None:
            print(f"    FAIL: benchmark crashed")
            all_passed = False
            continue

        base_time = base["median_time"]
        curr_time = result["median_time"]
        regression = (curr_time - base_time) / base_time

        status = "PASS" if regression <= threshold else "FAIL"
        symbol = "+" if regression > 0 else ""

        print(f"    Baseline: {base_time*1000:8.1f}ms")
        print(f"    Current:  {curr_time*1000:8.1f}ms ({symbol}{regression*100:.1f}%)")

        if result['iterations'] is not None and base.get('iterations') is not None:
            print(f"    Iters:    {result['iterations']} (baseline: {base['iterations']})")

        if status == "FAIL":
            print(f"    >>> REGRESSION: {regression*100:.1f}% exceeds {threshold*100:.0f}% threshold")
            all_passed = False
            # Primary case is a hard failure
            if name == "primary":
                print(f"    >>> PRIMARY BENCHMARK FAILED - this is a release blocker")
        else:
            print(f"    [{status}]")

    print("\n" + "=" * 70)
    if all_passed:
        print("  RESULT: ALL PERFORMANCE CHECKS PASSED")
    else:
        print("  RESULT: PERFORMANCE REGRESSION DETECTED")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Performance regression test for mgcv_rust")
    parser.add_argument("--establish", action="store_true", help="Establish new baseline")
    parser.add_argument("--threshold", type=float, default=0.20, help="Regression threshold (default: 0.20 = 20%%)")
    parser.add_argument("--ci", action="store_true", help="CI mode: relaxed threshold")
    args = parser.parse_args()

    if args.establish:
        establish_baseline()
        sys.exit(0)

    passed = run_regression_test(threshold=args.threshold, ci_mode=args.ci)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
