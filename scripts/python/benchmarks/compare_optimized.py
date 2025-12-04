#!/usr/bin/env python3
"""
Compare baseline vs optimized GAM implementation.
Tests the performance improvements from caching and algorithm improvements.
"""

import numpy as np
import time
import json
from typing import Dict, Tuple
import sys

import mgcv_rust


def generate_test_data(n, d, seed=42):
    """Generate test data for GAM."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = np.zeros(n)
    for i in range(d):
        if i % 3 == 0:
            y += np.sin(2 * np.pi * X[:, i])
        elif i % 3 == 1:
            y += (X[:, i] - 0.5) ** 2
        else:
            y += np.exp(-5 * (X[:, i] - 0.5) ** 2)
    y += np.random.normal(0, 0.1, n)
    return X, y


def run_comparison_test(n, d, k_values, n_repeats=5):
    """Compare baseline vs optimized for a single scenario."""
    X, y = generate_test_data(n, d)

    # Baseline timings
    baseline_times = []
    for _ in range(n_repeats):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        try:
            result_baseline = gam.fit_auto(X, y, k=k_values, method="REML", bs="cr")
            end = time.perf_counter()
            baseline_times.append(end - start)
        except Exception as e:
            print(f"Baseline failed: {e}")
            return None

    # Optimized timings
    optimized_times = []
    for _ in range(n_repeats):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        try:
            result_optimized = gam.fit_auto_optimized(X, y, k=k_values, method="REML", bs="cr")
            end = time.perf_counter()
            optimized_times.append(end - start)
        except Exception as e:
            print(f"Optimized failed: {e}")
            return None

    baseline_mean = np.mean(baseline_times)
    optimized_mean = np.mean(optimized_times)
    speedup = baseline_mean / optimized_mean

    # Check results are similar
    r2_baseline = 1 - (result_baseline['deviance'] / np.var(y) / len(y))
    r2_optimized = 1 - (result_optimized['deviance'] / np.var(y) / len(y))

    fitted_baseline = result_baseline['fitted_values']
    fitted_optimized = result_optimized['fitted_values']
    max_diff = np.max(np.abs(fitted_baseline - fitted_optimized))

    return {
        "n": n,
        "d": d,
        "k": k_values,
        "baseline_time": baseline_mean,
        "baseline_std": np.std(baseline_times),
        "optimized_time": optimized_mean,
        "optimized_std": np.std(optimized_times),
        "speedup": speedup,
        "r2_baseline": float(r2_baseline),
        "r2_optimized": float(r2_optimized),
        "max_fitted_diff": float(max_diff),
    }


def run_comprehensive_comparison():
    """Run comprehensive comparison across various scenarios."""

    print("="*80)
    print("Baseline vs Optimized GAM Performance Comparison")
    print("="*80)
    print()

    test_configs = [
        # Small problems
        {"n": 50, "d": 1, "k": [10]},
        {"n": 100, "d": 1, "k": [10]},
        {"n": 200, "d": 1, "k": [10]},

        # Medium problems
        {"n": 500, "d": 1, "k": [10]},
        {"n": 1000, "d": 1, "k": [10]},
        {"n": 1000, "d": 1, "k": [20]},
        {"n": 1000, "d": 1, "k": [30]},

        # Large problems
        {"n": 2000, "d": 1, "k": [10]},
        {"n": 5000, "d": 1, "k": [10]},
        {"n": 5000, "d": 1, "k": [20]},

        # Multi-dimensional
        {"n": 500, "d": 2, "k": [10, 10]},
        {"n": 1000, "d": 2, "k": [10, 10]},
        {"n": 500, "d": 3, "k": [10, 10, 10]},
        {"n": 1000, "d": 3, "k": [10, 10, 10]},
        {"n": 500, "d": 5, "k": [8] * 5},
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        n, d, k = config["n"], config["d"], config["k"]
        print(f"Test {i}/{len(test_configs)}: n={n:5d}, d={d:2d}, k={k}", end=" ... ")
        sys.stdout.flush()

        result = run_comparison_test(n, d, k)

        if result:
            speedup = result['speedup']
            speedup_pct = (speedup - 1) * 100

            if speedup > 1:
                print(f"✓ {speedup:.2f}x faster ({speedup_pct:+.1f}%)")
            else:
                print(f"⚠ {speedup:.2f}x (slower)")

            print(f"   Baseline: {result['baseline_time']:.4f}s ± {result['baseline_std']:.4f}s")
            print(f"   Optimized: {result['optimized_time']:.4f}s ± {result['optimized_std']:.4f}s")
            print(f"   R² difference: {abs(result['r2_baseline'] - result['r2_optimized']):.6f}")
            print(f"   Max fitted value difference: {result['max_fitted_diff']:.6e}")

            results.append(result)
        else:
            print("✗ Failed")
        print()

    # Summary statistics
    if results:
        print("="*80)
        print("Summary Statistics")
        print("="*80)

        speedups = [r['speedup'] for r in results]
        print(f"\nSpeedup statistics (n={len(speedups)}):")
        print(f"  Mean speedup:     {np.mean(speedups):.2f}x")
        print(f"  Median speedup:   {np.median(speedups):.2f}x")
        print(f"  Min speedup:      {np.min(speedups):.2f}x")
        print(f"  Max speedup:      {np.max(speedups):.2f}x")
        print(f"  Std dev:          {np.std(speedups):.2f}x")

        # Categorize by problem size
        small = [r for r in results if r['n'] <= 200]
        medium = [r for r in results if 200 < r['n'] <= 2000]
        large = [r for r in results if r['n'] > 2000]

        if small:
            print(f"\nSmall problems (n≤200): {np.mean([r['speedup'] for r in small]):.2f}x avg speedup")
        if medium:
            print(f"Medium problems (200<n≤2000): {np.mean([r['speedup'] for r in medium]):.2f}x avg speedup")
        if large:
            print(f"Large problems (n>2000): {np.mean([r['speedup'] for r in large]):.2f}x avg speedup")

        # Check correctness
        max_r2_diff = max(abs(r['r2_baseline'] - r['r2_optimized']) for r in results)
        max_fitted_diff = max(r['max_fitted_diff'] for r in results)

        print(f"\nCorrectness check:")
        print(f"  Max R² difference:     {max_r2_diff:.6f}")
        print(f"  Max fitted value diff: {max_fitted_diff:.6e}")

        if max_r2_diff < 1e-3 and max_fitted_diff < 1e-3:
            print(f"  ✓ Results match within tolerance!")
        else:
            print(f"  ⚠ Results may differ significantly")

        # Save results
        with open("optimization_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: optimization_comparison.json")

    return results


def main():
    results = run_comprehensive_comparison()

    if results:
        speedups = [r['speedup'] for r in results]
        avg_speedup = np.mean(speedups)

        print("\n" + "="*80)
        if avg_speedup > 1.2:
            print(f"🎉 SUCCESS! Average speedup: {avg_speedup:.2f}x ({(avg_speedup-1)*100:.1f}% faster)")
        elif avg_speedup > 1.0:
            print(f"✓ Slight improvement: {avg_speedup:.2f}x ({(avg_speedup-1)*100:.1f}% faster)")
        else:
            print(f"⚠ No improvement: {avg_speedup:.2f}x")
        print("="*80)
        return 0
    else:
        print("Failed to run benchmarks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
