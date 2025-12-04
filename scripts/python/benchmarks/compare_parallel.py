#!/usr/bin/env python3
"""
Compare baseline vs optimized vs parallel GAM implementations.
Tests performance improvements from parallelization for multi-dimensional GAMs.
"""

import numpy as np
import time
import json
from typing import Dict, List
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
    """Compare baseline vs optimized vs parallel for a single scenario."""
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

    # Parallel timings
    parallel_times = []
    for _ in range(n_repeats):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        try:
            result_parallel = gam.fit_auto_parallel(X, y, k=k_values, method="REML", bs="cr")
            end = time.perf_counter()
            parallel_times.append(end - start)
        except Exception as e:
            print(f"Parallel failed: {e}")
            return None

    baseline_mean = np.mean(baseline_times)
    optimized_mean = np.mean(optimized_times)
    parallel_mean = np.mean(parallel_times)

    speedup_optimized = baseline_mean / optimized_mean
    speedup_parallel = baseline_mean / parallel_mean
    speedup_parallel_vs_opt = optimized_mean / parallel_mean

    # Check results are similar
    r2_baseline = 1 - (result_baseline['deviance'] / np.var(y) / len(y))
    r2_optimized = 1 - (result_optimized['deviance'] / np.var(y) / len(y))
    r2_parallel = 1 - (result_parallel['deviance'] / np.var(y) / len(y))

    fitted_baseline = result_baseline['fitted_values']
    fitted_optimized = result_optimized['fitted_values']
    fitted_parallel = result_parallel['fitted_values']

    max_diff_opt = np.max(np.abs(fitted_baseline - fitted_optimized))
    max_diff_par = np.max(np.abs(fitted_baseline - fitted_parallel))

    return {
        "n": n,
        "d": d,
        "k": k_values,
        "baseline_time": baseline_mean,
        "baseline_std": np.std(baseline_times),
        "optimized_time": optimized_mean,
        "optimized_std": np.std(optimized_times),
        "parallel_time": parallel_mean,
        "parallel_std": np.std(parallel_times),
        "speedup_optimized": speedup_optimized,
        "speedup_parallel": speedup_parallel,
        "speedup_parallel_vs_optimized": speedup_parallel_vs_opt,
        "r2_baseline": float(r2_baseline),
        "r2_optimized": float(r2_optimized),
        "r2_parallel": float(r2_parallel),
        "max_fitted_diff_optimized": float(max_diff_opt),
        "max_fitted_diff_parallel": float(max_diff_par),
    }


def run_comprehensive_comparison():
    """Run comprehensive comparison across various scenarios."""

    print("="*80)
    print("Baseline vs Optimized vs Parallel GAM Performance Comparison")
    print("="*80)
    print()

    test_configs = [
        # Small problems (d=1) - overhead may hurt parallelization
        {"n": 100, "d": 1, "k": [10]},
        {"n": 500, "d": 1, "k": [10]},
        {"n": 1000, "d": 1, "k": [10]},
        {"n": 2000, "d": 1, "k": [20]},

        # Multi-dimensional (d=2-3) - moderate parallelization benefit
        {"n": 500, "d": 2, "k": [10, 10]},
        {"n": 1000, "d": 2, "k": [10, 10]},
        {"n": 500, "d": 3, "k": [10, 10, 10]},
        {"n": 1000, "d": 3, "k": [10, 10, 10]},

        # Higher dimensions (d=5-10) - should show parallelization benefits
        {"n": 500, "d": 5, "k": [8] * 5},
        {"n": 1000, "d": 5, "k": [8] * 5},
        {"n": 500, "d": 7, "k": [8] * 7},
        {"n": 1000, "d": 7, "k": [8] * 7},
        {"n": 500, "d": 10, "k": [8] * 10},
        {"n": 1000, "d": 10, "k": [8] * 10},
    ]

    results = []

    for i, config in enumerate(test_configs, 1):
        n, d, k = config["n"], config["d"], config["k"]
        print(f"Test {i}/{len(test_configs)}: n={n:5d}, d={d:2d}, k={k}", end=" ... ")
        sys.stdout.flush()

        result = run_comparison_test(n, d, k)

        if result:
            speedup_opt = result['speedup_optimized']
            speedup_par = result['speedup_parallel']
            speedup_par_vs_opt = result['speedup_parallel_vs_optimized']

            print(f"✓")
            print(f"   Baseline:  {result['baseline_time']:.4f}s ± {result['baseline_std']:.4f}s")
            print(f"   Optimized: {result['optimized_time']:.4f}s ± {result['optimized_std']:.4f}s ({speedup_opt:.2f}x)")
            print(f"   Parallel:  {result['parallel_time']:.4f}s ± {result['parallel_std']:.4f}s ({speedup_par:.2f}x vs baseline, {speedup_par_vs_opt:.2f}x vs optimized)")

            if speedup_par_vs_opt > 1.0:
                print(f"   🎉 Parallel is {speedup_par_vs_opt:.2f}x faster than optimized!")
            elif speedup_par_vs_opt < 0.95:
                print(f"   ⚠ Parallel overhead ({1/speedup_par_vs_opt:.2f}x slower than optimized)")
            else:
                print(f"   → Similar performance to optimized")

            results.append(result)
        else:
            print("✗ Failed")
        print()

    # Summary statistics
    if results:
        print("="*80)
        print("Summary Statistics")
        print("="*80)

        # Overall speedups
        speedups_opt = [r['speedup_optimized'] for r in results]
        speedups_par = [r['speedup_parallel'] for r in results]
        speedups_par_vs_opt = [r['speedup_parallel_vs_optimized'] for r in results]

        print(f"\nOverall speedup statistics (n={len(results)}):")
        print(f"  Optimized vs Baseline:")
        print(f"    Mean:   {np.mean(speedups_opt):.2f}x")
        print(f"    Median: {np.median(speedups_opt):.2f}x")
        print(f"  Parallel vs Baseline:")
        print(f"    Mean:   {np.mean(speedups_par):.2f}x")
        print(f"    Median: {np.median(speedups_par):.2f}x")
        print(f"  Parallel vs Optimized:")
        print(f"    Mean:   {np.mean(speedups_par_vs_opt):.2f}x")
        print(f"    Median: {np.median(speedups_par_vs_opt):.2f}x")

        # Categorize by dimensionality
        d1 = [r for r in results if r['d'] == 1]
        d2_3 = [r for r in results if 2 <= r['d'] <= 3]
        d5_plus = [r for r in results if r['d'] >= 5]

        print(f"\nSpeedup by dimensionality (Parallel vs Optimized):")
        if d1:
            avg = np.mean([r['speedup_parallel_vs_optimized'] for r in d1])
            print(f"  d=1:      {avg:.2f}x (overhead likely)")
        if d2_3:
            avg = np.mean([r['speedup_parallel_vs_optimized'] for r in d2_3])
            print(f"  d=2-3:    {avg:.2f}x (moderate benefit)")
        if d5_plus:
            avg = np.mean([r['speedup_parallel_vs_optimized'] for r in d5_plus])
            print(f"  d>=5:     {avg:.2f}x (best for parallelization)")

        # Check correctness
        max_r2_diff_opt = max(abs(r['r2_baseline'] - r['r2_optimized']) for r in results)
        max_r2_diff_par = max(abs(r['r2_baseline'] - r['r2_parallel']) for r in results)
        max_fitted_diff_opt = max(r['max_fitted_diff_optimized'] for r in results)
        max_fitted_diff_par = max(r['max_fitted_diff_parallel'] for r in results)

        print(f"\nCorrectness check:")
        print(f"  Optimized:")
        print(f"    Max R² difference:     {max_r2_diff_opt:.6f}")
        print(f"    Max fitted value diff: {max_fitted_diff_opt:.6e}")
        print(f"  Parallel:")
        print(f"    Max R² difference:     {max_r2_diff_par:.6f}")
        print(f"    Max fitted value diff: {max_fitted_diff_par:.6e}")

        if max_r2_diff_opt < 1e-3 and max_r2_diff_par < 1e-3:
            print(f"  ✓ All results match within tolerance!")
        else:
            print(f"  ⚠ Results may differ significantly")

        # Save results
        with open("parallel_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: parallel_comparison.json")

    return results


def main():
    results = run_comprehensive_comparison()

    if results:
        speedups_par_vs_opt = [r['speedup_parallel_vs_optimized'] for r in results]
        avg_speedup = np.mean(speedups_par_vs_opt)

        # Check if parallelization is worthwhile
        d5_plus = [r for r in results if r['d'] >= 5]
        if d5_plus:
            avg_high_d = np.mean([r['speedup_parallel_vs_optimized'] for r in d5_plus])

            print("\n" + "="*80)
            print("🎯 Parallelization Recommendation:")
            print("="*80)
            if avg_high_d > 1.2:
                print(f"✓ Use parallel version for d >= 5 ({avg_high_d:.2f}x faster)")
                print(f"✓ Use optimized version for d < 5 (less overhead)")
            elif avg_high_d > 1.0:
                print(f"→ Marginal benefit for high-d ({avg_high_d:.2f}x)")
                print(f"  Optimized version is safer default")
            else:
                print(f"✗ Parallel overhead dominates (only {avg_high_d:.2f}x)")
                print(f"  Stick with optimized version")
            print("="*80)

        return 0
    else:
        print("Failed to run benchmarks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
