#!/usr/bin/env python3
"""
Test to identify n=2500 anomaly by comparing fit_auto vs fit_auto_optimized.
"""

import numpy as np
import time
import mgcv_rust

def generate_4d_data(n, seed=42):
    """Generate 4D test data."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, 4))

    y_true = (
        np.sin(2 * np.pi * X[:, 0]) * 2.0 +
        np.cos(4 * np.pi * X[:, 1]) * 1.5 +
        X[:, 2] * 0.5
    )
    y = y_true + np.random.normal(0, 0.1, size=n)
    return X, y

def benchmark_method(X, y, method_name, k=16, n_iters=10):
    """Benchmark a specific fit method."""
    k_list = [k] * 4

    # Warm-up
    gam = mgcv_rust.GAM()
    if method_name == 'optimized':
        gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
    else:
        gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')

    # Timing runs
    times = []
    for i in range(n_iters):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        if method_name == 'optimized':
            gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
        else:
            gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.array(times)

def main():
    print("="*70)
    print("Testing fit_auto vs fit_auto_optimized at different scales")
    print("="*70)

    sample_sizes = [500, 1500, 2500, 5000]
    k = 16

    results = []

    for n in sample_sizes:
        print(f"\n{'='*70}")
        print(f"n={n}")
        print(f"{'='*70}")

        X, y = generate_4d_data(n)

        # Test both methods
        times_normal = benchmark_method(X, y, 'normal', k=k)
        times_optimized = benchmark_method(X, y, 'optimized', k=k)

        mean_normal = np.mean(times_normal)
        std_normal = np.std(times_normal)
        mean_optimized = np.mean(times_optimized)
        std_optimized = np.std(times_optimized)

        speedup = mean_normal / mean_optimized

        print(f"fit_auto:           {mean_normal:8.2f} ± {std_normal:6.2f} ms")
        print(f"fit_auto_optimized: {mean_optimized:8.2f} ± {std_optimized:6.2f} ms")
        print(f"Speedup:            {speedup:8.2f}x")

        results.append({
            'n': n,
            'normal': mean_normal,
            'optimized': mean_optimized,
            'speedup': speedup
        })

    # Analyze scaling
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS")
    print(f"{'='*70}")
    print(f"\n{'n':>6}  {'fit_auto':>12}  {'optimized':>12}  {'Speedup':>10}")
    print("-"*70)

    for r in results:
        print(f"{r['n']:6d}  {r['normal']:12.2f}  {r['optimized']:12.2f}  {r['speedup']:10.2f}x")

    # Check for anomalies in scaling
    print(f"\n{'='*70}")
    print("SCALING RATIOS (vs previous size)")
    print(f"{'='*70}")

    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]

        n_ratio = curr['n'] / prev['n']
        normal_ratio = curr['normal'] / prev['normal']
        opt_ratio = curr['optimized'] / prev['optimized']

        print(f"\nn={prev['n']} → n={curr['n']} (sample ratio: {n_ratio:.2f}x)")
        print(f"  fit_auto time ratio:           {normal_ratio:.2f}x", end="")
        if normal_ratio > n_ratio * 1.3:
            print("  ⚠ SLOWER than expected")
        elif normal_ratio < n_ratio * 0.7:
            print("  ✓ FASTER than expected")
        else:
            print("  ✓ Normal")

        print(f"  fit_auto_optimized time ratio: {opt_ratio:.2f}x", end="")
        if opt_ratio > n_ratio * 1.3:
            print("  ⚠ SLOWER than expected")
        elif opt_ratio < n_ratio * 0.7:
            print("  ✓ FASTER than expected")
        else:
            print("  ✓ Normal")

if __name__ == "__main__":
    main()
