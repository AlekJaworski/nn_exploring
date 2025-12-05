#!/usr/bin/env python3
"""
Profile mgcv_rust to find bottlenecks
"""

import numpy as np
import time
import mgcv_rust
from collections import defaultdict

def profile_gam_fit():
    """Profile a single GAM fit with detailed timing"""

    # Use same data as the test
    np.random.seed(42)
    n, d, k = 500, 4, 12

    X = np.random.uniform(0, 1, size=(n, d))

    # True effects
    effect_1 = np.sin(2 * np.pi * X[:, 0])
    effect_2 = (X[:, 1] - 0.5) ** 2
    effect_3 = X[:, 2]
    effect_4 = np.zeros(n)

    y_true = effect_1 + effect_2 + effect_3 + effect_4
    y = y_true + np.random.normal(0, 0.3, n)

    print("="*70)
    print("Performance Profile: n=500, d=4, k=12")
    print("="*70)

    # Time the full fit
    gam = mgcv_rust.GAM()

    start = time.perf_counter()
    result = gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
    total_time = (time.perf_counter() - start) * 1000

    print(f"\nTotal fit time: {total_time:.2f} ms")
    print(f"Deviance: {result['deviance']:.6f}")
    print(f"Lambda values: {result['lambda']}")

    # Run multiple iterations to get stable timings
    print("\n" + "="*70)
    print("Running 20 iterations for stable timing...")
    print("="*70)

    times = []
    for i in range(20):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
        times.append((time.perf_counter() - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nTiming statistics (20 runs):")
    print(f"  Mean:   {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min:    {min_time:.2f} ms")
    print(f"  Max:    {max_time:.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")

    # Compare with optimized version
    print("\n" + "="*70)
    print("Comparing fit_auto vs fit_auto_optimized")
    print("="*70)

    times_auto = []
    times_opt = []

    for i in range(10):
        # fit_auto
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
        times_auto.append((time.perf_counter() - start) * 1000)

        # fit_auto_optimized
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        gam.fit_auto_optimized(X, y, k=[k]*d, method='REML', bs='cr')
        times_opt.append((time.perf_counter() - start) * 1000)

    print(f"\nfit_auto:           {np.mean(times_auto):.2f} ± {np.std(times_auto):.2f} ms")
    print(f"fit_auto_optimized: {np.mean(times_opt):.2f} ± {np.std(times_opt):.2f} ms")
    print(f"Speedup:            {np.mean(times_auto)/np.mean(times_opt):.2f}x")

    return mean_time

if __name__ == "__main__":
    profile_gam_fit()
