#!/usr/bin/env python3
"""Benchmark mgcv_rust vs R mgcv on high-dimensional problems."""

import mgcv_rust
import numpy as np
import time
import sys

def bench(n, d, k, seed=123, runs=3):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)

    times = []
    result = None
    for r in range(runs):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        result = gam.fit_auto_optimized(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    median_ms = np.median(times) * 1000
    iters = result.get('iterations', '?') if result else '?'
    return median_ms, iters

# Extended configs with higher dimensions
configs = [
    # Original losing configs (small n)
    (1000, 2, 10),
    (1000, 4, 10),
    # Higher dimensional configs
    (5000, 10, 10),
    (5000, 10, 12),
    (5000, 12, 10),
    (10000, 8, 10),
    (10000, 10, 10),
    (10000, 12, 10),
    (20000, 4, 10),
    (20000, 8, 10),
    (20000, 10, 10),
]

print("=" * 70)
print("  HIGH-DIMENSIONAL BENCHMARK")
print("  Testing scaling with d=10-12, k=10-12, n=5000-20000")
print("=" * 70)
print()

header = f"{'Config':<30s} | {'Rust(ms)':>10s} | {'p=d*k':>6s}"
print(header)
print("-" * len(header))

for n, d, k in configs:
    sys.stdout.flush()
    try:
        ms, iters = bench(n, d, k)
        p = d * k
        config_str = f"n={n:>6d}, d={d:>2d}, k={k:>2d}"
        print(f"{config_str:<30s} | {ms:>10.1f} | {p:>6d}")
    except Exception as e:
        config_str = f"n={n:>6d}, d={d:>2d}, k={k:>2d}"
        print(f"{config_str:<30s} | {'ERROR':>10s} | {str(e)[:30]}")

print()
print("Testing worst-case: d=16, k=15, n=5000 (p=240)...")
try:
    ms, iters = bench(5000, 16, 15, runs=1)
    print(f"  Completed in {ms:.1f}ms")
except Exception as e:
    print(f"  ERROR: {e}")
