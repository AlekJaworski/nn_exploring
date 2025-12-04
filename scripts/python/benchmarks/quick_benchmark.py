#!/usr/bin/env python3
"""Quick benchmark to test optimizations."""

import numpy as np
import time
import mgcv_rust

def benchmark_size(n, k, n_runs=5):
    """Benchmark a specific problem size."""
    np.random.seed(42)
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.flatten()) + np.random.normal(0, 0.1, n)

    times = []
    for _ in range(n_runs):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        result = gam.fit_auto(x, y, k=[k], method='REML', bs='cr')
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times)
    }

print("Quick Benchmark - Optimized mgcvrust")
print("=" * 60)

test_cases = [
    (1000, 15),
    (2000, 20),
    (5000, 20),
    (7000, 20),
]

for n, k in test_cases:
    result = benchmark_size(n, k)
    print(f"n={n:5d}, k={k:2d}: {result['mean']:.4f}s ± {result['std']:.4f}s  (min: {result['min']:.4f}s)")
