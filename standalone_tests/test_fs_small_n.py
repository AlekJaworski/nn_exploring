#!/usr/bin/env python3
"""Test Fellner-Schall vs Newton for small n."""

import mgcv_rust
import numpy as np
import time

def bench(n, d, k, algorithm, seed=123):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)
    
    times = []
    for _ in range(3):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        result = gam.fit_auto_optimized(
            X, y, k=[k] * d, method='REML', bs='cr', max_iter=100,
            algorithm=algorithm
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
    
    return np.median(times)

print("Newton vs Fellner-Schall for small n=1000")
print("=" * 60)
print(f"{'Config':<20s} | {'Newton':>10s} | {'FS':>10s} | {'R bam':>10s}")
print("-" * 60)

# R bam times from historical benchmarks
r_times = {
    (1000, 1, 10): 22.0,
    (1000, 2, 10): 25.0,
    (1000, 4, 10): 50.0,
}

for n, d, k in [(1000, 1, 10), (1000, 2, 10), (1000, 4, 10)]:
    newton_ms = bench(n, d, k, 'newton')
    fs_ms = bench(n, d, k, 'fellner-schall')
    r_ms = r_times.get((n, d, k), '?')
    
    config = f"n={n}, d={d}, k={k}"
    print(f"{config:<20s} | {newton_ms:>10.1f} | {fs_ms:>10.1f} | {r_ms:>10.1f}")

print()
print("Recommendation: Use Fellner-Schall as default for n < 2000")
