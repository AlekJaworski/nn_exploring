#!/usr/bin/env python3
"""Test extreme dimensions to find scaling limits."""

import mgcv_rust
import numpy as np
import time
import sys

def bench(n, d, k, seed=42, runs=1):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)
    
    times = []
    for r in range(runs):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        try:
            result = gam.fit_auto_optimized(X, y, k=[k] * d, method='REML', bs='cr', max_iter=50)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
        except Exception as e:
            return None, str(e)
    
    median_ms = np.median(times) * 1000 if times else None
    return median_ms, None

print("Testing extreme dimensions...")
print()

test_cases = [
    # Moderate high-d
    (5000, 10, 10, "baseline high-d"),
    (5000, 12, 10, "more dims"),
    (5000, 10, 12, "more knots (problematic)"),
    
    # Very high-d (may fail or be very slow)
    (5000, 15, 10, "extreme d=15"),
    (10000, 12, 10, "large n, high d"),
    
    # High-k tests
    (5000, 6, 15, "high k=15"),
    (5000, 6, 20, "very high k=20"),
]

for n, d, k, desc in test_cases:
    p = d * k
    print(f"{desc}: n={n}, d={d}, k={k}, p={p}... ", end="", flush=True)
    
    ms, err = bench(n, d, k)
    if err:
        print(f"ERROR: {err[:60]}")
    elif ms:
        print(f"{ms:.1f}ms")
    else:
        print("FAILED")

print()
print("Testing if problem is specific to k>=12...")
for k in [8, 9, 10, 11, 12, 13]:
    print(f"  k={k}: ", end="", flush=True)
    ms, err = bench(5000, 10, k)
    if ms:
        print(f"{ms:.1f}ms")
    else:
        print(f"ERROR: {err[:40] if err else 'failed'}")
