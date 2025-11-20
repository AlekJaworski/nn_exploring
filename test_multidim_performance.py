#!/usr/bin/env python3
"""
Performance test for multidimensional GAM after gradient scaling fix.

Tests convergence speed and accuracy for various dimensional cases.
"""

import numpy as np
import mgcv_rust
import os
import time

# Enable profiling to see iteration counts
os.environ['MGCV_PROFILE'] = '1'

print("=" * 80)
print("MULTIDIMENSIONAL PERFORMANCE TEST")
print("After gradient scaling fix: (n-total_rank)/rank_i")
print("=" * 80)

test_cases = [
    ("2D", 2, 500, 10),
    ("3D", 3, 500, 10),
    ("4D", 4, 500, 10),
    ("4D Large", 4, 1000, 12),
    ("6D", 6, 500, 10),
]

results = []

for name, ndim, n, k in test_cases:
    print(f"\n{'=' * 80}")
    print(f"Test: {name} (ndim={ndim}, n={n}, k={k})")
    print(f"{'=' * 80}")

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(n, ndim)

    # True function: mix of sin, quadratic, linear
    y_true = np.sin(X[:, 0])
    if ndim >= 2:
        y_true += 0.5 * X[:, 1]**2
    if ndim >= 3:
        y_true += 0.3 * X[:, 2]
    if ndim >= 4:
        y_true += np.cos(X[:, 3])
    if ndim >= 5:
        y_true += 0.2 * X[:, 4]
    if ndim >= 6:
        y_true += np.sin(2 * X[:, 5])

    y = y_true + np.random.randn(n) * 0.1

    # Fit GAM
    gam = mgcv_rust.GAM()
    k_list = [k] * ndim

    start_time = time.time()
    result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
    elapsed = time.time() - start_time

    # Make predictions
    pred = gam.predict(X)

    # Compute metrics
    rss = np.sum((y - pred)**2)
    tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss/tss

    print(f"\nResults:")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Lambda: {result['lambda']}")
    print(f"  Deviance: {result.get('deviance', 'N/A')}")
    print(f"  R²: {r2:.4f}")
    print(f"  Prediction MSE: {np.mean((y - pred)**2):.6f}")

    results.append({
        'name': name,
        'ndim': ndim,
        'n': n,
        'k': k,
        'time': elapsed,
        'lambda': result['lambda'],
        'r2': r2,
    })

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Test':<12} | {'Dims':>4} | {'n':>5} | {'k':>3} | {'Time (s)':>9} | {'R²':>6}")
print("-" * 80)

for r in results:
    print(f"{r['name']:<12} | {r['ndim']:>4} | {r['n']:>5} | {r['k']:>3} | {r['time']:>9.3f} | {r['r2']:>6.4f}")

print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS")
print("=" * 80)
print("""
Check the profiling output above for iteration counts.

Expected behavior after gradient fix:
- Should converge in 5-10 iterations (vs 20-100 before fix)
- Gradient should start at ~100-300 range (vs 3-4 before fix)
- No oscillation in late iterations
- R² should be reasonable (>0.5 for these synthetic datasets)

Key metrics:
- 2D/3D: Should be very fast (<1s)
- 4D: Should converge quickly (~1-2s)
- 6D: More iterations expected but should still converge
""")
