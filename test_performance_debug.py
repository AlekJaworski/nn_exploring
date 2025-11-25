#!/usr/bin/env python3
"""
Quick diagnostic script to identify performance issues.
"""

import numpy as np
import time
import mgcv_rust

print("="*70)
print("Performance Diagnostic Test")
print("="*70)

# Small test
n = 100
X = np.random.randn(n, 2)
y = np.random.randn(n)

print(f"\nQuick test: n={n}, dimensions=2, k=10")
print("Running 5 iterations...")

times = []
for i in range(5):
    gam = mgcv_rust.GAM()
    start = time.perf_counter()
    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    end = time.perf_counter()
    times.append((end - start) * 1000)
    print(f"  Iteration {i+1}: {times[-1]:.2f} ms")

mean_time = np.mean(times)
print(f"\nMean time: {mean_time:.2f} ms")

print("\nDiagnostic:")
if mean_time < 50:
    print("✓ Performance is GOOD - likely using release build")
elif mean_time < 200:
    print("⚠ Performance is OK - might be using release build")
else:
    print("❌ Performance is POOR - likely using DEBUG build!")
    print("\nTo fix:")
    print("  1. Rebuild with release flag:")
    print("     maturin build --release --features 'python,blas'")
    print("  2. Reinstall:")
    print("     pip install --force-reinstall target/wheels/mgcv_rust-*.whl")

# Test if optimized version is faster
print("\n" + "="*70)
print("Testing fit_auto vs fit_auto_optimized")
print("="*70)

X_test = np.random.randn(200, 2)
y_test = np.random.randn(200)

print("\nTesting fit_auto...")
gam1 = mgcv_rust.GAM()
start = time.perf_counter()
gam1.fit_auto(X_test, y_test, k=[10, 10], method='REML', bs='cr')
time_auto = (time.perf_counter() - start) * 1000

print("\nTesting fit_auto_optimized...")
gam2 = mgcv_rust.GAM()
start = time.perf_counter()
gam2.fit_auto_optimized(X_test, y_test, k=[10, 10], method='REML', bs='cr')
time_optimized = (time.perf_counter() - start) * 1000

print(f"\nResults:")
print(f"  fit_auto:           {time_auto:.2f} ms")
print(f"  fit_auto_optimized: {time_optimized:.2f} ms")
print(f"  Speedup:            {time_auto/time_optimized:.2f}x")

if time_optimized < time_auto:
    print(f"\n✓ fit_auto_optimized is {time_auto/time_optimized:.2f}x faster!")
    print("  Consider using fit_auto_optimized in production code.")
else:
    print("\n⚠ No significant speedup from optimization")
