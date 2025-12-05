#!/usr/bin/env python3
"""
Performance diagnostic to identify if you're using debug build.
"""

import numpy as np
import time
import mgcv_rust

print("="*70)
print("mgcv_rust Performance Diagnostic")
print("="*70)

# Quick benchmark
n, d, k = 200, 2, 10
X = np.random.randn(n, d)
y = np.random.randn(n)

print(f"\nQuick benchmark: n={n}, dimensions={d}, k={k}")
print("Running 3 iterations...")

times = []
for i in range(3):
    gam = mgcv_rust.GAM()
    start = time.perf_counter()
    gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.1f} ms")

mean_time = np.mean(times)
print(f"\nAverage: {mean_time:.1f} ms")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if mean_time < 50:
    print("✅ EXCELLENT - Using optimized release build")
    print("   Your performance should match or beat R's mgcv")
elif mean_time < 150:
    print("✓ GOOD - Using release build")
    print("  Performance is decent")
elif mean_time < 500:
    print("⚠ SLOW - Possible issues:")
    print("  1. Using debug build (most likely)")
    print("  2. BLAS not properly linked")
    print("  3. Old CPU without AVX instructions")
else:
    print("❌ VERY SLOW - Using DEBUG build!")
    print("")
    print("  Your build is NOT optimized for production.")
    print("")
    print("  TO FIX:")
    print("  --------")
    print("  1. Clean old builds:")
    print("     cargo clean")
    print("")
    print("  2. Rebuild with --release flag:")
    print("     maturin build --release --features 'python,blas'")
    print("")
    print("  3. Reinstall:")
    print("     pip install --force-reinstall target/wheels/mgcv_rust-*.whl")
    print("")
    print("  Expected speedup: 10-20x faster after rebuilding")

print("\n" + "="*70)
print("Build Check")
print("="*70)

# Check if we can detect build type (not always possible from Python)
import os
import sys

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

# Try to check the wheel filename
import importlib.util
spec = importlib.util.find_spec("mgcv_rust")
if spec and spec.origin:
    print(f"Module location: {spec.origin}")
    if "debug" in spec.origin.lower():
        print("  ⚠ Contains 'debug' in path - might be debug build")
    else:
        print("  ✓ Path looks OK")

print("\n" + "="*70)
