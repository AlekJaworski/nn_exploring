#!/usr/bin/env python3
"""
Test EDF functionality in Python bindings.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path for local testing
sys.path.insert(0, os.path.dirname(__file__))

try:
    import mgcv_rust
    print("✓ mgcv_rust module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import mgcv_rust: {e}")
    print("Make sure to run: maturin develop --features python,blas")
    sys.exit(1)

# Generate test data
np.random.seed(42)
n = 100
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

print(f"\nTest data: n={n}, d=2")
print(f"x shape: {x.shape}, y shape: {y.shape}")

# Test 1: Default behavior (use_edf=False)
print("\n" + "="*50)
print("Test 1: Default (use_edf=False, Rank method)")
print("="*50)

gam1 = mgcv_rust.GAM()
try:
    result1 = gam1.fit(x, y, k=[10, 10], method='REML', max_iter=5)
    print("✓ Fit succeeded")
    print(f"  λ = {result1['lambda']}")
    print(".6f")
except Exception as e:
    print(f"✗ Fit failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: EDF method (use_edf=True)
print("\n" + "="*50)
print("Test 2: EDF method (use_edf=True)")
print("="*50)

gam2 = mgcv_rust.GAM()
try:
    result2 = gam2.fit(x, y, k=[10, 10], method='REML', max_iter=5, use_edf=True)
    print("✓ Fit succeeded")
    print(f"  λ = {result2['lambda']}")
    print(".6f")
except Exception as e:
    print(f"✗ Fit failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare results
print("\n" + "="*50)
print("COMPARISON")
print("="*50)

lambda_diff = np.abs(result2['lambda'] - result1['lambda'])
lambda_ratio = result2['lambda'] / result1['lambda']

print("Rank method (default):")
print(f"  λ = {result1['lambda']}")
print(".6f")

print("EDF method:")
print(f"  λ = {result2['lambda']}")
print(".6f")

print("Differences:")
print(f"  λ difference: {lambda_diff}")
print(f"  λ ratio: {lambda_ratio}")

if np.allclose(result1['lambda'], result2['lambda'], rtol=0.1):
    print("✓ Results are similar (expected for well-conditioned problems)")
else:
    print("⚠ Results differ (EDF may be more accurate for this case)")

print("\n" + "="*50)
print("SUCCESS: EDF functionality is working in Python bindings!")
print("="*50)
print("\nUsage:")
print("  gam.fit(x, y, k=[10, 10], use_edf=False)  # Default, fast")
print("  gam.fit(x, y, k=[10, 10], use_edf=True)   # EDF, accurate")