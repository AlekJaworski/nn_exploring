#!/usr/bin/env python3
"""Test if block-wise QR produces correct results."""

import numpy as np
import mgcv_rust

# Test case where we know the answer
np.random.seed(42)
n = 2000
k = 20

x = np.linspace(0, 1, n).reshape(-1, 1)
y = np.sin(2 * np.pi * x.flatten()) + np.random.normal(0, 0.1, n)

print("Testing block-wise QR correctness")
print("=" * 60)
print(f"n={n}, k={k}")

gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[k], method='REML', bs='cr')

print(f"Lambda: {result['lambda'][0]:.6f}")
print(f"Fitted: {result['fitted']}")
print()
print("Expected lambda from R: ~20.76")
print()
if abs(result['lambda'][0] - 20.76) > 1.0:
    print("❌ ERROR: Lambda is way off! Block-wise QR has a bug.")
else:
    print("✓ Lambda looks correct")
