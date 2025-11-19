#!/usr/bin/env python3
"""
Test sp parameterization on a simple 2D case with gradient profiling.
"""
import numpy as np
import mgcv_rust
import os

# Enable profiling
os.environ['MGCV_PROFILE'] = '1'

print("=" * 80)
print("Testing sp parameterization on 2D GAM")
print("=" * 80)

# Simple 2D test case
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
y = np.sin(X[:, 0]) + 0.5*X[:, 1]**2 + 0.1*np.random.randn(n)

print(f"\nData: n={n}, 2D with k=[8,8]")
print("\nFitting GAM with sp parameterization...")
print("-" * 80)

gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[8, 8], method='REML', bs='cr')

print("-" * 80)
print("\nResults:")
print(f"  Lambda: {result.get('lambda', result.get('all_lambdas', []))}")

# Get predictions and compute R²
preds = gam.predict(X)
r2 = 1 - np.var(y - preds) / np.var(y)
print(f"  R²: {r2:.4f}")

print("\n" + "=" * 80)
print("SUCCESS: Check gradient values in profiling output above")
print("Target: gradients should be < 100 (not 10^28!)")
print("=" * 80)
