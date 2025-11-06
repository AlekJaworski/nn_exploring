#!/usr/bin/env python3
"""
Test boundary behavior WITHIN training range (no extrapolation)
"""

import numpy as np
import mgcv_rust

print("=" * 70)
print("Test: Fit and predict on SAME data (no extrapolation)")
print("=" * 70)

np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.2 * np.random.randn(n)

X = x.reshape(-1, 1)

# Revert the padding by fitting with exact range
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[10], method='REML')

print(f"✓ Fit successful with λ = {result['lambda']:.6f}")

# Predict on the SAME data (no extrapolation)
y_pred = gam.predict(X)

# Check predictions at boundaries vs middle
print(f"\nPredictions (first 5):")
for i in range(5):
    print(f"  x={x[i]:.3f}: pred={y_pred[i]:.4f}, true={y_true[i]:.4f}, obs={y[i]:.4f}")

print(f"\nPredictions (last 5):")
for i in range(95, 100):
    print(f"  x={x[i]:.3f}: pred={y_pred[i]:.4f}, true={y_true[i]:.4f}, obs={y[i]:.4f}")

# Check errors
errors = np.abs(y_pred - y_true)
print(f"\nError analysis:")
print(f"  Mean error:      {np.mean(errors):.4f}")
print(f"  Max error:       {np.max(errors):.4f}")
print(f"  Max error at:    x={x[np.argmax(errors)]:.3f}")

# Check boundary vs middle errors
boundary_indices = list(range(5)) + list(range(95, 100))
middle_indices = list(range(45, 55))

boundary_errors = errors[boundary_indices]
middle_errors = errors[middle_indices]

print(f"\n  Boundary error (mean): {np.mean(boundary_errors):.4f}")
print(f"  Middle error (mean):   {np.mean(middle_errors):.4f}")

if np.mean(boundary_errors) > 2 * np.mean(middle_errors):
    print(f"\n⚠ WARNING: Boundary errors are much larger than middle!")
    print(f"⚠ This indicates boundary constraint issues WITHIN the training range")
else:
    print(f"\n✓ Boundary behavior within training range is reasonable")

# Visualize the fitted function
print(f"\nFitted values at key points:")
quartiles = [0, 25, 50, 75, 99]
for i in quartiles:
    print(f"  x={x[i]:.3f} (position {i}): pred={y_pred[i]:.4f}")
