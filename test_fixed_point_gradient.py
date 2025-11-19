#!/usr/bin/env python3
"""
Compare our gradient computation with mgcv at a fixed point.
"""
import numpy as np
import mgcv_rust
import os

# Enable debug output
os.environ['MGCV_GRAD_DEBUG'] = '1'

# Load the fixed point data
data = np.loadtxt('fixed_point_data.csv', delimiter=',', skiprows=1)
X = data[:, :2]  # x1, x2
y = data[:, 2]   # y

print("=" * 80)
print("FIXED POINT GRADIENT COMPARISON")
print("=" * 80)
print(f"\nData: n={len(y)}, 2D")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\n" + "=" * 80)
print("mgcv's values at solution:")
print("=" * 80)
print("S.scale:   [70.87, 173.11]")
print("sp:        [1.470, 1.564]")
print("lambda:    [104.18, 270.66]")
print("gradient:  [~10^-13, ~10^-11, ~10^-9]  (near zero)")
print("iterations: 5")

print("\n" + "=" * 80)
print("Our gradient computation:")
print("=" * 80)

# Fit with our implementation
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[8, 8], method='REML', bs='cr')

print("\nFinal results:")
print(f"Lambda: {result.get('lambda', result.get('all_lambdas', []))}")

# Get predictions
preds = gam.predict(X)
r2 = 1 - np.var(y - preds) / np.var(y)
print(f"R²: {r2:.4f}")

print("\n" + "=" * 80)
print("Check debug output above for:")
print("=" * 80)
print("1. trace_unscaled values - should be moderate (~10^0 to 10^2)")
print("2. P matrix norm - should be small (~10^0 to 10^1)")
print("3. Gradient values - should be ~10^-6, not 10^28!")
print("=" * 80)
