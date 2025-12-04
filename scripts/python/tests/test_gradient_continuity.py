#!/usr/bin/env python3
"""
Test gradient continuity at boundaries
The gradient should be continuous across the boundary
"""

import numpy as np
import mgcv_rust

print("="*70)
print("Testing Gradient Continuity at Boundaries")
print("="*70)

# Fit to linear function
np.random.seed(42)
x_train = np.linspace(0.3, 0.7, 50)
y_train = 2*x_train + 1 + 0.05*np.random.randn(50)

gam = mgcv_rust.GAM()
result = gam.fit_auto(x_train.reshape(-1, 1), y_train, k=[8], method='REML')

print(f"\nLinear function: y = 2x + 1")
print(f"Training range: [0.3, 0.7]")
print(f"λ = {result['lambda']:.6f}")

# Create fine grid around boundaries
print(f"\n{'='*70}")
print("Left Boundary (x = 0.3)")
print("="*70)

x_left = np.array([0.25, 0.28, 0.29, 0.30, 0.31, 0.32, 0.35])
X_left = x_left.reshape(-1, 1)
y_left = gam.predict(X_left)

print(f"\n{'x':>6s} {'Region':>10s} {'Prediction':>12s} {'Gradient':>12s}")
print("-"*50)
for i, x in enumerate(x_left):
    region = "EXTRAP" if x < 0.3 else "IN RANGE"
    pred = y_left[i]

    # Compute gradient using forward difference
    if i < len(x_left) - 1:
        grad = (y_left[i+1] - y_left[i]) / (x_left[i+1] - x_left[i])
        print(f"{x:6.3f} {region:>10s} {pred:12.6f} {grad:12.6f}")
    else:
        print(f"{x:6.3f} {region:>10s} {pred:12.6f}")

print(f"\n{'='*70}")
print("Right Boundary (x = 0.7)")
print("="*70)

x_right = np.array([0.65, 0.68, 0.69, 0.70, 0.71, 0.72, 0.75])
X_right = x_right.reshape(-1, 1)
y_right = gam.predict(X_right)

print(f"\n{'x':>6s} {'Region':>10s} {'Prediction':>12s} {'Gradient':>12s}")
print("-"*50)
for i, x in enumerate(x_right):
    region = "IN RANGE" if x <= 0.7 else "EXTRAP"
    pred = y_right[i]

    # Compute gradient using forward difference
    if i < len(x_right) - 1:
        grad = (y_right[i+1] - y_right[i]) / (x_right[i+1] - x_right[i])
        print(f"{x:6.3f} {region:>10s} {pred:12.6f} {grad:12.6f}")
    else:
        print(f"{x:6.3f} {region:>10s} {pred:12.6f}")

# Check gradient jump across boundaries
print(f"\n{'='*70}")
print("Gradient Discontinuity Analysis")
print("="*70)

# Left boundary: gradient just before vs just after 0.3
grad_before_left = (y_left[3] - y_left[2]) / (x_left[3] - x_left[2])  # 0.29->0.30
grad_after_left = (y_left[4] - y_left[3]) / (x_left[4] - x_left[3])   # 0.30->0.31
jump_left = abs(grad_after_left - grad_before_left)

print(f"\nLeft boundary (x=0.3):")
print(f"  Gradient before: {grad_before_left:.6f}")
print(f"  Gradient after:  {grad_after_left:.6f}")
print(f"  Jump:            {jump_left:.6f} {'✓ CONTINUOUS' if jump_left < 0.1 else '✗ DISCONTINUOUS'}")

# Right boundary: gradient just before vs just after 0.7
grad_before_right = (y_right[3] - y_right[2]) / (x_right[3] - x_right[2])  # 0.69->0.70
grad_after_right = (y_right[4] - y_right[3]) / (x_right[4] - x_right[3])   # 0.70->0.71
jump_right = abs(grad_after_right - grad_before_right)

print(f"\nRight boundary (x=0.7):")
print(f"  Gradient before: {grad_before_right:.6f}")
print(f"  Gradient after:  {grad_after_right:.6f}")
print(f"  Jump:            {jump_right:.6f} {'✓ CONTINUOUS' if jump_right < 0.1 else '✗ DISCONTINUOUS'}")

# Check if extrapolation is truly linear (constant gradient in extrap region)
print(f"\n{'='*70}")
print("Linearity Check (gradient should be constant in extrapolation)")
print("="*70)

# Left extrapolation region
grad_extrap_left_1 = (y_left[1] - y_left[0]) / (x_left[1] - x_left[0])  # 0.25->0.28
grad_extrap_left_2 = (y_left[2] - y_left[1]) / (x_left[2] - x_left[1])  # 0.28->0.29
extrap_left_var = abs(grad_extrap_left_2 - grad_extrap_left_1)

print(f"\nLeft extrapolation gradients:")
print(f"  0.25->0.28: {grad_extrap_left_1:.6f}")
print(f"  0.28->0.29: {grad_extrap_left_2:.6f}")
print(f"  Variation:  {extrap_left_var:.6f} {'✓ LINEAR' if extrap_left_var < 0.01 else '✗ NON-LINEAR'}")

# Right extrapolation region
grad_extrap_right_1 = (y_right[5] - y_right[4]) / (x_right[5] - x_right[4])  # 0.71->0.72
grad_extrap_right_2 = (y_right[6] - y_right[5]) / (x_right[6] - x_right[5])  # 0.72->0.75
extrap_right_var = abs(grad_extrap_right_2 - grad_extrap_right_1)

print(f"\nRight extrapolation gradients:")
print(f"  0.71->0.72: {grad_extrap_right_1:.6f}")
print(f"  0.72->0.75: {grad_extrap_right_2:.6f}")
print(f"  Variation:  {extrap_right_var:.6f} {'✓ LINEAR' if extrap_right_var < 0.01 else '✗ NON-LINEAR'}")
