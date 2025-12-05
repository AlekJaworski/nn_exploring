#!/usr/bin/env python3
"""
Test linear extrapolation with correct gradient
"""

import numpy as np
import mgcv_rust

print("="*70)
print("Testing Linear Extrapolation Gradient")
print("="*70)

# Test 1: Linear function (should extrapolate perfectly)
np.random.seed(42)
x_train = np.linspace(0.3, 0.7, 50)
y_train = 2*x_train + 1 + 0.05*np.random.randn(50)

print(f"\nTest 1: Linear function y = 2x + 1")
print(f"Training range: [{x_train.min():.2f}, {x_train.max():.2f}]")

gam = mgcv_rust.GAM()
result = gam.fit_auto(x_train.reshape(-1, 1), y_train, k=[6], method='REML')
print(f"λ = {result['lambda']:.6f}")

# Test points
x_test = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
X_test = x_test.reshape(-1, 1)
y_pred = gam.predict(X_test)
y_true = 2*x_test + 1

print(f"\n{'x':>6s} {'Region':>8s} {'True':>8s} {'Pred':>8s} {'Error':>8s}")
print("-"*50)
for i, x in enumerate(x_test):
    region = "LEFT" if x < 0.3 else ("RIGHT" if x > 0.7 else "IN")
    error = abs(y_pred[i] - y_true[i])
    print(f"{x:6.2f} {region:>8s} {y_true[i]:8.4f} {y_pred[i]:8.4f} {error:8.4f}")

# Check gradients at boundaries
print(f"\n{'='*70}")
print("Gradient Check")
print("="*70)

# Estimate gradient from predictions
# Left extrapolation: between x=0.2 and x=0.3
left_grad_pred = (y_pred[2] - y_pred[1]) / (x_test[2] - x_test[1])  # 0.3 - 0.2
print(f"\nLeft side (x < 0.3):")
print(f"  True gradient:      2.0000")
print(f"  Predicted gradient: {left_grad_pred:.4f}")
print(f"  Error: {abs(left_grad_pred - 2.0):.4f}")

# Right extrapolation: between x=0.7 and x=0.8
right_grad_pred = (y_pred[6] - y_pred[5]) / (x_test[6] - x_test[5])  # 1.0 - 0.8
print(f"\nRight side (x > 0.7):")
print(f"  True gradient:      2.0000")
print(f"  Predicted gradient: {right_grad_pred:.4f}")
print(f"  Error: {abs(right_grad_pred - 2.0):.4f}")

# Within range: between x=0.3 and x=0.5
in_grad_pred = (y_pred[3] - y_pred[2]) / (x_test[3] - x_test[2])  # 0.5 - 0.3
print(f"\nWithin range [0.3, 0.7]:")
print(f"  True gradient:      2.0000")
print(f"  Predicted gradient: {in_grad_pred:.4f}")
print(f"  Error: {abs(in_grad_pred - 2.0):.4f}")

# Test 2: Sine wave (more complex)
print(f"\n{'='*70}")
print("Test 2: Sine wave y = sin(2πx)")
print("="*70)

x_train2 = np.linspace(0.2, 0.8, 100)
y_train2 = np.sin(2*np.pi*x_train2) + 0.1*np.random.randn(100)

gam2 = mgcv_rust.GAM()
result2 = gam2.fit_auto(x_train2.reshape(-1, 1), y_train2, k=[10], method='REML')

x_test2 = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
X_test2 = x_test2.reshape(-1, 1)
y_pred2 = gam2.predict(X_test2)
y_true2 = np.sin(2*np.pi*x_test2)

print(f"\n{'x':>6s} {'Region':>8s} {'True':>8s} {'Pred':>8s} {'Error':>8s}")
print("-"*50)
for i, x in enumerate(x_test2):
    region = "LEFT" if x < 0.2 else ("RIGHT" if x > 0.8 else "IN")
    error = abs(y_pred2[i] - y_true2[i])
    print(f"{x:6.2f} {region:>8s} {y_true2[i]:8.4f} {y_pred2[i]:8.4f} {error:8.4f}")

# Check that extrapolation is not constant (gradient should be non-zero)
left_extrap_grad = (y_pred2[1] - y_pred2[0]) / (x_test2[1] - x_test2[0])
right_extrap_grad = (y_pred2[4] - y_pred2[3]) / (x_test2[4] - x_test2[3])

print(f"\nExtrapolation gradients (should be non-zero):")
print(f"  Left:  {left_extrap_grad:.4f}")
print(f"  Right: {right_extrap_grad:.4f}")

if abs(left_extrap_grad) > 0.1 and abs(right_extrap_grad) > 0.1:
    print(f"\n✓ PASS: Extrapolation has non-zero gradient")
else:
    print(f"\n✗ FAIL: Extrapolation appears constant")
