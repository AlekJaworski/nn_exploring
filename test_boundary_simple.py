#!/usr/bin/env python3
"""
Simple boundary test
"""

import numpy as np
import mgcv_rust

# Test 1: Simple sine wave with full range
print("=" * 70)
print("Test 1: Full range fit")
print("=" * 70)

np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(n)

gam = mgcv_rust.GAM()
X = x.reshape(-1, 1)

try:
    result = gam.fit_auto(X, y, k=[10], method='REML')
    print(f"✓ Fit successful with λ = {result['lambda']:.6f}")

    # Predict on extended range
    x_test = np.linspace(-0.1, 1.1, 50)
    X_test = x_test.reshape(-1, 1)
    y_pred = gam.predict(X_test)

    print(f"\nPredictions at boundaries:")
    print(f"  x=-0.1: {y_pred[0]:.4f}")
    print(f"  x=0.0:  {y_pred[5]:.4f}")
    print(f"  x=0.5:  {y_pred[25]:.4f}")
    print(f"  x=1.0:  {y_pred[45]:.4f}")
    print(f"  x=1.1:  {y_pred[49]:.4f}")

    # Check for zeros
    if np.any(np.abs(y_pred) < 1e-10):
        print("\n⚠ WARNING: Found zero predictions!")
        zero_indices = np.where(np.abs(y_pred) < 1e-10)[0]
        print(f"  Zero at indices: {zero_indices}")
        print(f"  Corresponding x values: {x_test[zero_indices]}")
    else:
        print("\n✓ No zero predictions - boundary issue fixed!")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Smaller k value
print("\n" + "=" * 70)
print("Test 2: Smaller k value (k=6)")
print("=" * 70)

try:
    gam2 = mgcv_rust.GAM()
    result2 = gam2.fit_auto(X, y, k=[6], method='REML')
    print(f"✓ Fit successful with λ = {result2['lambda']:.6f}")

    y_pred2 = gam2.predict(X_test)
    print(f"\nBoundary predictions:")
    print(f"  x=-0.1: {y_pred2[0]:.4f}")
    print(f"  x=1.1:  {y_pred2[49]:.4f}")

except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Training on subset, predict on full range
print("\n" + "=" * 70)
print("Test 3: Train on [0.3, 0.7], predict on [0, 1]")
print("=" * 70)

x_train = np.linspace(0.3, 0.7, 50)
y_train = np.sin(2 * np.pi * x_train) + 0.2 * np.random.randn(50)
X_train = x_train.reshape(-1, 1)

try:
    gam3 = mgcv_rust.GAM()
    result3 = gam3.fit_auto(X_train, y_train, k=[6], method='REML')
    print(f"✓ Fit successful with λ = {result3['lambda']:.6f}")

    x_test_full = np.linspace(0, 1, 50)
    X_test_full = x_test_full.reshape(-1, 1)
    y_pred3 = gam3.predict(X_test_full)

    print(f"\nPredictions:")
    print(f"  x=0.0 (outside): {y_pred3[0]:.4f}")
    print(f"  x=0.3 (edge):    {y_pred3[15]:.4f}")
    print(f"  x=0.5 (middle):  {y_pred3[25]:.4f}")
    print(f"  x=0.7 (edge):    {y_pred3[35]:.4f}")
    print(f"  x=1.0 (outside): {y_pred3[49]:.4f}")

    if np.abs(y_pred3[0]) < 1e-10 or np.abs(y_pred3[49]) < 1e-10:
        print("\n⚠ Still getting zeros at boundaries")
    else:
        print("\n✓ Good extrapolation behavior!")

except Exception as e:
    print(f"✗ Error: {e}")
