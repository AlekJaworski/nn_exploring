#!/usr/bin/env python3
"""
Test to visualize boundary behavior in GAMs
This should reveal any sharp drops at the edges
"""

import numpy as np
import mgcv_rust

def test_boundary_extrapolation():
    """Test how the GAM behaves near and outside the boundary"""
    print("=" * 70)
    print("Testing Boundary Behavior")
    print("=" * 70)

    np.random.seed(42)

    # Training data: only in middle range [0.2, 0.8]
    n_train = 100
    x_train = np.linspace(0.2, 0.8, n_train)
    y_train = np.sin(2 * np.pi * x_train) + 0.2 * np.random.randn(n_train)

    # Test data: includes boundaries and extrapolation [0, 1]
    n_test = 200
    x_test = np.linspace(0.0, 1.0, n_test)
    y_true = np.sin(2 * np.pi * x_test)

    print(f"\nTraining range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Test range:     [{x_test.min():.2f}, {x_test.max():.2f}]")

    # Fit GAM
    gam = mgcv_rust.GAM()
    X_train = x_train.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)

    result = gam.fit_auto(X_train, y_train, k=[15], method='REML')
    print(f"\nλ = {result['lambda']:.6f}")

    # Predict on test data
    y_pred = gam.predict(X_test)

    # Check for boundary issues
    print("\n" + "=" * 70)
    print("Boundary Analysis")
    print("=" * 70)

    # Find predictions at boundaries
    left_boundary = np.where(x_test <= 0.2)[0]
    right_boundary = np.where(x_test >= 0.8)[0]
    middle = np.where((x_test > 0.2) & (x_test < 0.8))[0]

    print(f"\nLeft boundary region (x ≤ 0.2):")
    print(f"  Predictions: {y_pred[left_boundary][:5]}")
    print(f"  True values: {y_true[left_boundary][:5]}")

    print(f"\nMiddle region (0.2 < x < 0.8):")
    print(f"  Predictions: {y_pred[middle][:5]}")
    print(f"  True values: {y_true[middle][:5]}")

    print(f"\nRight boundary region (x ≥ 0.8):")
    print(f"  Predictions: {y_pred[right_boundary][:5]}")
    print(f"  True values: {y_true[right_boundary][:5]}")

    # Check for sharp drops (large derivatives at boundaries)
    def compute_derivative(x, y):
        """Compute numerical derivative"""
        dx = x[1] - x[0]
        dy = np.diff(y) / dx
        return dy

    dy = compute_derivative(x_test, y_pred)

    print(f"\n" + "=" * 70)
    print("Derivative Analysis (looking for sharp changes)")
    print("=" * 70)

    # Derivatives at different regions
    left_deriv = dy[left_boundary[:-1]]
    middle_deriv = dy[middle[:-1]]
    right_deriv = dy[right_boundary[:-1]]

    print(f"\nLeft boundary derivatives:")
    print(f"  Mean: {np.mean(np.abs(left_deriv)):.4f}")
    print(f"  Max:  {np.max(np.abs(left_deriv)):.4f}")
    print(f"  First 5: {left_deriv[:5]}")

    print(f"\nMiddle derivatives:")
    print(f"  Mean: {np.mean(np.abs(middle_deriv)):.4f}")
    print(f"  Max:  {np.max(np.abs(middle_deriv)):.4f}")

    print(f"\nRight boundary derivatives:")
    print(f"  Mean: {np.mean(np.abs(right_deriv)):.4f}")
    print(f"  Max:  {np.max(np.abs(right_deriv)):.4f}")
    print(f"  Last 5: {right_deriv[-5:]}")

    # Check for sudden jumps (second derivative)
    d2y = np.diff(dy)
    print(f"\n" + "=" * 70)
    print("Second Derivative Analysis (curvature)")
    print("=" * 70)
    print(f"Max |d²y/dx²|: {np.max(np.abs(d2y)):.4f}")
    print(f"At boundaries: {d2y[left_boundary[:-2]][:3]} ... {d2y[right_boundary[:-2]][-3:]}")

    # Warning if boundary behavior is problematic
    boundary_deriv_mean = (np.mean(np.abs(left_deriv)) + np.mean(np.abs(right_deriv))) / 2
    middle_deriv_mean = np.mean(np.abs(middle_deriv))

    print(f"\n" + "=" * 70)
    if boundary_deriv_mean > 2 * middle_deriv_mean:
        print("⚠ WARNING: Boundary derivatives are much larger than middle!")
        print("⚠ This indicates potential sharp drops at boundaries.")
    else:
        print("✓ Boundary behavior appears reasonable")
    print("=" * 70)

    # Output data for plotting
    print(f"\nSample predictions for visualization:")
    for i in [0, 10, 50, 100, 150, 190, 199]:
        print(f"  x={x_test[i]:.3f}: pred={y_pred[i]:.4f}, true={y_true[i]:.4f}, diff={y_pred[i]-y_true[i]:.4f}")

def test_constant_extrapolation():
    """Test with a simple linear function to see extrapolation behavior"""
    print("\n\n" + "=" * 70)
    print("Testing Extrapolation with Linear Function")
    print("=" * 70)

    np.random.seed(42)

    # Training: linear in middle
    n_train = 50
    x_train = np.linspace(0.3, 0.7, n_train)
    y_train = 2 * x_train + 1 + 0.1 * np.random.randn(n_train)

    # Test: wider range
    x_test = np.linspace(0.0, 1.0, 100)
    y_true = 2 * x_test + 1

    gam = mgcv_rust.GAM()
    result = gam.fit_auto(x_train.reshape(-1, 1), y_train, k=[10], method='REML')
    y_pred = gam.predict(x_test.reshape(-1, 1))

    print(f"\nLinear function: y = 2x + 1")
    print(f"Training range: [{x_train.min():.2f}, {x_train.max():.2f}]")

    print(f"\nExtrapolation errors:")
    print(f"  At x=0.0:  pred={y_pred[0]:.4f}, true={y_true[0]:.4f}, error={np.abs(y_pred[0]-y_true[0]):.4f}")
    print(f"  At x=0.3:  pred={y_pred[30]:.4f}, true={y_true[30]:.4f}, error={np.abs(y_pred[30]-y_true[30]):.4f}")
    print(f"  At x=0.7:  pred={y_pred[70]:.4f}, true={y_true[70]:.4f}, error={np.abs(y_pred[70]-y_true[70]):.4f}")
    print(f"  At x=1.0:  pred={y_pred[99]:.4f}, true={y_true[99]:.4f}, error={np.abs(y_pred[99]-y_true[99]):.4f}")

    # Check if extrapolation is reasonable (should continue linearly)
    left_slope = (y_pred[30] - y_pred[0]) / 0.3
    right_slope = (y_pred[99] - y_pred[70]) / 0.3
    true_slope = 2.0

    print(f"\nSlope analysis:")
    print(f"  True slope:          {true_slope:.4f}")
    print(f"  Left extrapolation:  {left_slope:.4f}")
    print(f"  Right extrapolation: {right_slope:.4f}")

    if abs(left_slope - true_slope) > 0.5 or abs(right_slope - true_slope) > 0.5:
        print(f"\n⚠ WARNING: Extrapolation slopes differ significantly from true slope!")
        print(f"⚠ This indicates boundary constraint issues.")
    else:
        print(f"\n✓ Extrapolation appears reasonable")

if __name__ == "__main__":
    test_boundary_extrapolation()
    test_constant_extrapolation()
