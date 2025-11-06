#!/usr/bin/env python3
"""
Simple 1-variable GAM test
Tests fitting, prediction, and extrapolation
"""

import numpy as np
import mgcv_rust

def main():
    print("="*70)
    print("1-Variable GAM Test")
    print("="*70)

    # Generate training data: sine wave in [0, 1]
    np.random.seed(42)
    n_train = 100
    x_train = np.linspace(0, 1, n_train)
    y_true = np.sin(2 * np.pi * x_train)
    noise = 0.2 * np.random.randn(n_train)
    y_train = y_true + noise

    print(f"\nTraining data:")
    print(f"  Function: y = sin(2πx) + noise")
    print(f"  Range: x ∈ [0, 1]")
    print(f"  N = {n_train} observations")
    print(f"  Noise: σ = 0.2")

    # Fit GAM
    print(f"\n{'='*70}")
    print("Fitting GAM")
    print("="*70)

    X_train = x_train.reshape(-1, 1)
    gam = mgcv_rust.GAM()

    result = gam.fit_auto(X_train, y_train, k=[10], method='REML')

    print(f"\n✓ Fit successful!")
    print(f"  λ = {result['lambda']:.6f}")
    print(f"  Deviance = {result['deviance']:.6f}")

    # Predictions on training data
    y_pred_train = gam.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_pred_train - y_true)**2))
    train_r2 = 1 - np.var(y_pred_train - y_true) / np.var(y_true)

    print(f"\nTraining set performance:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    # Test predictions within range
    print(f"\n{'='*70}")
    print("Predictions Within Training Range")
    print("="*70)

    x_test_in = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    X_test_in = x_test_in.reshape(-1, 1)
    y_pred_in = gam.predict(X_test_in)
    y_true_in = np.sin(2 * np.pi * x_test_in)

    print(f"\n{'x':>6s} {'True':>8s} {'Pred':>8s} {'Error':>8s}")
    print("-"*35)
    for i, x in enumerate(x_test_in):
        error = abs(y_pred_in[i] - y_true_in[i])
        print(f"{x:6.2f} {y_true_in[i]:8.4f} {y_pred_in[i]:8.4f} {error:8.4f}")

    # Test extrapolation beyond training range
    print(f"\n{'='*70}")
    print("Extrapolation Test (Beyond Training Range)")
    print("="*70)

    x_test_out = np.array([-0.2, -0.1, 1.1, 1.2])
    X_test_out = x_test_out.reshape(-1, 1)
    y_pred_out = gam.predict(X_test_out)
    y_true_out = np.sin(2 * np.pi * x_test_out)

    print(f"\n{'x':>6s} {'True':>8s} {'Pred':>8s} {'Zero?':>8s}")
    print("-"*35)
    all_good = True
    for i, x in enumerate(x_test_out):
        is_zero = abs(y_pred_out[i]) < 1e-6
        status = "✗ ZERO" if is_zero else "✓ OK"
        print(f"{x:6.2f} {y_true_out[i]:8.4f} {y_pred_out[i]:8.4f} {status:>8s}")
        if is_zero:
            all_good = False

    if all_good:
        print(f"\n✓ PASS: Extrapolation working (no zeros)")
    else:
        print(f"\n✗ FAIL: Getting zero predictions outside range")

    # Full range visualization
    print(f"\n{'='*70}")
    print("Full Range Predictions")
    print("="*70)

    x_full = np.linspace(-0.2, 1.2, 100)
    X_full = x_full.reshape(-1, 1)
    y_pred_full = gam.predict(X_full)
    y_true_full = np.sin(2 * np.pi * x_full)

    # Check for issues
    has_zeros = np.any(np.abs(y_pred_full) < 1e-6)
    has_nans = np.any(np.isnan(y_pred_full))

    print(f"\nSample predictions across full range:")
    sample_idx = [0, 10, 30, 50, 70, 90, 99]
    for idx in sample_idx:
        x = x_full[idx]
        region = "LEFT" if x < 0 else ("RIGHT" if x > 1 else "IN")
        print(f"  x={x:5.2f} ({region:>5s}): pred={y_pred_full[idx]:7.4f}, true={y_true_full[idx]:7.4f}")

    print(f"\nDiagnostics:")
    print(f"  Contains zeros: {has_zeros}")
    print(f"  Contains NaNs:  {has_nans}")
    print(f"  Min prediction: {np.min(y_pred_full):.4f}")
    print(f"  Max prediction: {np.max(y_pred_full):.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)

    checks = {
        "Fit succeeded": True,
        "R² > 0.8": train_r2 > 0.8,
        "No zero predictions": not has_zeros,
        "No NaN predictions": not has_nans,
        "Extrapolation works": all_good
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    all_passed = all(checks.values())
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠ SOME TESTS FAILED")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
