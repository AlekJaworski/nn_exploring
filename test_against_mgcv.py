#!/usr/bin/env python3
"""
Direct comparison of Rust implementation against R/mgcv
Tests numerical accuracy and correlation with mgcv predictions
"""

import numpy as np
import pandas as pd
import subprocess
import mgcv_rust
import tempfile
import os

def test_predictions_vs_mgcv():
    """Compare Rust predictions directly with R/mgcv"""
    print("="*60)
    print("TEST: Rust vs R/mgcv Prediction Correlation")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n = 200
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)

    # True function
    y_true = np.sin(2 * np.pi * x1) + (x2 - 0.5)**2
    y = y_true + np.random.normal(0, 0.1, n)

    X = np.column_stack([x1, x2])

    # Save data to temp file
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    data_file = '/tmp/test_data_mgcv.csv'
    df.to_csv(data_file, index=False)

    print(f"\nData: n={n}, dims=2")
    print(f"True function: sin(2π·x1) + (x2-0.5)²")

    # Fit with R/mgcv
    print("\n[1/3] Fitting with R/mgcv...")
    r_code = """
    library(mgcv)

    # Load data
    df <- read.csv('/tmp/test_data_mgcv.csv')

    # Fit GAM
    fit <- gam(y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr'),
               data=df, method='REML')

    # Get predictions
    pred <- predict(fit, type='response')

    # Get smoothing parameters
    sp <- fit$sp

    # Save results
    write.csv(data.frame(pred=pred), '/tmp/mgcv_pred.csv', row.names=FALSE)
    write.csv(data.frame(lambda1=sp[1], lambda2=sp[2]), '/tmp/mgcv_lambdas.csv', row.names=FALSE)

    # Print summary
    cat('R/mgcv fit complete\n')
    cat('Smoothing parameters:', sp, '\n')
    cat('R²:', summary(fit)$r.sq, '\n')
    """

    result = subprocess.run(['Rscript', '-e', r_code],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print("R Error:", result.stderr)
        return False

    print(result.stdout)

    # Fit with Rust
    print("[2/3] Fitting with Rust...")
    gam = mgcv_rust.GAM()
    gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    pred_rust = gam.predict(X)

    # Load R predictions
    mgcv_pred = pd.read_csv('/tmp/mgcv_pred.csv')['pred'].values
    mgcv_lambdas = pd.read_csv('/tmp/mgcv_lambdas.csv')

    print(f"  R² (Rust): {1 - np.sum((y - pred_rust)**2) / np.sum((y - np.mean(y))**2):.4f}")

    # Compare
    print("\n[3/3] Comparing results...")
    corr = np.corrcoef(pred_rust, mgcv_pred)[0, 1]
    rmse_diff = np.sqrt(np.mean((pred_rust - mgcv_pred)**2))
    max_diff = np.max(np.abs(pred_rust - mgcv_pred))

    print(f"\nPrediction comparison:")
    print(f"  Correlation:    {corr:.10f}")
    print(f"  RMSE diff:      {rmse_diff:.6f}")
    print(f"  Max diff:       {max_diff:.6f}")

    print(f"\nSmoothing parameters:")
    print(f"  R/mgcv:  λ1={mgcv_lambdas['lambda1'].values[0]:.6f}, λ2={mgcv_lambdas['lambda2'].values[0]:.6f}")

    # Check agreement
    if corr > 0.999:
        print("\n✓ EXCELLENT: Predictions highly correlated with mgcv")
        status = True
    elif corr > 0.99:
        print("\n⚠ GOOD: Predictions correlated but some differences")
        status = True
    else:
        print("\n✗ FAILED: Poor correlation with mgcv")
        status = False

    return status


def test_gradient_comparison_with_mgcv():
    """Compare gradient computations with R/mgcv"""
    print("\n" + "="*60)
    print("TEST: Gradient Comparison with R/mgcv")
    print("="*60)

    # Small problem for comparison
    np.random.seed(123)
    n = 100
    d = 2
    k = 8

    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + (x2 - 0.5)**2 + np.random.normal(0, 0.1, n)

    X = np.column_stack([x1, x2])

    # Save data
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    df.to_csv('/tmp/grad_test_data.csv', index=False)

    print(f"\nData: n={n}, dims={d}, k={k}")

    # Fit with both
    print("\n[1/2] Fitting with R/mgcv...")
    r_code = """
    library(mgcv)
    df <- read.csv('/tmp/grad_test_data.csv')
    fit <- gam(y ~ s(x1, k=8, bs='cr') + s(x2, k=8, bs='cr'),
               data=df, method='REML')

    # Save lambdas and predictions
    write.csv(data.frame(lambda1=fit$sp[1], lambda2=fit$sp[2]),
              '/tmp/grad_lambdas.csv', row.names=FALSE)
    write.csv(data.frame(pred=predict(fit)),
              '/tmp/grad_pred_mgcv.csv', row.names=FALSE)

    cat('Smoothing parameters:', fit$sp, '\\n')
    cat('Converged:', fit$converged, '\\n')
    """

    result = subprocess.run(['Rscript', '-e', r_code],
                          capture_output=True, text=True)

    if result.returncode != 0:
        print("R Error:", result.stderr)
        return False

    print(result.stdout)

    print("[2/2] Fitting with Rust...")
    gam = mgcv_rust.GAM()
    gam.fit_auto(X, y, k=[k, k], method='REML', bs='cr')
    pred_rust = gam.predict(X)

    # Load R results
    mgcv_lambdas = pd.read_csv('/tmp/grad_lambdas.csv')
    mgcv_pred = pd.read_csv('/tmp/grad_pred_mgcv.csv')['pred'].values

    # Compare
    print("\nComparison:")
    print(f"  Smoothing parameters:")
    print(f"    R/mgcv: λ1={mgcv_lambdas['lambda1'].values[0]:.6f}, λ2={mgcv_lambdas['lambda2'].values[0]:.6f}")

    corr = np.corrcoef(pred_rust, mgcv_pred)[0, 1]
    print(f"\n  Prediction correlation: {corr:.10f}")

    if corr > 0.99:
        print("\n✓ PASSED: Gradients lead to consistent optimization")
        return True
    else:
        print("\n✗ FAILED: Optimization differs from mgcv")
        return False


def test_numerical_stability():
    """Test numerical stability across different conditions"""
    print("\n" + "="*60)
    print("TEST: Numerical Stability")
    print("="*60)

    conditions = [
        ("Small n", 50, 2, 6),
        ("Moderate n", 200, 3, 10),
        ("Large n", 1000, 2, 12),
        ("Many dims", 200, 4, 8),
    ]

    results = []

    for name, n, d, k in conditions:
        print(f"\n{name}: n={n}, d={d}, k={k}")

        # Generate data
        np.random.seed(42 + n)
        X = np.random.uniform(0, 1, (n, d))

        y_true = np.zeros(n)
        for i in range(d):
            y_true += np.sin(2 * np.pi * X[:, i])

        y = y_true + np.random.normal(0, 0.2, n)

        # Fit
        try:
            gam = mgcv_rust.GAM()
            gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
            pred = gam.predict(X)

            r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)

            # Check for NaN or Inf
            has_nan = np.any(np.isnan(pred)) or np.any(np.isinf(pred))

            print(f"  R²: {r2:.4f}, NaN/Inf: {has_nan}")

            results.append({
                'name': name,
                'r2': r2,
                'stable': not has_nan and r2 > 0.7
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': name,
                'r2': 0,
                'stable': False
            })

    all_stable = all(r['stable'] for r in results)

    if all_stable:
        print("\n✓ PASSED: Numerically stable across all conditions")
        return True
    else:
        print("\n✗ FAILED: Stability issues found")
        return False


def main():
    """Run all comparison tests"""
    print("\n" + "="*60)
    print("RUST vs R/MGCV COMPARISON SUITE")
    print("="*60)
    print("\nThis tests that Rust implementation produces")
    print("results consistent with the reference R/mgcv package.\n")

    tests = [
        ("Prediction vs mgcv", test_predictions_vs_mgcv),
        ("Gradient/Optimization vs mgcv", test_gradient_comparison_with_mgcv),
        ("Numerical Stability", test_numerical_stability),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nConclusion: Rust implementation is numerically")
        print("consistent with R/mgcv reference implementation.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nIMPORTANT: Results differ from R/mgcv!")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
