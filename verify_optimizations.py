#!/usr/bin/env python3
"""
Comprehensive verification script for gradient optimizations
Tests correctness, correlations, and numerical accuracy
"""

import numpy as np
import mgcv_rust
import time

def generate_test_data(n=1000, d=4, k=10, seed=42):
    """Generate test data for GAM"""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, d))

    # True smooth effects
    y_true = np.sin(2 * np.pi * X[:, 0])

    if d > 1:
        y_true += (X[:, 1] - 0.5) ** 2

    if d > 2:
        y_true += X[:, 2]

    if d > 3:
        y_true += np.cos(3 * np.pi * X[:, 3])

    # Add remaining dimensions with smaller effects
    for i in range(4, d):
        y_true += 0.1 * np.sin(2 * np.pi * X[:, i])

    y = y_true + np.random.normal(0, 0.3, n)
    return X, y, y_true


def test_gradient_consistency():
    """Test that different gradient implementations give same results"""
    print("="*60)
    print("TEST 1: Gradient Method Consistency")
    print("="*60)

    n, d, k = 500, 4, 10
    X, y, _ = generate_test_data(n, d, k)

    # Test with simple GAM fitting
    gam1 = mgcv_rust.GAM()
    gam2 = mgcv_rust.GAM()

    k_list = [k] * d

    print(f"\nConfiguration: n={n}, dims={d}, k={k}")
    print("\nFitting with standard method...")
    gam1.fit_auto(X, y, k=k_list, method='REML', bs='cr')

    print("Fitting with optimized method...")
    gam2.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')

    # Get predictions
    pred1 = gam1.predict(X)
    pred2 = gam2.predict(X)

    # Check correlation
    corr = np.corrcoef(pred1, pred2)[0, 1]
    rmse_diff = np.sqrt(np.mean((pred1 - pred2)**2))
    max_diff = np.max(np.abs(pred1 - pred2))

    print(f"\nPrediction comparison:")
    print(f"  Correlation:    {corr:.10f}")
    print(f"  RMSE diff:      {rmse_diff:.2e}")
    print(f"  Max diff:       {max_diff:.2e}")

    # Pass/fail (based on predictions)
    passed = (corr > 0.9999 and rmse_diff < 1e-6)

    if passed:
        print("\n✓ PASSED: Methods produce consistent results")
    else:
        print("\n✗ FAILED: Methods diverge!")

    return passed


def test_prediction_accuracy():
    """Test prediction accuracy on known smooth functions"""
    print("\n" + "="*60)
    print("TEST 2: Prediction Accuracy")
    print("="*60)

    n, d, k = 1000, 4, 12
    X, y, y_true = generate_test_data(n, d, k)

    gam = mgcv_rust.GAM()
    k_list = [k] * d

    print(f"\nConfiguration: n={n}, dims={d}, k={k}")
    print("Fitting GAM...")

    start = time.time()
    gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')
    fit_time = time.time() - start

    pred = gam.predict(X)

    # Calculate R²
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot

    # Calculate R² against true function
    ss_res_true = np.sum((y_true - pred)**2)
    ss_tot_true = np.sum((y_true - np.mean(y_true))**2)
    r2_true = 1 - ss_res_true / ss_tot_true

    print(f"\nResults:")
    print(f"  Fit time:        {fit_time*1000:.2f} ms")
    print(f"  R² (vs noisy):   {r2:.4f}")
    print(f"  R² (vs true):    {r2_true:.4f}")
    print(f"  RMSE (vs noisy): {np.sqrt(np.mean((y - pred)**2)):.4f}")
    print(f"  RMSE (vs true):  {np.sqrt(np.mean((y_true - pred)**2)):.4f}")

    passed = (r2 > 0.8 and r2_true > 0.9)

    if passed:
        print("\n✓ PASSED: Good prediction accuracy")
    else:
        print("\n✗ FAILED: Poor prediction accuracy")

    return passed


def test_multiple_problem_sizes():
    """Test consistency across different problem sizes"""
    print("\n" + "="*60)
    print("TEST 3: Consistency Across Problem Sizes")
    print("="*60)

    configs = [
        (200, 2, 8),
        (500, 4, 10),
        (1000, 4, 12),
        (2000, 6, 10),
    ]

    results = []

    for n, d, k in configs:
        X, y, _ = generate_test_data(n, d, k, seed=42+n)

        gam1 = mgcv_rust.GAM()
        gam2 = mgcv_rust.GAM()

        k_list = [k] * d

        print(f"\nTesting n={n}, dims={d}, k={k}...")

        # Time both methods
        start = time.time()
        gam1.fit_auto(X, y, k=k_list, method='REML', bs='cr')
        time1 = time.time() - start

        start = time.time()
        gam2.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
        time2 = time.time() - start

        pred1 = gam1.predict(X)
        pred2 = gam2.predict(X)

        corr = np.corrcoef(pred1, pred2)[0, 1]
        diff = np.max(np.abs(pred1 - pred2))

        print(f"  Standard:  {time1*1000:6.2f} ms")
        print(f"  Optimized: {time2*1000:6.2f} ms")
        print(f"  Correlation: {corr:.10f}")
        print(f"  Max diff:    {diff:.2e}")

        results.append({
            'n': n, 'd': d, 'k': k,
            'corr': corr,
            'diff': diff,
            'time1': time1,
            'time2': time2
        })

    # Check all passed
    all_passed = all(r['corr'] > 0.9999 and r['diff'] < 1e-5 for r in results)

    if all_passed:
        print("\n✓ PASSED: Consistent results across all problem sizes")
    else:
        print("\n✗ FAILED: Inconsistencies found")

    return all_passed


def benchmark_performance():
    """Benchmark performance"""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)

    n, d, k = 1000, 4, 12
    X, y, _ = generate_test_data(n, d, k)

    k_list = [k] * d
    n_iter = 20

    print(f"\nConfiguration: n={n}, dims={d}, k={k}")
    print(f"Iterations: {n_iter}\n")

    # Benchmark standard
    times1 = []
    for i in range(n_iter):
        gam = mgcv_rust.GAM()
        start = time.time()
        gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')
        times1.append(time.time() - start)
        if (i + 1) % 5 == 0:
            print(f"  Standard: {i+1}/{n_iter}...")

    # Benchmark optimized
    times2 = []
    for i in range(n_iter):
        gam = mgcv_rust.GAM()
        start = time.time()
        gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
        times2.append(time.time() - start)
        if (i + 1) % 5 == 0:
            print(f"  Optimized: {i+1}/{n_iter}...")

    times1 = np.array(times1) * 1000
    times2 = np.array(times2) * 1000

    print(f"\nStandard method:")
    print(f"  Mean: {np.mean(times1):.2f} ms")
    print(f"  Std:  {np.std(times1):.2f} ms")

    print(f"\nOptimized method:")
    print(f"  Mean: {np.mean(times2):.2f} ms")
    print(f"  Std:  {np.std(times2):.2f} ms")

    speedup = np.mean(times1) / np.mean(times2)
    print(f"\nSpeedup: {speedup:.2f}x")

    if abs(speedup - 1.0) < 0.2:
        print("(Similar performance is expected - optimizations are in gradient computation)")

    return True


def main():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("GAM OPTIMIZATION VERIFICATION SUITE")
    print("="*60)
    print()

    tests = [
        ("Gradient Consistency", test_gradient_consistency),
        ("Prediction Accuracy", test_prediction_accuracy),
        ("Multi-Size Consistency", test_multiple_problem_sizes),
        ("Performance Benchmark", benchmark_performance),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
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
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
