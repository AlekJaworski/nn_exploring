#!/usr/bin/env python3
"""
Test the new simplified fit() API
"""

import numpy as np
from mgcv_rust import GAM

def test_simple_fit():
    """Test the new simple fit() API"""
    print("Testing simplified fit() API...")

    # Generate test data: sin(2πx) with noise
    np.random.seed(42)
    n = 500
    d = 2
    X = np.random.uniform(0, 1, (n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] - 0.5)**2 + np.random.normal(0, 0.3, n)

    # Test 1: Simplest possible usage (most common case)
    print("\n1. Simplest usage: gam.fit(X, y, k=[10, 10])")
    gam = GAM()
    result = gam.fit(X, y, k=[10, 10])

    print(f"   ✓ Fitted successfully")
    print(f"   Lambda values: {result['lambda']}")
    print(f"   Deviance: {result['deviance']:.3f}")

    # Test 2: With optional parameters
    print("\n2. With optional parameters: bs='bs'")
    gam2 = GAM()
    result2 = gam2.fit(X, y, k=[10, 10], bs='bs')

    print(f"   ✓ Fitted successfully")
    print(f"   Lambda values: {result2['lambda']}")
    print(f"   Deviance: {result2['deviance']:.3f}")

    # Test 3: Single-dimensional case
    print("\n3. Single dimension: X.shape = (n, 1)")
    X_1d = X[:, 0:1]
    y_1d = np.sin(2 * np.pi * X_1d[:, 0]) + np.random.normal(0, 0.3, n)

    gam3 = GAM()
    result3 = gam3.fit(X_1d, y_1d, k=[15])

    print(f"   ✓ Fitted successfully")
    print(f"   Lambda: {result3['lambda'][0]:.3f}")
    print(f"   Deviance: {result3['deviance']:.3f}")

    # Test 4: Prediction
    print("\n4. Prediction with fitted model")
    X_test = np.random.uniform(0, 1, (10, d))
    predictions = gam.predict(X_test)

    print(f"   ✓ Predicted successfully")
    print(f"   First 3 predictions: {predictions[:3]}")

    print("\n✅ All tests passed!")
    print("The new simplified API works correctly!")
    return True

def compare_with_old_api():
    """Compare new fit() with fit_auto_optimized() to ensure they're the same"""
    print("\nComparing new fit() with fit_auto_optimized()...")

    np.random.seed(123)
    n = 200
    X = np.random.uniform(0, 1, (n, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + np.random.normal(0, 0.2, n)

    # New API
    gam1 = GAM()
    result1 = gam1.fit(X, y, k=[10, 10])

    # Old API (should give same results)
    gam2 = GAM()
    result2 = gam2.fit_auto_optimized(X, y, k=[10, 10], method='REML')

    # Compare
    lambda_diff = np.max(np.abs(result1['lambda'] - result2['lambda']))
    dev_diff = abs(result1['deviance'] - result2['deviance'])

    print(f"   Lambda difference: {lambda_diff:.2e}")
    print(f"   Deviance difference: {dev_diff:.2e}")

    if lambda_diff < 1e-10 and dev_diff < 1e-10:
        print("   ✅ Results are identical!")
        return True
    else:
        print("   ⚠️  Results differ!")
        return False

if __name__ == "__main__":
    test_simple_fit()
    compare_with_old_api()
