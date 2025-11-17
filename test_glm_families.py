#!/usr/bin/env python3
"""
Test GAM with different GLM families (distributions).

Tests all supported families:
- Gaussian (identity link) - continuous data
- Binomial (logit link) - binary/proportion data
- Poisson (log link) - count data
- Gamma (inverse link) - positive continuous data
"""

import numpy as np
import sys

try:
    import mgcv_rust
    print("✓ mgcv_rust available")
except ImportError as e:
    print(f"✗ Error: mgcv_rust not available: {e}")
    sys.exit(1)


def test_gaussian():
    """Test Gaussian family (default)."""
    print("\n" + "="*70)
    print("Test 1: Gaussian Family (continuous data)")
    print("="*70)

    np.random.seed(42)
    n = 200
    X = np.random.uniform(0, 1, (n, 2))

    # True function: sine + quadratic
    y_true = np.sin(2 * np.pi * X[:, 0]) + (X[:, 1] - 0.5)**2
    y = y_true + np.random.normal(0, 0.2, n)

    # Fit with Gaussian family
    gam = mgcv_rust.GAM(family='gaussian')
    print(f"Family: {gam.get_family()}")

    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    pred = gam.predict(X)

    # Check fit quality
    corr = np.corrcoef(y, pred)[0, 1]
    rmse = np.sqrt(np.mean((y - pred)**2))

    print(f"Lambda values: {result['lambda']}")
    print(f"Deviance: {result['deviance']:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"RMSE: {rmse:.4f}")

    if corr > 0.95 and rmse < 0.3:
        print("✓ PASS: Gaussian family working correctly")
        return True
    else:
        print("✗ FAIL: Poor fit quality")
        return False


def test_binomial():
    """Test Binomial family (binary classification)."""
    print("\n" + "="*70)
    print("Test 2: Binomial Family (binary data)")
    print("="*70)

    np.random.seed(123)
    n = 300
    X = np.random.uniform(0, 1, (n, 2))

    # True probability function
    p_true = 1 / (1 + np.exp(-(2 * X[:, 0] - 1 + np.sin(3 * np.pi * X[:, 1]))))

    # Generate binary outcomes
    y = (np.random.random(n) < p_true).astype(float)

    print(f"Binary outcomes: {np.sum(y)}/{n} = {np.mean(y):.2f}")

    # Fit with Binomial family
    gam = mgcv_rust.GAM(family='binomial')
    print(f"Family: {gam.get_family()}")

    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    pred = gam.predict(X)

    # Predictions should be probabilities [0, 1]
    print(f"\nPrediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Lambda values: {result['lambda']}")
    print(f"Deviance: {result['deviance']:.4f}")

    # Check classification accuracy
    pred_class = (pred > 0.5).astype(float)
    accuracy = np.mean(pred_class == y)

    # Check AUC-like metric
    from scipy import stats
    corr_spearman = stats.spearmanr(p_true, pred)[0]

    print(f"Classification accuracy: {accuracy:.4f}")
    print(f"Correlation with true prob: {corr_spearman:.4f}")

    if 0 <= pred.min() <= 1 and 0 <= pred.max() <= 1 and corr_spearman > 0.6:
        print("✓ PASS: Binomial family working correctly")
        return True
    else:
        print("✗ FAIL: Issues with binomial predictions")
        return False


def test_poisson():
    """Test Poisson family (count data)."""
    print("\n" + "="*70)
    print("Test 3: Poisson Family (count data)")
    print("="*70)

    np.random.seed(456)
    n = 250
    X = np.random.uniform(0, 1, (n, 2))

    # True rate function (must be positive)
    lambda_true = np.exp(1 + 2 * X[:, 0] + np.sin(4 * np.pi * X[:, 1]))

    # Generate Poisson counts (need float for GAM)
    y = np.random.poisson(lambda_true).astype(np.float64)

    print(f"Count statistics: mean={np.mean(y):.2f}, max={np.max(y)}, min={np.min(y)}")

    # Fit with Poisson family
    gam = mgcv_rust.GAM(family='poisson')
    print(f"Family: {gam.get_family()}")

    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    pred = gam.predict(X)

    print(f"\nPrediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Lambda values: {result['lambda']}")
    print(f"Deviance: {result['deviance']:.4f}")

    # Check predictions are positive
    all_positive = np.all(pred > 0)

    # Check correlation with true rates
    corr = np.corrcoef(lambda_true, pred)[0, 1]

    # Relative error
    rel_error = np.mean(np.abs(pred - y) / (y + 1))

    print(f"All predictions positive: {all_positive}")
    print(f"Correlation with true rate: {corr:.4f}")
    print(f"Mean relative error: {rel_error:.4f}")

    if all_positive and corr > 0.7:
        print("✓ PASS: Poisson family working correctly")
        return True
    else:
        print("✗ FAIL: Issues with Poisson predictions")
        return False


def test_gamma():
    """Test Gamma family (positive continuous data)."""
    print("\n" + "="*70)
    print("Test 4: Gamma Family (positive continuous data)")
    print("="*70)

    np.random.seed(789)
    n = 200
    X = np.random.uniform(0, 1, (n, 2))

    # True mean function (must be positive)
    mu_true = np.exp(0.5 + X[:, 0] + 0.5 * np.sin(3 * np.pi * X[:, 1]))

    # Generate Gamma data (shape parameter = 2 for moderate variance)
    shape = 2.0
    y = np.random.gamma(shape, mu_true / shape).astype(np.float64)

    print(f"Data statistics: mean={np.mean(y):.2f}, max={np.max(y):.2f}, min={np.min(y):.2f}")

    # Fit with Gamma family
    gam = mgcv_rust.GAM(family='gamma')
    print(f"Family: {gam.get_family()}")

    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='cr')
    pred = gam.predict(X)

    print(f"\nPrediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Lambda values: {result['lambda']}")
    print(f"Deviance: {result['deviance']:.4f}")

    # Check predictions are positive
    all_positive = np.all(pred > 0)

    # Check correlation with true mean
    corr = np.corrcoef(mu_true, pred)[0, 1]

    # Relative error (Gamma has variance proportional to mean^2)
    rel_error = np.mean(np.abs(pred - y) / y)

    print(f"All predictions positive: {all_positive}")
    print(f"Correlation with true mean: {corr:.4f}")
    print(f"Mean relative error: {rel_error:.4f}")

    if all_positive and corr > 0.7:
        print("✓ PASS: Gamma family working correctly")
        return True
    else:
        print("✗ FAIL: Issues with Gamma predictions")
        return False


def test_family_api():
    """Test family API and error handling."""
    print("\n" + "="*70)
    print("Test 5: Family API and Error Handling")
    print("="*70)

    # Test default (Gaussian)
    gam1 = mgcv_rust.GAM()
    print(f"Default family: {gam1.get_family()}")
    assert gam1.get_family() == 'gaussian', "Default should be gaussian"

    # Test explicit families
    for family in ['gaussian', 'binomial', 'poisson', 'gamma']:
        gam = mgcv_rust.GAM(family=family)
        assert gam.get_family() == family, f"Family mismatch for {family}"
        print(f"✓ {family}: {gam.get_family()}")

    # Test invalid family
    try:
        gam = mgcv_rust.GAM(family='invalid')
        print("✗ FAIL: Should have raised error for invalid family")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid family: {e}")

    print("✓ PASS: Family API working correctly")
    return True


def main():
    """Run all GLM family tests."""
    print("="*70)
    print("GLM Family Test Suite")
    print("="*70)
    print("\nTesting all supported distributions and link functions:")
    print("  - Gaussian (identity link)")
    print("  - Binomial (logit link)")
    print("  - Poisson (log link)")
    print("  - Gamma (inverse link)")

    results = {}

    # Run all tests
    results['gaussian'] = test_gaussian()
    results['binomial'] = test_binomial()
    results['poisson'] = test_poisson()
    results['gamma'] = test_gamma()
    results['api'] = test_family_api()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.capitalize():20s}: {status}")

    all_passed = all(results.values())
    print("="*70)

    if all_passed:
        print("✓ ALL TESTS PASSED - All GLM families working!")
        return 0
    else:
        print("⚠ SOME TESTS FAILED - Check results above")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
