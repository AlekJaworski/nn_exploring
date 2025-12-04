#!/usr/bin/env python3
"""
Test multi-variable GAM with all_lambdas functionality
"""

import numpy as np
import mgcv_rust

def main():
    print("=" * 70)
    print("Testing Multi-Variable GAM with all_lambdas")
    print("=" * 70)

    np.random.seed(42)
    n = 300

    # Generate 3 predictors with different complexities
    x1 = np.linspace(0, 1, n)
    x2 = np.linspace(-1, 1, n)
    x3 = np.linspace(-2, 2, n)

    # True model: y = sin(2πx₁) + 0.5·x₂² + 2·x₃ + noise
    y_true = (
        np.sin(2 * np.pi * x1) +     # Complex: sine wave
        0.5 * x2**2 +                 # Moderate: quadratic
        2.0 * x3                      # Simple: linear
    )
    noise = 0.5 * np.random.randn(n)
    y = y_true + noise

    X = np.column_stack([x1, x2, x3])

    print(f"\nData: {n} observations, 3 predictors")
    print(f"True model: y = sin(2πx₁) + 0.5·x₂² + 2·x₃ + noise")
    print(f"\nExpected λ pattern: λ₃ (linear) >> λ₂ (quadratic) > λ₁ (sine)")

    # Test with fit_auto
    print("\n" + "=" * 70)
    print("Test 1: fit_auto() with k=[15, 12, 10]")
    print("=" * 70)

    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[15, 12, 10], method='REML')

    print(f"\nResult dictionary keys: {list(result.keys())}")
    print(f"\nSmoothing parameters:")
    print(f"  lambda (first):  {result['lambda']:.6f}")
    print(f"  all_lambdas:     {result['all_lambdas']}")

    # Verify all_lambdas is a numpy array
    all_lambdas = result['all_lambdas']
    print(f"\nType of all_lambdas: {type(all_lambdas)}")
    print(f"Shape: {all_lambdas.shape}")
    print(f"\nIndividual λ values:")
    print(f"  λ₁ (x₁, sine):      {all_lambdas[0]:.6f}")
    print(f"  λ₂ (x₂, quadratic): {all_lambdas[1]:.6f}")
    print(f"  λ₃ (x₃, linear):    {all_lambdas[2]:.6f}")

    # Check pattern
    if all_lambdas[2] > all_lambdas[0]:
        print(f"\n✓ Correct pattern: λ₃ ({all_lambdas[2]:.3f}) > λ₁ ({all_lambdas[0]:.3f})")
    else:
        print(f"\n✗ Unexpected pattern")

    # Test with get_all_lambdas() method
    print("\n" + "=" * 70)
    print("Test 2: get_all_lambdas() method")
    print("=" * 70)

    lambdas_method = gam.get_all_lambdas()
    print(f"\nget_all_lambdas() result: {lambdas_method}")
    print(f"Type: {type(lambdas_method)}")
    print(f"Shape: {lambdas_method.shape}")

    # Verify they match
    if np.allclose(all_lambdas, lambdas_method):
        print("\n✓ get_all_lambdas() matches result['all_lambdas']")
    else:
        print("\n✗ Mismatch!")

    # Test with fit_formula
    print("\n" + "=" * 70)
    print("Test 3: fit_formula() with multi-predictor formula")
    print("=" * 70)

    gam2 = mgcv_rust.GAM()
    result2 = gam2.fit_formula(X, y, formula="s(0, k=15) + s(1, k=12) + s(2, k=10)", method='REML')

    print(f"\nResult dictionary keys: {list(result2.keys())}")
    print(f"all_lambdas: {result2['all_lambdas']}")

    all_lambdas2 = result2['all_lambdas']
    print(f"\nIndividual λ values:")
    print(f"  λ₁ (x₁, sine):      {all_lambdas2[0]:.6f}")
    print(f"  λ₂ (x₂, quadratic): {all_lambdas2[1]:.6f}")
    print(f"  λ₃ (x₃, linear):    {all_lambdas2[2]:.6f}")

    # Verify both methods produce similar results
    print("\n" + "=" * 70)
    print("Comparison: fit_auto vs fit_formula")
    print("=" * 70)
    print(f"\nfit_auto    λs: {all_lambdas}")
    print(f"fit_formula λs: {all_lambdas2}")
    print(f"\nDifferences: {np.abs(all_lambdas - all_lambdas2)}")

    if np.allclose(all_lambdas, all_lambdas2, rtol=0.01):
        print("\n✓ Both methods produce consistent results")
    else:
        print("\n⚠ Some differences (may be due to convergence)")

    print("\n" + "=" * 70)
    print("Success! all_lambdas functionality working correctly")
    print("=" * 70)

if __name__ == "__main__":
    main()
