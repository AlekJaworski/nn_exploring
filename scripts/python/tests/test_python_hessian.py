#!/usr/bin/env python3
"""
Test the pure Python corrected Hessian implementation.

This shows that the fix works even without compiling Rust.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/nn_exploring')

from reml_hessian_corrected import reml_hessian_corrected, estimate_rank
from validate_hessian_numerical import reml_gradient_ift, numerical_hessian

def test_analytical_vs_numerical():
    """Compare analytical Hessian against numerical differentiation."""
    print("="*70)
    print("TEST: Analytical Hessian vs Numerical Differentiation")
    print("="*70)

    # Generate test data
    np.random.seed(42)
    n = 100
    p1, p2 = 10, 10
    p = p1 + p2

    X = np.random.randn(n, p)
    y = np.random.randn(n)
    w = np.ones(n)

    # Penalty matrices
    def second_diff_penalty(k):
        D = np.zeros((k-2, k))
        for i in range(k-2):
            D[i, i:i+3] = [1, -2, 1]
        return D.T @ D

    S1 = np.zeros((p, p))
    S1[:p1, :p1] = second_diff_penalty(p1)
    S2 = np.zeros((p, p))
    S2[p1:, p1:] = second_diff_penalty(p2)
    penalties = [S1, S2]

    lambdas = np.array([5.0, 3.0])

    print(f"\nComputing analytical Hessian...")
    H_analytical = reml_hessian_corrected(y, X, w, lambdas, penalties)

    print(f"\nComputing numerical Hessian (finite differences)...")
    H_numerical = numerical_hessian(y, X, w, lambdas, penalties, eps=1e-5)

    print(f"\nComparison:")
    print(f"  Analytical Hessian:\n{H_analytical}")
    print(f"  Numerical Hessian:\n{H_numerical}")

    diff = H_analytical - H_numerical
    max_diff = np.max(np.abs(diff))
    rel_error = max_diff / np.max(np.abs(H_numerical))

    print(f"\n  Max absolute difference: {max_diff:.6e}")
    print(f"  Relative error: {rel_error*100:.3f}%")

    if rel_error < 0.01:  # Less than 1%
        print(f"  ✓ PASS: Analytical matches numerical (<1% error)")
        return True
    else:
        print(f"  ✗ FAIL: Error too large (>{rel_error*100:.1f}%)")
        return False

def test_descent_direction():
    """Test that Hessian gives valid descent direction."""
    print("\n" + "="*70)
    print("TEST: Descent Direction Validation")
    print("="*70)

    # Generate test data
    np.random.seed(42)
    n = 100
    p1, p2 = 10, 10
    p = p1 + p2

    X = np.random.randn(n, p)
    y = np.random.randn(n)
    w = np.ones(n)

    def second_diff_penalty(k):
        D = np.zeros((k-2, k))
        for i in range(k-2):
            D[i, i:i+3] = [1, -2, 1]
        return D.T @ D

    S1 = np.zeros((p, p))
    S1[:p1, :p1] = second_diff_penalty(p1)
    S2 = np.zeros((p, p))
    S2[p1:, p1:] = second_diff_penalty(p2)
    penalties = [S1, S2]

    lambdas = np.array([5.0, 3.0])

    # Compute gradient and Hessian
    grad = reml_gradient_ift(y, X, w, lambdas, penalties)
    H = reml_hessian_corrected(y, X, w, lambdas, penalties)

    # Newton step: Δρ = -H⁻¹·g
    delta_rho = -np.linalg.solve(H, grad)

    # Check descent: g'·Δρ < 0
    descent_check = grad @ delta_rho

    print(f"\n  Gradient: {grad}")
    print(f"  Newton step: {delta_rho}")
    print(f"  Descent check (g'·Δρ): {descent_check:.6e}")

    if descent_check < 0:
        print(f"  ✓ PASS: Valid descent direction (g'·Δρ < 0)")
        return True
    else:
        print(f"  ✗ FAIL: NOT a descent direction (g'·Δρ > 0)")
        return False

def test_positive_definite():
    """Test that Hessian is positive definite."""
    print("\n" + "="*70)
    print("TEST: Positive Definiteness")
    print("="*70)

    # Generate test data
    np.random.seed(42)
    n = 100
    p1, p2 = 10, 10
    p = p1 + p2

    X = np.random.randn(n, p)
    y = np.random.randn(n)
    w = np.ones(n)

    def second_diff_penalty(k):
        D = np.zeros((k-2, k))
        for i in range(k-2):
            D[i, i:i+3] = [1, -2, 1]
        return D.T @ D

    S1 = np.zeros((p, p))
    S1[:p1, :p1] = second_diff_penalty(p1)
    S2 = np.zeros((p, p))
    S2[p1:, p1:] = second_diff_penalty(p2)
    penalties = [S1, S2]

    lambdas = np.array([5.0, 3.0])

    H = reml_hessian_corrected(y, X, w, lambdas, penalties)

    eigenvalues = np.linalg.eigvalsh(H)
    print(f"\n  Eigenvalues: {eigenvalues}")

    if np.all(eigenvalues > 0):
        print(f"  ✓ PASS: All eigenvalues positive (Hessian is PD)")
        return True
    else:
        print(f"  ✗ FAIL: Some eigenvalues non-positive")
        return False

def main():
    print("="*70)
    print("PURE PYTHON CORRECTED HESSIAN VALIDATION")
    print("="*70)
    print("\nThis tests the corrected Hessian formula in pure Python,")
    print("demonstrating the fix works even without Rust compilation.\n")

    results = {
        'Analytical vs Numerical': test_analytical_vs_numerical(),
        'Descent Direction': test_descent_direction(),
        'Positive Definite': test_positive_definite(),
    }

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:30s} {status}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("The corrected Hessian formula is working correctly!")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*70)

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
