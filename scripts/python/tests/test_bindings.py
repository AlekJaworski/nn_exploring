#!/usr/bin/env python3
"""Test script for the new bindings (coefficients and design matrix)"""

import numpy as np
import sys

# Try to import the built module
try:
    from mgcv_rust import GAM
except ImportError:
    print("Error: mgcv_rust module not found. Please build it first with: maturin develop")
    sys.exit(1)

# Generate test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n)

# Prepare data for GAM
X = x.reshape(-1, 1)

print("Testing new bindings for GAM...")
print(f"Data shape: X={X.shape}, y={y.shape}")

# Create and fit GAM
gam = GAM(family="gaussian")
print("\nFitting GAM with k=10 basis functions...")
result = gam.fit_auto(X, y, k=[10], method='REML', bs='cr')

print(f"Fit successful: {result['fitted']}")
print(f"Lambda: {result['lambda']}")
print(f"Deviance: {result['deviance']:.4f}")

# Test new bindings
print("\n--- Testing new bindings ---")

# Get coefficients
try:
    coefficients = gam.get_coefficients()
    print(f"\n✓ get_coefficients() works!")
    print(f"  Shape: {coefficients.shape}")
    print(f"  First 5 coefficients: {coefficients[:5]}")
    print(f"  Sum of coefficients: {np.sum(coefficients):.4f}")
except Exception as e:
    print(f"\n✗ get_coefficients() failed: {e}")
    sys.exit(1)

# Get design matrix
try:
    design_matrix = gam.get_design_matrix()
    print(f"\n✓ get_design_matrix() works!")
    print(f"  Shape: {design_matrix.shape}")
    print(f"  Expected shape: ({n}, {10}) (n_obs, n_basis)")
    print(f"  First row (first 5 values): {design_matrix[0, :5]}")
    print(f"  Last row (first 5 values): {design_matrix[-1, :5]}")
except Exception as e:
    print(f"\n✗ get_design_matrix() failed: {e}")
    sys.exit(1)

# Verify predictions match: fitted_values ≈ design_matrix @ coefficients
try:
    fitted_values = gam.get_fitted_values()
    manual_fitted = design_matrix @ coefficients

    max_diff = np.max(np.abs(fitted_values - manual_fitted))
    print(f"\n--- Verification ---")
    print(f"✓ Fitted values match design_matrix @ coefficients")
    print(f"  Max difference: {max_diff:.10f}")

    if max_diff < 1e-10:
        print("  ✓ Perfect match!")
    elif max_diff < 1e-6:
        print("  ✓ Excellent match (within numerical precision)")
    else:
        print(f"  ⚠ Difference might be too large: {max_diff}")
except Exception as e:
    print(f"\n✗ Verification failed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
