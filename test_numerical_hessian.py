#!/usr/bin/env python3
"""
Compute Hessian numerically using finite differences and compare to mgcv.

This will show us what the correct Hessian values should be, independent of the formula.
"""

import numpy as np
import mgcv_rust
import os

# Load data
x = np.loadtxt('/tmp/unit_x.csv', delimiter=',', skiprows=1)
y = np.loadtxt('/tmp/unit_y.csv', delimiter=',', skiprows=1)

print("=" * 80)
print("NUMERICAL HESSIAN COMPUTATION")
print("=" * 80)

# Disable debug output for clean REML computation
if 'MGCV_GRAD_DEBUG' in os.environ:
    del os.environ['MGCV_GRAD_DEBUG']
if 'MGCV_PROFILE' in os.environ:
    del os.environ['MGCV_PROFILE']

# Test at specific λ values
test_lambdas = [0.1, 0.1]  # Intermediate value

print(f"\nTesting at λ = {test_lambdas}")

# Compute REML and gradient at base point
gam = mgcv_rust.GAM()
result = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=test_lambdas)
reml_base = result['reml']
print(f"REML at base: {reml_base:.8f}")

# Compute numerical gradient using finite differences
h = 1e-5  # Step size for finite differences
gradient_numerical = []

for i in range(2):
    lambda_plus = test_lambdas.copy()
    lambda_plus[i] += h

    result_plus = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=lambda_plus)
    reml_plus = result_plus['reml']

    grad_i = (reml_plus - reml_base) / h
    gradient_numerical.append(grad_i)
    print(f"∂REML/∂λ_{i} ≈ {grad_i:.6f}")

# Compute numerical Hessian using finite differences
print(f"\nNumerical Hessian:")
hessian_numerical = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        # Compute ∂²REML/∂λ_i∂λ_j using centered differences

        # Point (+h_i, +h_j)
        lambda_pp = test_lambdas.copy()
        lambda_pp[i] += h
        lambda_pp[j] += h
        reml_pp = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=lambda_pp)['reml']

        # Point (+h_i, -h_j)
        lambda_pm = test_lambdas.copy()
        lambda_pm[i] += h
        lambda_pm[j] -= h
        reml_pm = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=lambda_pm)['reml']

        # Point (-h_i, +h_j)
        lambda_mp = test_lambdas.copy()
        lambda_mp[i] -= h
        lambda_mp[j] += h
        reml_mp = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=lambda_mp)['reml']

        # Point (-h_i, -h_j)
        lambda_mm = test_lambdas.copy()
        lambda_mm[i] -= h
        lambda_mm[j] -= h
        reml_mm = gam.fit_fixed_lambda(x, y, k=[10, 10], method='REML', bs='cr', lambda_values=lambda_mm)['reml']

        hess_ij = (reml_pp - reml_pm - reml_mp + reml_mm) / (4 * h * h)
        hessian_numerical[i, j] = hess_ij

print(hessian_numerical)
print(f"\nDiagonal: {np.diag(hessian_numerical)}")

print("\n" + "=" * 80)
print("Now with ρ = log(λ) parameterization:")
print("=" * 80)

# Convert to ρ-space derivatives
# ∂REML/∂ρ_i = ∂REML/∂λ_i * λ_i
grad_rho = [gradient_numerical[i] * test_lambdas[i] for i in range(2)]
print(f"∂REML/∂ρ: {grad_rho}")

# ∂²REML/∂ρ_i∂ρ_j = λ_i * λ_j * ∂²REML/∂λ_i∂λ_j + δ_ij * λ_i * ∂REML/∂λ_i
hess_rho = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        hess_rho[i, j] = test_lambdas[i] * test_lambdas[j] * hessian_numerical[i, j]
        if i == j:
            hess_rho[i, j] += test_lambdas[i] * gradient_numerical[i]

print("Hessian (ρ-space):")
print(hess_rho)
print(f"Diagonal: {np.diag(hess_rho)}")

print("\n" + "=" * 80)
print("CONCLUSIONS:")
print("=" * 80)
print("This numerical Hessian should match what mgcv computes.")
print("If our analytical Hessian doesn't match, we have the wrong formula.")
