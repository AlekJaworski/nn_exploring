#!/usr/bin/env python3
"""
Extract our penalty matrices for comparison with R mgcv
"""

import numpy as np
import mgcv_rust

# Same parameters as R script
np.random.seed(42)
n = 500
k = 20
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
X = x.reshape(-1, 1)

print("=" * 70)
print("Extracting mgcv_rust Penalty Matrices")
print("=" * 70)

# CR
gam_cr = mgcv_rust.GAM()
result_cr = gam_cr.fit_auto(X, y, k=[k], method='REML', bs='cr')

# Access penalty matrix through internal state (if available)
# For now, let's just compute it manually using our basis functions
from mgcv_rust import GAM

# Create a simple test to get penalty matrix structure
n_test = 100
k_test = 10
x_test = np.linspace(0, 1, n_test).reshape(-1, 1)
y_test = np.sin(2 * np.pi * x_test.ravel()) + np.random.normal(0, 0.1, n_test)

print("\nTest case (n=100, k=10):")

# BS
gam_bs_test = GAM()
result_bs_test = gam_bs_test.fit_auto(x_test, y_test, k=[k_test], method='REML', bs='bs')
print(f"  BS Lambda: {result_bs_test['lambda']:.6f}")

# CR
gam_cr_test = GAM()
result_cr_test = gam_cr_test.fit_auto(x_test, y_test, k=[k_test], method='REML', bs='cr')
print(f"  CR Lambda: {result_cr_test['lambda']:.6f}")

print("\nMain test case (n=500, k=20):")
print(f"  CR Lambda: {result_cr['lambda']:.6f}")
print(f"  BS Lambda: {result_bs['lambda'] if 'lambda' in result_bs else 'N/A'}")

# Let me check if there's a way to extract penalty matrices
# For now, I'll create a minimal example to compute penalties directly

print("\n" + "=" * 70)
print("ISSUE DETECTED")
print("=" * 70)
print("\nLambda values are still very far from mgcv:")
print(f"  CR: mgcv = 16.45, ours = {result_cr['lambda']:.6f}")
print("\nThis suggests our penalty matrices have different STRUCTURE")
print("or SCALE from mgcv, not just normalization.")
