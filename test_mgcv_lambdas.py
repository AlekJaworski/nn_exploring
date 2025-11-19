#!/usr/bin/env python3
"""
Test our gradient computation at mgcv's optimal lambda values
"""
import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

import numpy as np
import mgcv_rust

# Setup data
np.random.seed(42)
n = 1000
x = np.random.randn(n, 4)
y = np.sin(x[:, 0]) + 0.5*x[:, 1]**2 + np.cos(x[:, 2]) + 0.3*x[:, 3] + np.random.randn(n)*0.1

# Use mgcv's optimal lambdas
optimal_lambdas = [15.16574, 9.358342, 17.88683, 142612.6]

print("Testing gradient computation at mgcv's optimal lambdas:")
print(f"Lambdas: {optimal_lambdas}")

# Create GAM and compute gradient at these lambdas
gam = mgcv_rust.GAM()
smooth_terms = [
    {'type': 'cr', 'variable': 0, 'k': 16},
    {'type': 'cr', 'variable': 1, 'k': 16},
    {'type': 'cr', 'variable': 2, 'k': 16},
    {'type': 'cr', 'variable': 3, 'k': 16}
]

# Fit with auto to get initial setup, but then evaluate gradient at optimal lambdas
result = gam.fit_auto(x, y, k=[16]*4, method='REML', bs='cr')

print(f"\nNote: Our implementation has 16 basis functions per smooth (64 total)")
print(f"mgcv has 15 basis functions per smooth (61 total including intercept)")
print(f"This dimensional mismatch is why we can't directly compare values yet.")
