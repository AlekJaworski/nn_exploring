#!/usr/bin/env python3
"""Test multi-dimensional GAM with profiling."""

import numpy as np
import mgcv_rust
import os

# Enable profiling
os.environ['MGCV_PROFILE'] = '1'

def generate_data(n, n_dims, seed=42):
    """Generate test data."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, n_dims))
    y = np.zeros(n)

    if n_dims >= 1:
        y += np.sin(2 * np.pi * X[:, 0])
    if n_dims >= 2:
        y += 0.5 * np.cos(3 * np.pi * X[:, 1])
    if n_dims >= 3:
        y += 0.3 * (X[:, 2] ** 2)

    for i in range(3, n_dims):
        y += 0.1 * np.sin(np.pi * X[:, i])

    y += np.random.normal(0, 0.2, n)
    return X, y

n, n_dims, k = 1000, 3, 10

print(f"Testing multi-dimensional GAM with profiling: n={n}, dimensions={n_dims}, k={k}")
print()

X, y = generate_data(n, n_dims)

gam = mgcv_rust.GAM()
print("Fitting GAM...")
print()

result = gam.fit_auto(X, y, k=[k] * n_dims, method='REML', bs='cr')

print()
print("=" * 80)
print("RESULTS:")
print(f"Lambdas: {result['lambda']}")
print(f"Fitted: {result['fitted']}")
