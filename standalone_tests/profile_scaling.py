#!/usr/bin/env python3
"""Profile where time is spent in high-dimensional cases."""

import mgcv_rust
import numpy as np
import time
import os

# Enable profiling
os.environ['MGCV_PROFILE'] = '1'

def profile_run(n, d, k):
    np.random.seed(42)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)
    
    gam = mgcv_rust.GAM()
    result = gam.fit_auto_optimized(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
    return result

print("Profiling n=5000, d=10, k=10 (baseline)...")
profile_run(5000, 10, 10)
print()

print("Profiling n=5000, d=10, k=12 (where we jump to 400ms)...")
profile_run(5000, 10, 12)
print()

print("Profiling n=10000, d=10, k=10...")
profile_run(10000, 10, 10)
