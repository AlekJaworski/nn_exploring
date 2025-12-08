#!/usr/bin/env python3
"""
Profile a single large-n case to see where time is spent.
"""

import numpy as np
import mgcv_rust
import time

# Test case: n=5000, d=1 (worst single-variable case)
n = 5000
np.random.seed(42)
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
X = x.reshape(-1, 1)

print(f"Profiling n={n}, d=1, k=20")
print("=" * 80)

# Run with profiling enabled
import os
os.environ['MGCV_PROFILE'] = '1'

gam = mgcv_rust.GAM()

start = time.perf_counter()
result = gam.fit_auto(X, y, k=[20], method='REML', bs='cr', max_iter=10)
end = time.perf_counter()

print(f"\nTotal time: {end - start:.4f}s")
print(f"Lambda: {result['lambda'][0]:.6f}")
print(f"Deviance: {result.get('deviance', 'N/A')}")
