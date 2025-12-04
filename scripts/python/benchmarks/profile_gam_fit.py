#!/usr/bin/env python3
"""
Profile where time is spent in the GAM fitting process.
"""

import numpy as np
import time
import mgcv_rust

def profile_gam_fit():
    """Profile a single GAM fit."""
    np.random.seed(42)
    n = 5000
    k = 20

    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.flatten()) + np.random.normal(0, 0.1, n)

    print(f"Profiling n={n}, k={k}")
    print("=" * 60)

    gam = mgcv_rust.GAM()

    # Time the full fit
    start = time.perf_counter()
    result = gam.fit_auto(x, y, k=[k], method='REML', bs='cr', max_iter=10)
    end = time.perf_counter()

    total_time = end - start

    print(f"Total time: {total_time:.4f}s")
    print(f"Lambda: {result['lambda'][0]:.6f}")
    print(f"Fitted: {result['fitted']}")
    print()
    print("Note: Run with MGCV_PROFILE=1 in Rust code to see detailed breakdown")

if __name__ == '__main__':
    profile_gam_fit()
