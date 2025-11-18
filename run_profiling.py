"""
Run profiling to identify bottlenecks
"""
import numpy as np
import mgcv_rust
import os
import sys

# Enable profiling
os.environ['MGCV_PROFILE'] = '1'

np.random.seed(42)

def profile_different_sizes():
    sizes = [500, 1500, 2500, 5000]
    k = 16

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"n = {n}")
        print('='*60)

        x = np.random.randn(n, 4)
        y_true = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.cos(x[:, 2]) + 0.3 * x[:, 3]
        y = y_true + np.random.randn(n) * 0.1

        k_list = [k] * 4

        # Run once with profiling
        gam = mgcv_rust.GAM()
        result = gam.fit_auto_optimized(x, y, k=k_list, method='REML', bs='cr')

        print(f"Lambda: {result['lambda']}")
        sys.stderr.flush()

if __name__ == "__main__":
    profile_different_sizes()
