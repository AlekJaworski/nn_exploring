"""
Profile GAM fitting to find hotspots
"""
import numpy as np
import mgcv_rust
import cProfile
import pstats
import io

np.random.seed(42)

def profile_gam():
    # Use n=1500 for faster profiling
    n = 1500
    k = 16

    x = np.random.randn(n, 4)
    y_true = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.cos(x[:, 2]) + 0.3 * x[:, 3]
    y = y_true + np.random.randn(n) * 0.1

    k_list = [k] * 4

    # Profile the fit
    gam = mgcv_rust.GAM()
    result = gam.fit_auto_optimized(x, y, k=k_list, method='REML', bs='cr')

    print(f"Fit completed. Lambda: {result['lambda']}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Run multiple times to get better statistics
    for _ in range(5):
        profile_gam()

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())
