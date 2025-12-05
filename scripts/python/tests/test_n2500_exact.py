#!/usr/bin/env python3
"""
Exact reproduction of compare_with_mgcv.py benchmark for n=2500 only.
"""

import numpy as np
import time
import mgcv_rust

def generate_4d_data(n=500, noise_level=0.2, seed=42):
    """Generate 4D test data - EXACT copy from compare_with_mgcv.py."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, 4))

    effect_1 = np.sin(2 * np.pi * X[:, 0])
    effect_2 = (X[:, 1] - 0.5) ** 2
    effect_3 = X[:, 2]
    effect_4 = np.zeros(n)

    y_true = effect_1 + effect_2 + effect_3 + effect_4
    y = y_true + np.random.normal(0, noise_level, n)

    return X, y, y_true

def benchmark_mgcv_rust(X, y, k, n_iters=10):
    """Benchmark mgcv_rust - EXACT copy from compare_with_mgcv.py."""
    print(f"  Benchmarking mgcv_rust...")

    k_list = [k] * X.shape[1]

    # Warm-up
    gam = mgcv_rust.GAM()
    result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')

    # Timing runs
    times = []
    for i in range(n_iters):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)

    # Final model
    gam = mgcv_rust.GAM()
    result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
    pred = gam.predict(X)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times,
        'pred': pred,
        'lambdas': result['lambda'],
        'deviance': result['deviance']
    }

def main():
    n = 2500
    k = 16

    print(f"Exact reproduction of compare_with_mgcv.py benchmark")
    print(f"n={n}, k={k}")
    print()

    # Generate data EXACTLY as compare_with_mgcv.py does
    X, y, y_true = generate_4d_data(n, noise_level=0.2, seed=42)

    # Benchmark EXACTLY as compare_with_mgcv.py does
    results = benchmark_mgcv_rust(X, y, k, n_iters=10)

    print(f"\nResults:")
    print(f"  Mean time: {results['mean']:7.2f} ± {results['std']:5.2f} ms")
    print(f"  Min time:  {results['min']:7.2f} ms")
    print(f"  Max time:  {results['max']:7.2f} ms")
    print(f"\nAll times (ms): {results['times']}")
    print(f"\nLambdas: {results['lambdas']}")
    print(f"Deviance: {results['deviance']:.6f}")

    # Compute RMSE
    residuals = y_true - results['pred']
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"RMSE (vs truth): {rmse:.6f}")

if __name__ == "__main__":
    main()
