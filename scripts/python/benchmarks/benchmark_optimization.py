#!/usr/bin/env python3
"""
Benchmark script to compare fit_auto vs fit_auto_optimized
"""

import numpy as np
import time
import sys

try:
    import mgcv_rust
    print("✓ mgcv_rust is available")
except ImportError as e:
    print(f"✗ Error: mgcv_rust not available: {e}")
    print("  Build with: maturin build --release && pip install target/wheels/*.whl")
    sys.exit(1)


def generate_4d_data(n=500, noise_level=0.3, seed=42):
    """Generate 4D test data"""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, 4))

    effect_1 = np.sin(2 * np.pi * X[:, 0])
    effect_2 = (X[:, 1] - 0.5) ** 2
    effect_3 = X[:, 2]
    effect_4 = np.zeros(n)

    y_true = effect_1 + effect_2 + effect_3 + effect_4
    y = y_true + np.random.normal(0, noise_level, n)

    return X, y


def benchmark_method(method_name, fit_func, X, y, k, n_iters=50):
    """Benchmark a fitting method"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {method_name}")
    print(f"{'='*60}")

    times = []
    k_list = [k] * X.shape[1]

    for i in range(n_iters):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        fit_func(gam, X, y, k_list)
        end = time.perf_counter()
        times.append(end - start)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_iters} iterations complete...")

    times = np.array(times) * 1000  # Convert to ms

    print(f"\n{method_name} results:")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
    }


def fit_standard(gam, X, y, k_list):
    """Standard fitting"""
    gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')


def fit_optimized(gam, X, y, k_list):
    """Optimized fitting"""
    gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')


def main():
    print("="*60)
    print("GAM Optimization Benchmark")
    print("="*60)

    # Generate data
    n = 500
    k = 12
    noise_level = 0.3
    n_iters = 50

    print(f"\nParameters:")
    print(f"  n = {n} observations")
    print(f"  dimensions = 4")
    print(f"  k = {k} basis functions per dimension")
    print(f"  iterations = {n_iters}")

    X, y = generate_4d_data(n, noise_level)

    # Benchmark standard version
    results_standard = benchmark_method(
        "fit_auto (standard)",
        fit_standard,
        X, y, k, n_iters
    )

    # Benchmark optimized version
    results_optimized = benchmark_method(
        "fit_auto_optimized",
        fit_optimized,
        X, y, k, n_iters
    )

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    speedup = results_standard['mean'] / results_optimized['mean']

    print(f"\nStandard version:  {results_standard['mean']:.2f} ± {results_standard['std']:.2f} ms")
    print(f"Optimized version: {results_optimized['mean']:.2f} ± {results_optimized['std']:.2f} ms")
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup > 1.1:
        print(f"✓ Optimized version is {speedup:.2f}x FASTER")
    elif speedup < 0.9:
        print(f"⚠ Optimized version is {1/speedup:.2f}x SLOWER")
    else:
        print("≈ Performance is similar (within 10%)")

    # Verify correctness
    print(f"\n{'='*60}")
    print("CORRECTNESS CHECK")
    print(f"{'='*60}")

    gam_std = mgcv_rust.GAM()
    result_std = gam_std.fit_auto(X, y, k=[k]*4, method='REML', bs='cr')
    pred_std = gam_std.predict(X)

    gam_opt = mgcv_rust.GAM()
    result_opt = gam_opt.fit_auto_optimized(X, y, k=[k]*4, method='REML', bs='cr')
    pred_opt = gam_opt.predict(X)

    # Compare predictions
    corr = np.corrcoef(pred_std, pred_opt)[0, 1]
    rmse_diff = np.sqrt(np.mean((pred_std - pred_opt)**2))
    max_diff = np.max(np.abs(pred_std - pred_opt))

    print(f"Prediction correlation: {corr:.8f}")
    print(f"Prediction RMSE diff:   {rmse_diff:.8f}")
    print(f"Prediction max diff:    {max_diff:.8f}")

    if corr > 0.9999 and rmse_diff < 1e-6:
        print("\n✓ Results are numerically equivalent")
    else:
        print("\n⚠ Results differ - investigate further")

    return 0


if __name__ == "__main__":
    sys.exit(main())
