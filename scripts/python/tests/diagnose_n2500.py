#!/usr/bin/env python3
"""
Diagnostic script to investigate n=2500 performance anomaly.

This script:
1. Runs benchmarks for n=500, 1500, 2500, 5000
2. Profiles each run to identify bottlenecks
3. Compares timing breakdown across sample sizes
4. Identifies why n=2500 is specifically slow
"""

import numpy as np
import time
import sys
import traceback
from typing import Dict, List, Tuple

try:
    import mgcv_rust
except ImportError:
    print("Error: mgcv_rust not found. Build with: maturin develop --release")
    sys.exit(1)


def generate_test_data(n: int, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 4D test data matching scaling tests."""
    np.random.seed(42)

    # 4D input
    X = np.random.uniform(0, 1, size=(n, 4))

    # True function: nonlinear x1 and x2, linear x3, noise x4
    y_true = (
        np.sin(2 * np.pi * X[:, 0]) * 2.0 +  # x1: strong nonlinear
        np.cos(4 * np.pi * X[:, 1]) * 1.5 +  # x2: strong nonlinear
        X[:, 2] * 0.5                         # x3: weak linear (should get large lambda)
        # x4: not used (pure noise, should get very large lambda)
    )

    # Add noise
    y = y_true + np.random.normal(0, 0.1, size=n)

    return X, y


def profile_single_fit(X: np.ndarray, y: np.ndarray, k: int = 16) -> Dict[str, float]:
    """Profile a single GAM fit, breaking down time by component."""
    n = len(y)

    print(f"\n{'='*70}")
    print(f"Profiling n={n}, k={k}")
    print(f"{'='*70}")

    results = {
        'n': n,
        'k': k,
        'total_time': 0.0,
        'fit_time': 0.0,
        'predict_time': 0.0,
    }

    # Measure total fit time
    start_total = time.time()

    try:
        # Create GAM with 4 smooths
        start_fit = time.time()
        gam = mgcv_rust.GAM()
        k_list = [k] * 4  # 4 dimensions

        # Fit the model
        result = gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')
        end_fit = time.time()
        results['fit_time'] = end_fit - start_fit

        # Measure prediction time
        start_pred = time.time()
        predictions = gam.predict(X)
        end_pred = time.time()
        results['predict_time'] = end_pred - start_pred

        end_total = time.time()
        results['total_time'] = end_total - start_total

        # Get lambda estimates
        if 'smoothing_parameters' in result:
            lambdas = result['smoothing_parameters']
            print(f"\nLambda estimates:")
            for i, lam in enumerate(lambdas):
                print(f"  x{i+1}: {lam:12.2f}")

        # Compute accuracy metrics
        residuals = y - predictions
        rmse = np.sqrt(np.mean(residuals**2))

        print(f"\nAccuracy:")
        print(f"  RMSE: {rmse:.6f}")

        print(f"\nTiming breakdown:")
        print(f"  Fit time:     {results['fit_time']*1000:8.2f} ms")
        print(f"  Predict time: {results['predict_time']*1000:8.2f} ms")
        print(f"  Total time:   {results['total_time']*1000:8.2f} ms")

        # Time per sample
        time_per_sample = results['fit_time'] * 1000 / n
        print(f"  Time/sample:  {time_per_sample:8.4f} ms")

        results['rmse'] = rmse
        results['time_per_sample'] = time_per_sample
        results['success'] = True

    except Exception as e:
        print(f"\nError during fit:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)

    return results


def compare_scaling(sample_sizes: List[int], k: int = 16, n_iter: int = 5):
    """Compare performance across different sample sizes."""
    print(f"\n{'='*70}")
    print(f"SCALING COMPARISON")
    print(f"{'='*70}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Basis functions: k={k}")
    print(f"Iterations: {n_iter}")

    all_results = []

    for n in sample_sizes:
        print(f"\n\nBenchmarking n={n}...")

        # Generate data
        X, y = generate_test_data(n, k)

        # Run multiple iterations
        times = []
        k_list = [k] * 4  # 4 dimensions
        for i in range(n_iter):
            start = time.time()
            try:
                gam = mgcv_rust.GAM()
                result = gam.fit_auto(X, y, k=k_list, method='REML', bs='cr')
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
            except Exception as e:
                print(f"  Iteration {i+1} failed: {e}")
                continue

        if times:
            mean_time = np.mean(times)
            std_time = np.std(times)
            time_per_sample = mean_time / n

            print(f"  Time: {mean_time:8.2f} ± {std_time:6.2f} ms")
            print(f"  Time/sample: {time_per_sample:8.4f} ms")

            all_results.append({
                'n': n,
                'mean_time': mean_time,
                'std_time': std_time,
                'time_per_sample': time_per_sample,
                'times': times
            })

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'n':>6}  {'Time (ms)':>12}  {'Time/sample':>12}  {'Ratio to n=500':>15}")
    print(f"{'-'*70}")

    baseline_time = None
    for result in all_results:
        n = result['n']
        mean_time = result['mean_time']
        time_per_sample = result['time_per_sample']

        if baseline_time is None:
            baseline_time = mean_time
            ratio = 1.0
        else:
            ratio = mean_time / baseline_time

        print(f"{n:6d}  {mean_time:12.2f}  {time_per_sample:12.4f}  {ratio:15.2f}x")

    # Identify anomaly
    print(f"\n{'='*70}")
    print(f"ANOMALY DETECTION")
    print(f"{'='*70}")

    # Expected scaling: approximately O(n) or O(n log n) for well-optimized code
    # Compare actual vs expected
    if len(all_results) >= 2:
        for i in range(1, len(all_results)):
            prev = all_results[i-1]
            curr = all_results[i]

            n_ratio = curr['n'] / prev['n']
            time_ratio = curr['mean_time'] / prev['mean_time']
            expected_ratio = n_ratio  # Assuming O(n) scaling

            anomaly_factor = time_ratio / expected_ratio

            status = "✓ Normal" if anomaly_factor < 1.5 else "✗ ANOMALY"

            print(f"n={prev['n']} → n={curr['n']}:")
            print(f"  Sample size ratio: {n_ratio:.2f}x")
            print(f"  Time ratio:        {time_ratio:.2f}x")
            print(f"  Expected (O(n)):   {expected_ratio:.2f}x")
            print(f"  Anomaly factor:    {anomaly_factor:.2f}x  {status}")
            print()


def main():
    """Run diagnostic analysis."""
    print("="*70)
    print("n=2500 Performance Anomaly Diagnostic")
    print("="*70)

    # Sample sizes to test
    sample_sizes = [500, 1500, 2500, 5000]
    k = 16

    # Run scaling comparison
    compare_scaling(sample_sizes, k=k, n_iter=10)

    # Detailed profiling of n=2500 specifically
    print(f"\n\n{'='*70}")
    print("DETAILED PROFILE: n=2500")
    print(f"{'='*70}")

    X, y = generate_test_data(2500, k)
    profile_single_fit(X, y, k)

    print(f"\n\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
Based on the analysis above:

1. If n=2500 shows 2x+ slower time ratio vs expected:
   → Likely REML convergence issue or memory allocation problem
   → Check REML iteration count (add instrumentation to Rust code)
   → Profile memory allocations

2. If time/sample increases significantly at n=2500:
   → Cache effects or memory bandwidth issue
   → Check matrix operation sizes (2500 might exceed cache)

3. If lambda estimates differ significantly from other sizes:
   → REML optimization taking different path
   → May need better initialization or convergence criteria

Next steps:
- Add REML iteration logging to src/smooth.rs
- Run with RUST_LOG=debug to see internal timings
- Consider profiling with `perf` or `flamegraph`
""")


if __name__ == "__main__":
    main()
