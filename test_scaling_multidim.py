#!/usr/bin/env python3
"""
Comprehensive scaling test for multidimensional GAM inference.

Tests performance and accuracy across different:
- Sample sizes: 500, 1500, 2500, 5000
- Basis dimensions: k=16 per dimension
- 4D data with mixed effects
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys

try:
    import mgcv_rust
    print("✓ mgcv_rust available")
except ImportError as e:
    print(f"✗ Error: mgcv_rust not available: {e}")
    sys.exit(1)


def generate_4d_data(n=500, noise_level=0.2, seed=42):
    """
    Generate 4D test data with specific effects:
    - x1: Sinusoidal effect
    - x2: Parabolic effect
    - x3: Linear effect
    - x4: Constant noise (no effect)
    """
    np.random.seed(seed)

    # Generate 4 features uniformly in [0, 1]
    X = np.random.uniform(0, 1, size=(n, 4))

    # True effects
    effect_1 = np.sin(2 * np.pi * X[:, 0])
    effect_2 = (X[:, 1] - 0.5) ** 2
    effect_3 = X[:, 2]
    effect_4 = np.zeros(n)

    # Combined true function
    y_true = effect_1 + effect_2 + effect_3 + effect_4

    # Add noise
    y = y_true + np.random.normal(0, noise_level, n)

    return X, y, y_true


def run_scaling_test(sample_sizes, k, n_iters=10):
    """
    Run scaling tests across different sample sizes.

    Args:
        sample_sizes: List of sample sizes to test
        k: Number of basis functions per dimension
        n_iters: Number of iterations for timing
    """
    results = []

    print("="*70)
    print(f"SCALING TEST: k={k} basis functions per dimension")
    print("="*70)
    print()

    for n in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Sample size: n={n}")
        print(f"{'='*70}")

        # Generate data
        X, y, y_true = generate_4d_data(n=n)
        k_list = [k] * 4

        # Warm-up run
        print("Warming up...")
        gam = mgcv_rust.GAM()
        result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')

        # Timing runs
        print(f"Running {n_iters} iterations for timing...")
        times = []
        for i in range(n_iters):
            gam = mgcv_rust.GAM()
            start = time.perf_counter()
            result = gam.fit_auto_optimized(X, y, k=k_list, method='REML', bs='cr')
            end = time.perf_counter()
            times.append(end - start)

            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{n_iters} complete...")

        times = np.array(times) * 1000  # Convert to ms

        # Compute accuracy metrics
        pred = gam.predict(X)
        rmse_pred = np.sqrt(np.mean((y - pred)**2))
        rmse_true = np.sqrt(np.mean((y_true - pred)**2))
        r2 = 1 - np.var(y - pred) / np.var(y)

        # Get model info
        lambdas = result['lambda']
        deviance = result['deviance']

        # Store results
        result_dict = {
            'n': n,
            'k': k,
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_min': np.min(times),
            'time_max': np.max(times),
            'rmse_pred': rmse_pred,
            'rmse_true': rmse_true,
            'r2': r2,
            'lambdas': lambdas,
            'deviance': deviance,
        }
        results.append(result_dict)

        # Print summary
        print(f"\nResults for n={n}:")
        print(f"  Timing:")
        print(f"    Mean:   {np.mean(times):7.2f} ms")
        print(f"    Std:    {np.std(times):7.2f} ms")
        print(f"    Min:    {np.min(times):7.2f} ms")
        print(f"    Max:    {np.max(times):7.2f} ms")
        print(f"  Accuracy:")
        print(f"    RMSE (vs data):  {rmse_pred:.4f}")
        print(f"    RMSE (vs truth): {rmse_true:.4f}")
        print(f"    R²:              {r2:.4f}")
        print(f"  Model:")
        print(f"    Deviance: {deviance:.2f}")
        print(f"    Lambdas:  [{', '.join([f'{l:.4f}' for l in lambdas])}]")

    return results


def analyze_scaling(results):
    """Analyze scaling behavior."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)

    ns = [r['n'] for r in results]
    times = [r['time_mean'] for r in results]
    rmses_true = [r['rmse_true'] for r in results]
    r2s = [r['r2'] for r in results]

    # Compute scaling exponent (assuming T ~ n^alpha)
    if len(ns) >= 2:
        log_ns = np.log(ns)
        log_times = np.log(times)
        # Simple linear regression
        A = np.vstack([log_ns, np.ones(len(log_ns))]).T
        alpha, log_c = np.linalg.lstsq(A, log_times, rcond=None)[0]

        print(f"\nTime complexity: T ≈ {np.exp(log_c):.2f} * n^{alpha:.2f}")
        print(f"(Theoretical for dense matrices: n^3)")
        print()

    # Time per sample
    print("Time per sample:")
    for r in results:
        time_per_sample = r['time_mean'] / r['n']
        print(f"  n={r['n']:5d}: {time_per_sample:7.4f} ms/sample")

    # Accuracy vs sample size
    print("\nAccuracy vs sample size:")
    for r in results:
        print(f"  n={r['n']:5d}: RMSE={r['rmse_true']:.4f}, R²={r['r2']:.4f}")

    # Lambda stability
    print("\nLambda estimates (should be similar across sample sizes):")
    for r in results:
        lambda_str = ', '.join([f'{l:.4f}' for l in r['lambdas']])
        print(f"  n={r['n']:5d}: [{lambda_str}]")


def create_visualizations(results):
    """Create visualization plots."""
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)

    ns = [r['n'] for r in results]
    times_mean = [r['time_mean'] for r in results]
    times_std = [r['time_std'] for r in results]
    rmses_true = [r['rmse_true'] for r in results]
    r2s = [r['r2'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Timing vs sample size
    ax = axes[0, 0]
    ax.errorbar(ns, times_mean, yerr=times_std, marker='o', capsize=5, linewidth=2)
    ax.set_xlabel('Sample size (n)', fontsize=12)
    ax.set_ylabel('Fit time (ms)', fontsize=12)
    ax.set_title('Performance Scaling (k=16 per dimension)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add trend line
    log_ns = np.log(np.array(ns))
    log_times = np.log(np.array(times_mean))
    A = np.vstack([log_ns, np.ones(len(log_ns))]).T
    alpha, log_c = np.linalg.lstsq(A, log_times, rcond=None)[0]
    trend = np.exp(log_c) * np.array(ns)**alpha
    ax.plot(ns, trend, 'r--', label=f'T ~ n^{alpha:.2f}', linewidth=1.5, alpha=0.7)
    ax.legend()

    # 2. Time per sample
    ax = axes[0, 1]
    time_per_sample = [t/n for t, n in zip(times_mean, ns)]
    ax.plot(ns, time_per_sample, marker='s', linewidth=2)
    ax.set_xlabel('Sample size (n)', fontsize=12)
    ax.set_ylabel('Time per sample (ms)', fontsize=12)
    ax.set_title('Efficiency per Sample', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Accuracy vs sample size
    ax = axes[1, 0]
    ax.plot(ns, rmses_true, marker='o', linewidth=2, label='RMSE (vs truth)')
    ax.set_xlabel('Sample size (n)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Accuracy vs Sample Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. R² vs sample size
    ax = axes[1, 1]
    ax.plot(ns, r2s, marker='o', linewidth=2, color='green')
    ax.set_xlabel('Sample size (n)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Model Fit Quality', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.0])

    plt.tight_layout()
    output_file = 'scaling_test_results.png'
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved visualization to {output_file}")


def generate_report(results, k):
    """Generate comprehensive report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE REPORT")
    print("="*70)

    print(f"\nTest Configuration:")
    print(f"  Basis functions: k={k} per dimension")
    print(f"  Dimensions: 4")
    print(f"  Sample sizes: {[r['n'] for r in results]}")
    print(f"  Iterations per size: 10")

    print(f"\nPerformance Summary:")
    print(f"  {'n':>6s}  {'Time (ms)':>12s}  {'Std (ms)':>10s}  {'ms/sample':>12s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*12}")
    for r in results:
        time_per_sample = r['time_mean'] / r['n']
        print(f"  {r['n']:6d}  {r['time_mean']:12.2f}  {r['time_std']:10.2f}  {time_per_sample:12.4f}")

    # Compute scaling factor between consecutive sizes
    print(f"\nScaling Ratios:")
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        time_ratio = curr['time_mean'] / prev['time_mean']
        size_ratio = curr['n'] / prev['n']
        print(f"  n={prev['n']} → n={curr['n']}: {time_ratio:.2f}x time for {size_ratio:.2f}x data")

    print(f"\nAccuracy Summary:")
    print(f"  {'n':>6s}  {'RMSE (pred)':>12s}  {'RMSE (true)':>12s}  {'R²':>8s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in results:
        print(f"  {r['n']:6d}  {r['rmse_pred']:12.4f}  {r['rmse_true']:12.4f}  {r['r2']:8.4f}")

    print(f"\nKey Findings:")

    # Best accuracy
    best_r2_idx = np.argmax([r['r2'] for r in results])
    print(f"  Best R²: {results[best_r2_idx]['r2']:.4f} at n={results[best_r2_idx]['n']}")

    # Speed comparison
    fastest_per_sample_idx = np.argmin([r['time_mean']/r['n'] for r in results])
    print(f"  Most efficient: {results[fastest_per_sample_idx]['time_mean']/results[fastest_per_sample_idx]['n']:.4f} ms/sample at n={results[fastest_per_sample_idx]['n']}")

    # Scaling behavior
    if len(results) >= 2:
        log_ns = np.log([r['n'] for r in results])
        log_times = np.log([r['time_mean'] for r in results])
        A = np.vstack([log_ns, np.ones(len(log_ns))]).T
        alpha, _ = np.linalg.lstsq(A, log_times, rcond=None)[0]
        print(f"  Empirical complexity: O(n^{alpha:.2f})")


def main():
    """Main test function."""
    print("="*70)
    print("MULTIDIMENSIONAL GAM SCALING TEST")
    print("="*70)

    # Test configuration
    sample_sizes = [500, 1500, 2500, 5000]
    k = 16
    n_iters = 10

    print(f"\nConfiguration:")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Basis functions: k={k} per dimension")
    print(f"  Dimensions: 4")
    print(f"  Iterations per size: {n_iters}")
    print()

    # Run tests
    results = run_scaling_test(sample_sizes, k, n_iters)

    # Analyze
    analyze_scaling(results)

    # Visualize
    create_visualizations(results)

    # Generate report
    generate_report(results, k)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\n✓ Results saved to scaling_test_results.png")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
