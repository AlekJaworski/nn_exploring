#!/usr/bin/env python3
"""
Comprehensive benchmark and analysis tool for GAM performance.
Compares performance across different scenarios and implementation variants.
"""

import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List
import sys

try:
    import mgcv_rust
    RUST_AVAILABLE = True
except ImportError:
    print("mgcv_rust not available")
    RUST_AVAILABLE = False


def generate_test_data(n, d, seed=42):
    """Generate test data for GAM."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = np.zeros(n)
    for i in range(d):
        if i % 3 == 0:
            y += np.sin(2 * np.pi * X[:, i])
        elif i % 3 == 1:
            y += (X[:, i] - 0.5) ** 2
        else:
            y += np.exp(-5 * (X[:, i] - 0.5) ** 2)
    y += np.random.normal(0, 0.1, n)
    return X, y


def run_timing_test(n, d, k_values, n_repeats=3):
    """Run timing test with multiple repeats."""
    X, y = generate_test_data(n, d)

    times = []
    for _ in range(n_repeats):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        try:
            result = gam.fit_auto(X, y, k=k_values, method="REML", bs="cr")
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            print(f"Error: {e}")
            return None

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def profile_components():
    """Profile individual components to identify bottlenecks."""
    print("\n" + "="*70)
    print("Profiling Individual Components")
    print("="*70)

    # Test different n values with fixed d and k
    print("\nScaling with n (d=1, k=10):")
    n_values = [50, 100, 200, 500, 1000, 2000, 5000]
    for n in n_values:
        result = run_timing_test(n, 1, [10])
        if result:
            print(f"  n={n:5d}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")

    # Test different k values with fixed n and d
    print("\nScaling with k (n=1000, d=1):")
    k_values = [5, 10, 15, 20, 25, 30]
    for k in k_values:
        result = run_timing_test(1000, 1, [k])
        if result:
            print(f"  k={k:2d}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")

    # Test different d values with fixed n and k
    print("\nScaling with d (n=500, k=10 for each dim):")
    d_values = [1, 2, 3, 5, 10]
    for d in d_values:
        result = run_timing_test(500, d, [10] * d)
        if result:
            print(f"  d={d:2d}: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")


def analyze_complexity():
    """Analyze computational complexity."""
    print("\n" + "="*70)
    print("Computational Complexity Analysis")
    print("="*70)

    # Measure n scaling
    print("\nAnalyzing O(n) scaling:")
    n_values = [100, 200, 400, 800, 1600]
    times = []
    for n in n_values:
        result = run_timing_test(n, 1, [10], n_repeats=5)
        if result:
            times.append(result['mean_time'])
            ratio = result['mean_time'] / times[0] if times else 1.0
            expected_linear = n / n_values[0]
            expected_quadratic = (n / n_values[0]) ** 2
            expected_cubic = (n / n_values[0]) ** 3
            print(f"  n={n:4d}: {result['mean_time']:.4f}s | "
                  f"ratio={ratio:.2f}x (linear={expected_linear:.2f}x, "
                  f"quad={expected_quadratic:.2f}x, cubic={expected_cubic:.2f}x)")

    # Measure k scaling
    print("\nAnalyzing O(k) scaling:")
    k_values = [5, 10, 15, 20, 25, 30]
    times = []
    for k in k_values:
        result = run_timing_test(500, 1, [k], n_repeats=5)
        if result:
            times.append(result['mean_time'])
            ratio = result['mean_time'] / times[0] if times else 1.0
            expected_linear = k / k_values[0]
            expected_quadratic = (k / k_values[0]) ** 2
            expected_cubic = (k / k_values[0]) ** 3
            print(f"  k={k:2d}: {result['mean_time']:.4f}s | "
                  f"ratio={ratio:.2f}x (linear={expected_linear:.2f}x, "
                  f"quad={expected_quadratic:.2f}x, cubic={expected_cubic:.2f}x)")


def create_visualizations():
    """Create performance visualization plots."""
    print("\n" + "="*70)
    print("Creating Performance Visualizations")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: n scaling
    n_values = [50, 100, 200, 500, 1000, 2000, 5000]
    times_n = []
    for n in n_values:
        result = run_timing_test(n, 1, [10])
        if result:
            times_n.append(result['mean_time'])
        else:
            times_n.append(np.nan)

    axes[0, 0].plot(n_values, times_n, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of data points (n)', fontsize=12)
    axes[0, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_title('Scaling with n (d=1, k=10)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')

    # Plot 2: k scaling
    k_values = [5, 10, 15, 20, 25, 30]
    times_k = []
    for k in k_values:
        result = run_timing_test(1000, 1, [k])
        if result:
            times_k.append(result['mean_time'])
        else:
            times_k.append(np.nan)

    axes[0, 1].plot(k_values, times_k, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Basis dimension (k)', fontsize=12)
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 1].set_title('Scaling with k (n=1000, d=1)', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: d scaling
    d_values = [1, 2, 3, 5, 10]
    times_d = []
    for d in d_values:
        result = run_timing_test(500, d, [10] * d)
        if result:
            times_d.append(result['mean_time'])
        else:
            times_d.append(np.nan)

    axes[1, 0].plot(d_values, times_d, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Number of dimensions (d)', fontsize=12)
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_title('Scaling with d (n=500, k=10/dim)', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Heatmap of n vs k
    n_test = [100, 500, 1000, 2000]
    k_test = [5, 10, 15, 20]
    heatmap_data = np.zeros((len(n_test), len(k_test)))

    for i, n in enumerate(n_test):
        for j, k in enumerate(k_test):
            result = run_timing_test(n, 1, [k], n_repeats=1)
            if result:
                heatmap_data[i, j] = result['mean_time']
            else:
                heatmap_data[i, j] = np.nan

    im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    axes[1, 1].set_xticks(range(len(k_test)))
    axes[1, 1].set_yticks(range(len(n_test)))
    axes[1, 1].set_xticklabels(k_test)
    axes[1, 1].set_yticklabels(n_test)
    axes[1, 1].set_xlabel('Basis dimension (k)', fontsize=12)
    axes[1, 1].set_ylabel('Number of data points (n)', fontsize=12)
    axes[1, 1].set_title('Time (s) for different n and k', fontsize=14)
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=150)
    print("Saved visualization to performance_analysis.png")


def main():
    if not RUST_AVAILABLE:
        print("Error: mgcv_rust not available. Build with 'maturin develop --release'")
        return 1

    print("="*70)
    print("GAM Performance Analysis Tool")
    print("="*70)

    profile_components()
    analyze_complexity()
    create_visualizations()

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
