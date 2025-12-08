#!/usr/bin/env python3
"""
Benchmark large-n cases specifically to identify bottlenecks.
Focus on n >= 2000 where performance degrades.
"""

import numpy as np
import subprocess
import time
import json
import tempfile
import os

import mgcv_rust


def generate_test_data(n, d=1, seed=42):
    """Generate synthetic test data."""
    np.random.seed(seed)
    if d == 1:
        x = np.linspace(0, 1, n)
        y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
        return x.reshape(-1, 1), y
    else:
        X = np.random.uniform(0, 1, (n, d))
        y = np.zeros(n)
        for i in range(d):
            y += np.sin(2 * np.pi * X[:, i])
        y += np.random.normal(0, 0.1, n)
        return X, y


def benchmark_rust(X, y, k_values, n_runs=3):
    """Benchmark Rust implementation."""
    times = []
    for _ in range(n_runs):
        gam = mgcv_rust.GAM()

        start = time.perf_counter()
        result = gam.fit_auto(X, y, k=k_values, method='REML', bs='cr', max_iter=10)
        end = time.perf_counter()

        times.append(end - start)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
    }


def benchmark_r(X, y, k_values, n_runs=3):
    """Benchmark R mgcv."""
    n, d = X.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, 'data.csv')
        result_file = os.path.join(tmpdir, 'results.json')

        header = ','.join([f'x{i}' for i in range(d)] + ['y'])
        np.savetxt(data_file, np.column_stack([X, y]), delimiter=',',
                   header=header, comments='')

        terms = ' + '.join([f's(x{i}, k={k}, bs="cr")' for i, k in enumerate(k_values)])
        formula = f'y ~ {terms}'

        r_script = f"""
library(mgcv)

data <- read.csv('{data_file}')

times <- numeric({n_runs})
for (i in 1:{n_runs}) {{
    start_time <- Sys.time()
    fit <- gam({formula}, data=data, method="REML")
    end_time <- Sys.time()
    times[i] <- as.numeric(end_time - start_time)
}}

results <- list(
    mean_time = mean(times),
    std_time = sd(times),
    min_time = min(times)
)

library(jsonlite)
write_json(results, '{result_file}', auto_unbox=TRUE, digits=10)
"""

        r_script_file = os.path.join(tmpdir, 'benchmark.R')
        with open(r_script_file, 'w') as f:
            f.write(r_script)

        try:
            subprocess.run(['Rscript', '--vanilla', r_script_file],
                         check=True, capture_output=True, text=True,
                         timeout=300)

            with open(result_file, 'r') as f:
                results = json.load(f)

            return results
        except Exception as e:
            print(f"R benchmark failed: {e}")
            return None


def main():
    print("=" * 80)
    print("LARGE-N PERFORMANCE ANALYSIS")
    print("=" * 80)
    print()

    # Test configurations: (n, d, k_values)
    configs = [
        # Single variable, large n
        (2000, 1, [20]),
        (5000, 1, [20]),
        (10000, 1, [20]),
        (20000, 1, [20]),

        # Multi-variable, large n
        (2000, 2, [10, 10]),
        (5000, 2, [10, 10]),
        (10000, 2, [10, 10]),

        # High-dimensional
        (2000, 4, [10, 10, 10, 10]),
        (5000, 4, [10, 10, 10, 10]),

        # Very high-dimensional
        (2000, 8, [10]*8),
        (5000, 8, [10]*8),
    ]

    results = []

    for n, d, k_values in configs:
        print(f"\nTesting: n={n}, d={d}, k={k_values}")
        print("-" * 80)

        X, y = generate_test_data(n, d)

        # Rust
        print("  Running Rust benchmark...")
        rust_result = benchmark_rust(X, y, k_values, n_runs=3)

        # R (skip for very large cases to save time)
        if n <= 10000:
            print("  Running R benchmark...")
            r_result = benchmark_r(X, y, k_values, n_runs=3)
        else:
            print("  Skipping R (too slow for n={})".format(n))
            r_result = None

        if r_result:
            speedup = r_result['mean_time'] / rust_result['mean_time']
            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")
            print(f"  R:      {r_result['mean_time']:.4f}s ± {r_result['std_time']:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")

        results.append({
            'n': n,
            'd': d,
            'k': k_values,
            'rust': rust_result,
            'r': r_result,
            'speedup': r_result['mean_time'] / rust_result['mean_time'] if r_result else None
        })

    # Summary by problem size
    print("\n" + "=" * 80)
    print("SUMMARY: Performance by Problem Size")
    print("=" * 80)

    for result in results:
        n = result['n']
        d = result['d']
        speedup = result['speedup']
        rust_time = result['rust']['mean_time']

        if speedup:
            print(f"n={n:5d}, d={d}, Rust={rust_time:7.4f}s, Speedup={speedup:5.2f}x")
        else:
            print(f"n={n:5d}, d={d}, Rust={rust_time:7.4f}s, Speedup=N/A")

    # Save results
    with open('large_n_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: large_n_benchmark_results.json")


if __name__ == '__main__':
    main()
