#!/usr/bin/env python3
"""
Performance comparison between mgcvrust (Rust with PyO3 bindings) and R's mgcv.

This script benchmarks:
1. Single-variable GAMs with various problem sizes
2. Multi-variable GAMs
3. Different basis types (CR splines)
"""

import numpy as np
import subprocess
import time
import json
import tempfile
import os
from pathlib import Path

# Import the Rust module
import mgcv_rust


def generate_test_data(n, seed=42):
    """Generate synthetic test data."""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
    return x, y


def generate_multidim_data(n, d, seed=42):
    """Generate multidimensional test data."""
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))

    # y = sum of smooth functions of each variable
    y = np.zeros(n)
    for i in range(d):
        y += np.sin(2 * np.pi * X[:, i])

    y += np.random.normal(0, 0.1, n)
    return X, y


def benchmark_rust_1d(x, y, k=10, method='REML', n_runs=5):
    """Benchmark Rust implementation for 1D GAM."""
    X = x.reshape(-1, 1)

    times = []
    for _ in range(n_runs):
        gam = mgcv_rust.GAM()

        start = time.perf_counter()
        result = gam.fit_auto(X, y, k=[k], method=method, bs='cr', max_iter=10)
        end = time.perf_counter()

        times.append(end - start)

    # Get final fit for results
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[k], method=method, bs='cr', max_iter=10)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'lambda': float(result['lambda'][0]),
        'deviance': float(result.get('deviance', np.nan)),
        'fitted': result.get('fitted', False)
    }


def benchmark_rust_multidim(X, y, k_values, method='REML', n_runs=3):
    """Benchmark Rust implementation for multidimensional GAM."""
    times = []
    for _ in range(n_runs):
        gam = mgcv_rust.GAM()

        start = time.perf_counter()
        result = gam.fit_auto(X, y, k=k_values, method=method, bs='cr', max_iter=10)
        end = time.perf_counter()

        times.append(end - start)

    # Get final fit for results
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=k_values, method=method, bs='cr', max_iter=10)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'lambdas': [float(l) for l in result['lambda']],
        'deviance': float(result.get('deviance', np.nan)),
        'fitted': result.get('fitted', False)
    }


def benchmark_r_1d(x, y, k=10, method='REML', n_runs=5):
    """Benchmark R mgcv for 1D GAM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data
        data_file = os.path.join(tmpdir, 'data.csv')
        result_file = os.path.join(tmpdir, 'results.json')

        np.savetxt(data_file, np.column_stack([x, y]), delimiter=',',
                   header='x,y', comments='')

        # Create R script
        r_script = f"""
library(mgcv)

# Read data
data <- read.csv('{data_file}')
x <- data$x
y <- data$y

# Benchmark
times <- numeric({n_runs})
for (i in 1:{n_runs}) {{
    start_time <- Sys.time()
    fit <- gam(y ~ s(x, k={k}, bs="cr"), method="{method}")
    end_time <- Sys.time()
    times[i] <- as.numeric(end_time - start_time)
}}

# Final fit for results
fit <- gam(y ~ s(x, k={k}, bs="cr"), method="{method}")

# Save results
results <- list(
    mean_time = mean(times),
    std_time = sd(times),
    min_time = min(times),
    lambda = fit$sp[[1]],
    deviance = deviance(fit),
    converged = fit$converged
)

library(jsonlite)
write_json(results, '{result_file}', auto_unbox=TRUE, digits=10)
"""

        # Run R script
        r_script_file = os.path.join(tmpdir, 'benchmark.R')
        with open(r_script_file, 'w') as f:
            f.write(r_script)

        try:
            subprocess.run(['Rscript', '--vanilla', r_script_file],
                         check=True, capture_output=True, text=True,
                         timeout=60)

            # Read results
            with open(result_file, 'r') as f:
                results = json.load(f)

            return results
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"R benchmark failed: {e}")
            return None


def benchmark_r_multidim(X, y, k_values, method='REML', n_runs=3):
    """Benchmark R mgcv for multidimensional GAM."""
    n, d = X.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data
        data_file = os.path.join(tmpdir, 'data.csv')
        result_file = os.path.join(tmpdir, 'results.json')

        # Create DataFrame with column names
        header = ','.join([f'x{i}' for i in range(d)] + ['y'])
        np.savetxt(data_file, np.column_stack([X, y]), delimiter=',',
                   header=header, comments='')

        # Create formula
        terms = ' + '.join([f's(x{i}, k={k}, bs="cr")' for i, k in enumerate(k_values)])
        formula = f'y ~ {terms}'

        # Create R script
        r_script = f"""
library(mgcv)

# Read data
data <- read.csv('{data_file}')

# Benchmark
times <- numeric({n_runs})
for (i in 1:{n_runs}) {{
    start_time <- Sys.time()
    fit <- gam({formula}, data=data, method="{method}")
    end_time <- Sys.time()
    times[i] <- as.numeric(end_time - start_time)
}}

# Final fit for results
fit <- gam({formula}, data=data, method="{method}")

# Save results
results <- list(
    mean_time = mean(times),
    std_time = sd(times),
    min_time = min(times),
    lambdas = as.numeric(fit$sp),
    deviance = deviance(fit),
    converged = fit$converged
)

library(jsonlite)
write_json(results, '{result_file}', auto_unbox=FALSE, digits=10)
"""

        # Run R script
        r_script_file = os.path.join(tmpdir, 'benchmark.R')
        with open(r_script_file, 'w') as f:
            f.write(r_script)

        try:
            result = subprocess.run(['Rscript', '--vanilla', r_script_file],
                         check=True, capture_output=True, text=True,
                         timeout=120)

            # Read results
            with open(result_file, 'r') as f:
                results = json.load(f)

            return results
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"R benchmark failed: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
            return None


def run_benchmarks():
    """Run all benchmarks and display results."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON: mgcvrust (Rust) vs mgcv (R)")
    print("=" * 80)
    print()

    results = {}

    # Test 1: Single variable, various sizes
    print("Test 1: Single Variable GAMs - Various Problem Sizes")
    print("-" * 80)

    test_sizes = [
        (100, 10),
        (500, 10),
        (1000, 15),
        (2000, 20),
        (5000, 20),
    ]

    results['single_var'] = []

    for n, k in test_sizes:
        print(f"\nProblem size: n={n}, k={k}")
        x, y = generate_test_data(n)

        # Benchmark Rust
        print("  Running Rust benchmark...")
        rust_result = benchmark_rust_1d(x, y, k=k, n_runs=5)

        # Benchmark R
        print("  Running R benchmark...")
        r_result = benchmark_r_1d(x, y, k=k, n_runs=5)

        if r_result is not None:
            speedup = r_result['mean_time'] / rust_result['mean_time']

            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s  (λ={rust_result['lambda']:.6f})")
            print(f"  R:      {r_result['mean_time']:.4f}s ± {r_result['std_time']:.4f}s  (λ={r_result['lambda']:.6f})")
            print(f"  Speedup: {speedup:.2f}x")

            results['single_var'].append({
                'n': n,
                'k': k,
                'rust': rust_result,
                'r': r_result,
                'speedup': speedup
            })
        else:
            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")
            print(f"  R:      FAILED")

            results['single_var'].append({
                'n': n,
                'k': k,
                'rust': rust_result,
                'r': None,
                'speedup': None
            })

    # Test 2: Multi-variable GAMs
    print("\n" + "=" * 80)
    print("Test 2: Multi-Variable GAMs")
    print("-" * 80)

    multi_tests = [
        (500, 2, [10, 10]),
        (1000, 3, [10, 10, 10]),
        (2000, 4, [10, 10, 10, 10]),
    ]

    results['multi_var'] = []

    for n, d, k_values in multi_tests:
        print(f"\nProblem size: n={n}, d={d}, k={k_values}")
        X, y = generate_multidim_data(n, d)

        # Benchmark Rust
        print("  Running Rust benchmark...")
        rust_result = benchmark_rust_multidim(X, y, k_values, n_runs=3)

        # Benchmark R
        print("  Running R benchmark...")
        r_result = benchmark_r_multidim(X, y, k_values, n_runs=3)

        if r_result is not None:
            # Convert to float in case R returned arrays
            r_mean = float(r_result['mean_time']) if not isinstance(r_result['mean_time'], list) else float(r_result['mean_time'][0])
            r_std = float(r_result['std_time']) if not isinstance(r_result['std_time'], list) else float(r_result['std_time'][0])

            speedup = r_mean / rust_result['mean_time']

            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")
            print(f"  R:      {r_mean:.4f}s ± {r_std:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")

            results['multi_var'].append({
                'n': n,
                'd': d,
                'k_values': k_values,
                'rust': rust_result,
                'r': r_result,
                'speedup': speedup
            })
        else:
            print(f"  Rust:   {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")
            print(f"  R:      FAILED")

            results['multi_var'].append({
                'n': n,
                'd': d,
                'k_values': k_values,
                'rust': rust_result,
                'r': None,
                'speedup': None
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average speedup for single variable
    single_var_speedups = [r['speedup'] for r in results['single_var'] if r['speedup'] is not None]
    if single_var_speedups:
        print(f"\nSingle Variable GAMs:")
        print(f"  Average speedup: {np.mean(single_var_speedups):.2f}x")
        print(f"  Min speedup:     {np.min(single_var_speedups):.2f}x")
        print(f"  Max speedup:     {np.max(single_var_speedups):.2f}x")

    # Average speedup for multi-variable
    multi_var_speedups = [r['speedup'] for r in results['multi_var'] if r['speedup'] is not None]
    if multi_var_speedups:
        print(f"\nMulti-Variable GAMs:")
        print(f"  Average speedup: {np.mean(multi_var_speedups):.2f}x")
        print(f"  Min speedup:     {np.min(multi_var_speedups):.2f}x")
        print(f"  Max speedup:     {np.max(multi_var_speedups):.2f}x")

    # Save results
    output_file = 'rust_vs_r_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print()


if __name__ == '__main__':
    # Check if R is available
    try:
        subprocess.run(['Rscript', '--version'], capture_output=True, check=True)
        print("R detected. Running full benchmarks...")
        print()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: R not found. Will only benchmark Rust implementation.")
        print()

    run_benchmarks()
