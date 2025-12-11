#!/usr/bin/env python3
"""Benchmark multi-dimensional GAM performance (Rust vs R)."""

import numpy as np
import time
import mgcv_rust

# Check if rpy2 is available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    HAS_R = True
    mgcv = importr('mgcv')
    stats = importr('stats')
except ImportError:
    HAS_R = False
    print("Warning: rpy2 not available, skipping R benchmarks")

def generate_multidim_data(n, n_dims, seed=42):
    """Generate synthetic multi-dimensional data.

    True function: y = sin(2πx₁) + 0.5*cos(3πx₂) + 0.3*x₃² + noise
                     + smaller contributions from other dimensions
    """
    np.random.seed(seed)

    # Generate predictors in [0, 1]
    X = np.random.uniform(0, 1, size=(n, n_dims))

    # Generate true function
    y = np.zeros(n)

    # Main effects
    if n_dims >= 1:
        y += np.sin(2 * np.pi * X[:, 0])
    if n_dims >= 2:
        y += 0.5 * np.cos(3 * np.pi * X[:, 1])
    if n_dims >= 3:
        y += 0.3 * (X[:, 2] ** 2)
    if n_dims >= 4:
        y += 0.2 * np.exp(-5 * (X[:, 3] - 0.5) ** 2)

    # Smaller contributions from remaining dimensions
    for i in range(4, n_dims):
        y += 0.1 * np.sin(np.pi * X[:, i])

    # Add noise
    y += np.random.normal(0, 0.2, n)

    return X, y


def benchmark_rust_multidim(X, y, k, n_runs=3, verbose=False):
    """Benchmark Rust multi-dimensional GAM."""
    n, n_dims = X.shape

    times = []
    lambdas_list = []
    iteration_counts = []

    for run in range(n_runs):
        gam = mgcv_rust.GAM()

        start = time.perf_counter()
        result = gam.fit_auto(X, y, k=[k] * n_dims, method='REML', bs='cr', max_iter=100)
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        lambdas_list.append(result['lambda'])

        # Try to get iteration count if available
        if 'iterations' in result:
            iteration_counts.append(result['iterations'])

        if verbose:
            print(f"  Run {run+1}: {elapsed:.4f}s, λ={result['lambda'][:3]}...")

    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_lambdas = np.mean(lambdas_list, axis=0)

    result = {
        'mean_time': mean_time,
        'std_time': std_time,
        'lambdas': mean_lambdas,
        'times': times,
        'all_lambdas': lambdas_list,
    }

    if iteration_counts:
        result['iterations'] = np.mean(iteration_counts)

    return result


def benchmark_r_multidim(X, y, k, n_runs=3, verbose=False):
    """Benchmark R's mgcv multi-dimensional GAM."""
    if not HAS_R:
        return None

    n, n_dims = X.shape

    # Build GAM formula: y ~ s(x1, bs='cr', k=k) + s(x2, bs='cr', k=k) + ...
    formula_parts = ["y ~ "]
    for i in range(n_dims):
        if i > 0:
            formula_parts.append(" + ")
        formula_parts.append(f"s(x{i+1}, bs='cr', k={k})")
    formula_str = "".join(formula_parts)

    times = []
    lambdas_list = []
    iteration_counts = []

    for run in range(n_runs):
        # Create R data frame
        r_data = ro.r['data.frame']
        data_dict = {'y': y}
        for i in range(n_dims):
            data_dict[f'x{i+1}'] = X[:, i]
        df = r_data(**data_dict)

        start = time.perf_counter()
        fit = mgcv.gam(ro.r(f'as.formula("{formula_str}")'), data=df, method='REML')
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        # Extract smoothing parameters
        sp = np.array(fit.rx2('sp'))
        lambdas_list.append(sp)

        # Try to get iteration count
        try:
            outer_info = fit.rx2('outer.info')
            if outer_info is not None:
                iter_count = np.array(outer_info.rx2('iter'))[0]
                iteration_counts.append(iter_count)
        except:
            pass

        if verbose:
            print(f"  Run {run+1}: {elapsed:.4f}s, λ={sp[:3]}...")

    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_lambdas = np.mean(lambdas_list, axis=0)

    result = {
        'mean_time': mean_time,
        'std_time': std_time,
        'lambdas': mean_lambdas,
        'times': times,
        'all_lambdas': lambdas_list,
    }

    if iteration_counts:
        result['iterations'] = np.mean(iteration_counts)

    return result


def main():
    print("=" * 80)
    print("MULTI-DIMENSIONAL GAM BENCHMARK")
    print("=" * 80)
    print()

    # Test configurations
    configs = [
        (6000, 8, 10),
        (6000, 10, 10),
    ]

    for n, n_dims, k in configs:
        print(f"Configuration: n={n}, dimensions={n_dims}, k={k}")
        print("-" * 80)

        # Generate data
        print(f"Generating {n_dims}-dimensional data...")
        X, y = generate_multidim_data(n, n_dims)
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        print()

        # Benchmark Rust
        print("Benchmarking Rust (3 runs)...")
        rust_result = benchmark_rust_multidim(X, y, k, n_runs=3, verbose=True)
        print(f"  Mean: {rust_result['mean_time']:.4f}s ± {rust_result['std_time']:.4f}s")
        print(f"  Lambdas: {rust_result['lambdas']}")
        if 'iterations' in rust_result:
            print(f"  Iterations: {rust_result['iterations']:.1f}")
        print()

        # Benchmark R
        if HAS_R:
            print("Benchmarking R's mgcv (3 runs)...")
            r_result = benchmark_r_multidim(X, y, k, n_runs=3, verbose=True)
            print(f"  Mean: {r_result['mean_time']:.4f}s ± {r_result['std_time']:.4f}s")
            print(f"  Lambdas: {r_result['lambdas']}")
            if 'iterations' in r_result:
                print(f"  Iterations: {r_result['iterations']:.1f}")
            print()

            # Compare
            print("COMPARISON:")
            print("-" * 80)
            speedup = r_result['mean_time'] / rust_result['mean_time']
            pct_diff = (rust_result['mean_time'] - r_result['mean_time']) / r_result['mean_time'] * 100

            print(f"  Rust time:  {rust_result['mean_time']:.4f}s")
            print(f"  R time:     {r_result['mean_time']:.4f}s")
            print(f"  Speedup:    {speedup:.2f}x")
            print(f"  Difference: {pct_diff:+.1f}%")

            if 'iterations' in rust_result and 'iterations' in r_result:
                iter_ratio = rust_result['iterations'] / r_result['iterations']
                time_per_iter_rust = rust_result['mean_time'] / rust_result['iterations']
                time_per_iter_r = r_result['mean_time'] / r_result['iterations']
                per_iter_ratio = time_per_iter_rust / time_per_iter_r

                print()
                print("ITERATION ANALYSIS:")
                print(f"  Rust iterations:     {rust_result['iterations']:.1f}")
                print(f"  R iterations:        {r_result['iterations']:.1f}")
                print(f"  Iteration ratio:     {iter_ratio:.2f}x")
                print(f"  Rust time/iter:      {time_per_iter_rust*1000:.1f}ms")
                print(f"  R time/iter:         {time_per_iter_r*1000:.1f}ms")
                print(f"  Per-iter speed:      {per_iter_ratio:.2f}x")
                print()

                if per_iter_ratio > 1.2:
                    print("  → BOTTLENECK: Per-iteration time is slower")
                elif iter_ratio > 1.2:
                    print("  → BOTTLENECK: Number of iterations is higher")
                else:
                    print("  → Performance is roughly equivalent")

            # Lambda comparison
            print()
            print("LAMBDA COMPARISON:")
            lambda_diffs = np.abs(rust_result['lambdas'] - r_result['lambdas']) / r_result['lambdas'] * 100
            for i, (r_lam, rust_lam, diff) in enumerate(zip(r_result['lambdas'], rust_result['lambdas'], lambda_diffs)):
                status = "✓" if diff < 5 else "⚠"
                print(f"  Dim {i+1}: Rust={rust_lam:8.2f}, R={r_lam:8.2f}, diff={diff:5.1f}% {status}")

        print()
        print("=" * 80)
        print()


if __name__ == '__main__':
    main()
