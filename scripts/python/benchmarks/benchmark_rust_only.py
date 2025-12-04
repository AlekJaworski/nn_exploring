#!/usr/bin/env python3
"""Benchmark Rust implementation only (no R comparison)."""

import numpy as np
import time
import mgcv_rust

def benchmark_rust(n, k, n_runs=5):
    """Benchmark Rust GAM fitting."""
    np.random.seed(42)
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.flatten()) + np.random.normal(0, 0.1, n)

    times = []
    lambdas = []

    for _ in range(n_runs):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        result = gam.fit_auto(x, y, k=[k], method='REML', bs='cr')
        end = time.perf_counter()
        times.append(end - start)
        lambdas.append(result['lambda'][0])

    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_lambda = np.mean(lambdas)

    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'lambda': mean_lambda,
        'times': times
    }

print("=" * 80)
print("RUST PERFORMANCE BENCHMARK")
print("=" * 80)
print()

test_cases = [
    (100, 10),
    (500, 10),
    (1000, 15),
    (2000, 20),
    (5000, 20),
]

results = []
for n, k in test_cases:
    print(f"Testing n={n}, k={k}...")
    result = benchmark_rust(n, k)
    results.append((n, k, result))
    print(f"  Rust:   {result['mean_time']:.4f}s ± {result['std_time']:.4f}s  (λ={result['lambda']:.6f})")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Comparison vs benchmark_final.txt R results:")
print()

# R results from benchmark_final.txt
r_results = {
    100: (0.0518, 1.223084),
    500: (0.0566, 1.413070),
    1000: (0.0771, 7.739346),
    2000: (0.1058, 20.762955),
    5000: (0.1720, 25.699706),
}

print(f"{'n':>6} {'k':>3} {'Rust Time':>12} {'R Time':>12} {'Speedup':>10} {'Rust λ':>12} {'R λ':>12}")
print("-" * 80)
for n, k, result in results:
    rust_time = result['mean_time']
    rust_lambda = result['lambda']
    if n in r_results:
        r_time, r_lambda = r_results[n]
        speedup = r_time / rust_time
        print(f"{n:>6} {k:>3} {rust_time:>10.4f}s {r_time:>10.4f}s {speedup:>9.2f}x {rust_lambda:>11.2f} {r_lambda:>11.2f}")
    else:
        print(f"{n:>6} {k:>3} {rust_time:>10.4f}s {'N/A':>10} {'N/A':>9} {rust_lambda:>11.2f} {'N/A':>11}")
print()

# Calculate performance for n >= 2000
print("Performance for n >= 2000:")
for n, k, result in results:
    if n >= 2000 and n in r_results:
        rust_time = result['mean_time']
        r_time, _ = r_results[n]
        speedup = r_time / rust_time
        pct_diff = (rust_time - r_time) / r_time * 100
        status = "✓ WITHIN 10%" if abs(pct_diff) <= 10 else "✗ NOT WITHIN 10%"
        print(f"  n={n}: {speedup:.2f}x ({pct_diff:+.1f}% vs R) {status}")
