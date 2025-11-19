#!/usr/bin/env python3
"""
Benchmark multidimensional GAM performance vs mgcv for different data sizes.
Tests n=500, 1500, 2500, 5000 with 4D data.
"""
import numpy as np
import mgcv_rust
import time
import subprocess

def benchmark_our_gam(n, k_values=[16, 16, 16, 16]):
    """Benchmark our GAM implementation"""
    np.random.seed(42)

    # Generate 4D test data
    X = np.random.randn(n, 4)
    y = (np.sin(X[:, 0]) + 0.5*X[:, 1]**2 +
         np.cos(X[:, 2]) + 0.3*X[:, 3] +
         0.1*np.random.randn(n))

    # Save data for mgcv comparison
    np.savetxt('/tmp/bench_x.csv', X)
    np.savetxt('/tmp/bench_y.csv', y)

    # Benchmark our implementation
    start = time.time()
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=k_values, method='REML', bs='cr')
    elapsed = time.time() - start

    # Get predictions and compute R²
    preds = gam.predict(X)
    r2 = 1 - np.var(y - preds) / np.var(y)

    lam = result.get('lambda', result.get('all_lambdas', [0]*4))

    return {
        'time': elapsed,
        'r2': r2,
        'lambda': lam,
        'n': n,
        'k': k_values
    }

def benchmark_mgcv():
    """Run mgcv benchmark using R script"""
    try:
        result = subprocess.run(
            ['Rscript', 'benchmark_mgcv.R'],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output
        lines = result.stdout.strip().split('\n')
        time_line = [l for l in lines if 'Time:' in l]
        iter_line = [l for l in lines if 'Iterations:' in l]

        if time_line:
            mgcv_time = float(time_line[0].split(':')[1].strip().replace('s', ''))
        else:
            mgcv_time = None

        if iter_line:
            mgcv_iter = int(iter_line[0].split(':')[1].strip())
        else:
            mgcv_iter = None

        return {
            'time': mgcv_time,
            'iterations': mgcv_iter
        }
    except Exception as e:
        return {'time': None, 'iterations': None, 'error': str(e)}

print("=" * 80)
print("MULTIDIMENSIONAL GAM SCALING BENCHMARK: Our Implementation vs mgcv")
print("=" * 80)
print("\nTest configuration: 4D GAM with k=[16,16,16,16], cubic regression splines")
print()

results = []

for n in [500, 1500, 2500, 5000]:
    print(f"\n{'─' * 80}")
    print(f"n = {n}")
    print(f"{'─' * 80}")

    # Benchmark our implementation
    print("Running our GAM...", end=' ', flush=True)
    our_result = benchmark_our_gam(n)
    print(f"✓ {our_result['time']:.3f}s")

    # Benchmark mgcv
    print("Running mgcv...    ", end=' ', flush=True)
    mgcv_result = benchmark_mgcv()
    if mgcv_result['time'] is not None:
        print(f"✓ {mgcv_result['time']:.3f}s")
    else:
        print(f"✗ Failed: {mgcv_result.get('error', 'Unknown error')}")

    # Compute speedup
    if mgcv_result['time'] is not None and mgcv_result['time'] > 0:
        speedup = mgcv_result['time'] / our_result['time']
    else:
        speedup = None

    # Print comparison
    print(f"\nResults for n={n}:")
    print(f"  Our GAM:")
    print(f"    Time:   {our_result['time']:.3f}s")
    print(f"    R²:     {our_result['r2']:.4f}")
    print(f"    λ:      {our_result['lambda']}")

    if mgcv_result['time'] is not None:
        print(f"  mgcv:")
        print(f"    Time:   {mgcv_result['time']:.3f}s")
        if mgcv_result['iterations'] is not None:
            print(f"    Iters:  {mgcv_result['iterations']}")
        print(f"  Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    results.append({
        'n': n,
        'our_time': our_result['time'],
        'our_r2': our_result['r2'],
        'mgcv_time': mgcv_result['time'],
        'mgcv_iter': mgcv_result.get('iterations'),
        'speedup': speedup
    })

# Print summary table
print(f"\n{'=' * 80}")
print("SUMMARY TABLE")
print(f"{'=' * 80}")
print(f"{'n':>6} │ {'Our Time':>10} │ {'mgcv Time':>10} │ {'Speedup':>10} │ {'R²':>8}")
print(f"{'─' * 6}┼{'─' * 12}┼{'─' * 12}┼{'─' * 12}┼{'─' * 10}")

for r in results:
    speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
    mgcv_time_str = f"{r['mgcv_time']:.3f}s" if r['mgcv_time'] else "N/A"
    print(f"{r['n']:6d} │ {r['our_time']:>9.3f}s │ {mgcv_time_str:>10} │ {speedup_str:>10} │ {r['our_r2']:>8.4f}")

print(f"{'=' * 80}")

# Print conclusions
print("\nCONCLUSIONS:")
if all(r['speedup'] for r in results):
    avg_speedup = np.mean([r['speedup'] for r in results if r['speedup']])
    print(f"✓ Average speedup over mgcv: {avg_speedup:.2f}x")
    print(f"✓ Fit quality: R² > {min(r['our_r2'] for r in results):.2f} across all test sizes")
    print(f"✓ Scaling: {results[0]['our_time']:.3f}s → {results[-1]['our_time']:.3f}s ({results[-1]['our_time']/results[0]['our_time']:.1f}x for 10x more data)")
else:
    print("⚠ Some mgcv benchmarks failed - partial results shown")

print()
