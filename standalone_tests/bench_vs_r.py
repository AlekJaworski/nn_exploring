#!/usr/bin/env python3
"""Benchmark mgcv_rust vs historical R mgcv (gam/bam) times."""

import mgcv_rust
import numpy as np
import time
import sys

def bench(n, d, k, seed=123, runs=3):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)

    times = []
    result = None
    for r in range(runs):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        result = gam.fit_auto_optimized(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    median_ms = np.median(times) * 1000
    iters = result.get('iterations', '?') if result else '?'
    return median_ms, iters


# Historical bam/gam times from COMPREHENSIVE_BENCHMARK_RESULTS.md
# These were measured on the same machine with R 4.3.3, mgcv 1.9-1
hist = {
    (1000, 1, 10): (22.0, 65.0),
    (1000, 2, 10): (25.0, 50.0),
    (1000, 4, 10): (50.0, 127.0),
    (2000, 1, 10): (17.0, 42.0),
    (2000, 2, 10): (27.0, 89.0),
    (2000, 4, 10): (61.0, 156.0),
    (2000, 8,  8): (95.0, 420.0),
    (5000, 1, 10): (20.0, 81.0),
    (5000, 2, 10): (29.0, 139.0),
    (5000, 4,  8): (46.0, 209.0),
    (5000, 8,  8): (174.0, 858.0),
    (10000, 1, 10): (29.0, 166.0),
    (10000, 2, 10): (42.0, 242.0),
    (10000, 4,  8): (67.0, 445.0),
}

configs = [
    (1000, 1, 10),
    (1000, 2, 10),
    (1000, 4, 10),
    (2000, 1, 10),
    (2000, 2, 10),
    (2000, 4, 10),
    (2000, 8, 8),
    (5000, 1, 10),
    (5000, 2, 10),
    (5000, 4, 8),
    (5000, 8, 8),
    (10000, 1, 10),
    (10000, 2, 10),
    (10000, 4, 8),
]

# Also run fresh R benchmarks if requested
run_r = "--run-r" in sys.argv

print("=" * 90)
print("  RUST vs R PERFORMANCE COMPARISON")
print("  (R times from historical benchmarks unless --run-r specified)")
print("=" * 90)
print()

header = f"{'Config':<22s} | {'Rust(ms)':>9s} | {'iters':>5s} | {'bam(ms)':>8s} | {'vs bam':>8s} | {'gam(ms)':>8s} | {'vs gam':>8s}"
print(header)
print("-" * len(header))

wins_bam = 0
losses_bam = 0
rust_times = []
bam_times = []

for n, d, k in configs:
    sys.stdout.flush()
    ms, iters = bench(n, d, k)
    bam_ms, gam_ms = hist.get((n, d, k), (None, None))

    if bam_ms:
        ratio_bam = ms / bam_ms
        if ratio_bam < 1.0:
            vs_bam = f"{1/ratio_bam:.1f}x WIN"
            wins_bam += 1
        else:
            vs_bam = f"{ratio_bam:.1f}x LOSE"
            losses_bam += 1
        rust_times.append(ms)
        bam_times.append(bam_ms)
    else:
        vs_bam = "?"

    if gam_ms:
        ratio_gam = ms / gam_ms
        if ratio_gam < 1.0:
            vs_gam = f"{1/ratio_gam:.1f}x WIN"
        else:
            vs_gam = f"{ratio_gam:.1f}x LOSE"
    else:
        vs_gam = "?"

    bam_str = f"{bam_ms:.0f}" if bam_ms else "?"
    gam_str = f"{gam_ms:.0f}" if gam_ms else "?"

    config_str = f"n={n:>5d}, d={d}, k={k:>2d}"
    print(f"{config_str:<22s} | {ms:>9.1f} | {str(iters):>5s} | {bam_str:>8s} | {vs_bam:>8s} | {gam_str:>8s} | {vs_gam:>8s}")

print()
print(f"vs bam(): {wins_bam} wins, {losses_bam} losses out of {wins_bam + losses_bam}")
if rust_times:
    geomean_ratio = np.exp(np.mean(np.log(np.array(rust_times) / np.array(bam_times))))
    print(f"Geometric mean ratio (Rust/bam): {geomean_ratio:.2f}x  ({'Rust faster' if geomean_ratio < 1 else 'bam faster'})")
print()

if run_r:
    print("Running fresh R benchmarks...")
    import subprocess
    r_script = "scripts/r/benchmarks/benchmark_comprehensive.R"
    result = subprocess.run(["Rscript", r_script], capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print("R stderr:", result.stderr[:500])
