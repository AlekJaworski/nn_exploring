#!/usr/bin/env python3
"""Benchmark Fellner-Schall vs Newton vs R bam()."""

import mgcv_rust
import numpy as np
import time
import sys

def bench(n, d, k, algo, seed=123, runs=3):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = sum(np.sin(2 * np.pi * X[:, i]) for i in range(d)) + np.random.normal(0, 0.3, n)

    times = []
    for r in range(runs):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        result = gam.fit_auto_optimized(
            X, y, k=[k] * d, method='REML', bs='cr', max_iter=100, algorithm=algo
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    return np.median(times) * 1000


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

# Fresh R bam() times measured on this machine
bam_times = {
    (1000, 1, 10): 31, (1000, 2, 10): 30, (1000, 4, 10): 57,
    (2000, 1, 10): 25, (2000, 2, 10): 36, (2000, 4, 10): 78, (2000, 8, 8): 99,
    (5000, 1, 10): 28, (5000, 2, 10): 43, (5000, 4, 8): 91, (5000, 8, 8): 205,
    (10000, 1, 10): 41, (10000, 2, 10): 56, (10000, 4, 8): 94,
}

print("=" * 85)
print("  FELLNER-SCHALL vs NEWTON vs R bam()")
print("=" * 85)
print()
header = "{:<22s} | {:>9s} | {:>9s} | {:>6s} | {:>10s}".format(
    "Config", "Newton", "F-S", "bam", "FS vs bam"
)
print(header)
print("-" * len(header))

fs_wins = 0
fs_losses = 0

for n, d, k in configs:
    sys.stdout.flush()
    newton_ms = bench(n, d, k, 'newton')
    fs_ms = bench(n, d, k, 'fellner-schall')
    bam_ms = bam_times.get((n, d, k), 0)

    if bam_ms > 0:
        ratio = fs_ms / bam_ms
        if ratio < 1.0:
            tag = "{:.1f}x WIN".format(1 / ratio)
            fs_wins += 1
        else:
            tag = "{:.1f}x LOSE".format(ratio)
            fs_losses += 1
    else:
        tag = "?"

    print("{:<22s} | {:>9.1f} | {:>9.1f} | {:>6d} | {:>10s}".format(
        "n={:>5d}, d={}, k={:>2d}".format(n, d, k),
        newton_ms, fs_ms, bam_ms, tag
    ))

print()
print("Fellner-Schall vs bam(): {} wins, {} losses".format(fs_wins, fs_losses))
