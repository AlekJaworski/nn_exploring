#!/usr/bin/env python3
"""Comprehensive benchmark: Rust (Newton + FS) vs R (gam + bam).

Runs fresh timings for all four methods across standard configs.
Uses 5 runs with 1 warm-up for stable medians.
Requires: mgcv_rust, rpy2, R with mgcv installed.
"""

import mgcv_rust
import numpy as np
import time
import sys

# Try to import R with new rpy2 API
try:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import get_conversion, localconverter
    from rpy2.robjects import numpy2ri

    _base_cv = get_conversion()
    _np_cv = numpy2ri.converter
    _r_converter = _base_cv + _np_cv

    with localconverter(_r_converter):
        ro.r('suppressMessages(library(mgcv))')
    HAS_R = True
except Exception as e:
    print(f"Warning: R not available ({e}), will skip R benchmarks")
    HAS_R = False

WARMUP = 1
RUNS = 5  # median of 5 (after 1 warmup)


def gen_data(n, d, k, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (n, d))
    y = np.zeros(n)
    for i in range(d):
        y += np.sin(2 * np.pi * X[:, i])
    y += np.random.normal(0, 0.3, n)
    return X, y


def bench_rust(X, y, d, k, algo, warmup=WARMUP, runs=RUNS):
    """Benchmark mgcv_rust with given algorithm."""
    times = []
    deviance = None
    for i in range(warmup + runs):
        gam = mgcv_rust.GAM()
        t0 = time.perf_counter()
        result = gam.fit_auto_optimized(
            X, y, k=[k] * d, method='REML', bs='cr', max_iter=100, algorithm=algo
        )
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            times.append(elapsed)
        deviance = result.get('deviance', None)
    return np.median(times) * 1000, deviance


def bench_r_gam(X, y, d, k, warmup=WARMUP, runs=RUNS):
    """Benchmark R's gam() with REML."""
    if not HAS_R:
        return None, None

    with localconverter(_r_converter):
        ro.globalenv['X_r'] = X
        ro.globalenv['y_r'] = y

        smooth_terms = [f"s(X{i+1}, k={k}, bs='cr')" for i in range(d)]
        formula_str = "y_r ~ " + " + ".join(smooth_terms)

        df_cols = "data.frame(y_r=y_r"
        for i in range(d):
            df_cols += f", X{i+1}=X_r[,{i+1}]"
        df_cols += ")"
        ro.r(f'df <- {df_cols}')

        total = warmup + runs
        ro.r(f'''
            times <- numeric({total})
            for (i in 1:{total}) {{
                t0 <- proc.time()
                fit <- gam({formula_str}, data=df, method="REML")
                times[i] <- (proc.time() - t0)[3]
            }}
        ''')
        all_times = np.array(ro.r('times'))
        timed = all_times[warmup:]
        deviance = ro.r('deviance(fit)')[0]

    return np.median(timed) * 1000, deviance


def bench_r_bam(X, y, d, k, warmup=WARMUP, runs=RUNS):
    """Benchmark R's bam() with fREML."""
    if not HAS_R:
        return None, None

    with localconverter(_r_converter):
        ro.globalenv['X_r'] = X
        ro.globalenv['y_r'] = y

        smooth_terms = [f"s(X{i+1}, k={k}, bs='cr')" for i in range(d)]
        formula_str = "y_r ~ " + " + ".join(smooth_terms)

        df_cols = "data.frame(y_r=y_r"
        for i in range(d):
            df_cols += f", X{i+1}=X_r[,{i+1}]"
        df_cols += ")"
        ro.r(f'df <- {df_cols}')

        total = warmup + runs
        ro.r(f'''
            times <- numeric({total})
            for (i in 1:{total}) {{
                t0 <- proc.time()
                fit <- bam({formula_str}, data=df, method="fREML")
                times[i] <- (proc.time() - t0)[3]
            }}
        ''')
        all_times = np.array(ro.r('times'))
        timed = all_times[warmup:]
        deviance = ro.r('deviance(fit)')[0]

    return np.median(timed) * 1000, deviance


configs = [
    (500,  1, 10),
    (1000, 1, 10),
    (1000, 2, 10),
    (1000, 4, 10),
    (2000, 1, 10),
    (2000, 2, 10),
    (2000, 4, 10),
    (2000, 8,  8),
    (5000, 1, 10),
    (5000, 2, 10),
    (5000, 4,  8),
    (5000, 8,  8),
    (10000, 1, 10),
    (10000, 2, 10),
    (10000, 4,  8),
]

print()
print("=" * 140)
print(f"  COMPREHENSIVE BENCHMARK: Rust (Newton / FS) vs R (gam / bam)")
print(f"  {WARMUP} warmup + median of {RUNS} runs, seed=42")
print("=" * 140)
print()

header = (
    f"{'Config':<22s} | "
    f"{'Newton':>8s} | "
    f"{'FS':>8s} | "
    f"{'R gam':>8s} | "
    f"{'R bam':>8s} | "
    f"{'best':>10s} | "
    f"{'best/bam':>10s} | "
    f"{'best/gam':>10s}"
)
print(header)
print("-" * len(header))

best_vs_bam_wins = 0
best_vs_gam_wins = 0
total_configs = 0

rust_best_times = []
bam_times_list = []
gam_times_list = []

for n, d, k in configs:
    sys.stdout.flush()
    X, y = gen_data(n, d, k)

    # Rust benchmarks
    newton_ms, newton_dev = bench_rust(X, y, d, k, 'newton')
    fs_ms, fs_dev = bench_rust(X, y, d, k, 'fellner-schall')

    # R benchmarks
    gam_ms, gam_dev = bench_r_gam(X, y, d, k)
    bam_ms, bam_dev = bench_r_bam(X, y, d, k)

    # Best Rust time
    best_ms = min(newton_ms, fs_ms)
    best_label = "N" if newton_ms <= fs_ms else "FS"

    total_configs += 1

    # Compute ratios vs bam
    if bam_ms and bam_ms > 0:
        best_bam_ratio = best_ms / bam_ms
        if best_bam_ratio < 1.0:
            best_bam_str = f"{1/best_bam_ratio:.1f}x WIN"
            best_vs_bam_wins += 1
        else:
            best_bam_str = f"{best_bam_ratio:.1f}x LOSE"
        rust_best_times.append(best_ms)
        bam_times_list.append(bam_ms)
    else:
        best_bam_str = "?"

    # Compute ratios vs gam
    if gam_ms and gam_ms > 0:
        best_gam_ratio = best_ms / gam_ms
        if best_gam_ratio < 1.0:
            best_gam_str = f"{1/best_gam_ratio:.1f}x WIN"
            best_vs_gam_wins += 1
        else:
            best_gam_str = f"{best_gam_ratio:.1f}x LOSE"
        gam_times_list.append(gam_ms)
    else:
        best_gam_str = "?"

    config_str = f"n={n:>5d}, d={d}, k={k:>2d}"
    gam_str = f"{gam_ms:.1f}" if gam_ms else "N/A"
    bam_str = f"{bam_ms:.1f}" if bam_ms else "N/A"
    best_str = f"{best_ms:.1f}({best_label})"

    print(
        f"{config_str:<22s} | "
        f"{newton_ms:>8.1f} | "
        f"{fs_ms:>8.1f} | "
        f"{gam_str:>8s} | "
        f"{bam_str:>8s} | "
        f"{best_str:>10s} | "
        f"{best_bam_str:>10s} | "
        f"{best_gam_str:>10s}"
    )

print()
print(f"Best Rust vs bam(): {best_vs_bam_wins} wins / {total_configs} configs")
print(f"Best Rust vs gam(): {best_vs_gam_wins} wins / {total_configs} configs")
if rust_best_times and bam_times_list:
    geomean = np.exp(np.mean(np.log(np.array(rust_best_times) / np.array(bam_times_list))))
    print(f"Geometric mean (best Rust / bam): {geomean:.2f}x")
if rust_best_times and gam_times_list:
    geomean_gam = np.exp(np.mean(np.log(np.array(rust_best_times[:len(gam_times_list)]) / np.array(gam_times_list))))
    print(f"Geometric mean (best Rust / gam): {geomean_gam:.2f}x")
print()
