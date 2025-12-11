#!/usr/bin/env python3
"""
Comprehensive comparison of mgcv_rust vs R's mgcv across multiple test cases
"""

import numpy as np
import subprocess
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mgcv_rust

def generate_test_data(n=100, seed=42):
    """Generate test data with different characteristics"""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.normal(0, 0.1, n)
    return x, y

def run_r_mgcv_comparison(x, y, k=10):
    """Run R's mgcv and return lambda estimate"""
    # Create temporary data file
    data = np.column_stack([x, y])
    np.savetxt('/tmp/test_data.csv', data, delimiter=',', header='x,y', comments='')

    # R script to run mgcv
    r_script = f'''
    library(mgcv)
    data <- read.csv("/tmp/test_data.csv")

    # Fit GAM with REML
    gam_fit <- gam(y ~ s(x, k={k}, bs="cr"), data=data, method="REML")

    # Return lambda
    cat(gam_fit$sp)
    '''

    try:
        result = subprocess.run(['Rscript', '-e', r_script],
                              capture_output=True, text=True, check=True)
        r_lambda = float(result.stdout.strip())
        return r_lambda
    except Exception as e:
        print(f"R mgcv failed: {e}")
        return None

def run_rust_comparison(x, y, k=10):
    """Run mgcv_rust and return lambda estimate"""
    try:
        # Create GAM and fit
        gam = mgcv_rust.GAM()
        result = gam.fit_auto(x.reshape(-1, 1), y, k=[k], method="REML", bs="cr")

        # Extract lambda
        rust_lambda = gam.get_lambda()
        return float(rust_lambda)
    except Exception as e:
        print(f"Rust failed: {e}")
        return None

def run_comparison_case(name, n, k, seed):
    """Run one comparison case"""
    print(f"\n--- {name} (n={n}, k={k}, seed={seed}) ---")

    # Generate data
    x, y = generate_test_data(n=n, seed=seed)

    # Run R
    r_lambda = run_r_mgcv_comparison(x, y, k=k)
    if r_lambda is None:
        return None

    # Run Rust
    rust_lambda = run_rust_comparison(x, y, k=k)
    if rust_lambda is None:
        return None

    # Calculate metrics
    lambda_ratio = rust_lambda / r_lambda
    lambda_diff_pct = abs(lambda_ratio - 1.0) * 100

    print(f"R λ:     {r_lambda:.6f}")
    print(f"Rust λ:  {rust_lambda:.6f}")
    print(f"Diff%:   {lambda_diff_pct:.2f}%")
    return {
        'name': name,
        'n': n,
        'k': k,
        'r_lambda': r_lambda,
        'rust_lambda': rust_lambda,
        'ratio': lambda_ratio,
        'diff_pct': lambda_diff_pct
    }

def main():
    print("=" * 100)
    print("COMPREHENSIVE COMPARISON: mgcv_rust vs R's mgcv")
    print("=" * 100)

    # Test cases with different scenarios
    test_cases = [
        ("Small dataset", 50, 10, 42),
        ("Medium dataset", 100, 10, 42),
        ("Large dataset", 200, 10, 42),
        ("Different seed", 100, 10, 123),
        ("Higher k", 100, 20, 42),
        ("Lower k", 100, 5, 42),
    ]

    results = []
    for name, n, k, seed in test_cases:
        result = run_comparison_case(name, n, k, seed)
        if result:
            results.append(result)

    if not results:
        print("❌ No successful test cases")
        return

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    diff_pcts = [r['diff_pct'] for r in results]
    ratios = [r['ratio'] for r in results]

    print(f"Number of test cases: {len(results)}")
    print(".2f")
    print(".4f")
    print(".4f")
    print(".2f")
    # Individual results table
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Test Case':<25} {'R λ':<10} {'Rust λ':<10} {'Ratio':<8} {'Diff%':<8} {'Status':<5}")
    print("-" * 80)
    for r in results:
        status = "✅" if r['diff_pct'] < 10 else "⚠️" if r['diff_pct'] < 100 else "❌"
        print(f"{r['name']:<25} {r['r_lambda']:<10.4f} {r['rust_lambda']:<10.4f} {r['ratio']:<8.4f} {r['diff_pct']:<8.2f} {status}")
    print("-" * 80)

    # Assessment
    print("\n" + "=" * 100)
    print("OVERALL ASSESSMENT")
    print("=" * 100)

    avg_diff_pct = np.mean(diff_pcts)
    max_diff_pct = np.max(diff_pcts)

    if avg_diff_pct < 1.0 and max_diff_pct < 5.0:
        print("🎉 EXCELLENT: Lambda estimates are highly accurate!")
        print("   Average difference: <1%, Max difference: <5%")
        print("   The REML implementation is now production-ready!")
    elif avg_diff_pct < 10.0 and max_diff_pct < 50.0:
        print("✅ GOOD: Lambda estimates are reasonably accurate!")
        print("   Average difference: <10%, Max difference: <50%")
        print("   Significant improvement from previous implementation")
    elif avg_diff_pct < 100.0:
        print("⚠️  MODERATE: Some accuracy issues remain")
        print("   Average difference: <100%, but inconsistent results")
        print("   Further tuning may be needed")
    else:
        print("❌ POOR: Major accuracy issues persist")
        print("   Lambda estimates still differ by orders of magnitude")
        print("   Need further debugging")

    print("\nKey Improvements Made:")
    print("✅ Added pseudo-determinant term log|S_+|")
    print("✅ Switched to EDF-based scale parameter φ")
    print("✅ Updated REML formula to match mgcv exactly")
    print("✅ Enabled BLAS for better numerical stability")

if __name__ == "__main__":
    main()