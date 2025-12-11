#!/usr/bin/env python3
"""
Direct comparison of mgcv_rust vs R's mgcv on identical data
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
    """Generate test data matching R scripts"""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.normal(0, 0.1, n)
    return x, y

def run_r_mgcv_comparison(x, y):
    """Run R's mgcv and return lambda estimate"""
    # Create temporary data file
    data = np.column_stack([x, y])
    np.savetxt('/tmp/test_data.csv', data, delimiter=',', header='x,y', comments='')

    # R script to run mgcv
    r_script = '''
    library(mgcv)
    data <- read.csv("/tmp/test_data.csv")

    # Fit GAM with REML
    gam_fit <- gam(y ~ s(x, k=10, bs="cr"), data=data, method="REML")

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

def run_rust_comparison(x, y):
    """Run mgcv_rust and return lambda estimate"""
    try:
        # Create GAM and fit
        gam = mgcv_rust.GAM()
        result = gam.fit_auto(x.reshape(-1, 1), y, k=[10], method="REML", bs="cr")

        # Extract lambda
        rust_lambda = gam.get_lambda()
        return float(rust_lambda)
    except Exception as e:
        print(f"Rust failed: {e}")
        return None

def main():
    print("=" * 80)
    print("DIRECT COMPARISON: mgcv_rust vs R's mgcv")
    print("=" * 80)

    # Generate identical test data
    x, y = generate_test_data(n=100, seed=42)
    print(f"Test data: n={len(y)}, x range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"Data characteristics: y_mean={y.mean():.3f}, y_std={y.std():.3f}")

    # Run R's mgcv
    print("\nRunning R's mgcv...")
    r_lambda = run_r_mgcv_comparison(x, y)
    if r_lambda is None:
        print("❌ R mgcv failed - cannot proceed with comparison")
        return

    print(f"R mgcv optimal λ: {r_lambda:.6f}")
    # Run mgcv_rust
    print("\nRunning mgcv_rust...")
    rust_lambda = run_rust_comparison(x, y)
    if rust_lambda is None:
        print("❌ mgcv_rust failed - cannot proceed with comparison")
        return

    print(f"mgcv_rust optimal λ: {rust_lambda:.6f}")
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    lambda_ratio = rust_lambda / r_lambda
    lambda_diff = abs(rust_lambda - r_lambda)
    lambda_diff_pct = abs(lambda_ratio - 1.0) * 100

    print(f"R mgcv λ:     {r_lambda:.6f}")
    print(f"Rust λ:       {rust_lambda:.6f}")
    print(f"Ratio:        {lambda_ratio:.6f}")
    print(f"Abs diff:     {lambda_diff:.6f}")
    print(f"Diff %:       {lambda_diff_pct:.2f}%")
    # Assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT")
    print("=" * 60)

    if lambda_diff_pct < 1.0:  # Within 1%
        print("✅ EXCELLENT: Lambda estimates match within 1%")
        print("   The REML implementation is now highly accurate!")
    elif lambda_diff_pct < 10.0:  # Within 10%
        print("✅ GOOD: Lambda estimates match within 10%")
        print("   Significant improvement from previous ~100,000x differences")
    elif lambda_diff_pct < 100.0:  # Within 100%
        print("⚠️  MODERATE: Lambda estimates match within 100%")
        print("   Better than before but still room for improvement")
    else:
        print("❌ POOR: Lambda estimates still differ by orders of magnitude")
        print("   Need further debugging of REML implementation")

    # Additional context
    print("\nTechnical Notes:")
    print("- Both implementations use REML optimization")
    print("- Both use cubic regression splines with k=10")
    print("- Identical data and model specification")
    print("- mgcv_rust now includes pseudo-determinant terms")
    print("- mgcv_rust now uses EDF-based scale parameter")

if __name__ == "__main__":
    main()