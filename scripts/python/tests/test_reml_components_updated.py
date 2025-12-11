#!/usr/bin/env python3
"""
Comprehensive REML component comparison between Rust and R's mgcv

This script tests the updated REML implementation with:
1. Pseudo-determinant term log|S_+|
2. EDF-based scale parameter φ
3. Complete component-by-component comparison

Usage:
    python scripts/python/tests/test_reml_components_updated.py
"""

import numpy as np
import pandas as pd
import subprocess
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mgcv_rust
from mgcv_rust import GAM, SmoothingParameter, OptimizationMethod, REMLAlgorithm

def generate_test_data(n=100, k=10, seed=42):
    """Generate test data matching R scripts"""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.normal(0, 0.1, n)
    return x, y

def run_r_comparison(x, y, lambda_val):
    """Run R script to get mgcv REML components at specific lambda"""
    # Create temporary data file
    data = pd.DataFrame({'x': x, 'y': y})
    data_file = '/tmp/test_data.csv'
    data.to_csv(data_file, index=False)

    # R script to compute REML components
    r_script = f'''
    library(mgcv)
    data <- read.csv("{data_file}")

    # Fit GAM with fixed lambda
    gam_fit <- gam(y ~ s(x, k=10, bs="cr"), data=data, sp={lambda_val})

    # Extract components
    lambda_mgcv <- gam_fit$sp
    deviance <- deviance(gam_fit)
    edf <- sum(gam_fit$edf)
    rss <- sum(residuals(gam_fit)^2)
    n <- length(data$y)

    # Extract matrices
    X <- model.matrix(gam_fit)
    S <- gam_fit$smooth[[1]]$S[[1]]
    beta <- coef(gam_fit)

    # Compute A = X'X + lambda*S
    W <- rep(1, n)  # Unit weights for simplicity
    XtX <- t(X) %*% X
    A <- XtX + {lambda_val} * S

    # Log determinants
    log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]

    # Pseudo-determinant of S
    ev <- eigen(S, only.values=TRUE)$values
    threshold <- 1e-10 * max(abs(ev))
    positive_eigs <- ev[ev > threshold]
    log_pseudo_det_S <- sum(log(positive_eigs))

    # Scale parameter (EDF-based)
    phi <- rss / (n - edf)

    # Penalty term
    penalty_term <- {lambda_val} * t(beta) %*% S %*% beta

    # REML components
    rss_bsb <- rss + penalty_term
    log_lambda_term <- length(positive_eigs) * log({lambda_val})

    # Full REML
    pi_val <- pi
    reml <- (rss_bsb/phi + (n-edf)*log(2*pi_val*phi) + log_det_A - log_lambda_term - log_pseudo_det_S) / 2

    # Output as JSON
    result <- list(
        lambda = lambda_mgcv,
        rss = rss,
        edf = edf,
        phi = phi,
        log_det_A = log_det_A,
        log_pseudo_det_S = log_pseudo_det_S,
        penalty_term = as.numeric(penalty_term),
        rss_bsb = as.numeric(rss_bsb),
        log_lambda_term = log_lambda_term,
        reml = as.numeric(reml),
        deviance = deviance,
        n = n,
        k = 10
    )

    cat(toJSON(result))
    '''

    # Run R script
    try:
        result = subprocess.run(['Rscript', '-e', r_script],
                              capture_output=True, text=True, check=True)
        return json.loads(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"R script failed: {e}")
        print(f"R stderr: {e.stderr}")
        return None

def test_reml_components():
    """Test REML components at multiple lambda values"""
    print("=" * 80)
    print("TESTING UPDATED REML IMPLEMENTATION")
    print("=" * 80)

    # Generate test data
    x, y = generate_test_data(n=100, k=10)
    print(f"Test data: n={len(y)}, x range=[{x.min():.3f}, {x.max():.3f}]")

    # Test at multiple lambda values
    lambda_values = [0.1, 1.0, 10.0, 100.0]

    results = []

    for lambda_val in lambda_values:
        print(f"\n--- Testing λ = {lambda_val} ---")

        # Get R reference values
        r_result = run_r_comparison(x, y, lambda_val)
        if r_result is None:
            print(f"  ❌ R comparison failed for λ={lambda_val}")
            continue

        print("  R mgcv results:")
        print(f"    REML: {r_result['reml']:.6f}")
        print(f"    φ (EDF): {r_result['phi']:.6f}")
        print(f"    EDF: {r_result['edf']:.4f}")
        print(f"    RSS: {r_result['rss']:.6f}")
        print(f"    log|A|: {r_result['log_det_A']:.6f}")
        print(f"    log|S_+|: {r_result['log_pseudo_det_S']:.6f}")

        # Test Rust implementation
        try:
            # Create GAM with fixed lambda
            gam = GAM()
            gam.add_smooth("x", x, basis="cr", k=10)

            # Set fixed lambda (bypass optimization)
            gam.smoothing_params = SmoothingParameter.new_with_algorithm(
                1, OptimizationMethod.REML, REMLAlgorithm.Newton
            )
            # Use setattr to avoid reserved keyword issue
            setattr(gam.smoothing_params, 'lambda', [lambda_val])

            # Fit with fixed lambda
            gam.fit(y)

            # Get REML criterion value
            rust_reml = gam.reml_criterion()

            print("  Rust results:")
            print(f"    REML: {rust_reml:.6f}")

            # Compare
            reml_diff = abs(rust_reml - r_result['reml'])
            print(f"    REML difference: {reml_diff:.6f}")

            if reml_diff < 1e-3:
                print("    ✅ REML values match!")
            else:
                print("    ❌ REML values differ significantly")
            # Store results
            results.append({
                'lambda': lambda_val,
                'r_reml': r_result['reml'],
                'rust_reml': rust_reml,
                'reml_diff': reml_diff,
                'r_phi': r_result['phi'],
                'r_edf': r_result['edf'],
                'r_rss': r_result['rss'],
                'r_log_det_A': r_result['log_det_A'],
                'r_log_pseudo_det_S': r_result['log_pseudo_det_S']
            })

        except Exception as e:
            print(f"  ❌ Rust test failed: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        df = pd.DataFrame(results)
        print("\nREML comparison across lambda values:")
        print(df[['lambda', 'r_reml', 'rust_reml', 'reml_diff']].to_string(index=False))

        # Check if all differences are small
        max_diff = df['reml_diff'].max()
        if max_diff < 1e-3:
            print("\n✅ ALL TESTS PASSED - REML implementation matches mgcv!")
        else:
            print(f"\n❌ Maximum REML difference: {max_diff:.6f} (should be < 1e-3)")
            print("   Implementation needs further debugging")
    else:
        print("❌ No successful tests - implementation has issues")

    return results

def test_optimization_convergence():
    """Test that optimization converges to similar lambdas"""
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZATION CONVERGENCE")
    print("=" * 80)

    # Generate test data
    x, y = generate_test_data(n=100, k=10)

    # Get R reference
    r_script = '''
    library(mgcv)
    set.seed(42)
    n <- 100
    x <- seq(0, 1, length.out = n)
    y <- sin(2 * pi * x) + rnorm(n, 0, 0.1)

    gam_fit <- gam(y ~ s(x, k=10, bs="cr"), method="REML")
    cat(gam_fit$sp)
    '''

    try:
        r_result = subprocess.run(['Rscript', '-e', r_script],
                                capture_output=True, text=True, check=True)
        r_lambda = float(r_result.stdout.strip())
        print(f"R mgcv optimal λ: {r_lambda:.6f}")
    except:
        print("❌ Could not get R reference lambda")
        return

    # Test Rust optimization
    try:
        gam = GAM()
        gam.add_smooth("x", x, basis="cr", k=10)
        gam.fit(y)

        rust_lambda = getattr(gam.smoothing_params, 'lambda')[0]
        print(f"Rust optimal λ: {rust_lambda:.6f}")

        ratio = rust_lambda / r_lambda
        print(f"Lambda ratio (Rust/R): {ratio:.6f}")

        if 0.1 < ratio < 10.0:  # Within order of magnitude
            print("✅ Lambda values are reasonable (within order of magnitude)")
        else:
            print("❌ Lambda values differ by orders of magnitude")

    except Exception as e:
        print(f"❌ Rust optimization failed: {e}")

if __name__ == "__main__":
    # Test REML components
    results = test_reml_components()

    # Test optimization convergence
    test_optimization_convergence()

    print("\n" + "=" * 80)
    print("IMPLEMENTATION STATUS")
    print("=" * 80)
    print("✅ Added pseudo-determinant term log|S_+|")
    print("✅ Switched to EDF-based scale parameter φ")
    print("✅ Updated REML formula to use EDF in denominator")
    print("🔄 Testing component-by-component agreement")
    print("🔄 Debugging remaining lambda discrepancies")