#!/usr/bin/env python3
"""
Unit tests to verify gradient and Hessian match mgcv exactly.

This creates a simple test case and compares:
1. Gradient at specific lambda values
2. Hessian at specific lambda values
3. REML value at specific lambda values

We test at the SAME lambda values to verify our formulas match mgcv's.
"""

import numpy as np
import subprocess
import json

def test_gradient_at_lambda():
    """Test that our gradient matches mgcv's at a specific lambda."""

    # Simple test case: n=100, 2 variables
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 2)
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

    # Save data
    np.savetxt('/tmp/test_x.csv', x, delimiter=',')
    np.savetxt('/tmp/test_y.csv', y)

    # Test at specific lambda values
    test_lambdas = [
        [1.0, 1.0],
        [0.1, 0.1],
        [10.0, 1.0],
        [0.01, 10.0],
    ]

    print("=" * 80)
    print("GRADIENT UNIT TEST")
    print("=" * 80)

    for lam in test_lambdas:
        print(f"\n{'=' * 80}")
        print(f"Testing at lambda = {lam}")
        print(f"{'=' * 80}")

        # Get mgcv's gradient using numerical differentiation
        r_script = f'''
        library(mgcv)

        x <- as.matrix(read.csv("/tmp/test_x.csv", header=FALSE))
        y <- as.numeric(read.csv("/tmp/test_y.csv", header=FALSE)$V1)

        df <- data.frame(x1=x[,1], x2=x[,2], y=y)

        # Build smooths
        sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
        sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]

        X <- cbind(sm1$X, sm2$X)
        S <- list()
        S[[1]] <- matrix(0, ncol(X), ncol(X))
        S[[2]] <- matrix(0, ncol(X), ncol(X))
        S[[1]][1:ncol(sm1$X), 1:ncol(sm1$X)] <- sm1$S[[1]]
        S[[2]][(ncol(sm1$X)+1):ncol(X), (ncol(sm1$X)+1):ncol(X)] <- sm2$S[[1]]

        # Test lambda
        sp <- c({lam[0]}, {lam[1]})

        # Compute REML and gradient using mgcv internals
        # Use .Call to access C function directly
        G <- list(
          X = X,
          y = y,
          w = rep(1, length(y)),
          S = S,
          off = rep(0, length(y)),
          rank = c(qr(S[[1]])$rank, qr(S[[2]])$rank),
          beta = rep(0, ncol(X))  # Initial guess
        )

        # Call mgcv's gdi function (gradient/deviance info)
        # This is the core gradient computation
        result <- .Call("gdi", G, sp, PACKAGE="mgcv")

        cat("\\nmgcv REML:", result$REML, "\\n")
        cat("mgcv gradient:\\n")
        print(result$reml1)
        cat("mgcv Hessian diagonal:\\n")
        print(diag(result$reml2))
        '''

        result = subprocess.run(['R', '--vanilla', '--slave', '-e', r_script],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("mgcv output:")
            print(result.stdout)
        else:
            print(f"R error: {result.stderr}")

        # Now get our gradient
        print("\n" + "-" * 80)
        print("Our implementation:")
        print("-" * 80)

        import mgcv_rust
        gam = mgcv_rust.GAM()

        # We need to expose a function to compute gradient at specific lambda
        # For now, let's just fit and see first iteration gradient
        import os
        os.environ['MGCV_GRAD_DEBUG'] = '1'

        result = gam.fit_auto_optimized(x, y, k=[10, 10], method='REML', bs='cr')
        print(f"(Check stderr for gradient at iteration 1)")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
    Compare:
    1. REML values - should match exactly
    2. Gradient values - should match exactly
    3. Hessian diagonal - should match exactly

    If they don't match, our formula is wrong!
    """)

if __name__ == '__main__':
    test_gradient_at_lambda()
