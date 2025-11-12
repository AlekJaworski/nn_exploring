#!/usr/bin/env python3
"""
Check CR spline basis properties: column sums, rank, etc.
"""

import numpy as np
import sys
import subprocess

sys.path.insert(0, "target/release")

print("=" * 70)
print("CR Spline Basis Diagnostics")
print("=" * 70)

# Test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, n)

import mgcv_rust

for k in [5, 10, 20]:
    print(f"\n{'=' * 70}")
    print(f"k={k}")
    print(f"{'=' * 70}")

    # Fit with Rust
    X_input = x.reshape(-1, 1)
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X_input, y, k=[k], method='REML', bs='cr')

    print(f"\nRust:")
    print(f"  Lambda: {result['lambda']:.6f}")
    print(f"  Deviance: {result['deviance']:.4f}")
    print(f"  EDF: {result.get('edf', 'N/A')}")

    # Get mgcv basis
    r_script = f"""
library(mgcv)
set.seed(42)
n <- 100
x <- seq(0, 1, length.out=n)

# Get smooth construction
sm <- smoothCon(s(x, k={k}, bs='cr'), data=data.frame(x=x), knots=NULL)[[1]]

cat('Basis dimensions:', dim(sm$X), '\\n')
cat('Column sums min/max:', min(colSums(sm$X)), max(colSums(sm$X)), '\\n')
cat('Rank:', qr(sm$X)$rank, '\\n')

# Check if there's an intercept (first column constant)
cat('First col constant?', length(unique(round(sm$X[,1], 10))) == 1, '\\n')
if (length(unique(round(sm$X[,1], 10))) == 1) {{
    cat('First col value:', sm$X[1,1], '\\n')
}}

# Check for sum-to-zero
cat('Has sum-to-zero constraint?', !is.null(sm$C), '\\n')
if (!is.null(sm$C)) {{
    cat('Constraint matrix C dimensions:', dim(sm$C), '\\n')
    cat('Constraint matrix C:\\n')
    print(sm$C)
}}

# Penalty rank
cat('Penalty rank:', qr(sm$S[[1]])$rank, '\\n')
"""

    result_r = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                             capture_output=True, timeout=30)

    if result_r.returncode == 0:
        print(f"\nmgcv:")
        print(result_r.stdout)
    else:
        print(f"R error: {result_r.stderr}")

print("\n" + "=" * 70)
