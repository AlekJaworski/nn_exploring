#!/usr/bin/env python3
"""
Extract penalty matrices from Rust to compare with mgcv
"""

import numpy as np
import mgcv_rust

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    importr('mgcv')
    HAS_RPY2 = True
except:
    HAS_RPY2 = False
    print("Error: rpy2 required")
    exit(1)

# Test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.2 * np.random.randn(n)

def get_mgcv_penalty(k_val):
    """Get mgcv's penalty matrix"""
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_r'] = x
        ro.globalenv['k_val'] = k_val

        ro.r('''
            library(mgcv)
            # Create smooth object
            smooth <- smoothCon(s(x_r, k=k_val, bs="bs"), data=data.frame(x_r=x_r),
                               knots=NULL, absorb.cons=FALSE)[[1]]
            S <- smooth$S[[1]]
            X <- smooth$X
            knots_mgcv <- smooth$knots
        ''')

        S_mgcv = np.array(ro.r('S'))
        X_mgcv = np.array(ro.r('X'))
        knots_mgcv = np.array(ro.r('knots_mgcv'))

        return S_mgcv, X_mgcv, knots_mgcv

def test_penalty_comparison(k_val):
    """Compare penalty matrices"""
    print(f"\n{'='*80}")
    print(f"k = {k_val}")
    print(f"{'='*80}")

    S_mgcv, X_mgcv, knots_mgcv = get_mgcv_penalty(k_val)

    print(f"\nmgcv Penalty Matrix:")
    print(f"  Shape: {S_mgcv.shape}")
    print(f"  ||S||_F: {np.linalg.norm(S_mgcv, 'fro'):.6f}")
    print(f"  trace(S): {np.trace(S_mgcv):.6f}")
    print(f"  rank(S): {np.linalg.matrix_rank(S_mgcv)}")
    print(f"  max(S): {np.max(S_mgcv):.6f}")
    print(f"  min(S): {np.min(S_mgcv):.6f}")

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(S_mgcv)
    eigvals_sorted = np.sort(eigvals)[::-1]
    print(f"  Top 5 eigenvalues: {eigvals_sorted[:5]}")

    print(f"\nmgcv Knots:")
    print(f"  {knots_mgcv}")

    print(f"\nmgcv Basis Matrix:")
    print(f"  Shape: {X_mgcv.shape}")
    print(f"  ||X||_F: {np.linalg.norm(X_mgcv, 'fro'):.6f}")

    # Show sample of S
    print(f"\nS[0:3, 0:3]:")
    print(S_mgcv[:3, :3])

# Test different k values
for k in [5, 10, 20]:
    test_penalty_comparison(k)

print(f"\n{'='*80}")
print("Now we need to extract same from Rust...")
print("This requires exposing internal matrices from mgcv_rust library")
print(f"{'='*80}")
