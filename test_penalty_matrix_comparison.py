#!/usr/bin/env python3
"""
TDD Test: Compare penalty matrices between mgcv_rust and R's mgcv

This identifies if penalty matrix computation differs.
"""

import numpy as np
import mgcv_rust
import pytest

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    importr('mgcv')
    HAS_RPY2 = True
except:
    HAS_RPY2 = False
    pytest.skip("rpy2 not available", allow_module_level=True)

# Test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.2 * np.random.randn(n)

def get_mgcv_penalty_and_basis(k_val):
    """Get mgcv's penalty matrix and basis"""
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_r'] = x
        ro.globalenv['y_r'] = y
        ro.globalenv['k_val'] = k_val

        # Fit GAM
        ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="bs"), method="REML")')

        # Extract penalty matrix
        ro.r('''
            smooth_obj <- gam_fit$smooth[[1]]
            S_mgcv <- smooth_obj$S[[1]]
            X_full <- predict(gam_fit, type="lpmatrix")
            smooth_first <- smooth_obj$first.para
            smooth_last <- smooth_obj$last.para
            X_mgcv <- X_full[, smooth_first:smooth_last, drop=FALSE]
        ''')

        S_mgcv = np.array(ro.r('S_mgcv'))
        X_mgcv = np.array(ro.r('X_mgcv'))

        return S_mgcv, X_mgcv

def get_rust_penalty_and_basis(k_val):
    """Get rust's penalty matrix and basis - need to extract from library"""
    # Note: We need to add a method to mgcv_rust to expose these
    # For now, let's create them manually using the same algorithm

    # This is a placeholder - we'll need to expose internals from Rust
    import ctypes

    # For now, return None to indicate we need to implement this
    return None, None

@pytest.mark.parametrize("k_val", [5, 10, 20])
def test_penalty_matrix_values(k_val):
    """Test that penalty matrices match"""
    S_mgcv, X_mgcv = get_mgcv_penalty_and_basis(k_val)

    print(f"\n{'='*80}")
    print(f"k = {k_val}")
    print(f"{'='*80}")
    print(f"\nPenalty Matrix S (mgcv):")
    print(f"  Shape: {S_mgcv.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(S_mgcv, 'fro'):.6f}")
    print(f"  Max value: {np.max(np.abs(S_mgcv)):.6f}")
    print(f"  Trace: {np.trace(S_mgcv):.6f}")
    print(f"  Rank: {np.linalg.matrix_rank(S_mgcv)}")

    # Show first few rows/cols
    print(f"\n  S[0:3, 0:3]:")
    print(S_mgcv[:3, :3])

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(S_mgcv)
    eigvals_sorted = np.sort(eigvals)[::-1]
    print(f"\n  Top 5 eigenvalues: {eigvals_sorted[:5]}")
    print(f"  Number of zero eigenvalues (<1e-6): {np.sum(eigvals < 1e-6)}")

    print(f"\nBasis Matrix X (mgcv):")
    print(f"  Shape: {X_mgcv.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(X_mgcv, 'fro'):.6f}")

    # TODO: Compare with rust once we expose the internals

@pytest.mark.parametrize("k_val", [5, 10, 20])
def test_penalty_scaling_with_k(k_val):
    """Test how penalty matrix scales with k"""
    S_mgcv, _ = get_mgcv_penalty_and_basis(k_val)

    S_norm = np.linalg.norm(S_mgcv, 'fro')
    S_trace = np.trace(S_mgcv)
    k_actual = S_mgcv.shape[0]

    print(f"\nk={k_val} (actual dim={k_actual}):")
    print(f"  ||S||_F: {S_norm:.6f}")
    print(f"  trace(S): {S_trace:.6f}")
    print(f"  ||S||_F / k: {S_norm / k_actual:.6f}")
    print(f"  trace(S) / k: {S_trace / k_actual:.6f}")

    # Hypothesis: S norm grows linearly with k
    # If so, ||S||_F / k should be roughly constant

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
