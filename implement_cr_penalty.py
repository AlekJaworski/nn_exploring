#!/usr/bin/env python3
"""
Implement the correct CR spline penalty algorithm from mgcv
S = D' B^{-1} D
"""
import numpy as np
import pandas as pd
from scipy.linalg import solve

def cr_spline_penalty(knots):
    """
    Compute CR spline penalty matrix using mgcv's algorithm
    
    Args:
        knots: array of n knot locations (sorted)
    
    Returns:
        S: n x n penalty matrix
    """
    n = len(knots)
    
    # Step 1: Compute knot spacings h
    h = np.diff(knots)  # length n-1
    
    # Step 2: Build (n-2) x n matrix D
    # D[i,i] = 1/h[i]
    # D[i,i+1] = -1/h[i] - 1/h[i+1]
    # D[i,i+2] = 1/h[i+1]
    n2 = n - 2
    D = np.zeros((n2, n))
    for i in range(n2):
        D[i, i] = 1.0 / h[i]
        D[i, i+1] = -1.0/h[i] - 1.0/h[i+1]
        D[i, i+2] = 1.0 / h[i+1]
    
    # Step 3: Build tridiagonal matrix B (n2 x n2)
    # Leading diagonal: (h[i] + h[i+1])/3
    # Super/sub diagonal: h[i+1]/6
    ldB = np.array([(h[i] + h[i+1])/3.0 for i in range(n2)])
    sdB = np.array([h[i+1]/6.0 for i in range(n2-1)])
    
    # Build full B matrix (symmetric tridiagonal)
    B = np.diag(ldB)
    if n2 > 1:
        B += np.diag(sdB, 1) + np.diag(sdB, -1)
    
    # Step 4: Solve B * X = D for X = B^{-1}D
    B_inv_D = solve(B, D)
    
    # Step 5: Compute S = D' B^{-1} D
    S = D.T @ B_inv_D
    
    return S

# Test with the exact knots from mgcv
print("="*70)
print("TESTING CR SPLINE PENALTY IMPLEMENTATION")
print("="*70)

# Load data and fit to get knots
data = pd.read_csv('/tmp/test_data.csv')
x = data['x'].values

# mgcv uses quantile-based knots for k=10
k = 10
knots = np.quantile(x, np.linspace(0, 1, k))

print(f"\nKnots (n={k}):")
print(knots)

# Compute penalty
S = cr_spline_penalty(knots)

print(f"\nComputed Penalty Matrix:")
print(f"  Shape: {S.shape}")
print(f"  Frobenius norm: {np.linalg.norm(S, 'fro'):.6f}")
print(f"  Trace: {np.trace(S):.6f}")
print(f"  Diagonal: {np.diag(S)}")
print(f"\n  First row:")
print(f"    {S[0, :]}")

# Load mgcv penalty for comparison
mgcv_penalty = pd.read_csv('/tmp/mgcv_penalty_matrix.csv').values

print(f"\nmgcv Penalty Matrix:")
print(f"  Shape: {mgcv_penalty.shape}")
print(f"  Frobenius norm: {np.linalg.norm(mgcv_penalty, 'fro'):.6f}")
print(f"  Trace: {np.trace(mgcv_penalty):.6f}")

# Compare
if S.shape == mgcv_penalty.shape:
    diff = np.max(np.abs(S - mgcv_penalty))
    print(f"\n" + "="*70)
    print(f"COMPARISON")
    print("="*70)
    print(f"Max difference: {diff:.2e}")
    print(f"Match (< 1e-6): {diff < 1e-6}")
else:
    print(f"\n" + "="*70)
    print(f"Shape mismatch! Ours: {S.shape}, mgcv: {mgcv_penalty.shape}")
    print("="*70)
    print(f"Note: mgcv applies identifiability constraint, reducing to {mgcv_penalty.shape[0]} basis")
