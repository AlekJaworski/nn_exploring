#!/usr/bin/env python3
"""
Validate trace computation as a DAG - check every intermediate value.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("TRACE COMPUTATION DAG VALIDATION")
print("=" * 80)

# Load the matrices mgcv computed
try:
    X = pd.read_csv('/tmp/X_matrix.csv').values
    S1_full = pd.read_csv('/tmp/S1_full.csv').values
    S2_full = pd.read_csv('/tmp/S2_full.csv').values
    data = pd.read_csv('/tmp/trace_step_data.csv')

    print(f"\nLoaded matrices from mgcv:")
    print(f"  X: {X.shape}")
    print(f"  S1_full: {S1_full.shape}")
    print(f"  S2_full: {S2_full.shape}")
except:
    print("ERROR: Need to run test_trace_step_by_step.py first to generate matrices")
    import sys
    sys.exit(1)

# Test parameters
lambda1, lambda2 = 2.0, 3.0
n, p = X.shape

print(f"\nTest config: n={n}, p={p}, λ=[{lambda1}, {lambda2}]")

# ============================================================================
# STEP 1: Compute A = X'X + λ1·S1 + λ2·S2
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Compute A matrix")
print("=" * 80)

XtX = X.T @ X
A_mgcv = XtX + lambda1 * S1_full + lambda2 * S2_full

print(f"✓ A computed: {A_mgcv.shape}")
print(f"  A[0,0] = {A_mgcv[0,0]:.6f}")
print(f"  A diagonal: min={np.diag(A_mgcv).min():.6f}, max={np.diag(A_mgcv).max():.6f}")

# ============================================================================
# STEP 2: Compute A^{-1}
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Compute A^{-1}")
print("=" * 80)

Ainv_mgcv = np.linalg.inv(A_mgcv)

print(f"✓ A^{{-1}} computed: {Ainv_mgcv.shape}")
print(f"  A^{{-1}}[0,0] = {Ainv_mgcv[0,0]:.6f}")

# Verify: A · A^{-1} = I
identity_error = np.max(np.abs(A_mgcv @ Ainv_mgcv - np.eye(p)))
print(f"  Verification: ||A·A^{{-1}} - I|| = {identity_error:.2e}")
assert identity_error < 1e-10, "A^{-1} is wrong!"

# ============================================================================
# STEP 3: Compute TRUE trace (mgcv's answer)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Compute TRUE trace (what we should get)")
print("=" * 80)

# Method 1: tr(A^{-1}·λ·S)
trace1_true = np.trace(Ainv_mgcv @ (lambda1 * S1_full))
trace2_true = np.trace(Ainv_mgcv @ (lambda2 * S2_full))

print(f"✓ TRUE trace (via A^{{-1}}):")
print(f"  tr(A^{{-1}}·λ1·S1) = {trace1_true:.6f}")
print(f"  tr(A^{{-1}}·λ2·S2) = {trace2_true:.6f}")

# ============================================================================
# STEP 4: Build augmented Z matrix (like our QR code does)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Build augmented matrix Z")
print("=" * 80)

# For QR method, we need sqrt(S) matrices
# Use eigenvalue decomposition
def compute_sqrt_penalty(S):
    """Compute thin sqrt: S = L·L' where L is p×rank"""
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    # Keep only positive eigenvalues
    threshold = 1e-10 * max(eigenvalues)
    mask = eigenvalues > threshold

    pos_eigs = eigenvalues[mask]
    pos_vecs = eigenvectors[:, mask]

    # L = Q·sqrt(Λ)
    L = pos_vecs @ np.diag(np.sqrt(pos_eigs))

    rank = len(pos_eigs)
    print(f"  Eigenvalues: {len(eigenvalues)} total, {rank} positive")
    print(f"  sqrt(S) shape: {L.shape}")

    # Verify: L·L' = S
    S_reconstructed = L @ L.T
    reconstruction_error = np.max(np.abs(S_reconstructed - S))
    print(f"  Verification: ||L·L' - S|| = {reconstruction_error:.2e}")

    return L, rank

print("\nComputing sqrt(S1):")
L1, rank1 = compute_sqrt_penalty(S1_full)

print("\nComputing sqrt(S2):")
L2, rank2 = compute_sqrt_penalty(S2_full)

# Build Z = [X; sqrt(λ1)·L1'; sqrt(λ2)·L2']
print(f"\nBuilding augmented matrix Z:")
print(f"  X: {X.shape}")
print(f"  sqrt(λ1)·L1': {L1.T.shape} scaled")
print(f"  sqrt(λ2)·L2': {L2.T.shape} scaled")

Z = np.vstack([
    X,
    np.sqrt(lambda1) * L1.T,
    np.sqrt(lambda2) * L2.T
])

print(f"✓ Z shape: {Z.shape}")

# ============================================================================
# STEP 5: QR decomposition
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: QR decomposition")
print("=" * 80)

Q, R = np.linalg.qr(Z)

print(f"✓ QR computed:")
print(f"  Q: {Q.shape}")
print(f"  R: {R.shape}")
print(f"  R[0,0] = {R[0,0]:.6f}")

# Extract upper triangular part (first p rows)
R_upper = R[:p, :p]
print(f"  R_upper: {R_upper.shape}")

# ============================================================================
# STEP 6: Compute P = R^{-1}
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Compute P = R^{{-1}}")
print("=" * 80)

P = np.linalg.inv(R_upper)

print(f"✓ P computed: {P.shape}")
print(f"  P[0,0] = {P[0,0]:.6f}")

# Verify: P'P should equal A^{-1}
PtP = P.T @ P
ptp_error = np.max(np.abs(PtP - Ainv_mgcv))

print(f"\nVERIFICATION: P'P = A^{{-1}}?")
print(f"  ||P'P - A^{{-1}}|| = {ptp_error:.2e}")

if ptp_error < 1e-8:
    print(f"  ✅ P'P = A^{{-1}} (verified!)")
else:
    print(f"  ❌ P'P ≠ A^{{-1}} (ERROR!)")
    print(f"\n  This is the bug! P matrix is wrong.")
    import sys
    sys.exit(1)

# ============================================================================
# STEP 7: Compute trace via QR method
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Compute trace via tr(P'·S·P)")
print("=" * 80)

# Method used in our code: tr(P'·L·L'·P) = tr((P'·L)·(P'·L)')= sum((P'·L)²)

print("\nFor smooth 1:")
P_t_L1 = P.T @ L1  # p × rank1
print(f"  P'·L1 shape: {P_t_L1.shape}")
trace_term1_qr = np.sum(P_t_L1 ** 2)
trace1_qr = lambda1 * trace_term1_qr
print(f"  trace_term = sum((P'·L1)²) = {trace_term1_qr:.6f}")
print(f"  trace = λ1 · trace_term = {trace1_qr:.6f}")

print("\nFor smooth 2:")
P_t_L2 = P.T @ L2  # p × rank2
print(f"  P'·L2 shape: {P_t_L2.shape}")
trace_term2_qr = np.sum(P_t_L2 ** 2)
trace2_qr = lambda2 * trace_term2_qr
print(f"  trace_term = sum((P'·L2)²) = {trace_term2_qr:.6f}")
print(f"  trace = λ2 · trace_term = {trace2_qr:.6f}")

# ============================================================================
# STEP 8: COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(f"\nSmooth 1:")
print(f"  TRUE trace (A^{{-1}} method): {trace1_true:.6f}")
print(f"  QR trace (P'·S·P method):     {trace1_qr:.6f}")
print(f"  Difference:                    {abs(trace1_true - trace1_qr):.2e}")
print(f"  Ratio (QR/TRUE):               {trace1_qr / trace1_true:.6f}")

print(f"\nSmooth 2:")
print(f"  TRUE trace (A^{{-1}} method): {trace2_true:.6f}")
print(f"  QR trace (P'·S·P method):     {trace2_qr:.6f}")
print(f"  Difference:                    {abs(trace2_true - trace2_qr):.2e}")
print(f"  Ratio (QR/TRUE):               {trace2_qr / trace2_true:.6f}")

if abs(trace1_qr - trace1_true) < 1e-6 and abs(trace2_qr - trace2_true) < 1e-6:
    print(f"\n✅ QR METHOD GIVES CORRECT TRACE!")
    print(f"\nSo the trace computation ALGORITHM is correct.")
    print(f"If our Rust code gives different values, the bug must be in:")
    print(f"  1. How we build the Z matrix")
    print(f"  2. How we compute sqrt(S)")
    print(f"  3. How we compute P from QR")
else:
    print(f"\n❌ QR METHOD GIVES WRONG TRACE!")
    print(f"\nThis means the algorithm itself has an issue.")

# Save these values for comparison with Rust
np.savez('/tmp/trace_validation.npz',
         trace1_true=trace1_true,
         trace2_true=trace2_true,
         trace1_qr=trace1_qr,
         trace2_qr=trace2_qr,
         P=P,
         L1=L1,
         L2=L2,
         A=A_mgcv,
         Ainv=Ainv_mgcv)

print(f"\n✓ Saved validation data to /tmp/trace_validation.npz")
