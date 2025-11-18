#!/usr/bin/env python3
"""
Test to understand the P matrix issue.
Compare different ways of computing A^{-1} from QR decomposition.
"""

import numpy as np

# Create a simple test case
np.random.seed(42)
n, p = 100, 10

# Create X'X + λS (symmetric positive definite)
X = np.random.randn(n, p)
S = np.random.randn(p, p)
S = S.T @ S  # Make symmetric positive definite
lam = 0.5

A = X.T @ X + lam * S

# Method 1: Direct inverse
A_inv_direct = np.linalg.inv(A)

# Method 2: Via Cholesky (R'R = A)
R_chol = np.linalg.cholesky(A).T  # Upper triangular
A_inv_chol = np.linalg.inv(R_chol) @ np.linalg.inv(R_chol.T)

# Method 3: Via QR decomposition
# Build augmented matrix Z such that Z'Z = A
# For simplicity, use Cholesky: Z = R_chol.T
Z = R_chol.T
Q, R_qr = np.linalg.qr(Z)
# Now R_qr'R_qr should equal A
print("QR check - R_qr'R_qr vs A:")
print("  Max diff:", np.max(np.abs(R_qr.T @ R_qr - A)))

# Compute P = R_qr^{-1}
P_wrong = np.linalg.inv(R_qr)
A_inv_wrong = P_wrong.T @ P_wrong  # This is (R'R)^{-1}? NO!

# Actually: P'P = R^{-T} R^{-1} = (RR')^{-1}, not (R'R)^{-1}!
print("\nMethod comparison:")
print("  A_inv_direct Frobenius norm:", np.linalg.norm(A_inv_direct, 'fro'))
print("  A_inv_chol Frobenius norm:", np.linalg.norm(A_inv_chol, 'fro'))
print("  P_wrong'P_wrong Frobenius norm:", np.linalg.norm(A_inv_wrong, 'fro'))

print("\nP = R^{-1} statistics:")
print("  R_qr diagonal:", R_qr.diagonal()[:5])
print("  P_wrong diagonal:", P_wrong.diagonal()[:5])
print("  P_wrong Frobenius norm:", np.linalg.norm(P_wrong, 'fro'))

# Correct approach: P = (R'R)^{-1} = R^{-1} R'^{-1}
R_inv = np.linalg.inv(R_qr)
R_t_inv = np.linalg.inv(R_qr.T)
A_inv_correct = R_inv @ R_t_inv

print("\nCorrect inverse:")
print("  A_inv_correct Frobenius norm:", np.linalg.norm(A_inv_correct, 'fro'))
print("  Max diff from direct:", np.max(np.abs(A_inv_correct - A_inv_direct)))

# Test trace computation
print("\nTrace computation:")
trace_direct = np.trace(A_inv_direct @ S)
trace_formula = np.trace(R_inv.T @ S @ R_inv)  # tr(P'·S·P) where P=R^{-1}
print("  tr(A^{-1}·S) direct:", trace_direct)
print("  tr(R^{-T}·S·R^{-1}):", trace_formula)
print("  Difference:", abs(trace_direct - trace_formula))
