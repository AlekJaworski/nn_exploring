#!/usr/bin/env python3
"""
Extract intermediate values from mgcv algorithm for Rust debugging
Matches the exact parameters used in Rust test: num_basis=20, x=[0,1]
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky

# Parameters matching Rust test
x_min, x_max = 0.0, 1.0
k = 20
num_basis = 20
degree = 3
m2_order = 2
pord = degree - m2_order  # Should be 1

print("="*70)
print("EXTRACTING INTERMEDIATE VALUES FOR RUST DEBUG")
print("="*70)
print(f"\nParameters: num_basis={num_basis}, degree={degree}, pord={pord}")

# Create knots (mgcv style)
nk = k - degree + 1
x_range = x_max - x_min
xl = x_min - x_range * 0.001
xu = x_max + x_range * 0.001
dx = (xu - xl) / (nk - 1)
knots_ext = np.linspace(xl - degree*dx, xu + degree*dx, nk + 2*degree)

print(f"\nExtended knots: {len(knots_ext)} knots")
print(f"  First 4: {knots_ext[:4]}")
print(f"  Last 4: {knots_ext[-4:]}")

# Interior knots
k0 = knots_ext[degree:(degree + nk)]
print(f"\nInterior knots k0: {len(k0)} knots")
print(f"  k0[0]: {k0[0]:.10f}")
print(f"  k0[-1]: {k0[-1]:.10f}")

# h unscaled and scaled
h_unscaled = np.diff(k0)
h_scaled = h_unscaled / 2.0
print(f"\nh values:")
print(f"  h_unscaled[0]: {h_unscaled[0]:.10f}")
print(f"  h_scaled[0]: {h_scaled[0]:.10f}")
print(f"  len(h_scaled): {len(h_scaled)}")

# Evaluation points k1
h1 = np.repeat(h_unscaled / pord, pord)
k1 = np.cumsum(np.concatenate([[k0[0]], h1]))
print(f"\nEvaluation points k1: {len(k1)} points")
print(f"  k1[:3]: {k1[:3]}")
print(f"  k1[-3:]: {k1[-3:]}")

# Derivative matrix D
D = np.zeros((len(k1), num_basis))
for i in range(num_basis):
    c = np.zeros(len(knots_ext) - degree - 1)
    c[i] = 1.0
    D[:, i] = BSpline(knots_ext, c, degree)(k1, nu=m2_order)

print(f"\nDerivative matrix D: shape {D.shape}")
print(f"  D[0, :5]: {D[0, :5]}")
print(f"  D[-1, :5]: {D[-1, :5]}")

# W1 matrix
seq_vals = np.linspace(-1, 1, pord + 1)
vec = np.repeat(seq_vals, pord + 1) ** np.tile(np.arange(pord + 1), len(seq_vals))
powers_matrix = vec.reshape((pord + 1, pord + 1), order='F').T
P = np.linalg.inv(powers_matrix)
i1 = np.add.outer(np.arange(1, pord + 2), np.arange(1, pord + 2))
H = (1 + (-1)**(i1 - 2)) / (i1 - 1)
W1 = P.T @ H @ P

print(f"\nW1 matrix:")
print(f"  W1: {W1}")
print(f"  diag(W1): {np.diag(W1)}")

# ld vector
diag_W1 = np.diag(W1)
ld0 = np.tile(diag_W1, len(h_scaled)) * np.repeat(h_scaled, pord + 1)
print(f"\nld0: {len(ld0)} elements")
print(f"  ld0[:6]: {ld0[:6]}")
print(f"  ld0[-6:]: {ld0[-6:]}")

# Reindex
indices = np.concatenate([
    np.repeat(np.arange(1, pord + 1), len(h_scaled)) +
    np.tile(np.arange(len(h_scaled)) * (pord + 1), pord),
    [len(ld0)]
]) - 1
ld = ld0[indices.astype(int)]

# Overlaps
if len(h_scaled) > 1:
    i0 = np.arange(1, len(h_scaled)) * pord
    i2 = np.arange(1, len(h_scaled)) * (pord + 1) - 1
    ld[i0] += ld0[i2]

print(f"\nld (after overlaps): {len(ld)} elements")
print(f"  ld[:6]: {ld[:6]}")
print(f"  ld[-6:]: {ld[-6:]}")
print(f"  All ld: {', '.join(f'{x:.10f}' for x in ld)}")

# B matrix
B = np.zeros((pord + 1, len(ld)))
B[0, :] = ld

for kk in range(1, pord + 1):
    if kk < W1.shape[0]:
        diwk = np.diag(W1, kk)
        ind_len = len(ld) - kk
        pattern = np.concatenate([diwk, np.zeros(kk - 1)])
        values = (np.repeat(h_scaled, pord) * np.tile(pattern, len(h_scaled)))[:ind_len]
        B[kk, :ind_len] = values

print(f"\nB matrix (before Cholesky): shape {B.shape}")
print(f"  B[0, :6]: {B[0, :6]}")
if B.shape[0] > 1:
    print(f"  B[1, :6]: {B[1, :6]}")

# Reconstruct full B matrix
B_full = np.zeros((len(ld), len(ld)))
for i in range(pord + 1):
    for j in range(len(ld) - i):
        B_full[j, j + i] = B_full[j + i, j] = B[i, j]

print(f"\nB_full: shape {B_full.shape}")
print(f"  B_full positive definite: {np.all(np.linalg.eigvalsh(B_full) > 0)}")
print(f"  Min eigenvalue: {np.min(np.linalg.eigvalsh(B_full)):.10f}")

# Cholesky
L_upper = cholesky(B_full, lower=False)
B_chol = np.array([[L_upper[j, j + i] for j in range(len(ld) - i)] + [0]*i for i in range(pord + 1)])

print(f"\nB_chol (after Cholesky): shape {B_chol.shape}")
print(f"  B_chol[0, :6]: {B_chol[0, :6]}")
if B_chol.shape[0] > 1:
    print(f"  B_chol[1, :6]: {B_chol[1, :6]}")

# Apply weights
D1 = D * B_chol[0, :len(D), np.newaxis]
for kk in range(1, pord + 1):
    ind = len(D) - kk
    if ind > 0:
        D1[:ind, :] += D[kk:, :] * B_chol[kk, :ind, np.newaxis]

print(f"\nD1 (after weighting): shape {D1.shape}")
print(f"  D1[0, :5]: {D1[0, :5]}")
print(f"  D1[-1, :5]: {D1[-1, :5]}")

# Final penalty
S = D1.T @ D1

frobenius = np.linalg.norm(S, 'fro')
trace = np.trace(S)

print(f"\n" + "="*70)
print("FINAL RESULT")
print("="*70)
print(f"Frobenius: {frobenius:.1f}")
print(f"Trace: {trace:.1f}")
print(f"Expected: Frobenius=66901.7, Trace=221391.7")
print(f"Match: {abs(frobenius - 66901.7) < 1 and abs(trace - 221391.7) < 1}")

# Save key values for Rust comparison
print(f"\n" + "="*70)
print("KEY VALUES FOR RUST COMPARISON")
print("="*70)
print(f"len(ld) = {len(ld)}")
print(f"ld[0] = {ld[0]:.15f}")
print(f"ld[1] = {ld[1]:.15f}")
print(f"ld[5] = {ld[5]:.15f}")
print(f"B[0,0] = {B[0,0]:.15f}")
print(f"B[0,1] = {B[0,1]:.15f}")
print(f"B[1,0] = {B[1,0]:.15f}")
print(f"B_chol[0,0] = {B_chol[0,0]:.15f}")
print(f"B_chol[0,1] = {B_chol[0,1]:.15f}")
print(f"B_chol[1,0] = {B_chol[1,0]:.15f}")
