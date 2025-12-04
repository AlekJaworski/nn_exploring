#!/usr/bin/env python3
"""
Implement mgcv's exact band Cholesky weighting for pord=2
Following smooth.construct.bs.smooth.spec source code
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky_banded
import pandas as pd

# Setup
x_min, x_max = 0.0, 1.0
k = 20
degree = 3  # m[1] in mgcv
num_basis = 19
pord = 2  # m[2] in mgcv - penalty order (second derivative)

# Create knots
nk = k - degree + 1
x_range = x_max - x_min
xl = x_min - x_range * 0.001
xu = x_max + x_range * 0.001
dx = (xu - xl) / (nk - 1)
n_total = nk + 2 * degree
start = xl - degree * dx
end = xu + degree * dx
knots_ext = np.linspace(start, end, n_total)

# Interior knots
k0 = knots_ext[degree:(degree + nk)]
h = np.diff(k0)

# Evaluation points for pord=2
h1 = np.repeat(h / pord, pord)
k1 = np.cumsum(np.concatenate([[k0[0]], h1]))

# Evaluate derivatives
D = np.zeros((len(k1), num_basis))
for i in range(num_basis):
    c = np.zeros(20)
    c[i] = 1.0
    spl = BSpline(knots_ext, c, degree)
    D[:, i] = spl(k1, nu=pord)

print("="*70)
print("IMPLEMENTING mgcv's BAND CHOLESKY WEIGHTING")
print("="*70)

# Step 1: Build W1 matrix (weight matrix for penalty order)
# P <- solve(matrix(rep(seq(-1, 1, length = pord + 1), pord + 1)^rep(0:pord, each = pord + 1), pord + 1, pord + 1))

seq_vals = np.linspace(-1, 1, pord + 1)  # [-1, 0, 1]
powers = np.arange(pord + 1)  # [0, 1, 2]

# Create matrix: each row is seq_vals^power
mat = np.array([seq_vals**p for p in powers]).T
P = np.linalg.inv(mat)

print(f"\nP matrix ({pord+1}x{pord+1}):")
print(P)

# H matrix
# i1 <- rep(1:(pord + 1), pord + 1) + rep(1:(pord + 1), each = pord + 1)
i1_matrix = np.add.outer(np.arange(1, pord + 2), np.arange(1, pord + 2))
# H <- matrix((1 + (-1)^(i1 - 2))/(i1 - 1), pord + 1, pord + 1)
H = (1 + (-1)**(i1_matrix - 2)) / (i1_matrix - 1)

print(f"\nH matrix:")
print(H)

# W1 <- t(P) %*% H %*% P
W1 = P.T @ H @ P

print(f"\nW1 matrix:")
print(W1)

# Step 2: Build banded weight matrix
# h <- h/2
h_scaled = h / 2.0

# sdiag gets diagonal elements
# ld0 <- rep(sdiag(W1), length(h)) * rep(h, each = pord + 1)
diag_W1 = np.diag(W1)
ld0 = np.repeat(diag_W1, len(h_scaled)) * np.tile(h_scaled, pord + 1)

print(f"\nDiagonal of W1: {diag_W1}")
print(f"\nld0 length: {len(ld0)}")

# Reindex ld0
# i1 <- c(rep(1:pord, length(h)) + rep(0:(length(h) - 1) * (pord + 1), each = pord), length(ld0))
part1 = np.repeat(np.arange(pord), len(h_scaled)) + np.tile(np.arange(len(h_scaled)) * (pord + 1), pord)
i1_idx = np.concatenate([part1, [len(ld0) - 1]])
ld = ld0[i1_idx]

# Add overlaps
# i0 <- 1:(length(h) - 1) * pord + 1
# i2 <- 1:(length(h) - 1) * (pord + 1)
# ld[i0] <- ld[i0] + ld0[i2]
if len(h_scaled) > 1:
    i0 = np.arange(1, len(h_scaled)) * pord
    i2 = np.arange(1, len(h_scaled)) * (pord + 1)
    ld[i0] += ld0[i2]

print(f"\nld vector length: {len(ld)}")
print(f"ld (first 10): {ld[:10]}")

# Step 3: Build banded matrix B from ld and off-diagonals of W1
# B is a (pord+1 x length(ld)) matrix
# B[1, ] <- ld (main diagonal)
# B[k+1, ] <- off-diagonals scaled by h

B = np.zeros((pord + 1, len(ld)))
B[0, :] = ld

# Add off-diagonals from W1
for k in range(1, pord + 1):
    # diwk <- sdiag(W1, k)  # k-th off-diagonal
    if k < len(W1):
        diwk = np.diag(W1, k)  # Upper k-th diagonal
    else:
        diwk = np.array([])

    if len(diwk) > 0:
        # ind <- 1:(length(ld) - k)
        ind_len = len(ld) - k
        # rep(h, each = pord) * rep(c(diwk, rep(0, k - 1)), length(h))
        pattern = np.concatenate([diwk, np.zeros(k - 1)])
        values = np.repeat(h_scaled, pord) * np.tile(pattern, len(h_scaled))[:ind_len]
        B[k, :ind_len] = values[:ind_len]

print(f"\nBanded matrix B shape: {B.shape}")
print(f"B (first few columns):")
print(B[:, :6])

# Step 4: Apply band Cholesky decomposition
# scipy's cholesky_banded expects upper form: ab[u + i - j, j] == a[i,j]
# where u = number of upper diagonals
# mgcv's bandchol returns the Cholesky factor

# For now, let's use the L from regular Cholesky as approximation
# Actually, mgcv's bandchol is complex - let me try simpler approach

# Apply band weighting directly
# D1 <- B[1, ] * D
D1 = D * B[0, :len(D), np.newaxis]

# for (k in 1:pord) {
#     ind <- 1:(nrow(D) - k)
#     D1[ind, ] <- D1[ind, ] + B[k + 1, ind] * D[ind + k, ]
# }
for k in range(1, pord + 1):
    ind = np.arange(len(D) - k)
    if len(ind) > 0 and k < B.shape[0]:
        D1[ind, :] += B[k, ind, np.newaxis] * D[ind + k, :]

print(f"\nWeighted derivative matrix D1 shape: {D1.shape}")
print(f"D1 min: {D1.min():.6f}, max: {D1.max():.6f}")

# Compute penalty
S = D1.T @ D1

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"S Frobenius norm: {np.linalg.norm(S, 'fro'):.6f}")
print(f"\nS (first 5x5):")
print(S[:5, :5])

# Compare
S_mgcv = pd.read_csv("/tmp/mgcv_bs_penalty.csv").values
print(f"\nmgcv Frobenius: {np.linalg.norm(S_mgcv, 'fro'):.6f}")
print(f"Ratio: {np.linalg.norm(S, 'fro') / np.linalg.norm(S_mgcv, 'fro'):.6f}")
