#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky

x_min, x_max, k, degree, m2_order = 0.0, 1.0, 20, 3, 2
pord = degree - m2_order  # 1
num_basis = 20

# Knots
nk = k - degree + 1
x_range = x_max - x_min
xl, xu = x_min - x_range * 0.001, x_max + x_range * 0.001
dx = (xu - xl) / (nk - 1)
knots_ext = np.linspace(xl - degree*dx, xu + degree*dx, nk + 2*degree)

# Interior knots
k0 = knots_ext[degree:(degree + nk)]
h = np.diff(k0)  # DON'T scale yet!

# h1 for evaluation points
h1 = np.repeat(h / pord, pord)
k1 = np.cumsum(np.concatenate([[k0[0]], h1]))

# NOW scale h
h = h / 2.0

# Evaluate derivatives  
D = np.zeros((len(k1), num_basis))
for i in range(min(num_basis, len(knots_ext) - degree - 1)):
    c = np.zeros(len(knots_ext) - degree - 1)
    c[i] = 1.0
    D[:, i] = BSpline(knots_ext, c, degree)(k1, nu=m2_order)

# W1 - CORRECTED to match mgcv exactly
seq_vals = np.linspace(-1, 1, pord + 1)
# Build powers matrix exactly as mgcv does (column-major reshape, then transpose)
vec = np.repeat(seq_vals, pord + 1) ** np.tile(np.arange(pord + 1), len(seq_vals))
powers_matrix = vec.reshape((pord + 1, pord + 1), order='F').T
P = np.linalg.inv(powers_matrix)
i1 = np.add.outer(np.arange(1, pord + 2), np.arange(1, pord + 2))
H = (1 + (-1)**(i1 - 2)) / (i1 - 1)
W1 = P.T @ H @ P

# ld0 (uses scaled h!)
diag_W1 = np.diag(W1)
ld0 = np.tile(diag_W1, len(h)) * np.repeat(h, pord + 1)

# Reindex
idx = np.concatenate([
    np.repeat(np.arange(1, pord + 1), len(h)) + np.tile(np.arange(len(h)) * (pord + 1), pord),
    [len(ld0)]
]) - 1
ld = ld0[idx.astype(int)]

# Overlaps
if len(h) > 1:
    ld[np.arange(1, len(h)) * pord] += ld0[np.arange(1, len(h)) * (pord + 1) - 1]

# B matrix
B = np.zeros((pord + 1, len(ld)))
B[0, :] = ld
for kk in range(1, pord + 1):
    if kk < W1.shape[0]:
        diwk = np.diag(W1, kk)
        ind_len = len(ld) - kk
        pattern = np.concatenate([diwk, np.zeros(kk - 1)])
        B[kk, :ind_len] = (np.repeat(h, pord) * np.tile(pattern, len(h)))[:ind_len]

# Full matrix & Cholesky
B_full = np.zeros((len(ld), len(ld)))
for i in range(pord + 1):
    for j in range(len(ld) - i):
        B_full[j, j + i] = B_full[j + i, j] = B[i, j]

L = cholesky(B_full, lower=False)
B_chol = np.array([[L[j, j + i] for j in range(len(ld) - i)] + [0]*i for i in range(pord + 1)])

# Apply
D1 = D * B_chol[0, :len(D), np.newaxis]
for kk in range(1, pord + 1):
    D1[:len(D)-kk, :] += D[kk:, :] * B_chol[kk, :len(D)-kk, np.newaxis]

S = D1.T @ D1

print(f"Frobenius: {np.linalg.norm(S, 'fro'):.1f} (expect 66901.7)")
print(f"Trace: {np.trace(S):.1f} (expect 221391.7)")
print(f"Match: {abs(np.linalg.norm(S, 'fro') - 66901.7) < 1}")

