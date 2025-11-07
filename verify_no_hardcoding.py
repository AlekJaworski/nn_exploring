#!/usr/bin/env python3
"""
Verify Rust implementation works correctly with different parameters
by comparing against Python implementation
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky

def compute_mgcv_penalty(num_basis, x_min, x_max, deriv_order=2):
    """Compute penalty matrix using mgcv algorithm"""
    degree = 3
    pord = degree - deriv_order

    # Create knots
    k = num_basis
    nk = k - degree + 1
    x_range = x_max - x_min
    xl = x_min - x_range * 0.001
    xu = x_max + x_range * 0.001
    dx = (xu - xl) / (nk - 1)
    knots_ext = np.linspace(xl - degree*dx, xu + degree*dx, nk + 2*degree)

    # Interior knots
    k0 = knots_ext[degree:(degree + nk)]

    # h scaled
    h_unscaled = np.diff(k0)
    h_scaled = h_unscaled / 2.0

    # Evaluation points
    h1 = np.repeat(h_unscaled / pord, pord)
    k1 = np.cumsum(np.concatenate([[k0[0]], h1]))

    # Derivative matrix
    D = np.zeros((len(k1), num_basis))
    for i in range(num_basis):
        c = np.zeros(len(knots_ext) - degree - 1)
        c[i] = 1.0
        D[:, i] = BSpline(knots_ext, c, degree)(k1, nu=deriv_order)

    # W1 matrix
    seq_vals = np.linspace(-1, 1, pord + 1)
    vec = np.repeat(seq_vals, pord + 1) ** np.tile(np.arange(pord + 1), len(seq_vals))
    powers_matrix = vec.reshape((pord + 1, pord + 1), order='F').T
    P = np.linalg.inv(powers_matrix)
    i1 = np.add.outer(np.arange(1, pord + 2), np.arange(1, pord + 2))
    H = (1 + (-1)**(i1 - 2)) / (i1 - 1)
    W1 = P.T @ H @ P

    # ld vector
    diag_W1 = np.diag(W1)
    ld0 = np.tile(diag_W1, len(h_scaled)) * np.repeat(h_scaled, pord + 1)

    indices = np.concatenate([
        np.repeat(np.arange(1, pord + 1), len(h_scaled)) +
        np.tile(np.arange(len(h_scaled)) * (pord + 1), pord),
        [len(ld0)]
    ]) - 1
    ld = ld0[indices.astype(int)]

    if len(h_scaled) > 1:
        i0 = np.arange(1, len(h_scaled)) * pord
        i2 = np.arange(1, len(h_scaled)) * (pord + 1) - 1
        ld[i0] += ld0[i2]

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

    # Full B and Cholesky
    B_full = np.zeros((len(ld), len(ld)))
    for i in range(pord + 1):
        for j in range(len(ld) - i):
            B_full[j, j + i] = B_full[j + i, j] = B[i, j]

    L_upper = cholesky(B_full, lower=False)
    B_chol = np.array([[L_upper[j, j + i] for j in range(len(ld) - i)] + [0]*i for i in range(pord + 1)])

    # Apply weights
    D1 = D * B_chol[0, :len(D), np.newaxis]
    for kk in range(1, pord + 1):
        ind = len(D) - kk
        if ind > 0:
            D1[:ind, :] += D[kk:, :] * B_chol[kk, :ind, np.newaxis]

    # Final penalty
    S = D1.T @ D1

    return S

print("="*70)
print("VERIFYING NO HARDCODING - TESTING DIFFERENT PARAMETERS")
print("="*70)

# Test 1: num_basis=10, range=[0,1]
print("\n=== Test 1: num_basis=10, range=[0,1] ===")
S = compute_mgcv_penalty(10, 0.0, 1.0)
frob = np.linalg.norm(S, 'fro')
print(f"  Python Frobenius: {frob:.1f}")
print(f"  Rust Frobenius:   2872.9")
print(f"  Match: {abs(frob - 2872.9) < 1.0}")

# Test 2: num_basis=20, range=[0,2]
print("\n=== Test 2: num_basis=20, range=[0,2] ===")
S = compute_mgcv_penalty(20, 0.0, 2.0)
frob = np.linalg.norm(S, 'fro')
print(f"  Python Frobenius: {frob:.1f}")
print(f"  Rust Frobenius:   8362.7")
print(f"  Match: {abs(frob - 8362.7) < 1.0}")

# Test 3: num_basis=15, range=[-5,5]
print("\n=== Test 3: num_basis=15, range=[-5,5] ===")
S = compute_mgcv_penalty(15, -5.0, 5.0)
frob = np.linalg.norm(S, 'fro')
print(f"  Python Frobenius: {frob:.1f}")
print(f"  Rust Frobenius:   19.5")
print(f"  Match: {abs(frob - 19.5) < 1.0}")

# Test 4: num_basis=12, range=[0,0.1]
print("\n=== Test 4: num_basis=12, range=[0,0.1] ===")
S = compute_mgcv_penalty(12, 0.0, 0.1)
frob = np.linalg.norm(S, 'fro')
print(f"  Python Frobenius: {frob:.1f}")
print(f"  Rust Frobenius:   7038124.6")
print(f"  Match: {abs(frob - 7038124.6) / 7038124.6 < 0.01}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
