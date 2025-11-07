#!/usr/bin/env python3
"""
Test: Manually compute penalty using mgcv's approach
Evaluate B-spline derivatives at midpoints and compute S = D^T * W * D
"""

import numpy as np
from scipy.interpolate import BSpline

# mgcv parameters for k=20
x_min, x_max = 0.0, 1.0
k = 20
degree = 3
num_basis = 19

# Create mgcv-style knots (from our Rust function, verified to match)
nk = k - degree + 1  # 18
x_range = x_max - x_min
xl = x_min - x_range * 0.001  #  -0.001
xu = x_max + x_range * 0.001  #  1.001
dx = (xu - xl) / (nk - 1)  # spacing
n_total = nk + 2 * degree  # 24
start = xl - degree * dx
end = xu + degree * dx
knots_ext = np.linspace(start, end, n_total)

print("Extended knots:", len(knots_ext))
print("First few:", knots_ext[:5])
print("Last few:", knots_ext[-5:])

# Extract interior knots (for quadrature points)
# mgcv uses: k0 <- k[m[1] + 1:nk]
# m[1] = degree = 3
# So k0 = knots_ext[3:(3+nk)] = knots_ext[3:21]
k0 = knots_ext[degree:(degree + nk)]
print(f"\nInterior knots k0: {len(k0)} knots")
print(k0)

# Compute midpoints for evaluation (like mgcv for pord=0)
midpoints = (k0[:-1] + k0[1:]) / 2.0
print(f"\nMidpoints: {len(midpoints)}")
print(midpoints)

# Create B-spline basis and compute second derivatives at midpoints
# Note: scipy BSpline uses (knots, degree) convention
# Number of basis functions = len(knots) - degree - 1 = 24 - 3 - 1 = 20

# But we only want 19... mgcv must drop one for identifiability
# Let's use basis functions 0 through 18

D = np.zeros((len(midpoints), num_basis))

for i in range(num_basis):
    # Create coefficient vector (1 for basis i, 0 for others)
    # Need 20 coefficients for 20 basis functions
    c = np.zeros(20)
    c[i] = 1.0

    # Create B-spline
    spl = BSpline(knots_ext, c, degree)

    # Evaluate second derivative at midpoints
    D[:, i] = spl(midpoints, nu=2)

print(f"\nDerivative matrix D shape: {D.shape}")
print(f"D min: {D.min():.6f}, max: {D.max():.6f}")

# Apply mgcv weighting: multiply by sqrt(h) where h = diff(k0)
h = np.diff(k0)
print(f"\nKnot spacings h: {len(h)}")
print(f"h values: {h}")
print(f"h min: {h.min():.6f}, max: {h.max():.6f}")

# Weight each row of D by sqrt(h[i])
D_weighted = D * np.sqrt(h)[:, np.newaxis]

print(f"\nWeighted derivative matrix D_weighted:")
print(f"D_weighted min: {D_weighted.min():.6f}, max: {D_weighted.max():.6f}")

# Compute penalty: S = D^T * D
S = D_weighted.T @ D_weighted

print(f"\nPenalty matrix S shape: {S.shape}")
print(f"S trace: {np.trace(S):.6f}")
print(f"S Frobenius norm: {np.linalg.norm(S, 'fro'):.6f}")
print(f"S max row sum: {np.max(np.sum(np.abs(S), axis=1)):.6f}")

print(f"\nS (first 5x5):")
print(S[:5, :5])

# Compare with mgcv
import pandas as pd
S_mgcv = pd.read_csv("/tmp/mgcv_bs_penalty.csv").values

print(f"\n" + "="*70)
print("COMPARISON WITH MGCV")
print("="*70)
print(f"mgcv Frobenius: {np.linalg.norm(S_mgcv, 'fro'):.6f}")
print(f"Ours Frobenius: {np.linalg.norm(S, 'fro'):.6f}")
print(f"Ratio: {np.linalg.norm(S, 'fro') / np.linalg.norm(S_mgcv, 'fro'):.6f}")
