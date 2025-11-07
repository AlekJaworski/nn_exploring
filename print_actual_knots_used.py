#!/usr/bin/env python3
"""
Print the actual knots being used in the penalty calculation
"""

import numpy as np

# Recreate exactly what our Rust function does
def create_mgcv_bs_knots(x_min, x_max, num_basis, degree):
    k = num_basis + 1  # 19 + 1 = 20
    nk = k - degree + 1  # 20 - 3 + 1 = 18

    x_range = x_max - x_min  # 1.0
    xl = x_min - x_range * 0.001  # -0.001
    xu = x_max + x_range * 0.001  # 1.001

    dx = (xu - xl) / (nk - 1)  # 1.002 / 17 = 0.05894118

    n_total = nk + 2 * degree  # 18 + 6 = 24
    start = xl - degree * dx  # -0.001 - 3 * 0.05894118 = -0.177824
    end = xu + degree * dx  # 1.001 + 3 * 0.05894118 = 1.177824

    return np.linspace(start, end, n_total)

x_min, x_max = 0.0, 1.0
num_basis = 19
degree = 3

our_knots = create_mgcv_bs_knots(x_min, x_max, num_basis, degree)

print("Our computed knots:")
print(f"Count: {len(our_knots)}")
print(f"First 5: {our_knots[:5]}")
print(f"Last 5: {our_knots[-5:]}")
print(f"\nAll knots:")
for i, k in enumerate(our_knots):
    print(f"  [{i:2d}] {k:.8f}")

print(f"\n" + "=" * 70)
print("mgcv's actual knots:")
print("=" * 70)

mgcv_knots = np.array([
    -0.17782353, -0.11888235, -0.05994118, -0.00100000,  0.05794118,  0.11688235,
     0.17582353,  0.23476471,  0.29370588,  0.35264706,  0.41158824,  0.47052941,
     0.52947059,  0.58841176,  0.64735294,  0.70629412,  0.76523529,  0.82417647,
     0.88311765,  0.94205882,  1.00100000,  1.05994118,  1.11888235,  1.17782353
])

print(f"Count: {len(mgcv_knots)}")
for i, k in enumerate(mgcv_knots):
    print(f"  [{i:2d}] {k:.8f}")

print(f"\n" + "=" * 70)
print("Differences:")
print("=" * 70)
diff = np.abs(our_knots - mgcv_knots)
print(f"Max diff: {np.max(diff):.10f}")
print(f"Mean diff: {np.mean(diff):.10f}")

if np.max(diff) < 1e-6:
    print("\n✅ Knots match mgcv!")
else:
    print(f"\n❌ Knots don't match (max diff = {np.max(diff):.6e})")
