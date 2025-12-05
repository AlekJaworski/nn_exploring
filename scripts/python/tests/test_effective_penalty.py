#!/usr/bin/env python3
"""
Test if lambda × penalty gives equivalent effective smoothing
even if lambda values differ
"""

import numpy as np
import mgcv_rust

np.random.seed(42)
n = 500
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)
X = x.reshape(-1, 1)
k = 20

print("=" * 70)
print("Testing Effective Penalty: λ × S")
print("=" * 70)

# Fit both
gam_bs = mgcv_rust.GAM()
result_bs = gam_bs.fit_auto(X, y, k=[k], method='REML', bs='bs')
pred_bs = gam_bs.predict(X)

gam_cr = mgcv_rust.GAM()
result_cr = gam_cr.fit_auto(X, y, k=[k], method='REML', bs='cr')
pred_cr = gam_cr.predict(X)

print(f"\nBS:")
print(f"  Lambda: {result_bs['lambda']:.6f}")
print(f"  Deviance: {result_bs['deviance']:.4f}")

print(f"\nCR:")
print(f"  Lambda: {result_cr['lambda']:.6f}")
print(f"  Deviance: {result_cr['deviance']:.4f}")

# Compare predictions (if effective penalty is same, predictions should be similar)
pred_diff = np.abs(pred_bs - pred_cr)
print(f"\nPrediction Comparison:")
print(f"  Max diff: {np.max(pred_diff):.6f}")
print(f"  Mean diff: {np.mean(pred_diff):.6f}")
print(f"  RMS diff: {np.sqrt(np.mean(pred_diff**2)):.6f}")

# Compare smoothness (second derivatives)
def second_diff(y):
    return np.diff(y, n=2)

smooth_bs = np.sum(second_diff(pred_bs)**2)
smooth_cr = np.sum(second_diff(pred_cr)**2)

print(f"\nSmoothness (Σ(Δ²f)²):")
print(f"  BS: {smooth_bs:.6e}")
print(f"  CR: {smooth_cr:.6e}")
print(f"  Ratio: {smooth_bs / smooth_cr:.4f}")

if np.abs(smooth_bs / smooth_cr - 1.0) < 0.1:
    print("\n✅ Similar smoothness despite different λ values")
    print("   This means λ × S might be scaled correctly")
else:
    print("\n⚠️  Very different smoothness levels")
    print("   Penalty matrices or lambda values need adjustment")
