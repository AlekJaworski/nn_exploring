#!/usr/bin/env python3
"""
Test the gradient fix to verify:
1. Trace values are now ~10,000x larger (~900 instead of ~0.09)
2. Gradient magnitude matches expectations (~450 instead of ~7)
3. Convergence is faster
"""

import numpy as np
import os

# Enable debug output
os.environ['MGCV_GRAD_DEBUG'] = '1'

# Fixed seed for determinism
np.random.seed(42)
n = 1000
x = np.random.randn(n, 4)
y = np.sin(x[:, 0]) + 0.5*x[:, 1]**2 + np.cos(x[:, 2]) + 0.3*x[:, 3] + np.random.randn(n)*0.1

print("=" * 70)
print("GRADIENT FIX VERIFICATION TEST")
print("=" * 70)
print(f"\nData: n={n}, p=4, k=16 (CR splines)")
print(f"Seed: 42 (fixed for determinism)")
print()

# Import after setting environment variable
import mgcv_rust

print("Testing gradient computation with block extraction fix...")
print("-" * 70)

gam = mgcv_rust.GAM()
result = gam.fit_auto_optimized(x, y, k=[16]*4, method='REML', bs='cr')

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Final lambda: {result['lambda']}")
print(f"REML value: {result.get('reml', 'N/A')}")
print()
print("Expected results with the fix:")
print("  - Trace values should be ~900 (previously ~0.09)")
print("  - Gradient magnitude should be ~450 (previously ~7)")
print("  - Convergence should occur in ~5 iterations (previously ~28)")
print()
print("Check the debug output above for:")
print("  [QR_GRAD_DEBUG] trace=... should show large values (~900)")
print("  [QR_GRAD_DEBUG] gradient[0]=... should show ~450")
