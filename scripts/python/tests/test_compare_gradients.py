#!/usr/bin/env python3
"""
Compare our gradient with mgcv's at the SAME sp values.

mgcv's final values (n=100):
- sp = [5.694, 5.201]
- gradient = ~0 (3.9e-9)
- REML = -64.642

Test: compute our gradient at sp=[5.694, 5.201] and see if it's also ~0
"""

import numpy as np
import mgcv_rust
import os

# Load the exact same data mgcv used
x = np.loadtxt('/tmp/unit_x.csv', delimiter=',', skiprows=1)
y = np.loadtxt('/tmp/unit_y.csv', delimiter=',', skiprows=1)

print("=" * 80)
print("GRADIENT COMPARISON AT MGCV'S FINAL SP")
print("=" * 80)

print(f"\nData: n={len(y)}, p={x.shape[1]}")
print(f"mgcv's final sp: [5.694, 5.201]")
print(f"mgcv's final gradient: ~3.9e-9 (essentially zero)")

# Now fit with our implementation
gam = mgcv_rust.GAM()

# Enable debug to see gradients
os.environ['MGCV_GRAD_DEBUG'] = '1'
os.environ['MGCV_PROFILE'] = '1'

print("\n" + "=" * 80)
print("OUR IMPLEMENTATION")
print("=" * 80)

result = gam.fit_auto_optimized(x, y, k=[10, 10], method='REML', bs='cr')

print(f"\nOur final sp: {result['lambda']}")
print(f"\nCheck stderr above for gradient values at each iteration")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
Questions:
1. Does our gradient start at ~41.6 like mgcv? (YES from earlier tests)
2. Does our gradient converge to ~0 like mgcv? (NO - we stop at ~10)
3. What's different in iterations 3-5 that causes divergence?

mgcv iterations:
1: 41.61 -> 2: 29.86 -> 3: 5.07 -> 4: 0.24 -> 5: 0.0006

Our iterations (from earlier):
1: 41.91 -> 2: 41.63 -> 3: 27.09 -> ... -> 7: 10.09

After iteration 2, we diverge! mgcv goes 29.86->5.07, we go 41.63->27.09.

This suggests:
- Gradient formula might be OK initially
- Newton step / Hessian is WRONG
- Or both gradient AND Hessian have correlated errors
""")
