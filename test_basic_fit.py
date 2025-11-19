#!/usr/bin/env python3
"""
Basic test: verify GAM fitting works with current implementation.
"""

import numpy as np
import mgcv_rust

print("Testing basic GAM fit...")

# Simple test data
np.random.seed(42)
n = 100
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

# Fit GAM
gam = mgcv_rust.GAM()
result = gam.fit_auto_optimized(x, y, k=[10, 10], method='REML', bs='cr')

print(f"✓ Fit completed")
print(f"  Final λ: {result['lambda']}")
print(f"  Deviance: {result['deviance']}")
print(f"  Fitted values shape: {result['fitted'].shape}")

# Check predictions work
pred = gam.predict(x)
print(f"✓ Predictions work")
print(f"  Prediction shape: {pred.shape}")
print(f"  Prediction mean: {np.mean(pred):.3f}")

print("\n✅ All basic tests pass")
