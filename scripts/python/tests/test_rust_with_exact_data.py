#!/usr/bin/env python3
"""
Test Rust mgcv with exact same data as R
"""
import numpy as np
import pandas as pd
import mgcv_rust

print("="*70)
print("TESTING RUST WITH EXACT DATA")
print("="*70)

# Load exact data from CSV
data = pd.read_csv('/tmp/test_data.csv')
x = data['x'].values
y = data['y'].values

print(f"\nData loaded:")
print(f"  n = {len(x)}")
print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
print(f"  y mean: {y.mean():.6f}")
print(f"  y std: {y.std():.6f}")

# Fit with mgcv_rust
print(f"\nFitting GAM with k=10, bs='cr', method='REML'")
X = x.reshape(-1, 1)
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[10], method='REML', bs='cr')

print(f"\nRust Results:")
print(f"  Lambda: {result['lambda']:.6f}")
print(f"  Deviance: {result['deviance']:.6f}")

# Compare with mgcv
mgcv_values = pd.read_csv('/tmp/mgcv_reml_exact.csv')
lambda_mgcv = mgcv_values['lambda'].values[0]
deviance_mgcv = mgcv_values['deviance'].values[0]

print(f"\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"{'Metric':<30} {'Rust':>15} {'mgcv':>15} {'Ratio':>10}")
print("-"*70)
print(f"{'Lambda':<30} {result['lambda']:>15.6f} {lambda_mgcv:>15.6f} {result['lambda']/lambda_mgcv:>10.4f}")
print(f"{'Deviance':<30} {result['deviance']:>15.6f} {deviance_mgcv:>15.6f} {result['deviance']/deviance_mgcv:>10.4f}")
print("="*70)
print(f"\nLambda is off by a factor of {lambda_mgcv/result['lambda']:.1f}x")
