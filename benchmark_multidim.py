#!/usr/bin/env python3
"""Benchmark multidimensional GAM performance"""
import numpy as np
import mgcv_rust
import time

np.random.seed(42)

print("=" * 70)
print("MULTIDIMENSIONAL GAM PERFORMANCE BENCHMARK")
print("=" * 70)

# Test 2D
print("\n2D GAM (n=200, k=[16,16]):")
n = 200
X = np.column_stack([np.linspace(0, 1, n), np.linspace(-1, 1, n)])
y = np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + 0.2*np.random.randn(n)

start = time.time()
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[16,16], method='REML')
elapsed = time.time() - start

preds = gam.predict(X)
r2 = 1 - np.var(y - preds) / np.var(y)
lam = result.get('lambda', result.get('all_lambdas', 'N/A'))
print(f"  Time: {elapsed:.3f}s")
print(f"  R²: {r2:.4f}")
print(f"  λ: {lam}")

# Test 3D
print("\n3D GAM (n=200, k=[12,12,12]):")
X3 = np.random.randn(n, 3)
y3 = np.sin(X3[:,0]) + 0.5*X3[:,1]**2 + np.cos(X3[:,2]) + 0.1*np.random.randn(n)

start = time.time()
gam3 = mgcv_rust.GAM()
result3 = gam3.fit_auto(X3, y3, k=[12]*3, method='REML')
elapsed3 = time.time() - start

preds3 = gam3.predict(X3)
r2_3 = 1 - np.var(y3 - preds3) / np.var(y3)
lam3 = result3.get('lambda', result3.get('all_lambdas', 'N/A'))
print(f"  Time: {elapsed3:.3f}s")
print(f"  R²: {r2_3:.4f}")
print(f"  λ: {lam3}")

# Test 4D
print("\n4D GAM (n=200, k=[16,16,16,16]):")
X4 = np.random.randn(n, 4)
y4 = np.sin(X4[:,0]) + 0.5*X4[:,1]**2 + np.cos(X4[:,2]) + 0.3*X4[:,3] + 0.1*np.random.randn(n)

start = time.time()
gam4 = mgcv_rust.GAM()
result4 = gam4.fit_auto(X4, y4, k=[16]*4, method='REML')
elapsed4 = time.time() - start

preds4 = gam4.predict(X4)
r2_4 = 1 - np.var(y4 - preds4) / np.var(y4)
lam4 = result4.get('lambda', result4.get('all_lambdas', 'N/A'))
print(f"  Time: {elapsed4:.3f}s")
print(f"  R²: {r2_4:.4f}")
print(f"  λ: {lam4}")

# Test 5D
print("\n5D GAM (n=250, k=[10,10,10,10,10]):")
n5 = 250
X5 = np.random.randn(n5, 5)
y5 = (np.sin(X5[:,0]) + 0.5*X5[:,1]**2 + np.cos(X5[:,2]) +
      0.3*X5[:,3] + 0.2*X5[:,4]**3 + 0.1*np.random.randn(n5))

start = time.time()
gam5 = mgcv_rust.GAM()
result5 = gam5.fit_auto(X5, y5, k=[10]*5, method='REML')
elapsed5 = time.time() - start

preds5 = gam5.predict(X5)
r2_5 = 1 - np.var(y5 - preds5) / np.var(y5)
lam5 = result5.get('lambda', result5.get('all_lambdas', 'N/A'))
print(f"  Time: {elapsed5:.3f}s")
print(f"  R²: {r2_5:.4f}")
print(f"  λ: {lam5}")

print("\n" + "=" * 70)
print("✓ All multidimensional benchmarks completed successfully!")
print("=" * 70)
