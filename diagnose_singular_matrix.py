#!/usr/bin/env python3
"""
Diagnose the singular matrix error in multidimensional GAMs
"""

import numpy as np
import sys

# First test: check if mgcv_rust is importable
try:
    import mgcv_rust
    print("✓ mgcv_rust imported successfully")
except ImportError as e:
    print(f"✗ Failed to import mgcv_rust: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("DIAGNOSING SINGULAR MATRIX ERROR")
print("="*70)

# Test 1: Single predictor (should work)
print("\nTest 1: Single predictor")
print("-" * 40)

np.random.seed(42)
n = 100
x1 = np.random.uniform(0, 1, n).reshape(-1, 1)
y = np.sin(2*np.pi*x1.ravel()) + 0.2*np.random.randn(n)

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(x1, y, k=[10], method='REML')
    print("✓ 1D case works!")
    print(f"  Lambda: {result['lambda']:.6f}")
except Exception as e:
    print(f"✗ 1D case failed: {e}")

# Test 2: Two predictors with RANDOM data
print("\nTest 2: Two predictors (RANDOM data)")
print("-" * 40)

np.random.seed(42)
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(-1, 1, n)
X = np.column_stack([x1, x2])
y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(n)

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[10, 10], method='REML')
    print("✓ 2D case with random data works!")
    print(f"  Lambdas: {result['all_lambdas']}")
except Exception as e:
    print(f"✗ 2D case with random data failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Two predictors with LINSPACE data
print("\nTest 3: Two predictors (LINSPACE data - potentially collinear)")
print("-" * 40)

np.random.seed(42)
x1 = np.linspace(0, 1, n)
x2 = np.linspace(-1, 1, n)
X = np.column_stack([x1, x2])
y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(n)

print(f"  Correlation between x1 and x2: {np.corrcoef(x1, x2)[0,1]:.6f}")

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[10, 10], method='REML')
    print("✓ 2D case with linspace data works!")
    print(f"  Lambdas: {result['all_lambdas']}")
except Exception as e:
    print(f"✗ 2D case with linspace data failed: {e}")

# Test 4: Two predictors with BS basis (not CR)
print("\nTest 4: Two predictors with B-spline basis (bs='bs')")
print("-" * 40)

np.random.seed(42)
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(-1, 1, n)
X = np.column_stack([x1, x2])
y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(n)

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[10, 10], method='REML', bs='bs')
    print("✓ 2D case with bs='bs' works!")
    print(f"  Lambdas: {result['all_lambdas']}")
except Exception as e:
    print(f"✗ 2D case with bs='bs' failed: {e}")

# Test 5: Try with different k values
print("\nTest 5: Two predictors with smaller k values")
print("-" * 40)

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[5, 5], method='REML')
    print("✓ 2D case with k=[5,5] works!")
    print(f"  Lambdas: {result['all_lambdas']}")
except Exception as e:
    print(f"✗ 2D case with k=[5,5] failed: {e}")

# Test 6: Try with GCV instead of REML
print("\nTest 6: Two predictors with GCV method")
print("-" * 40)

try:
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[10, 10], method='GCV')
    print("✓ 2D case with GCV works!")
    print(f"  Lambdas: {result['all_lambdas']}")
except Exception as e:
    print(f"✗ 2D case with GCV failed: {e}")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print("\nThe tests above help identify:")
print("1. Whether the issue is specific to multidimensional cases")
print("2. Whether random vs linspace data makes a difference")
print("3. Whether the basis type (cr vs bs) affects it")
print("4. Whether the issue is specific to REML vs GCV")
print("5. Whether smaller k values avoid the problem")
print("\nNext steps based on results:")
print("- If random data works but linspace fails: collinearity issue")
print("- If bs works but cr fails: CR spline implementation issue")
print("- If small k works: numerical scaling issue")
print("- If GCV works but REML fails: REML gradient/Hessian bug")
