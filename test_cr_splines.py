#!/usr/bin/env python3
"""
Test cubic regression splines (cr basis) against mgcv
"""

import numpy as np
import mgcv_rust

# Test 1: Basic functionality
print("=" * 60)
print("Test 1: Basic CR Spline Functionality")
print("=" * 60)

np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n)

X = x.reshape(-1, 1)

# Fit with cr splines
gam_cr = mgcv_rust.GAM()
result_cr = gam_cr.fit_auto(X, y, k=[10], method='REML', bs='cr')
pred_cr = gam_cr.predict(X)

print(f"✓ CR splines fitted successfully")
print(f"  Lambda: {result_cr['lambda']:.6f}")
print(f"  Deviance: {result_cr['deviance']:.4f}")
print(f"  Number of basis functions: 10")

# Test 2: Compare with B-splines
print("\n" + "=" * 60)
print("Test 2: Compare CR vs BS")
print("=" * 60)

# Fit with B-splines
gam_bs = mgcv_rust.GAM()
result_bs = gam_bs.fit_auto(X, y, k=[10], method='REML', bs='bs')
pred_bs = gam_bs.predict(X)

print(f"CR splines:")
print(f"  Lambda: {result_cr['lambda']:.6f}")
print(f"  Deviance: {result_cr['deviance']:.4f}")
print(f"\nB-splines:")
print(f"  Lambda: {result_bs['lambda']:.6f}")
print(f"  Deviance: {result_bs['deviance']:.4f}")

# Compare predictions
correlation = np.corrcoef(pred_cr, pred_bs)[0, 1]
rmse_diff = np.sqrt(np.mean((pred_cr - pred_bs)**2))
print(f"\nComparison:")
print(f"  Correlation: {correlation:.6f}")
print(f"  RMSE difference: {rmse_diff:.6f}")

if correlation > 0.95:
    print("  ✓ Predictions are very similar (correlation > 0.95)")
else:
    print("  ⚠ Predictions differ more than expected")

# Test 3: Extrapolation behavior
print("\n" + "=" * 60)
print("Test 3: Extrapolation Behavior")
print("=" * 60)

x_extrap = np.linspace(-0.2, 1.2, 50)
X_extrap = x_extrap.reshape(-1, 1)
pred_extrap = gam_cr.predict(X_extrap)

has_zeros = np.any(np.abs(pred_extrap) < 1e-6)
print(f"Has zeros in extrapolation: {'❌ Yes' if has_zeros else '✅ No'}")
print(f"Min prediction: {pred_extrap.min():.6f}")
print(f"Max prediction: {pred_extrap.max():.6f}")

# Test 4: Multi-variable GAM with CR splines
print("\n" + "=" * 60)
print("Test 4: Multi-variable GAM with CR splines")
print("=" * 60)

np.random.seed(42)
n = 200
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)
y_true = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2)
y_multi = y_true + np.random.normal(0, 0.2, n)

X_multi = np.column_stack([x1, x2])

gam_multi = mgcv_rust.GAM()
result_multi = gam_multi.fit_auto(X_multi, y_multi, k=[10, 10], method='REML', bs='cr')
pred_multi = gam_multi.predict(X_multi)
all_lambdas = gam_multi.get_all_lambdas()

print(f"✓ Multi-variable CR GAM fitted successfully")
print(f"  Lambda 1: {all_lambdas[0]:.6f}")
print(f"  Lambda 2: {all_lambdas[1]:.6f}")
print(f"  Deviance: {result_multi['deviance']:.4f}")

# Evaluate fit quality
correlation_multi = np.corrcoef(pred_multi, y_true)[0, 1]
rmse_multi = np.sqrt(np.mean((pred_multi - y_true)**2))
print(f"\nFit quality:")
print(f"  Correlation with true function: {correlation_multi:.6f}")
print(f"  RMSE: {rmse_multi:.6f}")

if correlation_multi > 0.7:
    print("  ✓ Good fit (correlation > 0.7)")

# Test 5: Compare with mgcv (if rpy2 is available)
print("\n" + "=" * 60)
print("Test 5: Compare with R's mgcv (CR splines)")
print("=" * 60)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()

    # Import mgcv
    mgcv = importr('mgcv')

    print("✓ rpy2 and mgcv available")

    # Fit with R mgcv using cr splines
    ro.globalenv['x_r'] = x
    ro.globalenv['y_r'] = y
    ro.r('gam_fit <- gam(y_r ~ s(x_r, k=10, bs="cr"), method="REML")')
    pred_r = np.array(ro.r('predict(gam_fit)'))
    lambda_r = np.array(ro.r('gam_fit$sp'))[0]

    print(f"\nOur CR implementation:")
    print(f"  Lambda: {result_cr['lambda']:.6f}")
    print(f"\nR mgcv (CR):")
    print(f"  Lambda: {lambda_r:.6f}")

    # Compare predictions
    corr_mgcv = np.corrcoef(pred_cr, pred_r)[0, 1]
    rmse_mgcv = np.sqrt(np.mean((pred_cr - pred_r)**2))
    lambda_ratio = result_cr['lambda'] / lambda_r

    print(f"\nComparison with mgcv:")
    print(f"  Correlation: {corr_mgcv:.6f}")
    print(f"  RMSE difference: {rmse_mgcv:.6f}")
    print(f"  Lambda ratio (ours/mgcv): {lambda_ratio:.4f}")

    if corr_mgcv > 0.95:
        print("  ✓ Predictions match well (correlation > 0.95)")
    else:
        print("  ⚠ Predictions differ from mgcv")

    if 0.1 < lambda_ratio < 10.0:
        print("  ✓ Lambda values are similar (ratio within 0.1-10)")
    else:
        print("  ⚠ Lambda values differ significantly")

except ImportError:
    print("⚠ rpy2 not available - skipping mgcv comparison")
    print("  Install with: pip install rpy2")
except Exception as e:
    print(f"⚠ Error during mgcv comparison: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
