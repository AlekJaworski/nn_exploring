#!/usr/bin/env python3
"""
Test the sum-to-zero constraint implementation for CR splines
"""

import numpy as np
import sys

# Add the target directory to Python path
sys.path.insert(0, "target/debug")

try:
    import mgcv_rust

    print("=" * 70)
    print("Testing CR Spline with Sum-to-Zero Constraint")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + np.random.normal(0, 0.1, n)

    X = x.reshape(-1, 1)

    # Fit with CR splines (should now use k-1=9 basis functions with constraint)
    print("\nFitting GAM with CR splines (k=10, constrained to k-1=9)...")
    gam_cr = mgcv_rust.GAM()
    result_cr = gam_cr.fit_auto(X, y, k=[10], method='REML', bs='cr')

    print(f"\nResults:")
    print(f"  Lambda: {result_cr['lambda']:.6f}")
    print(f"  Deviance: {result_cr['deviance']:.4f}")
    print(f"  Number of basis functions (constrained): 9")

    # Check predictions
    pred_cr = gam_cr.predict(X)
    rmse = np.sqrt(np.mean((pred_cr - y_true)**2))
    correlation = np.corrcoef(pred_cr, y_true)[0, 1]

    print(f"\nFit quality:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation with true function: {correlation:.4f}")

    # Compare with mgcv (if rpy2 is available)
    print("\n" + "=" * 70)
    print("Comparing with R's mgcv")
    print("=" * 70)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        numpy2ri.activate()

        mgcv = importr('mgcv')

        # Fit with R mgcv using cr splines
        ro.globalenv['x_r'] = x
        ro.globalenv['y_r'] = y
        ro.r('gam_fit <- gam(y_r ~ s(x_r, k=10, bs="cr"), method="REML")')
        pred_r = np.array(ro.r('predict(gam_fit)'))
        lambda_r = np.array(ro.r('gam_fit$sp'))[0]
        deviance_r = np.array(ro.r('deviance(gam_fit)'))[0]

        print(f"\nOur implementation (Rust):")
        print(f"  Lambda: {result_cr['lambda']:.6f}")
        print(f"  Deviance: {result_cr['deviance']:.4f}")

        print(f"\nR mgcv (reference):")
        print(f"  Lambda: {lambda_r:.6f}")
        print(f"  Deviance: {deviance_r:.4f}")

        # Compare predictions
        corr_mgcv = np.corrcoef(pred_cr, pred_r)[0, 1]
        rmse_mgcv = np.sqrt(np.mean((pred_cr - pred_r)**2))
        lambda_ratio = result_cr['lambda'] / lambda_r
        deviance_ratio = result_cr['deviance'] / deviance_r

        print(f"\nComparison:")
        print(f"  Prediction correlation: {corr_mgcv:.6f}")
        print(f"  Prediction RMSE difference: {rmse_mgcv:.6f}")
        print(f"  Lambda ratio (ours/mgcv): {lambda_ratio:.4f}")
        print(f"  Deviance ratio (ours/mgcv): {deviance_ratio:.4f}")

        # Success criteria
        print(f"\nSuccess criteria:")
        if corr_mgcv > 0.99:
            print(f"  ✅ Predictions match well (correlation > 0.99)")
        else:
            print(f"  ⚠️  Predictions differ (correlation = {corr_mgcv:.4f})")

        if 0.5 < lambda_ratio < 2.0:
            print(f"  ✅ Lambda values are similar (ratio within 0.5-2.0)")
        elif 0.1 < lambda_ratio < 10.0:
            print(f"  ⚠️  Lambda values differ (ratio = {lambda_ratio:.4f})")
        else:
            print(f"  ❌ Lambda values differ significantly (ratio = {lambda_ratio:.4f})")

        if 0.9 < deviance_ratio < 1.1:
            print(f"  ✅ Deviance values match closely (ratio within 0.9-1.1)")
        elif 0.5 < deviance_ratio < 2.0:
            print(f"  ⚠️  Deviance values differ (ratio = {deviance_ratio:.4f})")
        else:
            print(f"  ❌ Deviance values differ significantly (ratio = {deviance_ratio:.4f})")

    except ImportError:
        print("⚠️ rpy2 not available - skipping mgcv comparison")
        print("  Install with: pip install rpy2")
    except Exception as e:
        print(f"⚠️ Error during mgcv comparison: {e}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

except ImportError as e:
    print(f"Error importing mgcv_rust: {e}")
    print("Make sure to build with: cargo build --features python")
    sys.exit(1)
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
