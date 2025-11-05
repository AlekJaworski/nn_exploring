#!/usr/bin/env python3
"""
Example showing the two convenient APIs for GAM fitting:
1. fit_auto(X, y, k=[...]) - Pythonic k-list approach
2. fit_formula(X, y, formula="s(0, k=10)") - R/mgcv-like formula approach
"""

import numpy as np
import mgcv_rust

def main():
    print("=" * 70)
    print("GAM Convenient APIs Demo")
    print("=" * 70)

    np.random.seed(42)
    n = 300

    # Generate data
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.5 * np.random.randn(n)

    X = x.reshape(-1, 1)

    print(f"\nData: {n} observations")
    print(f"True function: sin(2πx)")
    print(f"Noise level: 0.5\n")

    # ===================================================================
    # API 1: fit_auto with k list (Pythonic)
    # ===================================================================
    print("=" * 70)
    print("API 1: fit_auto(X, y, k=[15], method='GCV')")
    print("=" * 70)
    print("\nUsage:")
    print("  gam = GAM()")
    print("  result = gam.fit_auto(X, y, k=[15], method='GCV')")
    print("\nAdvantages:")
    print("  - Simple and Pythonic")
    print("  - Just specify k values for each column")
    print("  - Automatically determines x_min, x_max from data")
    print("\n" + "-" * 70)

    gam1 = mgcv_rust.GAM()
    result1 = gam1.fit_auto(X, y, k=[15], method='GCV')

    print(f"Results:")
    print(f"  Selected λ: {result1['lambda']:.6f}")
    print(f"  Deviance: {result1['deviance']:.4f}")
    print(f"  Fitted: {result1['fitted']}")

    # Compute RMSE
    y_pred1 = gam1.predict(X)
    rmse1 = np.sqrt(np.mean((y_pred1 - y_true)**2))
    print(f"  RMSE vs true: {rmse1:.4f}")

    # ===================================================================
    # API 2: fit_formula with R-like syntax
    # ===================================================================
    print("\n" + "=" * 70)
    print("API 2: fit_formula(X, y, formula='s(0, k=15)', method='GCV')")
    print("=" * 70)
    print("\nUsage:")
    print("  gam = GAM()")
    print("  result = gam.fit_formula(X, y, formula='s(0, k=15)', method='GCV')")
    print("\nAdvantages:")
    print("  - R/mgcv-like syntax - familiar to R users")
    print("  - Explicit about column indices")
    print("  - Can easily specify different k for different predictors")
    print("\n" + "-" * 70)

    gam2 = mgcv_rust.GAM()
    result2 = gam2.fit_formula(X, y, formula="s(0, k=15)", method='GCV')

    print(f"Results:")
    print(f"  Selected λ: {result2['lambda']:.6f}")
    print(f"  Deviance: {result2['deviance']:.4f}")
    print(f"  Fitted: {result2['fitted']}")

    # Compute RMSE
    y_pred2 = gam2.predict(X)
    rmse2 = np.sqrt(np.mean((y_pred2 - y_true)**2))
    print(f"  RMSE vs true: {rmse2:.4f}")

    # ===================================================================
    # Comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"\nBoth APIs produce identical results:")
    print(f"  fit_auto    λ: {result1['lambda']:.6f}")
    print(f"  fit_formula λ: {result2['lambda']:.6f}")
    print(f"  Difference: {abs(result1['lambda'] - result2['lambda']):.10f}")

    # ===================================================================
    # Different k values
    # ===================================================================
    print("\n" + "=" * 70)
    print("Testing different k values")
    print("=" * 70)

    k_values = [10, 15, 20, 25]
    print(f"\n{'k':<8} {'λ (GCV)':<12} {'λ (REML)':<12} {'RMSE':<10}")
    print("-" * 70)

    for k in k_values:
        # GCV
        gam_gcv = mgcv_rust.GAM()
        res_gcv = gam_gcv.fit_auto(X, y, k=[k], method='GCV')
        y_pred_gcv = gam_gcv.predict(X)
        rmse_gcv = np.sqrt(np.mean((y_pred_gcv - y_true)**2))

        # REML
        gam_reml = mgcv_rust.GAM()
        res_reml = gam_reml.fit_auto(X, y, k=[k], method='REML')

        print(f"{k:<8} {res_gcv['lambda']:<12.6f} {res_reml['lambda']:<12.6f} {rmse_gcv:<10.4f}")

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("""
1. fit_auto() is simpler - just pass k as a list
2. fit_formula() is more explicit and R-like
3. Both produce identical results
4. Higher k allows more flexibility (but λ controls actual smoothness)
5. REML typically selects lower λ than GCV (more smoothing)

Note: Multi-predictor GAMs (X with multiple columns) not yet implemented
      for automatic smoothing parameter selection.
    """)

if __name__ == "__main__":
    main()
