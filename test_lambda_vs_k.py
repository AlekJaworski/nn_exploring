#!/usr/bin/env python3
"""
Test lambda optimization across different k values to check for overfitting to k=10.

This script compares lambda values found by mgcv_rust vs R's mgcv for different
basis sizes (k values) to see if lambda tuning is specific to k=10.
"""

import numpy as np
import mgcv_rust
import matplotlib.pyplot as plt

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("Warning: rpy2 not available. Install with: pip install rpy2")
    exit(1)

def generate_test_data(n=100, noise_level=0.2, seed=42):
    """Generate test data: y = sin(2πx) + noise"""
    np.random.seed(seed)
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + noise_level * np.random.randn(n)
    return x, y, y_true

def test_lambda_for_k(k_value, x, y):
    """Test lambda value found by both rust and R mgcv for a given k"""

    # Fit with mgcv_rust
    X = x.reshape(-1, 1)
    gam_rust = mgcv_rust.GAM()
    result_rust = gam_rust.fit_auto(X, y, k=[k_value], method='REML')
    lambda_rust = result_rust['lambda']
    pred_rust = gam_rust.predict(X)
    deviance_rust = result_rust['deviance']

    # Fit with R mgcv
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_r'] = x
        ro.globalenv['y_r'] = y
        ro.globalenv['k_val'] = k_value
        ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="bs"), method="REML")')
        lambda_r = np.array(ro.r('gam_fit$sp'))[0]
        pred_r = np.array(ro.r('predict(gam_fit)'))
        deviance_r = np.array(ro.r('gam_fit$deviance'))[0]

    # Compare predictions
    corr = np.corrcoef(pred_rust, pred_r)[0, 1]
    rmse_diff = np.sqrt(np.mean((pred_rust - pred_r)**2))

    # Compute RMSE vs true function
    _, _, y_true = generate_test_data(len(x), 0.0, 42)  # Get true function
    rmse_rust_true = np.sqrt(np.mean((pred_rust - y_true)**2))
    rmse_r_true = np.sqrt(np.mean((pred_r - y_true)**2))

    return {
        'k': k_value,
        'lambda_rust': lambda_rust,
        'lambda_r': lambda_r,
        'lambda_ratio': lambda_rust / lambda_r,
        'corr': corr,
        'rmse_diff': rmse_diff,
        'deviance_rust': deviance_rust,
        'deviance_r': deviance_r,
        'rmse_rust_true': rmse_rust_true,
        'rmse_r_true': rmse_r_true,
        'pred_rust': pred_rust,
        'pred_r': pred_r,
    }

def main():
    print("=" * 80)
    print("Testing Lambda Optimization Across Different k Values")
    print("=" * 80)
    print()

    # Import R mgcv
    try:
        mgcv = importr('mgcv')
    except Exception as e:
        print(f"Error: R mgcv package not available: {e}")
        exit(1)

    # Generate test data
    print("Generating test data: y = sin(2πx) + noise")
    x, y, y_true = generate_test_data(n=100, noise_level=0.2, seed=42)
    print(f"  n = {len(x)}, noise_level = 0.2")
    print()

    # Test different k values
    k_values = [5, 7, 10, 12, 15, 20, 25, 30]
    results = []

    print("Testing different k values:")
    print("-" * 80)
    print(f"{'k':<5} {'λ_rust':<12} {'λ_mgcv':<12} {'Ratio':<8} {'Corr':<8} {'RMSE_diff':<10}")
    print("-" * 80)

    for k in k_values:
        result = test_lambda_for_k(k, x, y)
        results.append(result)

        print(f"{result['k']:<5} {result['lambda_rust']:<12.6f} {result['lambda_r']:<12.6f} "
              f"{result['lambda_ratio']:<8.4f} {result['corr']:<8.6f} {result['rmse_diff']:<10.6f}")

    print("-" * 80)
    print()

    # Analyze results
    print("Analysis:")
    print("-" * 80)

    # Check if lambda ratio varies significantly with k
    ratios = [r['lambda_ratio'] for r in results]
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_min = np.min(ratios)
    ratio_max = np.max(ratios)

    print(f"Lambda ratio (rust/mgcv) statistics:")
    print(f"  Mean:   {ratio_mean:.4f}")
    print(f"  Std:    {ratio_std:.4f}")
    print(f"  Min:    {ratio_min:.4f} (k={results[np.argmin(ratios)]['k']})")
    print(f"  Max:    {ratio_max:.4f} (k={results[np.argmax(ratios)]['k']})")
    print(f"  Range:  {ratio_max - ratio_min:.4f}")
    print()

    # Check prediction quality
    corrs = [r['corr'] for r in results]
    print(f"Prediction correlation:")
    print(f"  Mean:   {np.mean(corrs):.6f}")
    print(f"  Min:    {np.min(corrs):.6f} (k={results[np.argmin(corrs)]['k']})")
    print()

    # Check RMSE vs true function
    rmse_rust = [r['rmse_rust_true'] for r in results]
    rmse_r = [r['rmse_r_true'] for r in results]

    print(f"RMSE vs true function:")
    print(f"  Rust - Mean: {np.mean(rmse_rust):.6f}, Best k: {results[np.argmin(rmse_rust)]['k']}")
    print(f"  R    - Mean: {np.mean(rmse_r):.6f}, Best k: {results[np.argmin(rmse_r)]['k']}")
    print()

    # Warning if ratio varies too much
    if ratio_std > 0.5 or ratio_max / ratio_min > 2.0:
        print("⚠️  WARNING: Lambda ratio varies significantly with k!")
        print("   This suggests potential overfitting to a specific k value.")
        print()

    # Create visualizations
    create_plots(results, x, y_true)

    print("=" * 80)
    print("Plots saved to:")
    print("  - lambda_vs_k_comparison.png")
    print("  - predictions_comparison.png")
    print("=" * 80)

def create_plots(results, x, y_true):
    """Create visualization plots"""

    k_values = [r['k'] for r in results]
    lambda_rust = [r['lambda_rust'] for r in results]
    lambda_r = [r['lambda_r'] for r in results]
    lambda_ratio = [r['lambda_ratio'] for r in results]
    corr = [r['corr'] for r in results]
    rmse_rust = [r['rmse_rust_true'] for r in results]
    rmse_r = [r['rmse_r_true'] for r in results]

    # Figure 1: Lambda values and ratios
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Lambda values
    ax = axes[0, 0]
    ax.plot(k_values, lambda_rust, 'b-o', linewidth=2, markersize=8, label='mgcv_rust')
    ax.plot(k_values, lambda_r, 'r--s', linewidth=2, markersize=8, label='R mgcv')
    ax.set_xlabel('k (number of basis functions)', fontsize=12)
    ax.set_ylabel('λ (smoothing parameter)', fontsize=12)
    ax.set_title('Lambda Values vs k', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Lambda ratio
    ax = axes[0, 1]
    ax.plot(k_values, lambda_ratio, 'g-^', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('k (number of basis functions)', fontsize=12)
    ax.set_ylabel('Ratio (λ_rust / λ_mgcv)', fontsize=12)
    ax.set_title('Lambda Ratio vs k', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction correlation
    ax = axes[1, 0]
    ax.plot(k_values, corr, 'm-d', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Minimum acceptable (0.95)')
    ax.set_xlabel('k (number of basis functions)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Prediction Correlation vs k', fontsize=14)
    ax.set_ylim([0.9, 1.0])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: RMSE vs true function
    ax = axes[1, 1]
    ax.plot(k_values, rmse_rust, 'b-o', linewidth=2, markersize=8, label='mgcv_rust')
    ax.plot(k_values, rmse_r, 'r--s', linewidth=2, markersize=8, label='R mgcv')
    ax.set_xlabel('k (number of basis functions)', fontsize=12)
    ax.set_ylabel('RMSE vs true function', fontsize=12)
    ax.set_title('RMSE vs True Function', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lambda_vs_k_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: lambda_vs_k_comparison.png")

    # Figure 2: Predictions for different k values
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Show predictions for representative k values
    k_to_show = [5, 10, 15, 20, 25, 30]

    for idx, k in enumerate(k_to_show):
        if idx >= len(axes):
            break

        # Find result for this k
        result = next((r for r in results if r['k'] == k), None)
        if result is None:
            continue

        ax = axes[idx]
        ax.plot(x, y_true, 'k--', linewidth=1.5, label='True function', alpha=0.5)
        ax.plot(x, result['pred_rust'], 'b-', linewidth=2, label='mgcv_rust', alpha=0.8)
        ax.plot(x, result['pred_r'], 'r--', linewidth=2, label='R mgcv', alpha=0.8)

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'k={k}, λ_ratio={result["lambda_ratio"]:.3f}, corr={result["corr"]:.4f}',
                    fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: predictions_comparison.png")

if __name__ == '__main__':
    main()
