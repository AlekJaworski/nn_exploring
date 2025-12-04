#!/usr/bin/env python3
"""
Example using mgcv_rust from Python with matplotlib visualization
"""

import numpy as np
import matplotlib.pyplot as plt

# This will work after building with: maturin develop
import mgcv_rust

def main():
    print("=" * 60)
    print("GAM Example with Python + matplotlib")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n = 300
    x = np.linspace(0, 1, n)

    # True function: sine wave
    true_y = np.sin(2 * np.pi * x)

    # Add noise
    noise = 0.5 * np.random.randn(n)
    y = true_y + noise

    print(f"\nData: {n} observations")
    print(f"Signal: sin(2πx)")
    print(f"Noise level: 0.5")

    # Create GAM model
    gam = mgcv_rust.GAM()

    # Add cubic spline smooth (k=15 basis functions)
    gam.add_cubic_spline("x", num_basis=15, x_min=0.0, x_max=1.0)

    # Fit the model
    print("\nFitting GAM with GCV...")
    x_matrix = x.reshape(-1, 1)
    result = gam.fit(x_matrix, y, method="GCV", max_iter=10)

    print(f"  λ (smoothing parameter): {result['lambda']:.6f}")
    #
    #
    print(f"  Deviance: {result['deviance']:.4f}")

    # Make predictions
    x_pred = np.linspace(0, 1, 200).reshape(-1, 1)
    y_pred = gam.predict(x_pred)

    # Compute RMSE vs true function
    true_y_pred = np.sin(2 * np.pi * x_pred.flatten())
    rmse = np.sqrt(np.mean((y_pred - true_y_pred)**2))
    print(f"  RMSE vs true function: {rmse:.4f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Fitted curve
    ax = axes[0, 0]
    ax.scatter(x, y, alpha=0.3, s=10, label='Data', color='gray')
    ax.plot(x_pred, true_y_pred, 'r-', linewidth=2, label='True function', alpha=0.7)
    ax.plot(x_pred, y_pred, 'b-', linewidth=2, label='GAM fit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'GAM Fit (λ = {result["lambda"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax = axes[0, 1]
    fitted_values = gam.get_fitted_values()
    residuals = y - fitted_values
    ax.scatter(fitted_values, residuals, alpha=0.3, s=10)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)

    # Plot 3: Compare REML vs GCV
    print("\nComparing REML vs GCV...")

    # Fit with REML
    gam_reml = mgcv_rust.GAM()
    gam_reml.add_cubic_spline("x", num_basis=15, x_min=0.0, x_max=1.0)
    result_reml = gam_reml.fit(x_matrix, y, method="REML", max_iter=10)
    y_pred_reml = gam_reml.predict(x_pred)

    print(f"  REML λ: {result_reml['lambda']:.6f}")
    print(f"  GCV λ:  {result['lambda']:.6f}")

    ax = axes[1, 0]
    ax.scatter(x, y, alpha=0.3, s=10, label='Data', color='gray')
    ax.plot(x_pred, true_y_pred, 'r-', linewidth=2, label='True', alpha=0.7)
    ax.plot(x_pred, y_pred, 'b-', linewidth=2, label=f'GCV (λ={result["lambda"]:.3f})')
    ax.plot(x_pred, y_pred_reml, 'g--', linewidth=2, label=f'REML (λ={result_reml["lambda"]:.3f})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('REML vs GCV Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Lambda effect demonstration
    ax = axes[1, 1]

    # Test different lambda values manually by varying num_basis
    # (more basis → potentially different lambda selection)
    lambda_values = []
    basis_counts = [10, 15, 20, 25, 30]

    for k in basis_counts:
        gam_test = mgcv_rust.GAM()
        gam_test.add_cubic_spline("x", num_basis=k, x_min=0.0, x_max=1.0)
        res = gam_test.fit(x_matrix, y, method="GCV", max_iter=10)
        lambda_values.append(res['lambda'])

    ax.plot(basis_counts, lambda_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of basis functions (k)')
    ax.set_ylabel('Selected λ (GCV)')
    ax.set_title('Lambda Selection vs Basis Complexity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gam_python_example.png', dpi=150, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Plot saved to: gam_python_example.png")
    print("=" * 60)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
