import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import mgcv_rust
    return mo, np, plt, mgcv_rust


@app.cell
def __(mo):
    mo.md(
        """
        # 1-Variable GAM Test

        This notebook tests the GAM implementation with:
        - Fitting to a sine wave with noise
        - Predictions within training range
        - **Extrapolation beyond training range** (validates boundary fix)
        - Visual comparison of predictions vs true function
        """
    )
    return


@app.cell
def __(mo, np):
    mo.md("## Generate Training Data")

    # Generate training data: sine wave in [0, 1]
    np.random.seed(42)
    n_train = 100
    x_train = np.linspace(0, 1, n_train)
    y_true_train = np.sin(2 * np.pi * x_train)
    noise = 0.2 * np.random.randn(n_train)
    y_train = y_true_train + noise

    mo.md(f"""
    **Training Data:**
    - Function: y = sin(2πx) + noise
    - Range: x ∈ [0, 1]
    - N = {n_train} observations
    - Noise: σ = 0.2
    """)
    return n_train, x_train, y_true_train, y_train, noise


@app.cell
def __(mo, plt, x_train, y_train, y_true_train):
    mo.md("### Visualize Training Data")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(x_train, y_train, alpha=0.5, s=20, label='Training data (noisy)')
    ax1.plot(x_train, y_true_train, 'r-', linewidth=2, label='True function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Training Data: y = sin(2πx) + noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    fig1
    return ax1, fig1


@app.cell
def __(mo, mgcv_rust, x_train, y_train, np):
    mo.md("## Fit GAM")

    # Fit GAM
    X_train = x_train.reshape(-1, 1)
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X_train, y_train, k=[10], method='REML')

    # Training predictions
    y_pred_train = gam.predict(X_train)
    train_rmse = np.sqrt(np.mean((y_pred_train - y_train)**2))
    train_r2 = 1 - np.var(y_pred_train - y_train) / np.var(y_train)

    mo.md(f"""
    **Fit Results:**
    - λ = {result['lambda']:.6f}
    - Deviance = {result['deviance']:.6f}
    - RMSE = {train_rmse:.4f}
    - R² = {train_r2:.4f}
    """)
    return X_train, gam, result, y_pred_train, train_rmse, train_r2


@app.cell
def __(mo, plt, x_train, y_train, y_pred_train, y_true_train):
    mo.md("### GAM Fit Visualization")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.scatter(x_train, y_train, alpha=0.3, s=20, label='Training data', color='gray')
    ax2.plot(x_train, y_true_train, 'r-', linewidth=2, label='True function', alpha=0.7)
    ax2.plot(x_train, y_pred_train, 'b-', linewidth=2, label='GAM prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('GAM Fit (Training Range)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    fig2
    return ax2, fig2


@app.cell
def __(mo, gam, np):
    mo.md("## Extrapolation Test")

    # Full range including extrapolation
    x_full = np.linspace(-0.2, 1.2, 200)
    X_full = x_full.reshape(-1, 1)
    y_pred_full = gam.predict(X_full)
    y_true_full = np.sin(2 * np.pi * x_full)

    # Check for issues
    has_zeros = np.any(np.abs(y_pred_full) < 1e-6)
    has_nans = np.any(np.isnan(y_pred_full))

    # Extrapolation regions
    left_extrap = x_full < 0
    right_extrap = x_full > 1
    in_range = (x_full >= 0) & (x_full <= 1)

    mo.md(f"""
    **Full Range: x ∈ [-0.2, 1.2]**
    - Contains zeros: {has_zeros} {'❌' if has_zeros else '✅'}
    - Contains NaNs: {has_nans} {'❌' if has_nans else '✅'}
    - Min prediction: {np.min(y_pred_full):.4f}
    - Max prediction: {np.max(y_pred_full):.4f}
    """)
    return (
        X_full,
        has_nans,
        has_zeros,
        in_range,
        left_extrap,
        right_extrap,
        x_full,
        y_pred_full,
        y_true_full,
    )


@app.cell
def __(mo, plt, x_full, y_pred_full, y_true_full, left_extrap, right_extrap, in_range):
    mo.md("### Extrapolation Visualization")

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Plot true function
    ax3.plot(x_full, y_true_full, 'r-', linewidth=2, label='True function', alpha=0.7)

    # Plot predictions with different colors for regions
    ax3.plot(x_full[left_extrap], y_pred_full[left_extrap], 'b-', linewidth=2.5,
             label='GAM (left extrapolation)')
    ax3.plot(x_full[in_range], y_pred_full[in_range], 'g-', linewidth=2.5,
             label='GAM (training range)')
    ax3.plot(x_full[right_extrap], y_pred_full[right_extrap], 'm-', linewidth=2.5,
             label='GAM (right extrapolation)')

    # Mark training range boundaries
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=1, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvspan(-0.2, 0, alpha=0.1, color='blue', label='Extrapolation region')
    ax3.axvspan(1, 1.2, alpha=0.1, color='magenta')

    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('GAM Predictions: Training Range [0, 1] + Extrapolation', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    fig3
    return ax3, fig3


@app.cell
def __(mo, np, x_full, y_pred_full, y_true_full):
    mo.md("## Prediction Error Analysis")

    # Compute errors
    errors = np.abs(y_pred_full - y_true_full)

    # Split by region
    left_mask = x_full < 0
    in_mask = (x_full >= 0) & (x_full <= 1)
    right_mask = x_full > 1

    left_error = np.mean(errors[left_mask])
    in_error = np.mean(errors[in_mask])
    right_error = np.mean(errors[right_mask])

    mo.md(f"""
    **Mean Absolute Errors by Region:**
    - Left extrapolation (x < 0): {left_error:.4f}
    - Training range [0, 1]: {in_error:.4f}
    - Right extrapolation (x > 1): {right_error:.4f}
    """)
    return (
        errors,
        in_error,
        in_mask,
        left_error,
        left_mask,
        right_error,
        right_mask,
    )


@app.cell
def __(mo, plt, x_full, errors, left_mask, in_mask, right_mask):
    mo.md("### Error Distribution")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(x_full[left_mask], errors[left_mask], 'b-', linewidth=2, alpha=0.7, label='Left extrapolation')
    ax4.plot(x_full[in_mask], errors[in_mask], 'g-', linewidth=2, alpha=0.7, label='Training range')
    ax4.plot(x_full[right_mask], errors[right_mask], 'm-', linewidth=2, alpha=0.7, label='Right extrapolation')
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(x=1, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|Prediction Error|')
    ax4.set_title('Absolute Error vs Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    fig4
    return ax4, fig4


@app.cell
def __(mo, train_r2, has_zeros, has_nans, left_error, in_error, right_error):
    mo.md("## Summary")

    checks = {
        "Fit succeeded": True,
        "R² > 0.8": train_r2 > 0.8,
        "No zero predictions": not has_zeros,
        "No NaN predictions": not has_nans,
        "Left extrapolation reasonable": left_error < 0.5,
        "Training range accurate": in_error < 0.2,
        "Right extrapolation reasonable": right_error < 0.5,
    }

    # Create status table
    status_lines = []
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        status_lines.append(f"- {status} {check}")

    all_passed = all(checks.values())

    summary = "\n".join(status_lines)

    if all_passed:
        result_msg = "## 🎉 ALL TESTS PASSED!"
    else:
        result_msg = "## ⚠️ SOME TESTS FAILED"

    mo.md(f"""
    {result_msg}

    {summary}
    """)
    return all_passed, check, checks, passed, result_msg, status_lines, summary


if __name__ == "__main__":
    app.run()
