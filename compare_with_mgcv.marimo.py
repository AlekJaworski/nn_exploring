import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import mgcv_rust

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        numpy2ri.activate()
        HAS_RPY2 = True
    except ImportError:
        HAS_RPY2 = False

    if HAS_RPY2:
        try:
            mgcv = importr('mgcv')
            stats = importr('stats')
        except:
            HAS_RPY2 = False

    return mo, np, plt, mgcv_rust, ro, HAS_RPY2, mgcv, stats, numpy2ri


@app.cell
def __(mo, HAS_RPY2):
    if not HAS_RPY2:
        mo.md("""
        ⚠️ **rpy2 not available**

        Install with:
        ```bash
        pip install rpy2
        ```

        Also ensure R and mgcv package are installed:
        ```R
        install.packages("mgcv")
        ```
        """)
    else:
        mo.md("""
        # Comparing mgcv_rust vs R's mgcv

        This notebook compares the Rust implementation against R's mgcv package.
        """)
    return


@app.cell
def __(mo, np):
    mo.md("## Generate Test Data")

    # Interactive parameters
    n_points = mo.ui.slider(50, 200, value=100, label="Number of points")
    noise_level = mo.ui.slider(0.0, 0.5, value=0.2, step=0.05, label="Noise level")
    k_basis = mo.ui.slider(5, 20, value=10, label="Number of basis functions (k)")

    mo.vstack([n_points, noise_level, k_basis])
    return n_points, noise_level, k_basis


@app.cell
def __(np, n_points, noise_level):
    # Generate data
    np.random.seed(42)
    n = n_points.value
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x)
    noise = noise_level.value * np.random.randn(n)
    y = y_true + noise

    return n, x, y_true, y, noise


@app.cell
def __(mo, plt, x, y, y_true):
    mo.md("### Training Data")

    fig_data, ax_data = plt.subplots(figsize=(10, 5))
    ax_data.scatter(x, y, alpha=0.5, s=20, label='Data (with noise)', color='gray')
    ax_data.plot(x, y_true, 'r-', linewidth=2, label='True function', alpha=0.7)
    ax_data.set_xlabel('x')
    ax_data.set_ylabel('y')
    ax_data.set_title('Training Data: y = sin(2πx) + noise')
    ax_data.legend()
    ax_data.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_data
    return ax_data, fig_data


@app.cell
def __(mo, HAS_RPY2, mgcv_rust, x, y, k_basis, ro, np):
    if not HAS_RPY2:
        mo.md("⚠️ Cannot fit models without rpy2")
    else:
        mo.md("## Fit Models")

        # Fit with mgcv_rust
        X = x.reshape(-1, 1)
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X, y, k=[k_basis.value], method='REML')
        pred_rust = gam_rust.predict(X)
        lambda_rust = result_rust['lambda']

        # Fit with R mgcv
        ro.globalenv['x_r'] = x
        ro.globalenv['y_r'] = y
        ro.globalenv['k_val'] = k_basis.value
        ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="cr"), method="REML")')
        pred_r = np.array(ro.r('predict(gam_fit)'))
        lambda_r = np.array(ro.r('gam_fit$sp'))[0]

        mo.md(f"""
        **Fit Complete**

        | Implementation | λ (smoothing param) | Deviance |
        |----------------|---------------------|----------|
        | mgcv_rust      | {lambda_rust:.6f}   | {result_rust['deviance']:.4f} |
        | R mgcv         | {lambda_r:.6f}      | -        |
        | Ratio (Rust/R) | {lambda_rust/lambda_r:.4f} | - |
        """)
    return (
        X,
        gam_rust,
        result_rust,
        pred_rust,
        lambda_rust,
        pred_r,
        lambda_r,
    )


@app.cell
def __(mo, HAS_RPY2, plt, x, y, pred_rust, pred_r, y_true):
    if not HAS_RPY2:
        mo.md("")
    else:
        mo.md("### Prediction Comparison")

        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))

        ax_pred.scatter(x, y, alpha=0.3, s=20, label='Data', color='lightgray')
        ax_pred.plot(x, y_true, 'k--', linewidth=1.5, label='True function', alpha=0.5)
        ax_pred.plot(x, pred_rust, 'b-', linewidth=2.5, label='mgcv_rust', alpha=0.8)
        ax_pred.plot(x, pred_r, 'r--', linewidth=2, label='R mgcv', alpha=0.8)

        ax_pred.set_xlabel('x', fontsize=12)
        ax_pred.set_ylabel('y', fontsize=12)
        ax_pred.set_title('Prediction Comparison', fontsize=14)
        ax_pred.legend()
        ax_pred.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_pred
    return ax_pred, fig_pred


@app.cell
def __(mo, HAS_RPY2, np, pred_rust, pred_r):
    if not HAS_RPY2:
        mo.md("")
    else:
        # Compute comparison metrics
        corr = np.corrcoef(pred_rust, pred_r)[0, 1]
        rmse_diff = np.sqrt(np.mean((pred_rust - pred_r)**2))
        max_diff = np.max(np.abs(pred_rust - pred_r))

        mo.md(f"""
        ### Prediction Metrics

        | Metric | Value | Status |
        |--------|-------|--------|
        | Correlation | {corr:.6f} | {'✅ Excellent' if corr > 0.99 else '⚠️ Good' if corr > 0.95 else '❌ Poor'} |
        | RMSE difference | {rmse_diff:.6f} | {'✅ Low' if rmse_diff < 0.1 else '⚠️ Moderate'} |
        | Max difference | {max_diff:.6f} | {'✅ Low' if max_diff < 0.2 else '⚠️ Moderate'} |
        """)
    return corr, rmse_diff, max_diff


@app.cell
def __(mo, HAS_RPY2, plt, x, pred_rust, pred_r):
    if not HAS_RPY2:
        mo.md("")
    else:
        mo.md("### Residual Difference")

        diff = pred_rust - pred_r

        fig_diff, (ax_diff1, ax_diff2) = plt.subplots(1, 2, figsize=(14, 5))

        # Difference plot
        ax_diff1.plot(x, diff, 'g-', linewidth=2)
        ax_diff1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_diff1.set_xlabel('x')
        ax_diff1.set_ylabel('mgcv_rust - R mgcv')
        ax_diff1.set_title('Prediction Difference')
        ax_diff1.grid(True, alpha=0.3)

        # Scatter comparison
        ax_diff2.scatter(pred_r, pred_rust, alpha=0.5, s=30)

        # Perfect agreement line
        lim_min = min(pred_r.min(), pred_rust.min())
        lim_max = max(pred_r.max(), pred_rust.max())
        ax_diff2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--',
                     linewidth=2, label='Perfect agreement')

        ax_diff2.set_xlabel('R mgcv predictions')
        ax_diff2.set_ylabel('mgcv_rust predictions')
        ax_diff2.set_title('Prediction Scatter')
        ax_diff2.legend()
        ax_diff2.grid(True, alpha=0.3)
        ax_diff2.axis('equal')

        plt.tight_layout()
        fig_diff
    return diff, fig_diff, ax_diff1, ax_diff2, lim_min, lim_max


@app.cell
def __(mo, HAS_RPY2, np, gam_rust, ro):
    if not HAS_RPY2:
        mo.md("")
    else:
        mo.md("## Extrapolation Test")

        # Test extrapolation
        x_extrap = np.linspace(-0.2, 1.2, 100)
        X_extrap = x_extrap.reshape(-1, 1)

        pred_rust_extrap = gam_rust.predict(X_extrap)

        ro.globalenv['x_extrap'] = x_extrap
        ro.r('pred_r_extrap <- predict(gam_fit, newdata=data.frame(x_r=x_extrap))')
        pred_r_extrap = np.array(ro.r('pred_r_extrap'))

        y_true_extrap = np.sin(2 * np.pi * x_extrap)

        # Store for plotting
        _extrap_data = (x_extrap, pred_rust_extrap, pred_r_extrap, y_true_extrap)
    return (
        x_extrap,
        X_extrap,
        pred_rust_extrap,
        pred_r_extrap,
        y_true_extrap,
        _extrap_data,
    )


@app.cell
def __(mo, HAS_RPY2, plt, _extrap_data):
    if not HAS_RPY2:
        mo.md("")
    else:
        mo.md("### Extrapolation Comparison")

        x_extrap, pred_rust_extrap, pred_r_extrap, y_true_extrap = _extrap_data

        fig_extrap, ax_extrap = plt.subplots(figsize=(12, 6))

        # Mark training region
        ax_extrap.axvspan(0, 1, alpha=0.1, color='green', label='Training region')
        ax_extrap.axvspan(-0.2, 0, alpha=0.1, color='blue')
        ax_extrap.axvspan(1, 1.2, alpha=0.1, color='blue', label='Extrapolation region')

        ax_extrap.plot(x_extrap, y_true_extrap, 'k--', linewidth=1.5,
                      label='True function', alpha=0.5)
        ax_extrap.plot(x_extrap, pred_rust_extrap, 'b-', linewidth=2.5,
                      label='mgcv_rust', alpha=0.8)
        ax_extrap.plot(x_extrap, pred_r_extrap, 'r--', linewidth=2,
                      label='R mgcv', alpha=0.8)

        ax_extrap.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax_extrap.axvline(x=1, color='k', linestyle='--', alpha=0.5)

        ax_extrap.set_xlabel('x', fontsize=12)
        ax_extrap.set_ylabel('y', fontsize=12)
        ax_extrap.set_title('Extrapolation Comparison', fontsize=14)
        ax_extrap.legend(loc='best')
        ax_extrap.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_extrap
    return ax_extrap, fig_extrap


@app.cell
def __(mo, HAS_RPY2, np, pred_rust_extrap, pred_r_extrap):
    if not HAS_RPY2:
        mo.md("")
    else:
        # Check for zeros in extrapolation
        has_zeros_rust = np.any(np.abs(pred_rust_extrap) < 1e-6)
        has_zeros_r = np.any(np.abs(pred_r_extrap) < 1e-6)

        extrap_corr = np.corrcoef(pred_rust_extrap, pred_r_extrap)[0, 1]

        mo.md(f"""
        ### Extrapolation Quality

        | Check | mgcv_rust | R mgcv |
        |-------|-----------|--------|
        | Has zeros | {'❌ Yes' if has_zeros_rust else '✅ No'} | {'❌ Yes' if has_zeros_r else '✅ No'} |
        | Correlation | {extrap_corr:.6f} | - |

        {'✅ Both implementations extrapolate properly' if not has_zeros_rust and not has_zeros_r else '⚠️ Check extrapolation implementation'}
        """)
    return has_zeros_rust, has_zeros_r, extrap_corr


@app.cell
def __(mo):
    mo.md("""
    ## Summary

    This comparison shows:
    - How well mgcv_rust matches R's mgcv predictions
    - Smoothing parameter (λ) selection comparison
    - Extrapolation behavior

    **Expected results:**
    - Correlation > 0.95 (predictions match well)
    - λ ratio between 0.5-2.0 (similar smoothing)
    - No zeros in extrapolation regions
    """)
    return


if __name__ == "__main__":
    app.run()
