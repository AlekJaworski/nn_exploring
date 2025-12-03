import marimo

__generated_with = "0.12.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import mgcv_rust

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        numpy2ri.activate()
        importr('mgcv')  # Check mgcv is available
        HAS_RPY2 = True

        status_msg = """
        # Comparing mgcv_rust CR splines vs R's mgcv CR splines

        ✅ rpy2 and R mgcv available

        **Note:** Both implementations use `bs="cr"` (cubic regression splines).
        """
    except:
        HAS_RPY2 = False
        status_msg = """
        # Comparing mgcv_rust CR splines vs R's mgcv CR splines

        ⚠️ **rpy2 or R mgcv not available**

        Install with:
        ```bash
        pip install rpy2
        ```

        And install R mgcv:
        ```R
        install.packages("mgcv")
        ```
        """

    mo.md(status_msg)
    return HAS_RPY2, importr, mgcv_rust, mo, np, numpy2ri, plt, ro, status_msg


@app.cell
def _():
    # Interactive parameters
    n_points = 1000
    noise_level = 0.7
    k_basis = 12
    return k_basis, n_points, noise_level


@app.cell
def _(n_points, noise_level, np):
    np.random.seed(42)
    n = n_points
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x * (1 + x))
    y = y_true + np.random.normal(0, noise_level, n)
    return n, x, y, y_true


@app.cell
def _(mo, plt, x, y, y_true):
    mo.md("### Data Visualization")

    fig_data, ax_data = plt.subplots(figsize=(10, 4))
    ax_data.scatter(x, y, alpha=0.5, s=20, label='Noisy data')
    ax_data.plot(x, y_true, 'k--', linewidth=2, label='True function')
    ax_data.set_xlabel('x')
    ax_data.set_ylabel('y')
    ax_data.set_title('Test Data: y = sin(2πx) + noise')
    ax_data.legend()
    ax_data.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_data
    return ax_data, fig_data


@app.cell
def _(k_basis, mgcv_rust, mo, np, ro, x, y):
    mo.md("## Fit Models with CR Splines")

    # Fit with mgcv_rust using CR splines
    X = x.reshape(-1, 1)
    gam_rust = mgcv_rust.GAM()
    result_rust = gam_rust.fit_auto(X, y, k=[k_basis], method='REML', bs='cr')
    pred_rust = gam_rust.predict(X)
    lambda_rust = result_rust['lambda']

    # Fit with R mgcv using CR splines
    ro.globalenv['x_r'] = x
    ro.globalenv['y_r'] = y
    ro.globalenv['k_val'] = k_basis
    ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="cr"), method="REML")')
    pred_r = np.array(ro.r('predict(gam_fit)'))
    lambda_r = np.array(ro.r('gam_fit$sp'))[0]

    mo.md(f"""
    **Fit Complete** (both using CR splines)

    | Implementation | Basis | λ (smoothing) | Deviance |
    |----------------|-------|---------------|----------|
    | mgcv_rust      | CR (cubic regression) | {lambda_rust} | {result_rust['deviance']} |
    | R mgcv         | CR (cubic regression) | {lambda_r} | - |
    | Ratio (Rust/R) | - | {lambda_rust/lambda_r} | - |
    """)
    return X, gam_rust, lambda_r, lambda_rust, pred_r, pred_rust, result_rust


@app.cell
def _(mo, plt, pred_r, pred_rust, x, y, y_true):
    mo.md("### Prediction Comparison")

    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))

    ax_pred.scatter(x, y, alpha=0.3, s=20, label='Data', color='lightgray')
    ax_pred.plot(x, y_true, 'k--', linewidth=1.5, label='True function', alpha=0.5)
    ax_pred.plot(x, pred_rust, 'b-', linewidth=2.5, label='mgcv_rust (CR)', alpha=0.8)
    ax_pred.plot(x, pred_r, 'r--', linewidth=2, label='R mgcv (CR)', alpha=0.8)

    ax_pred.set_xlabel('x', fontsize=12)
    ax_pred.set_ylabel('y', fontsize=12)
    ax_pred.set_title('Prediction Comparison (CR splines)', fontsize=14)
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_pred
    return ax_pred, fig_pred


@app.cell
def _(mo, np, pred_r, pred_rust):
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
    return corr, max_diff, rmse_diff


@app.cell
def _(mo, plt, pred_r, pred_rust, x):
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
    return ax_diff1, ax_diff2, diff, fig_diff, lim_max, lim_min


@app.cell
def _(gam_rust, mo, np, ro):
    mo.md("## Extrapolation Test")

    # Test extrapolation
    x_extrap = np.linspace(-0.2, 1.2, 100)
    X_extrap = x_extrap.reshape(-1, 1)

    pred_rust_extrap = gam_rust.predict(X_extrap)

    ro.globalenv['x_extrap'] = x_extrap
    ro.r('pred_r_extrap <- predict(gam_fit, newdata=data.frame(x_r=x_extrap))')
    pred_r_extrap = np.array(ro.r('pred_r_extrap'))

    y_true_extrap = np.sin(2 * np.pi * x_extrap)
    return X_extrap, pred_r_extrap, pred_rust_extrap, x_extrap, y_true_extrap


@app.cell
def _(mo, plt, pred_r_extrap, pred_rust_extrap, x_extrap, y_true_extrap):
    mo.md("### Extrapolation Comparison")

    fig_extrap, ax_extrap = plt.subplots(figsize=(12, 6))

    # Mark training region
    ax_extrap.axvspan(0, 1, alpha=0.1, color='green', label='Training region')
    ax_extrap.axvspan(-0.2, 0, alpha=0.1, color='blue')
    ax_extrap.axvspan(1, 1.2, alpha=0.1, color='blue', label='Extrapolation region')

    ax_extrap.plot(x_extrap, y_true_extrap, 'k--', linewidth=1.5,
                  label='True function', alpha=0.5)
    ax_extrap.plot(x_extrap, pred_rust_extrap, 'b-', linewidth=2.5,
                  label='mgcv_rust (CR)', alpha=0.8)
    ax_extrap.plot(x_extrap, pred_r_extrap, 'r--', linewidth=2,
                  label='R mgcv (CR)', alpha=0.8)

    ax_extrap.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax_extrap.axvline(x=1, color='k', linestyle='--', alpha=0.5)

    ax_extrap.set_xlabel('x', fontsize=12)
    ax_extrap.set_ylabel('y', fontsize=12)
    ax_extrap.set_title('Extrapolation Comparison (CR splines)', fontsize=14)
    ax_extrap.legend(loc='best')
    ax_extrap.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_extrap
    return ax_extrap, fig_extrap


@app.cell
def _(mo, np, pred_r_extrap, pred_rust_extrap):
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
    return extrap_corr, has_zeros_r, has_zeros_rust


@app.cell
def _(X, k_basis, mgcv_rust, mo, np, plt, ro, y):
    mo.md("## Speed Comparison")

    import time

    # Number of iterations for timing
    n_iters = 100

    # Time mgcv_rust
    times_rust = []
    for _ in range(n_iters):
        gam_rust_speed = mgcv_rust.GAM()
        start = time.perf_counter()
        gam_rust_speed.fit_auto(X, y, k=[k_basis], method='REML', bs='cr')
        end = time.perf_counter()
        times_rust.append(end - start)

    # Time R mgcv
    times_r = []
    ro.globalenv['x_r'] = X[:, 0]
    ro.globalenv['y_r'] = y
    ro.globalenv['k_val'] = k_basis

    for _ in range(n_iters):
        start = time.perf_counter()
        ro.r('gam_fit_speed <- gam(y_r ~ s(x_r, k=k_val, bs="cr"), method="REML")')
        end = time.perf_counter()
        times_r.append(end - start)

    # Compute statistics
    mean_rust = np.mean(times_rust) * 1000  # Convert to ms
    std_rust = np.std(times_rust) * 1000
    mean_r = np.mean(times_r) * 1000
    std_r = np.std(times_r) * 1000
    speedup = mean_r / mean_rust

    # Create visualization
    fig_speed, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax_box.boxplot([np.array(times_rust) * 1000, np.array(times_r) * 1000],
                    labels=['mgcv_rust', 'R mgcv'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    ax_box.set_ylabel('Time (ms)', fontsize=12)
    ax_box.set_title(f'Timing Distribution ({n_iters} iterations)', fontsize=14)
    ax_box.grid(True, alpha=0.3, axis='y')

    # Bar plot
    implementations = ['mgcv_rust', 'R mgcv']
    means = [mean_rust, mean_r]
    stds = [std_rust, std_r]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax_bar.bar(implementations, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax_bar.set_ylabel('Time (ms)', fontsize=12)
    ax_bar.set_title('Mean Fit Time (± std)', fontsize=14)
    ax_bar.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.2f} ms',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    mo.md(f"""
    ### Performance Results ({n_iters} iterations)

    | Implementation | Mean Time | Std Dev | Status |
    |----------------|-----------|---------|--------|
    | mgcv_rust      | {mean_rust:.2f} ms | {std_rust:.2f} ms | {'🚀 Faster' if speedup > 1 else ''} |
    | R mgcv         | {mean_r:.2f} ms | {std_r:.2f} ms | {'🚀 Faster' if speedup < 1 else ''} |
    | **Speedup**    | **{speedup:.2f}x** | - | {'✅ Rust wins!' if speedup > 1 else '⚠️ R wins'} |

    {fig_speed}

    **Notes:**
    - Timing includes REML optimization (Newton's method)
    - Both implementations use same number of basis functions (k={k_basis})
    - R has overhead from rpy2 interface (slight disadvantage)
    - Rust benefits from compiled performance and optimized linear algebra
    """)
    return (
        ax_bar,
        ax_box,
        bar,
        bars,
        colors,
        end,
        fig_speed,
        gam_rust_speed,
        height,
        implementations,
        mean_r,
        mean_rust,
        mean_val,
        means,
        n_iters,
        speedup,
        start,
        std_r,
        std_rust,
        stds,
        time,
        times_r,
        times_rust,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## Summary

        This comparison shows:
        - How well mgcv_rust CR splines match R's mgcv CR splines
        - Smoothing parameter (λ) selection comparison
        - Extrapolation behavior
        - Performance comparison

        **Success criteria:**
        - Correlation > 0.95 (predictions match well)
        - λ ratio between 0.5-2.0 (similar smoothing)
        - No zeros in extrapolation regions

        **Note:** Both implementations use `bs="cr"` (cubic regression splines), which is mgcv's default.
        """
    )
    return


if __name__ == "__main__":
    app.run()
