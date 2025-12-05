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
        importr('mgcv')  # Check mgcv is available
        HAS_RPY2 = True

        status_msg = """
        # Comparing mgcv_rust vs R's mgcv

        ✅ rpy2 and R mgcv available

        **Note:** Using `bs="bs"` (B-splines) in R to match our implementation.
        """
    except:
        HAS_RPY2 = False
        status_msg = """
        # Comparing mgcv_rust vs R's mgcv

        ⚠️ **rpy2 or R mgcv not available**

        Install with:
        ```bash
        pip install rpy2
        ```

        Also ensure R and mgcv package are installed:
        ```R
        install.packages("mgcv")
        ```
        """

    mo.md(status_msg)

    return mo, np, plt, mgcv_rust, ro, HAS_RPY2


@app.cell
def __(np):
    def generate_test_data(n_samples=100, noise_level=0.2, true_fn=None, seed=42):
        """
        Generate test data for GAM fitting.

        Parameters:
        -----------
        n_samples : int, default=100
            Number of data points to generate
        noise_level : float, default=0.2
            Standard deviation of Gaussian noise to add
        true_fn : callable, default=None
            Function that takes x (array) and returns y_true (array).
            If None, uses y = sin(2πx) as default.
        seed : int, default=42
            Random seed for reproducibility

        Returns:
        --------
        dict with keys:
            'x': input points (1D array)
            'y': noisy observations (1D array)
            'y_true': true function values (1D array)
            'noise': noise added (1D array)
        """
        np.random.seed(seed)

        x = np.linspace(0, 1, n_samples)

        # Use default function if none provided
        if true_fn is None:
            y_true = np.sin(2 * np.pi * x)
        else:
            y_true = true_fn(x)

        noise = noise_level * np.random.randn(n_samples)
        y = y_true + noise

        return {
            'x': x,
            'y': y,
            'y_true': y_true,
            'noise': noise
        }

    return generate_test_data,


@app.cell
def __(mo, np, HAS_RPY2):
    if not HAS_RPY2:
        raise RuntimeError("rpy2 not available - cannot continue")

    mo.md("## Generate Test Data")

    # Interactive parameters
    n_points = mo.ui.slider(50, 200, value=100, label="Number of points")
    noise_level = mo.ui.slider(0.0, 0.5, value=0.2, step=0.05, label="Noise level")
    k_basis = mo.ui.slider(5, 20, value=10, label="Number of basis functions (k)")

    mo.vstack([n_points, noise_level, k_basis])
    return n_points, noise_level, k_basis


@app.cell
def __(generate_test_data, n_points, noise_level):
    # Generate data using the reusable function
    data = generate_test_data(
        n_samples=n_points.value,
        noise_level=noise_level.value,
        true_fn=None,  # Use default sin(2πx)
        seed=42
    )

    n = n_points.value
    x = data['x']
    y_true = data['y_true']
    y = data['y']
    noise = data['noise']

    return n, x, y_true, y, noise, data


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
def __(mo, mgcv_rust, x, y, k_basis, ro, np):
    mo.md("## Fit Models")

    # Fit with mgcv_rust
    X = x.reshape(-1, 1)
    gam_rust = mgcv_rust.GAM()
    result_rust = gam_rust.fit_auto(X, y, k=[k_basis.value], method='REML')
    pred_rust = gam_rust.predict(X)
    lambda_rust = result_rust['lambda']

    # Fit with R mgcv using B-splines (bs="bs") to match our implementation
    ro.globalenv['x_r'] = x
    ro.globalenv['y_r'] = y
    ro.globalenv['k_val'] = k_basis.value
    ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="bs"), method="REML")')
    pred_r = np.array(ro.r('predict(gam_fit)'))
    lambda_r = np.array(ro.r('gam_fit$sp'))[0]

    mo.md(f"""
    **Fit Complete**

    | Implementation | Basis | λ (smoothing) | Deviance |
    |----------------|-------|---------------|----------|
    | mgcv_rust      | B-splines | {lambda_rust:.6f} | {result_rust['deviance']:.4f} |
    | R mgcv         | B-splines (`bs="bs"`) | {lambda_r:.6f} | - |
    | Ratio (Rust/R) | - | {lambda_rust/lambda_r:.4f} | - |
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
def __(mo, plt, x, y, pred_rust, pred_r, y_true):
    mo.md("### Prediction Comparison")

    fig_pred, ax_pred = plt.subplots(figsize=(12, 5))

    ax_pred.scatter(x, y, alpha=0.3, s=20, label='Data', color='lightgray')
    ax_pred.plot(x, y_true, 'k--', linewidth=1.5, label='True function', alpha=0.5)
    ax_pred.plot(x, pred_rust, 'b-', linewidth=2.5, label='mgcv_rust', alpha=0.8)
    ax_pred.plot(x, pred_r, 'r--', linewidth=2, label='R mgcv', alpha=0.8)

    ax_pred.set_xlabel('x', fontsize=12)
    ax_pred.set_ylabel('y', fontsize=12)
    ax_pred.set_title('Prediction Comparison (B-splines)', fontsize=14)
    ax_pred.legend()
    ax_pred.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_pred
    return ax_pred, fig_pred


@app.cell
def __(mo, np, pred_rust, pred_r):
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
def __(mo, plt, x, pred_rust, pred_r):
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
def __(mo, np, gam_rust, ro):
    mo.md("## Extrapolation Test")

    # Test extrapolation
    x_extrap = np.linspace(-0.2, 1.2, 100)
    X_extrap = x_extrap.reshape(-1, 1)

    pred_rust_extrap = gam_rust.predict(X_extrap)

    ro.globalenv['x_extrap'] = x_extrap
    ro.r('pred_r_extrap <- predict(gam_fit, newdata=data.frame(x_r=x_extrap))')
    pred_r_extrap = np.array(ro.r('pred_r_extrap'))

    y_true_extrap = np.sin(2 * np.pi * x_extrap)

    return x_extrap, X_extrap, pred_rust_extrap, pred_r_extrap, y_true_extrap


@app.cell
def __(mo, plt, x_extrap, pred_rust_extrap, pred_r_extrap, y_true_extrap):
    mo.md("### Extrapolation Comparison")

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
def __(mo, np, pred_rust_extrap, pred_r_extrap):
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
def __(mo, generate_test_data, mgcv_rust, ro, np):
    import time

    mo.md("""
    ## Performance Comparison

    Comparing fitting times between mgcv_rust and R's mgcv across different sample sizes.
    """)

    # Test different sample sizes
    sample_sizes = [50, 100, 200, 500, 1000]
    times_rust = []
    times_r = []

    for n_test in sample_sizes:
        # Generate test data
        test_data = generate_test_data(n_samples=n_test, noise_level=0.2, seed=42)
        X_test = test_data['x'].reshape(-1, 1)
        y_test = test_data['y']

        # Time mgcv_rust
        start = time.time()
        gam_test_rust = mgcv_rust.GAM()
        gam_test_rust.fit_auto(X_test, y_test, k=[10], method='REML')
        time_rust = time.time() - start
        times_rust.append(time_rust)

        # Time R mgcv
        ro.globalenv['x_test'] = test_data['x']
        ro.globalenv['y_test'] = y_test
        start = time.time()
        ro.r('gam_test <- gam(y_test ~ s(x_test, k=10, bs="bs"), method="REML")')
        time_r = time.time() - start
        times_r.append(time_r)

    return sample_sizes, times_rust, times_r, time


@app.cell
def __(mo, plt, np, sample_sizes, times_rust, times_r):
    mo.md("### Timing Results")

    fig_time, (ax_time1, ax_time2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute times
    ax_time1.plot(sample_sizes, times_rust, 'b-o', linewidth=2, markersize=8, label='mgcv_rust')
    ax_time1.plot(sample_sizes, times_r, 'r--s', linewidth=2, markersize=8, label='R mgcv')
    ax_time1.set_xlabel('Number of samples', fontsize=12)
    ax_time1.set_ylabel('Time (seconds)', fontsize=12)
    ax_time1.set_title('Fitting Time Comparison', fontsize=14)
    ax_time1.legend()
    ax_time1.grid(True, alpha=0.3)
    ax_time1.set_xscale('log')
    ax_time1.set_yscale('log')

    # Speedup ratio
    speedup = np.array(times_r) / np.array(times_rust)
    ax_time2.plot(sample_sizes, speedup, 'g-^', linewidth=2, markersize=8)
    ax_time2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal performance')
    ax_time2.set_xlabel('Number of samples', fontsize=12)
    ax_time2.set_ylabel('Speedup (R time / Rust time)', fontsize=12)
    ax_time2.set_title('Performance Ratio', fontsize=14)
    ax_time2.legend()
    ax_time2.grid(True, alpha=0.3)
    ax_time2.set_xscale('log')

    plt.tight_layout()
    fig_time

    # Generate timing table
    timing_table = "| Samples | mgcv_rust | R mgcv | Speedup |\n"
    timing_table += "|---------|-----------|--------|----------|\n"
    for n, t_rust, t_r, sp in zip(sample_sizes, times_rust, times_r, speedup):
        timing_table += f"| {n} | {t_rust:.4f}s | {t_r:.4f}s | {sp:.2f}x |\n"

    mo.md(f"""
    ### Timing Summary

    {timing_table}

    **Average speedup:** {np.mean(speedup):.2f}x
    """)

    return fig_time, ax_time1, ax_time2, speedup, timing_table


@app.cell
def __(mo):
    mo.md("""
    ## Summary

    This comparison shows:
    - How well mgcv_rust matches R's mgcv predictions **using the same basis (B-splines)**
    - Smoothing parameter (λ) selection comparison
    - Extrapolation behavior
    - **Performance comparison** across different sample sizes

    **Expected results:**
    - Correlation > 0.95 (predictions match well)
    - λ ratio between 0.5-2.0 (similar smoothing)
    - No zeros in extrapolation regions
    - Competitive or better performance compared to R mgcv

    **Note:** We use `bs="bs"` in R to match our B-spline implementation, not `bs="cr"` (cubic regression splines).

    **Data generation:** All tests use a consistent, reusable `generate_test_data()` function that allows customization of:
    - Sample size
    - Noise level
    - True function (default: sin(2πx))
    - Random seed
    """)
    return


if __name__ == "__main__":
    app.run()
