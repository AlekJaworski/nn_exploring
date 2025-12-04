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
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        importr('mgcv')  # Check mgcv is available
        HAS_RPY2 = True

        status_msg = """
        # Benchmarking mgcv_rust vs R's mgcv: Multivariable GAMs

        ✅ rpy2 and R mgcv available

        **Note:** This notebook benchmarks fitting and prediction with multiple variables using CR splines.

        **Lambda values:** Note that mgcv_rust and R's mgcv use different internal scaling
        conventions, so lambda values may differ significantly while predictions match perfectly.
        See `LAMBDA_SCALING_EXPLANATION.md` for details.
        """
    except:
        HAS_RPY2 = False
        status_msg = """
        # Benchmarking mgcv_rust vs R's mgcv: Multivariable GAMs

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
    # Configuration parameters
    n_points = 1000
    noise_level = 0.7
    k_basis = 12
    n_vars = 3
    return k_basis, n_points, n_vars, noise_level


@app.cell
def _(mo, n_points, n_vars, noise_level, np):
    mo.md("## Generate Multivariable Test Data")

    np.random.seed(42)
    n = n_points
    n_v = n_vars

    # Generate random points in [0, 1]^n_vars space
    X_data = np.random.uniform(0, 1, size=(n, n_v))

    # Create a complex multivariate function
    # y = sin(2π*x1) + cos(2π*x2) + 0.5*sin(4π*x3) + ...
    y_true_data = np.zeros(n)
    for i in range(n_v):
        if i % 3 == 0:
            y_true_data += np.sin(2 * np.pi * X_data[:, i])
        elif i % 3 == 1:
            y_true_data += np.cos(2 * np.pi * X_data[:, i])
        else:
            y_true_data += 0.5 * np.sin(4 * np.pi * X_data[:, i])

    y_data = y_true_data + np.random.normal(0, noise_level, n)

    return X_data, n, n_v, y_data, y_true_data


@app.cell
def _(X_data, mo, n_v, plt, y_data):
    mo.md("### Data Visualization")

    # Create pairwise plots for first 3 variables
    n_plot_vars = min(3, n_v)
    fig_data, axes = plt.subplots(1, n_plot_vars, figsize=(5*n_plot_vars, 4))

    if n_plot_vars == 1:
        axes = [axes]

    for i in range(n_plot_vars):
        axes[i].scatter(X_data[:, i], y_data, alpha=0.5, s=20, label='Noisy data')
        axes[i].set_xlabel(f'x{i+1}')
        axes[i].set_ylabel('y')
        axes[i].set_title(f'y vs x{i+1}')
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    plt.tight_layout()

    mo.md(f"""
    **Generated {n_v} variables with {len(y_data)} observations**

    True function components:
    - Variables x1, x4, x7, ...: sin(2π·x)
    - Variables x2, x5, x8, ...: cos(2π·x)
    - Variables x3, x6, x9, ...: 0.5·sin(4π·x)
    """)

    fig_data
    return axes, fig_data, n_plot_vars


@app.cell
def _(X_data, k_basis, localconverter, mgcv_rust, mo, n_v, np, numpy2ri, ro, y_data):
    mo.md("## Fit Multivariable GAM Models")

    # Fit with mgcv_rust using CR splines for all variables
    k_list = [k_basis] * n_v
    gam_rust_multi = mgcv_rust.GAM()
    result_rust_multi = gam_rust_multi.fit_auto(X_data, y_data, k=k_list, method='REML', bs='cr')
    pred_rust_multi = gam_rust_multi.predict(X_data)
    lambda_rust_multi = result_rust_multi['lambda']

    # Fit with R mgcv using CR splines for all variables
    # Build the formula: y ~ s(x1, k=k, bs="cr") + s(x2, k=k, bs="cr") + ...
    with localconverter(ro.default_converter + numpy2ri.converter):
        for i in range(n_v):
            ro.globalenv[f'x{i+1}'] = X_data[:, i]
        ro.globalenv['y_r'] = y_data
        ro.globalenv['k_val'] = k_basis

        # Construct formula
        smooth_terms = [f's(x{i+1}, k=k_val, bs="cr")' for i in range(n_v)]
        formula = f'y_r ~ {" + ".join(smooth_terms)}'

        ro.r(f'gam_fit_multi <- gam({formula}, method="REML")')
        pred_r_multi = np.array(ro.r('predict(gam_fit_multi)'))
        lambda_r_multi = np.array(ro.r('gam_fit_multi$sp'))

    # Format lambda comparison
    lambda_table = "| Variable | λ (Rust) | λ (R) | Ratio |\n|----------|----------|-------|-------|\n"
    for i in range(n_v):
        l_r = lambda_rust_multi[i] if hasattr(lambda_rust_multi, '__len__') else lambda_rust_multi
        l_m = lambda_r_multi[i] if len(lambda_r_multi) > 1 else lambda_r_multi[0]
        ratio = l_r / l_m if l_m > 0 else float('inf')
        lambda_table += f"| x{i+1} | {l_r:.6f} | {l_m:.6f} | {ratio:.4f} |\n"

    mo.md(f"""
    **Multivariable Fit Complete** ({n_v} variables, each with CR splines)

    | Implementation | Deviance |
    |----------------|----------|
    | mgcv_rust      | {result_rust_multi['deviance']:.4f} |
    | R mgcv         | - |

    ### Smoothing Parameters (λ)
    {lambda_table}
    """)
    return (
        formula,
        gam_rust_multi,
        k_list,
        lambda_r_multi,
        lambda_rust_multi,
        pred_r_multi,
        pred_rust_multi,
        result_rust_multi,
        smooth_terms,
    )


@app.cell
def _(mo, np, plt, pred_r_multi, pred_rust_multi, y_data, y_true_data):
    mo.md("### Prediction Comparison")

    fig_pred, (ax_pred1, ax_pred2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predictions vs True values
    idx_sort = np.argsort(y_true_data)
    ax_pred1.scatter(range(len(y_data)), y_data[idx_sort], alpha=0.3, s=10,
                     label='Data', color='lightgray')
    ax_pred1.plot(range(len(y_true_data)), y_true_data[idx_sort], 'k--',
                  linewidth=1.5, label='True function', alpha=0.5)
    ax_pred1.plot(range(len(pred_rust_multi)), pred_rust_multi[idx_sort], 'b-',
                  linewidth=2, label='mgcv_rust', alpha=0.7)
    ax_pred1.plot(range(len(pred_r_multi)), pred_r_multi[idx_sort], 'r--',
                  linewidth=2, label='R mgcv', alpha=0.7)

    ax_pred1.set_xlabel('Observation (sorted by true value)', fontsize=11)
    ax_pred1.set_ylabel('y', fontsize=11)
    ax_pred1.set_title('Predictions vs True Values', fontsize=12)
    ax_pred1.legend()
    ax_pred1.grid(True, alpha=0.3)

    # Plot 2: Scatter comparison
    ax_pred2.scatter(pred_r_multi, pred_rust_multi, alpha=0.5, s=30, c='blue')
    lim_min = min(pred_r_multi.min(), pred_rust_multi.min())
    lim_max = max(pred_r_multi.max(), pred_rust_multi.max())
    ax_pred2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--',
                 linewidth=2, label='Perfect agreement')

    ax_pred2.set_xlabel('R mgcv predictions', fontsize=11)
    ax_pred2.set_ylabel('mgcv_rust predictions', fontsize=11)
    ax_pred2.set_title('Implementation Comparison', fontsize=12)
    ax_pred2.legend()
    ax_pred2.grid(True, alpha=0.3)
    ax_pred2.axis('equal')

    plt.tight_layout()
    fig_pred
    return ax_pred1, ax_pred2, fig_pred, idx_sort, lim_max, lim_min


@app.cell
def _(mo, np, pred_r_multi, pred_rust_multi, y_data, y_true_data):
    # Compute comparison metrics
    corr_multi = np.corrcoef(pred_rust_multi, pred_r_multi)[0, 1]
    rmse_diff_multi = np.sqrt(np.mean((pred_rust_multi - pred_r_multi)**2))
    max_diff_multi = np.max(np.abs(pred_rust_multi - pred_r_multi))

    # Compute prediction accuracy vs true function
    rmse_rust_true = np.sqrt(np.mean((pred_rust_multi - y_true_data)**2))
    rmse_r_true = np.sqrt(np.mean((pred_r_multi - y_true_data)**2))
    rmse_data_true = np.sqrt(np.mean((y_data - y_true_data)**2))

    mo.md(f"""
    ### Prediction Metrics

    **Agreement between implementations:**
    | Metric | Value | Status |
    |--------|-------|--------|
    | Correlation | {corr_multi:.6f} | {'✅ Excellent' if corr_multi > 0.99 else '⚠️ Good' if corr_multi > 0.95 else '❌ Poor'} |
    | RMSE difference | {rmse_diff_multi:.6f} | {'✅ Low' if rmse_diff_multi < 0.1 else '⚠️ Moderate'} |
    | Max difference | {max_diff_multi:.6f} | {'✅ Low' if max_diff_multi < 0.2 else '⚠️ Moderate'} |

    **Accuracy vs true function:**
    | Implementation | RMSE vs Truth |
    |----------------|---------------|
    | mgcv_rust      | {rmse_rust_true:.4f} |
    | R mgcv         | {rmse_r_true:.4f} |
    | Data noise     | {rmse_data_true:.4f} |
    """)
    return (
        corr_multi,
        max_diff_multi,
        rmse_data_true,
        rmse_diff_multi,
        rmse_r_true,
        rmse_rust_true,
    )


@app.cell
def _(mo, np, plt, pred_r_multi, pred_rust_multi):
    mo.md("### Residual Difference Analysis")

    diff_multi = pred_rust_multi - pred_r_multi

    fig_diff, (ax_diff1, ax_diff2) = plt.subplots(1, 2, figsize=(14, 5))

    # Difference plot
    ax_diff1.plot(diff_multi, 'g-', linewidth=1.5, alpha=0.7)
    ax_diff1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax_diff1.set_xlabel('Observation', fontsize=11)
    ax_diff1.set_ylabel('mgcv_rust - R mgcv', fontsize=11)
    ax_diff1.set_title('Prediction Difference', fontsize=12)
    ax_diff1.grid(True, alpha=0.3)

    # Histogram of differences
    ax_diff2.hist(diff_multi, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax_diff2.axvline(x=0, color='k', linestyle='--', linewidth=2, label='Zero difference')
    ax_diff2.axvline(x=np.mean(diff_multi), color='r', linestyle='--',
                     linewidth=2, label=f'Mean: {np.mean(diff_multi):.6f}')
    ax_diff2.set_xlabel('Prediction difference', fontsize=11)
    ax_diff2.set_ylabel('Frequency', fontsize=11)
    ax_diff2.set_title('Distribution of Differences', fontsize=12)
    ax_diff2.legend()
    ax_diff2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_diff
    return ax_diff1, ax_diff2, diff_multi, fig_diff


@app.cell
def _(gam_rust_multi, localconverter, mo, n_v, np, numpy2ri, ro):
    mo.md("## Extrapolation Test")

    # Generate test points with some values outside [0, 1]
    n_test = 200
    X_extrap = np.random.uniform(-0.2, 1.2, size=(n_test, n_v))

    # Predict with mgcv_rust
    pred_rust_extrap_multi = gam_rust_multi.predict(X_extrap)

    # Predict with R mgcv
    with localconverter(ro.default_converter + numpy2ri.converter):
        for i in range(n_v):
            ro.globalenv[f'x{i+1}_new'] = X_extrap[:, i]

        # Create newdata frame
        newdata_str = ", ".join([f'x{i+1}=x{i+1}_new' for i in range(n_v)])
        ro.r(f'pred_r_extrap_multi <- predict(gam_fit_multi, newdata=data.frame({newdata_str}))')
        pred_r_extrap_multi = np.array(ro.r('pred_r_extrap_multi'))

    # Compute true values
    y_true_extrap = np.zeros(n_test)
    for i in range(n_v):
        if i % 3 == 0:
            y_true_extrap += np.sin(2 * np.pi * X_extrap[:, i])
        elif i % 3 == 1:
            y_true_extrap += np.cos(2 * np.pi * X_extrap[:, i])
        else:
            y_true_extrap += 0.5 * np.sin(4 * np.pi * X_extrap[:, i])

    return (
        X_extrap,
        n_test,
        newdata_str,
        pred_r_extrap_multi,
        pred_rust_extrap_multi,
        y_true_extrap,
    )


@app.cell
def _(mo, np, plt, pred_r_extrap_multi, pred_rust_extrap_multi):
    mo.md("### Extrapolation Quality")

    # Check for zeros or anomalies
    has_zeros_rust_multi = np.any(np.abs(pred_rust_extrap_multi) < 1e-6)
    has_zeros_r_multi = np.any(np.abs(pred_r_extrap_multi) < 1e-6)
    has_nan_rust = np.any(np.isnan(pred_rust_extrap_multi))
    has_nan_r = np.any(np.isnan(pred_r_extrap_multi))

    extrap_corr_multi = np.corrcoef(pred_rust_extrap_multi, pred_r_extrap_multi)[0, 1]
    extrap_rmse = np.sqrt(np.mean((pred_rust_extrap_multi - pred_r_extrap_multi)**2))

    # Visualization
    fig_extrap, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax1.scatter(pred_r_extrap_multi, pred_rust_extrap_multi, alpha=0.5, s=30, c='purple')
    lim_min_e = min(pred_r_extrap_multi.min(), pred_rust_extrap_multi.min())
    lim_max_e = max(pred_r_extrap_multi.max(), pred_rust_extrap_multi.max())
    ax1.plot([lim_min_e, lim_max_e], [lim_min_e, lim_max_e], 'r--',
             linewidth=2, label='Perfect agreement')
    ax1.set_xlabel('R mgcv predictions', fontsize=11)
    ax1.set_ylabel('mgcv_rust predictions', fontsize=11)
    ax1.set_title('Extrapolation Predictions', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Difference histogram
    diff_extrap = pred_rust_extrap_multi - pred_r_extrap_multi
    ax2.hist(diff_extrap, bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax2.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(diff_extrap), color='r', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(diff_extrap):.6f}')
    ax2.set_xlabel('Prediction difference', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Extrapolation Differences', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    mo.md(f"""
    **Extrapolation Test Results** (includes points outside [0,1] range)

    | Check | mgcv_rust | R mgcv |
    |-------|-----------|--------|
    | Has zeros | {'❌ Yes' if has_zeros_rust_multi else '✅ No'} | {'❌ Yes' if has_zeros_r_multi else '✅ No'} |
    | Has NaN | {'❌ Yes' if has_nan_rust else '✅ No'} | {'❌ Yes' if has_nan_r else '✅ No'} |

    | Metric | Value |
    |--------|-------|
    | Correlation | {extrap_corr_multi:.6f} |
    | RMSE difference | {extrap_rmse:.6f} |

    {'✅ Both implementations extrapolate properly' if not has_zeros_rust_multi and not has_zeros_r_multi and not has_nan_rust and not has_nan_r else '⚠️ Check extrapolation implementation'}

    {fig_extrap}
    """)
    return (
        ax1,
        ax2,
        diff_extrap,
        extrap_corr_multi,
        extrap_rmse,
        fig_extrap,
        has_nan_r,
        has_nan_rust,
        has_zeros_r_multi,
        has_zeros_rust_multi,
        lim_max_e,
        lim_min_e,
    )


@app.cell
def _(X_data, k_list, localconverter, mgcv_rust, mo, n_v, np, numpy2ri, plt, ro, y_data):
    mo.md("## Performance Benchmark")

    import time

    # Number of iterations for timing
    n_iters = 100

    # Time mgcv_rust
    times_rust_multi = []
    for _ in range(n_iters):
        gam_rust_speed = mgcv_rust.GAM()
        start = time.perf_counter()
        gam_rust_speed.fit_auto(X_data, y_data, k=k_list, method='REML', bs='cr')
        end = time.perf_counter()
        times_rust_multi.append(end - start)

    # Time R mgcv
    times_r_multi = []
    with localconverter(ro.default_converter + numpy2ri.converter):
        for i in range(n_v):
            ro.globalenv[f'x{i+1}'] = X_data[:, i]
        ro.globalenv['y_r'] = y_data
        ro.globalenv['k_val'] = k_list[0]

        smooth_terms_bench = [f's(x{i+1}, k=k_val, bs="cr")' for i in range(n_v)]
        formula_bench = f'y_r ~ {" + ".join(smooth_terms_bench)}'

        for _ in range(n_iters):
            start = time.perf_counter()
            ro.r(f'gam_fit_speed <- gam({formula_bench}, method="REML")')
            end = time.perf_counter()
            times_r_multi.append(end - start)

    # Compute statistics
    mean_rust_multi = np.mean(times_rust_multi) * 1000  # Convert to ms
    std_rust_multi = np.std(times_rust_multi) * 1000
    mean_r_multi = np.mean(times_r_multi) * 1000
    std_r_multi = np.std(times_r_multi) * 1000
    speedup_multi = mean_r_multi / mean_rust_multi

    # Create visualization
    fig_speed, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    ax_box.boxplot([np.array(times_rust_multi) * 1000, np.array(times_r_multi) * 1000],
                    labels=['mgcv_rust', 'R mgcv'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    ax_box.set_ylabel('Time (ms)', fontsize=12)
    ax_box.set_title(f'Timing Distribution ({n_iters} iterations)', fontsize=14)
    ax_box.grid(True, alpha=0.3, axis='y')

    # Bar plot
    implementations = ['mgcv_rust', 'R mgcv']
    means = [mean_rust_multi, mean_r_multi]
    stds = [std_rust_multi, std_r_multi]
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
    ### Performance Results ({n_iters} iterations, {n_v} variables)

    | Implementation | Mean Time | Std Dev | Status |
    |----------------|-----------|---------|--------|
    | mgcv_rust      | {mean_rust_multi:.2f} ms | {std_rust_multi:.2f} ms | {'🚀 Faster' if speedup_multi > 1 else ''} |
    | R mgcv         | {mean_r_multi:.2f} ms | {std_r_multi:.2f} ms | {'🚀 Faster' if speedup_multi < 1 else ''} |
    | **Speedup**    | **{speedup_multi:.2f}x** | - | {'✅ Rust wins!' if speedup_multi > 1 else '⚠️ R wins'} |

    {fig_speed}

    **Notes:**
    - Timing includes REML optimization with {n_v} smoothing parameters
    - Each variable has {k_list[0]} basis functions
    - R has overhead from rpy2 interface
    - Performance gap typically increases with more variables
    """)

    return (
        ax_bar,
        ax_box,
        fig_speed,
        mean_r_multi,
        mean_rust_multi,
        n_iters,
        speedup_multi,
        std_r_multi,
        std_rust_multi,
        times_r_multi,
        times_rust_multi,
    )


@app.cell
def _(k_list, mo, n_v):
    mo.md(
        f"""
        ## Summary

        This benchmark compares mgcv_rust and R's mgcv for **multivariable GAM fitting** with:
        - **{n_v} input variables** (x1, x2, ..., x{n_v})
        - **{k_list[0]} basis functions per variable** (CR splines)
        - **REML optimization** for smoothing parameter selection

        ### What we tested:
        1. **Fitting**: Additive model with one smooth term per variable
        2. **Prediction**: Accuracy on training data
        3. **Extrapolation**: Behavior outside training domain
        4. **Performance**: Speed comparison across implementations

        ### Success criteria:
        - ✅ Correlation > 0.95 (implementations agree)
        - ✅ Similar smoothing parameters (λ values)
        - ✅ No numerical issues in extrapolation
        - ✅ Competitive or better performance

        ### Key insights:
        - Multivariable fitting scales differently in each implementation
        - Smoothing parameter selection is independent per variable
        - Both implementations use CR splines which extrapolate linearly
        """
    )
    return


if __name__ == "__main__":
    app.run()
