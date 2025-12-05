#!/usr/bin/env python3
"""
Standalone test script for 4-dimensional multidimensional inference.

Tests mgcv_rust vs R's mgcv on a dataset with:
- Feature 1: Sinusoidal effect (sin(2π*x))
- Feature 2: Parabolic effect ((x-0.5)^2)
- Feature 3: Linear effect (x)
- Feature 4: Constant noise (no effect)

This script verifies that multidimensional inference works correctly.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

# Check if mgcv_rust is available
try:
    import mgcv_rust

    print("✓ mgcv_rust is available")
except ImportError as e:
    print(f"✗ Error: mgcv_rust not available: {e}")
    print("  Build with: maturin build --release && pip install target/wheels/*.whl")
    sys.exit(1)

# Check if R and rpy2 are available
try:
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    # Use context manager for conversion instead of activate()
    importr("mgcv")
    HAS_RPY2 = True
    print("✓ rpy2 and R's mgcv are available")
except Exception as e:
    HAS_RPY2 = False
    print(f"⚠ Warning: rpy2 or R's mgcv not available: {e}")
    print("  Install with: pip install rpy2")
    print("  And install R's mgcv: R -e 'install.packages(\"mgcv\")'")
    print("  Continuing with mgcv_rust only...")


def generate_4d_data(n=500, noise_level=0.3, seed=42):
    """
    Generate 4D test data with specific effects:
    - x1: Sinusoidal effect
    - x2: Parabolic effect
    - x3: Linear effect
    - x4: Constant noise (no effect)
    """
    np.random.seed(seed)

    # Generate 4 features uniformly in [0, 1]
    X = np.random.uniform(0, 1, size=(n, 4))

    # True effects:
    # Feature 1: Sinusoidal - sin(2π*x)
    effect_1 = np.sin(2 * np.pi * X[:, 0])

    # Feature 2: Parabolic - (x - 0.5)^2
    effect_2 = (X[:, 1] - 0.5) ** 2

    # Feature 3: Linear - x
    effect_3 = X[:, 2]

    # Feature 4: No effect (just noise)
    effect_4 = np.zeros(n)

    # Combined true function
    y_true = effect_1 + effect_2 + effect_3 + effect_4

    # Add noise
    y = y_true + np.random.normal(0, noise_level, n)

    return X, y, y_true, effect_1, effect_2, effect_3, effect_4


def fit_mgcv_rust(X, y, k=12):
    """Fit GAM using mgcv_rust."""
    print("\n" + "=" * 70)
    print("Fitting with mgcv_rust")
    print("=" * 70)

    gam = mgcv_rust.GAM()
    k_list = [k] * X.shape[1]

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"k per dimension: {k_list}")

    result = gam.fit_auto_optimized(X, y, k=k_list, method="REML", bs="cr")
    pred = gam.predict(X)

    print(f"Deviance: {result['deviance']:.6f}")
    print(f"Lambda values: {result['lambda']}")

    return gam, result, pred


def fit_mgcv_r(X, y, k=12):
    """Fit GAM using R's mgcv."""
    if not HAS_RPY2:
        return None, None, None

    print("\n" + "=" * 70)
    print("Fitting with R's mgcv")
    print("=" * 70)

    n_vars = X.shape[1]

    # Set up R environment with proper conversion
    with localconverter(ro.default_converter + numpy2ri.converter):
        for i in range(n_vars):
            ro.globalenv[f"x{i + 1}"] = X[:, i]
        ro.globalenv["y_r"] = y
        ro.globalenv["k_val"] = k

        # Construct formula: y ~ s(x1, k=k, bs="cr") + s(x2, k=k, bs="cr") + ...
        smooth_terms = [f's(x{i + 1}, k=k_val, bs="cr")' for i in range(n_vars)]
        formula = f"y_r ~ {' + '.join(smooth_terms)}"

        print(f"Formula: {formula}")

        # Fit the model
        ro.r(f'gam_fit <- gam({formula}, method="REML")')

        # Get predictions
        pred_r = np.array(ro.r("predict(gam_fit)"))

        # Get smoothing parameters
        lambda_r = np.array(ro.r("gam_fit$sp"))

        # Get deviance
        deviance_r = np.array(ro.r("deviance(gam_fit)"))[0]

    print(f"Deviance: {deviance_r:.6f}")
    print(f"Lambda values: {lambda_r}")

    return None, {"deviance": deviance_r, "lambda": lambda_r}, pred_r


def compare_predictions(pred_rust, pred_r, y_true, y_data):
    """Compare predictions between implementations."""
    print("\n" + "=" * 70)
    print("Prediction Comparison")
    print("=" * 70)

    # Compute metrics vs true function
    rmse_rust_true = np.sqrt(np.mean((pred_rust - y_true) ** 2))
    rmse_data_true = np.sqrt(np.mean((y_data - y_true) ** 2))

    print(f"\nAccuracy vs true function:")
    print(f"  mgcv_rust RMSE: {rmse_rust_true:.6f}")
    print(f"  Data noise RMSE: {rmse_data_true:.6f}")

    if pred_r is not None:
        rmse_r_true = np.sqrt(np.mean((pred_r - y_true) ** 2))

        # Compare implementations
        corr = np.corrcoef(pred_rust, pred_r)[0, 1]
        rmse_diff = np.sqrt(np.mean((pred_rust - pred_r) ** 2))
        max_diff = np.max(np.abs(pred_rust - pred_r))

        print(f"  R mgcv RMSE:    {rmse_r_true:.6f}")
        print(f"\nAgreement between implementations:")
        print(
            f"  Correlation:    {corr:.6f} {'✓ Excellent' if corr > 0.99 else '⚠ Check'}"
        )
        print(
            f"  RMSE diff:      {rmse_diff:.6f} {'✓ Good' if rmse_diff < 0.1 else '⚠ Check'}"
        )
        print(
            f"  Max diff:       {max_diff:.6f} {'✓ Good' if max_diff < 0.2 else '⚠ Check'}"
        )

        return {
            "corr": corr,
            "rmse_diff": rmse_diff,
            "max_diff": max_diff,
            "rmse_rust": rmse_rust_true,
            "rmse_r": rmse_r_true,
        }
    else:
        return {
            "rmse_rust": rmse_rust_true,
        }


def visualize_results(X, y, y_true, pred_rust, pred_r, effects):
    """Create visualization plots."""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    effect_1, effect_2, effect_3, effect_4 = effects

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Overall predictions comparison
    ax1 = plt.subplot(3, 3, 1)
    idx_sort = np.argsort(y_true)
    ax1.scatter(
        range(len(y)), y[idx_sort], alpha=0.3, s=10, label="Data", color="lightgray"
    )
    ax1.plot(
        range(len(y_true)),
        y_true[idx_sort],
        "k--",
        linewidth=1.5,
        label="True",
        alpha=0.6,
    )
    ax1.plot(
        range(len(pred_rust)),
        pred_rust[idx_sort],
        "b-",
        linewidth=2,
        label="mgcv_rust",
        alpha=0.7,
    )
    if pred_r is not None:
        ax1.plot(
            range(len(pred_r)),
            pred_r[idx_sort],
            "r--",
            linewidth=2,
            label="R mgcv",
            alpha=0.7,
        )
    ax1.set_xlabel("Observation (sorted)")
    ax1.set_ylabel("y")
    ax1.set_title("Overall Predictions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot: Rust vs R (if available)
    if pred_r is not None:
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(pred_r, pred_rust, alpha=0.5, s=30, c="blue")
        lim_min = min(pred_r.min(), pred_rust.min())
        lim_max = max(pred_r.max(), pred_rust.max())
        ax2.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            "r--",
            linewidth=2,
            label="Perfect agreement",
        )
        ax2.set_xlabel("R mgcv predictions")
        ax2.set_ylabel("mgcv_rust predictions")
        ax2.set_title("Implementation Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis("equal")

        # 3. Difference histogram
        ax3 = plt.subplot(3, 3, 3)
        diff = pred_rust - pred_r
        ax3.hist(diff, bins=30, edgecolor="black", alpha=0.7, color="green")
        ax3.axvline(x=0, color="k", linestyle="--", linewidth=2)
        ax3.axvline(
            x=np.mean(diff),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(diff):.6f}",
        )
        ax3.set_xlabel("Prediction difference")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Difference Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

    # 4-7. Individual feature effects
    feature_names = ["x1: Sinusoidal", "x2: Parabolic", "x3: Linear", "x4: Noise"]
    true_effects = [effect_1, effect_2, effect_3, effect_4]

    for i in range(4):
        ax = plt.subplot(3, 3, 4 + i)
        idx_sort_x = np.argsort(X[:, i])
        ax.scatter(
            X[idx_sort_x, i],
            y[idx_sort_x],
            alpha=0.2,
            s=10,
            label="Data",
            color="lightgray",
        )
        ax.plot(
            X[idx_sort_x, i],
            true_effects[i][idx_sort_x],
            "k-",
            linewidth=2,
            label="True effect",
            alpha=0.7,
        )
        ax.set_xlabel(f"Feature {i + 1}")
        ax.set_ylabel("Effect")
        ax.set_title(feature_names[i])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 8. Residuals
    ax8 = plt.subplot(3, 3, 8)
    residuals = y - pred_rust
    ax8.scatter(pred_rust, residuals, alpha=0.5, s=20)
    ax8.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax8.set_xlabel("Fitted values")
    ax8.set_ylabel("Residuals")
    ax8.set_title("Residual Plot (mgcv_rust)")
    ax8.grid(True, alpha=0.3)

    # 9. Q-Q plot
    ax9 = plt.subplot(3, 3, 9)
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=ax9)
    ax9.set_title("Q-Q Plot (mgcv_rust)")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = "test_4d_multidim_results.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved visualization to {output_file}")


def benchmark_performance(X, y, k=12, n_iters=50):
    """Benchmark fitting performance."""
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)

    import time

    n_vars = X.shape[1]
    k_list = [k] * n_vars

    print(f"\nBenchmarking with {n_iters} iterations...")
    print(f"Data: n={len(y)}, dimensions={n_vars}, k={k} per dimension")

    # Benchmark mgcv_rust
    print("\nTiming mgcv_rust...")
    times_rust = []
    for i in range(n_iters):
        gam = mgcv_rust.GAM()
        start = time.perf_counter()
        gam.fit_auto_optimized(X, y, k=k_list, method="REML", bs="cr")
        end = time.perf_counter()
        times_rust.append(end - start)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_iters} iterations complete...")

    mean_rust = np.mean(times_rust) * 1000  # Convert to ms
    std_rust = np.std(times_rust) * 1000
    min_rust = np.min(times_rust) * 1000
    max_rust = np.max(times_rust) * 1000

    print(f"\nmgcv_rust results:")
    print(f"  Mean:   {mean_rust:.2f} ms")
    print(f"  Std:    {std_rust:.2f} ms")
    print(f"  Min:    {min_rust:.2f} ms")
    print(f"  Max:    {max_rust:.2f} ms")

    # Benchmark R's mgcv
    if HAS_RPY2:
        print("\nTiming R's mgcv...")
        times_r = []

        with localconverter(ro.default_converter + numpy2ri.converter):
            # Set up data once
            for i in range(n_vars):
                ro.globalenv[f"x{i + 1}"] = X[:, i]
            ro.globalenv["y_r"] = y
            ro.globalenv["k_val"] = k

            smooth_terms = [f's(x{i + 1}, k=k_val, bs="cr")' for i in range(n_vars)]
            formula = f"y_r ~ {' + '.join(smooth_terms)}"

            # Time fitting
            for i in range(n_iters):
                start = time.perf_counter()
                ro.r(f'gam_fit_bench <- gam({formula}, method="REML")')
                end = time.perf_counter()
                times_r.append(end - start)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{n_iters} iterations complete...")

        mean_r = np.mean(times_r) * 1000
        std_r = np.std(times_r) * 1000
        min_r = np.min(times_r) * 1000
        max_r = np.max(times_r) * 1000

        print(f"\nR's mgcv results:")
        print(f"  Mean:   {mean_r:.2f} ms")
        print(f"  Std:    {std_r:.2f} ms")
        print(f"  Min:    {min_r:.2f} ms")
        print(f"  Max:    {max_r:.2f} ms")

        # Compute speedup
        speedup = mean_r / mean_rust
        print(f"\n" + "=" * 70)
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"mgcv_rust is {speedup:.2f}x FASTER than R's mgcv ✓")
        else:
            print(f"R's mgcv is {1 / speedup:.2f}x faster than mgcv_rust")
        print("=" * 70)

        return {
            "rust_mean": mean_rust,
            "rust_std": std_rust,
            "r_mean": mean_r,
            "r_std": std_r,
            "speedup": speedup,
        }
    else:
        print("\nR comparison not available (rpy2 not installed)")
        return {
            "rust_mean": mean_rust,
            "rust_std": std_rust,
        }


def test_extrapolation(gam_rust, X_train, n_test=200):
    """Test extrapolation behavior."""
    print("\n" + "=" * 70)
    print("Extrapolation Test")
    print("=" * 70)

    np.random.seed(100)
    n_vars = X_train.shape[1]

    # Generate test points with some values outside [0, 1]
    X_test = np.random.uniform(-0.2, 1.2, size=(n_test, n_vars))

    # Predict with mgcv_rust
    pred_rust_extrap = gam_rust.predict(X_test)

    # Check for anomalies
    has_nan = np.any(np.isnan(pred_rust_extrap))
    has_zeros = np.any(np.abs(pred_rust_extrap) < 1e-10)

    print(f"Test points: {n_test}")
    print(f"Has NaN: {'❌ Yes' if has_nan else '✓ No'}")
    print(f"Has zeros: {'⚠ Yes' if has_zeros else '✓ No'}")
    print(
        f"Prediction range: [{pred_rust_extrap.min():.4f}, {pred_rust_extrap.max():.4f}]"
    )

    if HAS_RPY2:
        # Predict with R mgcv
        with localconverter(ro.default_converter + numpy2ri.converter):
            for i in range(n_vars):
                ro.globalenv[f"x{i + 1}_new"] = X_test[:, i]
            newdata_str = ", ".join([f"x{i + 1}=x{i + 1}_new" for i in range(n_vars)])
            ro.r(
                f"pred_r_extrap <- predict(gam_fit, newdata=data.frame({newdata_str}))"
            )
            pred_r_extrap = np.array(ro.r("pred_r_extrap"))

        corr_extrap = np.corrcoef(pred_rust_extrap, pred_r_extrap)[0, 1]
        rmse_extrap = np.sqrt(np.mean((pred_rust_extrap - pred_r_extrap) ** 2))

        print(f"\nAgreement with R mgcv:")
        print(f"  Correlation: {corr_extrap:.6f}")
        print(f"  RMSE diff:   {rmse_extrap:.6f}")


def main():
    """Main test function."""
    print("=" * 70)
    print("4D Multidimensional Inference Test")
    print("=" * 70)
    print("\nFeatures:")
    print("  1. Sinusoidal effect: sin(2π*x)")
    print("  2. Parabolic effect:  (x - 0.5)^2")
    print("  3. Linear effect:     x")
    print("  4. Constant noise:    (no effect)")

    # Generate data
    n = 2500
    k = 12
    noise_level = 0.3

    print(f"\nParameters:")
    print(f"  n = {n} observations")
    print(f"  k = {k} basis functions per dimension")
    print(f"  noise_level = {noise_level}")

    X, y, y_true, effect_1, effect_2, effect_3, effect_4 = generate_4d_data(
        n, noise_level
    )

    # Fit with mgcv_rust
    gam_rust, result_rust, pred_rust = fit_mgcv_rust(X, y, k)

    # Fit with R's mgcv
    _, result_r, pred_r = fit_mgcv_r(X, y, k)

    # Compare predictions
    metrics = compare_predictions(pred_rust, pred_r, y_true, y)

    # Compare lambda values
    if result_r is not None:
        print("\n" + "=" * 70)
        print("Smoothing Parameter Comparison")
        print("=" * 70)
        lambda_rust = result_rust["lambda"]
        lambda_r = result_r["lambda"]

        # Check if lambda_rust is a scalar (BUG!)
        if isinstance(lambda_rust, (int, float)):
            print(f"\n⚠️  WARNING: mgcv_rust returned a SINGLE lambda value!")
            print(f"   This is likely a BUG in multidimensional inference.")
            print(f"   Expected: {len(lambda_r)} lambda values (one per dimension)")
            print(f"   Got:      1 lambda value = {lambda_rust:.6f}")
            print(f"\n   R's mgcv lambda values:")
            for i, lam in enumerate(lambda_r):
                print(f"     x{i + 1}: {lam:.6f}")
        else:
            print(f"{'Feature':<15} {'λ (Rust)':<15} {'λ (R)':<15} {'Ratio':<10}")
            print("-" * 60)
            for i in range(len(lambda_rust)):
                ratio = (
                    lambda_rust[i] / lambda_r[i] if lambda_r[i] > 0 else float("inf")
                )
                print(
                    f"x{i + 1:<14} {lambda_rust[i]:<15.6f} {lambda_r[i]:<15.6f} {ratio:<10.4f}"
                )

    # Visualize
    effects = (effect_1, effect_2, effect_3, effect_4)
    visualize_results(X, y, y_true, pred_rust, pred_r, effects)

    # Test extrapolation
    test_extrapolation(gam_rust, X)

    # Benchmark performance
    perf_results = benchmark_performance(X, y, k=k, n_iters=50)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if pred_r is not None:
        status_corr = "✓ PASS" if metrics["corr"] > 0.99 else "❌ FAIL"
        status_rmse = "✓ PASS" if metrics["rmse_diff"] < 0.1 else "⚠ WARNING"
        status_max = "✓ PASS" if metrics["max_diff"] < 0.2 else "⚠ WARNING"

        print(f"Agreement tests:")
        print(f"  Correlation > 0.99:  {status_corr} ({metrics['corr']:.6f})")
        print(f"  RMSE diff < 0.1:     {status_rmse} ({metrics['rmse_diff']:.6f})")
        print(f"  Max diff < 0.2:      {status_max} ({metrics['max_diff']:.6f})")

        if "speedup" in perf_results:
            print(f"\nPerformance:")
            print(
                f"  mgcv_rust: {perf_results['rust_mean']:.2f} ± {perf_results['rust_std']:.2f} ms"
            )
            print(
                f"  R mgcv:    {perf_results['r_mean']:.2f} ± {perf_results['r_std']:.2f} ms"
            )
            print(
                f"  Speedup:   {perf_results['speedup']:.2f}x {'✓' if perf_results['speedup'] > 1 else ''}"
            )

        if metrics["corr"] > 0.99 and metrics["rmse_diff"] < 0.1:
            print(
                "\n✓ ALL TESTS PASSED - Multidimensional inference working correctly!"
            )
            return 0
        else:
            print("\n⚠ SOME TESTS FAILED - Check the results above")
            return 1
    else:
        print("✓ mgcv_rust completed successfully")
        print(
            f"  Mean fit time: {perf_results['rust_mean']:.2f} ± {perf_results['rust_std']:.2f} ms"
        )
        print("  (R comparison not available)")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
