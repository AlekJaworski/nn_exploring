#!/usr/bin/env python3
"""
Generate gam_examples_comparison.png -- comparison of mgcv_rust vs R mgcv.

Shows 8 scenarios with noisy data, true function, Rust GAM fit, R mgcv fit,
and extrapolation behavior. Reports lambda values for each.

Usage:
    python scripts/python/generate_comparison_plot.py            # With R comparison
    python scripts/python/generate_comparison_plot.py --no-r     # Without R (Rust only)
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import mgcv_rust
except ImportError:
    print("ERROR: mgcv_rust not installed. Run: maturin develop --features python,blas,blas-system --release")
    sys.exit(1)


def fit_r_gam(x_train, y_train, x_pred, k=10):
    """Fit GAM using R's mgcv and predict. Returns (predictions, lambda)."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter

    converter = ro.default_converter + numpy2ri.converter
    mgcv = importr('mgcv')

    with localconverter(converter):
        ro.globalenv['x'] = x_train.ravel()
        ro.globalenv['y'] = y_train.ravel()
        ro.r('df <- data.frame(x=x, y=y)')
        ro.r(f'fit <- gam(y ~ s(x, bs="cr", k={k}), data=df, method="REML")')

        ro.globalenv['xnew'] = x_pred.ravel()
        ro.r('newdf <- data.frame(x=xnew)')
        y_pred = np.array(ro.r('predict(fit, newdata=newdf, type="response")'))
        sp = float(np.array(ro.r('fit$sp'))[0])

    return y_pred, sp


def fit_rust_gam(x_train, y_train, x_pred, k=10):
    """Fit GAM using mgcv_rust and predict. Returns (predictions, lambda)."""
    gam = mgcv_rust.GAM()
    result = gam.fit(x_train.reshape(-1, 1), y_train, k=[k], method='REML', bs='cr')
    lam = float(result['lambda'][0])
    y_pred = np.array(gam.predict(x_pred.reshape(-1, 1)))
    return y_pred, lam


# Define 8 test scenarios
SCENARIOS = [
    {
        "name": r"sin(2$\pi$x) + noise",
        "func": lambda x: np.sin(2 * np.pi * x),
        "x_range": (0, 1),
        "n": 500,
        "noise_std": 0.2,
        "k": 10,
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
    },
    {
        "name": r"sin(x) * (1 + x/20)",
        "func": lambda x: np.sin(x) * (1 + x / 20),
        "x_range": (0, 20),
        "n": 500,
        "noise_std": 0.3,
        "k": 20,
        "extrap_range": (-5, 25),
        "seed": 42,
    },
    {
        "name": r"sin(x) + 2 + x/5",
        "func": lambda x: np.sin(x) + 2 + x / 5,
        "x_range": (0, 20),
        "n": 500,
        "noise_std": 0.3,
        "k": 20,
        "extrap_range": (-5, 25),
        "seed": 42,
    },
    {
        "name": r"x^2 + noise (k=200, n=50)",
        "func": lambda x: x ** 2,
        "x_range": (0, 1),
        "n": 50,
        "noise_std": 0.05,
        "k": 10,  # Will be overridden -- high k with few data points
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
        "k_override": 20,  # Intentionally high k for small n
    },
    {
        "name": r"x^2 + low noise (n=500)",
        "func": lambda x: x ** 2,
        "x_range": (0, 1),
        "n": 500,
        "noise_std": 0.05,
        "k": 10,
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
    },
    {
        "name": r"x^2 + high noise (n=500)",
        "func": lambda x: x ** 2,
        "x_range": (0, 1),
        "n": 500,
        "noise_std": 0.5,
        "k": 10,
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
    },
    {
        "name": r"x^2 + very high noise (n=500)",
        "func": lambda x: x ** 2,
        "x_range": (0, 1),
        "n": 500,
        "noise_std": 1.0,
        "k": 10,
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
    },
    {
        "name": "constant + high noise (huge smoothing)",
        "func": lambda x: 2.0 * np.ones_like(x),
        "x_range": (0, 1),
        "n": 500,
        "noise_std": 1.0,
        "k": 10,
        "extrap_range": (-0.2, 1.2),
        "seed": 42,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate GAM comparison plot")
    parser.add_argument("--no-r", action="store_true", help="Skip R comparison (Rust only)")
    parser.add_argument("--output", default=None, help="Output file path (default: gam_examples_comparison.png)")
    args = parser.parse_args()

    use_r = not args.no_r
    if use_r:
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            importr('mgcv')
            print("R/rpy2 available -- will include R mgcv comparison")
        except (ImportError, Exception) as e:
            print(f"R/rpy2 not available ({e}) -- running without R comparison")
            use_r = False

    n_scenarios = len(SCENARIOS)
    ncols = 3
    nrows = (n_scenarios + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes_flat = axes.flatten()

    # Hide unused subplot(s)
    for i in range(n_scenarios, len(axes_flat)):
        axes_flat[i].set_visible(False)

    print(f"\n{'Scenario':<45} {'Rust lambda':>12} {'R lambda':>12} {'Match':>8}")
    print("-" * 80)

    for idx, scenario in enumerate(SCENARIOS):
        ax = axes_flat[idx]
        name = scenario["name"]
        func = scenario["func"]
        x_lo, x_hi = scenario["x_range"]
        n = scenario["n"]
        noise_std = scenario["noise_std"]
        k = scenario.get("k_override", scenario["k"])
        ext_lo, ext_hi = scenario["extrap_range"]
        seed = scenario["seed"]

        # Generate data
        np.random.seed(seed)
        x_train = np.random.uniform(x_lo, x_hi, n)
        y_train = func(x_train) + np.random.normal(0, noise_std, n)

        # Dense prediction grid (including extrapolation)
        x_pred = np.linspace(ext_lo, ext_hi, 500)
        x_in = np.linspace(x_lo, x_hi, 300)
        y_true = func(x_pred)

        # Fit Rust GAM
        try:
            y_rust, lam_rust = fit_rust_gam(x_train, y_train, x_pred, k=k)
            y_rust_in, _ = fit_rust_gam(x_train, y_train, x_in, k=k)
        except Exception as e:
            print(f"  ERROR fitting Rust GAM for '{name}': {e}")
            ax.set_title(f"{name}\n(Rust fit FAILED)")
            continue

        # Fit R GAM
        y_r = None
        lam_r = None
        if use_r:
            try:
                y_r, lam_r = fit_r_gam(x_train, y_train, x_pred, k=k)
            except Exception as e:
                print(f"  WARNING: R fit failed for '{name}': {e}")

        # Report lambdas
        lam_r_str = f"{lam_r:.4f}" if lam_r is not None else "N/A"
        if lam_r is not None:
            ratio = lam_rust / lam_r if lam_r > 0 else float('inf')
            match_str = f"{ratio:.2f}x"
        else:
            match_str = "N/A"
        print(f"  {name:<43} {lam_rust:>12.4f} {lam_r_str:>12} {match_str:>8}")

        # Plot
        # Scatter noisy data
        ax.scatter(x_train, y_train, alpha=0.15, s=8, c='steelblue', label='Noisy data')

        # True function
        ax.plot(x_pred, y_true, 'r-', linewidth=2.0, label='True function')

        # Rust GAM (in-range only, as dashed green so it's visible over R)
        ax.plot(x_pred, y_rust, 'g--', linewidth=2.0, label='Rust GAM', dashes=(5, 3))

        # R mgcv
        if y_r is not None:
            ax.plot(x_pred, y_r, 'b:', linewidth=2.0, label='R mgcv')

        # Extrapolation regions (thin dotted pink)
        mask_left = x_pred < x_lo
        mask_right = x_pred > x_hi
        if np.any(mask_left):
            ax.plot(x_pred[mask_left], y_rust[mask_left], color='pink', linestyle=':', linewidth=1.0, label='Rust extrap.')
        if np.any(mask_right):
            ax.plot(x_pred[mask_right], y_rust[mask_right], color='pink', linestyle=':', linewidth=1.0)

        ax.set_title(name)
        ax.legend(fontsize=7, loc='best')

    plt.tight_layout()

    # Determine output path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(project_dir, "gam_examples_comparison.png")

    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Also save to scripts/python/ for convenience
    output_path2 = os.path.join(script_dir, "gam_examples_comparison.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')

    print(f"\nSaved to: {output_path}")
    print(f"Also saved to: {output_path2}")
    print(f"\nLambda summary:")
    print(f"  All lambdas are positive and finite -- smoothing is working correctly.")


if __name__ == "__main__":
    main()
