# Maturin Build Guide for mgcv_rust

This guide explains how to build and install the Python bindings for the mgcv_rust project using Maturin.

## Prerequisites

### System Dependencies

You need to have the following installed:

1. **Rust toolchain** (cargo and rustc)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Python 3.8+** with pip
   ```bash
   python --version
   pip --version
   ```

3. **OpenBLAS library** (required for the `blas` feature)
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libopenblas-dev

   # On macOS
   brew install openblas

   # On Fedora/RHEL
   sudo dnf install openblas-devel
   ```

### Python Dependencies

Install Maturin and patchelf:
```bash
pip install maturin patchelf
```

## Building the Project

### Option 1: Development Build (faster, for testing)

```bash
maturin build --features "python,blas"
```

This creates a debug build in `target/wheels/`.

### Option 2: Release Build (optimized, for production)

```bash
maturin build --release --features "python,blas"
```

This creates an optimized build in `target/wheels/`. The release build:
- Uses optimization level 3
- Applies link-time optimization (LTO)
- Strips debug symbols
- Results in much faster runtime performance

### Option 3: Development with Hot Reload

For rapid development, use:
```bash
maturin develop --features "python,blas"
```

This builds and installs the package in your current Python environment without creating a wheel file.

## Installing the Built Package

After building, install the wheel file:

```bash
pip install --force-reinstall target/wheels/mgcv_rust-*.whl
```

Or for a specific version:
```bash
pip install --force-reinstall target/wheels/mgcv_rust-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
```

## Using the Package in Python

Once installed, you can use it in your Python code:

```python
import numpy as np
from mgcv_rust import GAM

# Create a GAM model
gam = GAM(family="gaussian")

# Prepare your data
X = np.random.randn(100, 2)
y = np.random.randn(100)

# Fit with automatic basis setup
result = gam.fit_auto(
    X, y,
    k=[10, 10],           # Number of basis functions per variable
    method='REML',        # or 'GCV'
    bs='cr',              # 'cr' (cubic regression) or 'bs' (B-splines)
    max_iter=10
)

# Get results
print(f"Lambda: {result['lambda']}")
print(f"Fitted values: {result['fitted_values']}")

# Make predictions
predictions = gam.predict(X)
```

## Available Python Functions

### GAM Class Methods

- `GAM(family=None)` - Constructor (family: 'gaussian', 'binomial', 'poisson', 'gamma')
- `fit_auto(x, y, k, method, bs=None, max_iter=None)` - Fit with automatic basis setup
- `fit_auto_optimized(...)` - Optimized version with caching
- `fit_formula(x, y, formula, method, max_iter=None)` - Fit using formula syntax
- `predict(x)` - Make predictions
- `get_lambda()` - Get smoothing parameter
- `get_all_lambdas()` - Get all smoothing parameters (multi-variable)
- `get_fitted_values()` - Get fitted values
- `get_coefficients()` - Get model coefficients
- `get_design_matrix()` - Get design matrix

### Standalone Functions

- `compute_penalty_matrix(basis_type, num_basis, knots)` - Compute penalty matrix
- `evaluate_gradient(x, y, lambdas, k_values)` - Evaluate REML gradient
- `reml_gradient_multi_qr_py(...)` - REML gradient computation
- `reml_hessian_multi_qr_py(...)` - REML Hessian computation
- `newton_pirls_py(...)` - Newton-PIRLS optimization

## Troubleshooting

### Error: `unable to find library -lopenblas`

**Solution:** Install OpenBLAS development libraries:
```bash
sudo apt-get install libopenblas-dev
```

### Error: `Failed to execute 'patchelf'`

**Solution:** Install patchelf:
```bash
pip install patchelf
```

### Error: `maturin: command not found`

**Solution:** Install Maturin:
```bash
pip install maturin
```

### Build Warnings

The build may show warnings about unused imports and snake_case naming. These are non-critical and don't affect functionality.

## Project Configuration

### Cargo.toml Features

- `python` - Enables PyO3 Python bindings
- `blas` - Enables optimized linear algebra with OpenBLAS
- Default: Both features should be enabled for Python usage

### pyproject.toml

The project uses Maturin as the build backend:
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["python", "blas"]
module-name = "mgcv_rust"
```

## Performance Notes

1. **Always use release builds for benchmarking** - Debug builds are 10-100x slower
2. **The BLAS feature is essential** - It provides optimized matrix operations
3. **The optimized version (`fit_auto_optimized`)** uses caching for better performance

## CI/CD Integration

For automated builds in CI/CD:

```bash
# Install dependencies
pip install maturin patchelf
sudo apt-get install -y libopenblas-dev

# Build wheel
maturin build --release --features "python,blas"

# Install and test
pip install target/wheels/mgcv_rust-*.whl
python -c "import mgcv_rust; print('Success!')"
```

## Additional Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [PyO3 Guide](https://pyo3.rs/)
- [Project README](README.md)
