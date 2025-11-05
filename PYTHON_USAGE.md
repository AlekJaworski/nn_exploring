# Python Bindings for mgcv_rust

This library provides Python bindings for the Rust GAM implementation, allowing you to use mgcv-style GAMs from Python with matplotlib visualization.

## Installation

### Requirements

- Python 3.8+
- Rust toolchain (for building from source)
- maturin (Python package builder)

### Building and Installing

```bash
# Install maturin
pip install maturin

# Build and install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/mgcv_rust-*.whl
```

## Usage

There are **three ways** to fit GAMs, from simplest to most flexible:

### Option 1: Automatic with k list (Recommended - Pythonic)

```python
import numpy as np
import mgcv_rust

# Generate data
n = 300
x = np.linspace(0, 1, n).reshape(-1, 1)
y = np.sin(2 * np.pi * x.flatten()) + 0.5 * np.random.randn(n)

# Fit with automatic setup - just specify k!
gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[15], method="GCV")
print(f"Selected λ: {result['lambda']:.6f}")

# Make predictions
y_pred = gam.predict(x)
```

### Option 2: Formula-based (R/mgcv-style)

```python
# Fit with R-like formula
gam = mgcv_rust.GAM()
result = gam.fit_formula(x, y, formula="s(0, k=15)", method="GCV")
print(f"Selected λ: {result['lambda']:.6f}")

# Make predictions
y_pred = gam.predict(x)
```

### Option 3: Manual (Most flexible)

```python
# Manually add smooths then fit
gam = mgcv_rust.GAM()
gam.add_cubic_spline("x", num_basis=15, x_min=0.0, x_max=1.0)
result = gam.fit(x, y, method="GCV", max_iter=10)
print(f"Selected λ: {result['lambda']:.6f}")

# Make predictions
x_new = np.linspace(0, 1, 200).reshape(-1, 1)
y_pred = gam.predict(x_new)

# Get fitted values
y_fit = gam.get_fitted_values()
```

## Examples

### API Demo

See `python_api_demo.py` for comprehensive demonstration of all three APIs:
```bash
python python_api_demo.py
```

### Visualization Example

See `python_example.py` for a complete example with matplotlib visualization showing:
- GAM fit vs true function
- Residual plots
- REML vs GCV comparison
- Lambda selection as a function of basis complexity

Run it with:
```bash
python python_example.py
```

## API Reference

### GAM Class

#### Constructor

- **`GAM()`**: Create a new GAM model with Gaussian family

#### Fitting Methods (choose one)

- **`fit_auto(x, y, k, method, max_iter=10)`**: **Recommended** - Automatic fitting with k list
  - `x` (ndarray): Input data, shape (n, d)
  - `y` (ndarray): Response variable, shape (n,)
  - `k` (list[int]): List of basis dimensions for each column (like `k` in mgcv)
  - `method` (str): "GCV" or "REML"
  - `max_iter` (int, optional): Maximum iterations
  - Returns: dict with keys `lambda`, `fitted`, `deviance`
  - Example: `gam.fit_auto(X, y, k=[15], method='GCV')`
  - Note: Currently only single-predictor (d=1) supported

- **`fit_formula(x, y, formula, method, max_iter=10)`**: R/mgcv-style formula fitting
  - `x` (ndarray): Input data, shape (n, d)
  - `y` (ndarray): Response variable, shape (n,)
  - `formula` (str): Formula like `"s(0, k=15)"` or `"s(0, k=10) + s(1, k=20)"`
  - `method` (str): "GCV" or "REML"
  - `max_iter` (int, optional): Maximum iterations
  - Returns: dict with keys `lambda`, `fitted`, `deviance`
  - Example: `gam.fit_formula(X, y, formula="s(0, k=15)", method='GCV')`
  - Note: Column indices are 0-based; currently only single smooth supported

- **`fit(x, y, method, max_iter=10)`**: Manual fitting (requires pre-added smooths)
  - `x` (ndarray): Input data, shape (n, d)
  - `y` (ndarray): Response variable, shape (n,)
  - `method` (str): "GCV" or "REML"
  - `max_iter` (int, optional): Maximum iterations
  - Returns: dict with keys `lambda`, `fitted`, `deviance`
  - Use with `add_cubic_spline()` to manually configure smooths

#### Smooth Configuration (for manual fitting)

- **`add_cubic_spline(var_name, num_basis, x_min, x_max)`**: Add a cubic spline smooth term
  - `var_name` (str): Variable name
  - `num_basis` (int): Number of basis functions (like `k` in mgcv)
  - `x_min` (float): Minimum x value
  - `x_max` (float): Maximum x value

#### Prediction

- **`predict(x)`**: Make predictions
  - `x` (ndarray): Input data for prediction, shape (n, d)
  - Returns: ndarray of predictions, shape (n,)

#### Model Information

- **`get_lambda()`**: Get selected smoothing parameter
  - Returns: float

- **`get_fitted_values()`**: Get fitted values from training data
  - Returns: ndarray

## Performance

The Rust implementation provides significant speedup over pure Python implementations while maintaining the same statistical properties as R's mgcv package.

## Comparison with mgcv (R)

### Simple k-based approach

```python
# Python with mgcv_rust (Pythonic)
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[15], method='GCV')
```

```r
# R with mgcv
library(mgcv)
gam_model <- gam(y ~ s(x, k=15), method="GCV.Cp")
```

### Formula-based approach

```python
# Python with mgcv_rust (R-like)
gam = mgcv_rust.GAM()
result = gam.fit_formula(X, y, formula="s(0, k=15)", method='GCV')
```

```r
# R with mgcv
library(mgcv)
gam_model <- gam(y ~ s(x, k=15), method="GCV.Cp")
```

All approaches use the same underlying algorithms (PiRLS, GCV/REML) and should produce similar results.

## Current Limitations

- Multi-predictor GAMs (X with multiple columns): Not yet implemented for automatic smoothing parameter selection
- Only cubic spline basis available (thin plate splines coming soon)
- Only Gaussian family currently supported
