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

```python
import numpy as np
import mgcv_rust

# Generate data
n = 300
x = np.linspace(0, 1, n).reshape(-1, 1)
y = np.sin(2 * np.pi * x.flatten()) + 0.5 * np.random.randn(n)

# Create and fit GAM
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

## Example with Visualization

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

#### Methods

- **`GAM()`**: Create a new GAM model with Gaussian family

- **`add_cubic_spline(var_name, num_basis, x_min, x_max)`**: Add a cubic spline smooth term
  - `var_name` (str): Variable name
  - `num_basis` (int): Number of basis functions (like `k` in mgcv)
  - `x_min` (float): Minimum x value
  - `x_max` (float): Maximum x value

- **`fit(x, y, method, max_iter=10)`**: Fit the GAM
  - `x` (ndarray): Input data, shape (n, d)
  - `y` (ndarray): Response variable, shape (n,)
  - `method` (str): "GCV" or "REML"
  - `max_iter` (int, optional): Maximum iterations
  - Returns: dict with keys `lambda`, `converged`, `iterations`, `deviance`

- **`predict(x)`**: Make predictions
  - `x` (ndarray): Input data for prediction, shape (n, d)
  - Returns: ndarray of predictions, shape (n,)

- **`get_lambda()`**: Get selected smoothing parameter
  - Returns: float

- **`get_fitted_values()`**: Get fitted values from training data
  - Returns: ndarray

## Performance

The Rust implementation provides significant speedup over pure Python implementations while maintaining the same statistical properties as R's mgcv package.

## Comparison with mgcv (R)

```python
# Python with mgcv_rust
gam = mgcv_rust.GAM()
gam.add_cubic_spline("x", num_basis=15, x_min=0, x_max=1)
result = gam.fit(x, y, method="GCV")
```

```r
# R with mgcv
library(mgcv)
gam_model <- gam(y ~ s(x, k=15), method="GCV.Cp")
```

Both use the same underlying algorithms (PiRLS, GCV/REML) and should produce similar results.
