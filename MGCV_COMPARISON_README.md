# Comparing mgcv_rust with R's mgcv

This directory contains tests and notebooks to compare the Rust implementation with R's mgcv package.

## Prerequisites

### Install rpy2
```bash
pip install rpy2
```

### Install R and mgcv
Make sure you have R installed, then:
```R
install.packages("mgcv")
```

## Running the Tests

### 1. Unit Tests (test_mgcv_comparison.py)

Run comprehensive unit tests comparing predictions, lambda values, and extrapolation:

```bash
python3 test_mgcv_comparison.py
```

**What it tests:**
- ✅ 1-variable predictions match (correlation > 0.95)
- ✅ Smoothing parameters (λ) in similar range
- ✅ Linear function handling
- ✅ Multi-variable GAM predictions
- ✅ Extrapolation behavior (no zeros)
- ✅ Reproducibility
- ✅ fit_auto vs fit_formula consistency

**Expected output:**
```
test_1var_predictions_match ... ok
test_lambda_similar ... ok
test_linear_function_exact ... ok
test_multi_variable_predictions ... ok
test_extrapolation_behavior ... ok
test_reproducibility ... ok
test_fit_auto_vs_fit_formula ... ok
```

### 2. Interactive Marimo Notebook (compare_with_mgcv.marimo.py)

Launch the interactive comparison notebook:

```bash
marimo edit compare_with_mgcv.marimo.py
```

**Features:**
- 🎛️ Interactive sliders to adjust data and parameters
- 📊 Visual comparison of predictions
- 📈 Extrapolation visualization
- 📉 Residual analysis
- 🔍 Side-by-side metric comparison

## What to Expect

### Predictions
- Correlation between implementations: **> 0.95** (excellent match)
- RMSE difference: **< 0.1** (low)

### Smoothing Parameters (λ)
- Rust/R ratio: **0.5 - 2.0** (same order of magnitude)
- May differ due to:
  - Different penalty parameterizations
  - Numerical optimization differences
  - Knot placement strategies

### Extrapolation
- **Both should NOT produce zeros** outside training range
- Linear continuation with gradient from boundary
- Predictions should be similar in extrapolation regions

## Troubleshooting

### "rpy2 not available"
```bash
pip install rpy2
```

### "R mgcv package not available"
In R:
```R
install.packages("mgcv")
```

### "Cannot find R installation"
Set R_HOME environment variable:
```bash
export R_HOME=/usr/lib/R  # Adjust path for your system
```

### Check R installation:
```bash
which R
R --version
```

## Interpreting Results

### Good Results ✅
- Correlation > 0.95
- λ ratio between 0.5-2.0
- No zeros in extrapolation
- Predictions visually similar

### Potential Issues ⚠️
- Correlation < 0.9: Check data scaling
- λ ratio > 10 or < 0.1: Different parameterization
- Zeros in extrapolation: Boundary handling issue

## Files

- `test_mgcv_comparison.py`: Unit tests with rpy2
- `compare_with_mgcv.marimo.py`: Interactive visual comparison
- `MGCV_COMPARISON_README.md`: This file
