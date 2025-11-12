# Multidimensional GAM Test Suite

This document describes the comprehensive test suite for multidimensional Generalized Additive Models (GAMs) in mgcv_rust.

## Overview

Two test files have been created to ensure mgcv_rust correctly handles GAMs with multiple predictors:

1. **`test_multidimensional_internal.py`** - Internal consistency tests (no R required)
2. **`test_multidimensional_mgcv.py`** - Comparison tests with R's mgcv package (requires R and rpy2)

## Test Coverage

### Dimensionality Tests
- **2D cases**: Two-variable GAMs with various signal types
- **3D cases**: Three-variable GAMs with mixed complexities
- **4D cases**: Four-variable GAMs
- **5D cases**: Five-variable GAMs

### Complexity Combinations
- Linear functions (should have high λ values for strong smoothing)
- Quadratic functions (moderate λ values)
- Cubic functions (moderate to low λ values)
- Sine wave functions (low λ values for flexible fitting)
- Mixed complexity scenarios

### Key Test Scenarios
1. **All linear**: Tests that all λ values are high when signals are simple
2. **All complex**: Tests that all λ values are low when signals need flexibility
3. **Mixed complexity**: Tests that λ values are ordered by signal complexity
4. **Different basis sizes**: Tests with varying k values across predictors
5. **Small k values** (k=5): Edge case with limited basis functions
6. **Large k values** (k=20): Edge case with many basis functions
7. **Noisy data**: Tests behavior with high noise variance

### Consistency Tests
- **fit_auto vs fit_formula**: Ensures both APIs produce identical results
- **Reproducibility**: Ensures same data produces same results
- **get_all_lambdas()**: Tests method for retrieving all smoothing parameters

### Comparison Tests (with R mgcv)
- **Prediction correlation**: Ensures predictions are highly correlated (>0.90)
- **Lambda similarity**: Ensures smoothing parameters are in similar ranges
- **Lambda ordering**: Ensures complexity ordering matches between implementations
- **Model fit quality**: Ensures comparable RMSEs

## Current Status

⚠️ **IMPORTANT**: The multidimensional tests currently expose a **bug in the REML optimization** for cases with multiple smoothing parameters.

### Known Issue
When fitting GAMs with 2 or more predictors using REML, the Newton optimization encounters a singular matrix error. This affects:
- All 2D+ tests in `test_multidimensional_internal.py`
- All 2D+ tests in `test_multidimensional_mgcv.py`
- The `examples/multi_variable_gam.rs` example

### What Works
- ✅ 1D GAMs (single predictor) work correctly
- ✅ Test infrastructure is complete and ready
- ✅ Test data generation is correct
- ✅ Expected behaviors are well-defined

### What Needs Fixing
- ❌ Multi-lambda Newton optimization in REML (likely in `src/reml.rs` or `src/smooth.rs`)
- ❌ Hessian computation for multiple smoothing parameters
- ❌ Matrix operations in the multi-dimensional case

## Running the Tests

### Prerequisites

```bash
# Install Python dependencies
pip install numpy

# For comparison tests with R mgcv (optional)
pip install rpy2
# Also requires R with mgcv package installed
```

### Build mgcv_rust

```bash
# Build and install the Python package
pip install maturin
maturin build --release
pip install target/wheels/mgcv_rust-*.whl
```

### Run Internal Tests

```bash
# These test internal consistency (no R required)
python test_multidimensional_internal.py
```

### Run Comparison Tests

```bash
# These compare with R's mgcv (requires R and rpy2)
python test_multidimensional_mgcv.py
```

## Test Design Philosophy

The tests are designed to:

1. **Define expected behavior**: Each test clearly states what the implementation should do
2. **Test edge cases**: Cover boundary conditions and unusual scenarios
3. **Compare with reference**: Use R's mgcv as ground truth
4. **Be comprehensive**: Cover all dimensionalities from 1D to 5D+
5. **Be maintainable**: Clear naming and documentation
6. **Expose bugs**: Current failures indicate areas needing work

## Expected Test Behavior (Once Fixed)

Once the multidimensional REML bug is fixed, tests should verify:

### Lambda Patterns
For a 3D model with `y = sin(2πx₁) + 0.5x₂² + 2x₃ + noise`:
- λ₁ (sine) should be **low** - needs flexibility
- λ₂ (quadratic) should be **medium** - moderate smoothing
- λ₃ (linear) should be **high** - strong smoothing for simple signal

### Prediction Accuracy
- Predictions should correlate > 0.90 with R's mgcv
- RMSE difference should be small (< 0.5 for typical cases)
- Fits should be reasonable (RMSE < 1.0 for signal with noise σ=0.2)

### Consistency
- `fit_auto` and `fit_formula` should produce identical results (decimal=6)
- Results should be reproducible (decimal=10)
- All λ values should be positive and finite

## Debug Information

When debugging the singular matrix issue, check:

1. **Hessian computation**: Is the Hessian of REML w.r.t. log(λ) correct?
2. **Matrix conditioning**: Are matrices becoming ill-conditioned?
3. **Initial lambda values**: Are starting values reasonable?
4. **Penalty matrices**: Are individual penalty matrices correct?
5. **Block structure**: Is the block-diagonal structure preserved?

## Future Enhancements

After fixing the core multidimensional bug:

1. Add tests for different basis types ('bs', 'cr', 'tp')
2. Add tests for GCV optimization (currently tests use REML)
3. Add tests for prediction on new data
4. Add tests for extrapolation behavior
5. Add performance benchmarks
6. Add tests for very high-dimensional cases (10+ predictors)

## References

- Wood, S.N. (2011) "Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models"
- Wood, S.N. (2017) "Generalized Additive Models: An Introduction with R" (2nd ed)
- R package mgcv: https://cran.r-project.org/package=mgcv

## Contributing

When the multidimensional REML bug is fixed:

1. Run all tests to verify they pass
2. Check that lambda patterns match expectations
3. Verify predictions correlate highly with R mgcv (>0.90)
4. Ensure all consistency tests pass

The comprehensive test suite will ensure the fix works correctly across all scenarios.
