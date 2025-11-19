# Multidimensional GAM Performance Report

## Summary

Multidimensional GAMs work correctly and efficiently at commit `362e8a3` (before k-1 basis changes).

## Performance Results

### 2D GAM (n=200, k=[16,16])
- **Time**: 0.017s
- **R²**: 0.9396
- **λ**: [0.104, 0.0]

### 3D GAM (n=200, k=[12,12,12])
- **Time**: 0.027s
- **R²**: 0.9882
- **λ**: [0.082, 0.0, 0.0]

### 4D GAM (n=200, k=[16,16,16,16])
- **Time**: 0.090s
- **R²**: 0.9937
- **λ**: [0.089, 0.0, 0.0, 0.0]

### 5D GAM (n=250, k=[10,10,10,10,10])
- **Time**: 0.076s
- **R²**: 0.9837
- **λ**: [0.089, 0.0, 0.0, 0.0, 0.0]

## Observations

1. **Excellent Fit Quality**: R² values range from 0.94 to 0.99, indicating very good model fits
2. **Fast Performance**: Even 5D GAMs complete in <0.1s for n=250
3. **Scaling**: Time scales roughly linearly with dimensionality
   - 2D: 0.017s
   - 3D: 0.027s (1.6x)
   - 4D: 0.090s (5.3x from 2D, but higher k values)
   - 5D: 0.076s (competitive with 4D despite extra dimension)

4. **Lambda Convergence**: Most λ values converge to 0 except for the first smooth term
   - This suggests most smooths are being fit with minimal regularization
   - May need better initialization or convergence criteria

## Known Issues

### Regression in Later Commits

**IMPORTANT**: Commits after 362e8a3 introduced multidimensional bugs:

1. **Commit a0f3435** ("Add intercept and use k-1 basis functions"):
   - Breaks multidimensional GAMs
   - Design matrix dimension mismatch (missing intercept column)
   - Error: "inputs 200 × 30 and 31 × 1 are not compatible"

2. **Commit a1da133** ("Fix gradient catastrophe: implement proper constraint absorption"):
   - Further exacerbates issues
   - Index out of bounds errors
   - Affects even single-dimensional GAMs in some cases

### Root Causes

The k-1 basis and constraint absorption changes were implemented without adequate multidimensional testing. Issues include:

- Inconsistent handling of `num_basis` vs actual basis dimensions
- Missing intercept column in design matrix construction
- Penalty matrix dimension mismatches when different k values are used

## Recommendations

### Immediate

1. **Stay at commit 362e8a3** for production use with multidimensional data
2. **Add CI tests** for multidimensional GAMs before merging future changes
3. **Document** the multidimensional capability

### Future Work

1. **Fix k-1 basis implementation** to properly handle:
   - Intercept inclusion
   - Multiple smooths with different k values
   - Consistent dimension tracking

2. **Improve constraint absorption** to:
   - Match mgcv's penalty rank (currently rank k-2, should be k-1 after absorption)
   - Maintain compatibility with multidimensional cases

3. **Better lambda initialization**:
   - Currently most λ → 0, suggests poor initialization
   - Consider data-driven initialization like mgcv

4. **Add comprehensive test suite**:
   - Test 2D through 10D
   - Test different k values per dimension
   - Test edge cases (small n, large k, etc.)

## Benchmark Script

Run `benchmark_multidim.py` to reproduce these results.
