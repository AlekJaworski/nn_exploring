# Lambda Overfitting Investigation

**Date:** 2025-11-12
**Branch:** `claude/investigate-lambda-overfit-011CV4FUNztxt1YYxUtzDQBy`

## Problem Statement

The lambda (smoothing parameter) optimization in mgcv_rust appears to be overfit to k=10 (or thereabouts), producing significantly different lambda values compared to R's mgcv as k increases beyond 10.

## Investigation Methodology

Created `test_lambda_vs_k.py` to systematically compare lambda values between mgcv_rust and R's mgcv across different k values (5, 7, 10, 12, 15, 20, 25, 30).

**Test Data:**
- Function: y = sin(2πx) + noise
- n = 100 samples
- noise_level = 0.2
- Basis: B-splines (`bs="bs"` in R)
- Method: REML

## Key Findings

### 1. Lambda Ratio Varies Dramatically with k

| k | λ_rust | λ_mgcv | Ratio (Rust/mgcv) |
|---|--------|--------|-------------------|
| 5 | 0.001473 | 0.002723 | 0.5412 |
| 7 | 0.000581 | 0.081122 | 0.0072 |
| 10 | 0.000482 | 0.542645 | 0.0009 |
| 12 | 0.000465 | 1.301241 | 0.0004 |
| 15 | 0.000441 | 3.283078 | 0.0001 |
| 20 | 0.000419 | 9.246667 | 0.0000 |
| 25 | 0.000454 | 19.833367 | 0.0000 |
| 30 | 0.000481 | 36.861773 | 0.0000 |

**Statistics:**
- Mean ratio: 0.0687
- Std dev: 0.1786
- Min: 0.0000 (k=30)
- Max: 0.5412 (k=5)
- Range: 0.5412

### 2. Pattern Analysis

1. **mgcv_rust consistently finds lower lambdas** as k increases
   - For k=5: rust finds λ ≈ 0.0015
   - For k=30: rust finds λ ≈ 0.0005 (similar magnitude!)

2. **R's mgcv increases lambda dramatically with k**
   - For k=5: mgcv finds λ ≈ 0.0027
   - For k=30: mgcv finds λ ≈ 36.86 (increase of 4 orders of magnitude!)

3. **The ratio degrades exponentially**
   - k=5: ratio = 0.54 (reasonable agreement)
   - k=10: ratio = 0.0009 (1000x difference!)
   - k=30: ratio < 0.00001 (>100,000x difference!)

### 3. Prediction Quality

Despite the lambda differences, predictions remain highly correlated:
- Mean correlation: 0.9967
- Min correlation: 0.9758 (k=5)

However, **RMSE vs true function** shows R performs better:
- Rust RMSE mean: 0.0643
- R RMSE mean: 0.0604
- R finds optimal k=7, Rust finds optimal k=15

## Root Cause Analysis

### Suspected Issues

1. **REML Criterion Calculation**
   - The REML formula in `src/reml.rs` may be missing penalty scaling terms
   - mgcv applies additional transformations based on k and penalty matrix rank

2. **Penalty Matrix Scaling**
   - Penalty matrices may need to be scaled differently for different k values
   - mgcv normalizes penalties in a k-dependent way

3. **Newton Optimization**
   - Starting lambda may be inappropriate for large k
   - Gradient/Hessian calculations may not account for k-dependent scaling
   - Current initialization: `vec![0.1; num_smooths]` regardless of k

4. **Rank Estimation**
   - `estimate_rank()` in `reml.rs` may not correctly estimate penalty rank for all k
   - This affects the REML criterion directly through the log(λ) term

### Specific Code Locations

1. **`src/smooth.rs:26`**: Initial lambda = 0.1 (not k-dependent)
   ```rust
   lambda: vec![0.1; num_smooths],
   ```

2. **`src/reml.rs:136-139`**: Log lambda term in REML
   ```rust
   let log_lambda_term = if lambda > 1e-10 && rank_s > 0 {
       (rank_s as f64) * lambda.ln()
   } else {
       0.0
   };
   ```

3. **`src/reml.rs:9-41`**: Rank estimation
   ```rust
   fn estimate_rank(matrix: &Array2<f64>) -> usize {
       // May not work correctly for all penalty matrices
   }
   ```

4. **`src/penalty.rs:760-762`**: Penalty normalization
   ```rust
   let L = knots[n - 1] - knots[0];
   let normalization = 1.0 / L;
   let S_normalized = &S * normalization;
   ```

## Recommendations

### Immediate Actions

1. **Compare R's mgcv REML source code**
   - Extract exact REML formula from `fast-REML.r`
   - Verify all scaling factors and normalizations

2. **Test penalty matrix values**
   - Compare penalty matrices (S) between rust and mgcv for different k
   - Check if penalties need k-dependent scaling

3. **Improve initial lambda guess**
   - Use k-dependent initialization
   - Consider: `initial_lambda ∝ k^α` where α needs to be determined

4. **Add comprehensive REML tests**
   - Test REML criterion values directly against mgcv
   - For fixed lambda and k values, compare REML scores

### Long-term Improvements

1. **Grid search initialization**
   - Use coarse grid search over log(λ) before Newton optimization
   - Adapt grid range based on k value

2. **Adaptive step sizing**
   - Current max step = 4.0 in log space may be too large for high k
   - Consider k-dependent step limits

3. **Convergence diagnostics**
   - Add warnings when lambda ratio seems unusual
   - Track and report optimization trajectory

4. **Extensive validation suite**
   - Test across many k values (5-50)
   - Multiple data patterns (linear, nonlinear, periodic)
   - Various noise levels

## Next Steps

1. Create a detailed comparison test for REML criterion values
2. Extract and compare penalty matrices
3. Review mgcv source code for scaling factors
4. Implement fixes based on findings
5. Re-run comprehensive validation

## Files Created

- `test_lambda_vs_k.py`: Systematic lambda comparison script
- `lambda_vs_k_comparison.png`: Visualization of lambda patterns
- `predictions_comparison.png`: Prediction quality across k values

## Conclusion

The investigation confirms that **lambda optimization is indeed overfit to small k values** (particularly k ≈ 5-10). The issue becomes severe for k > 15, with lambda ratios dropping below 0.001.

This is a critical bug that affects model quality, as incorrect lambda values lead to either:
- **Over-smoothing** (lambda too high): Loss of signal, bias
- **Under-smoothing** (lambda too low): Overfitting, high variance

Based on the RMSE comparison, mgcv_rust is currently **under-smoothing** (lambda too low) for higher k values.
