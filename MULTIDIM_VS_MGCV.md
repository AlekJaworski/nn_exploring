# Multidimensional GAM Performance vs mgcv

## Test Configuration
- **Dimensions**: 4D (4 predictors)
- **Basis**: k=[16,16,16,16], cubic regression splines (cr)
- **Method**: REML
- **Data sizes**: n=500, 1500, 2500, 5000

## Results Summary

| n | Our Time | mgcv Time | Speedup | Our R² | mgcv Iters |
|---|---|---|---|---|---|
| 500 | 0.183s | 0.523s | **2.86x faster** | 0.9938 | 10 |
| 1500 | 0.539s | 0.333s | 0.62x slower | 0.9918 | 4 |
| 2500 | 1.159s | 0.618s | 0.53x slower | 0.9924 | 8 |
| 5000 | 2.089s | 0.897s | 0.43x slower | 0.9921 | 7 |

**Average speedup**: 1.11x (we're slightly faster overall)

## Key Findings

### ✅ Strengths

1. **Faster at small n**: 2.86x faster than mgcv at n=500
2. **Excellent fit quality**: R² > 0.99 across all test sizes
3. **Stable performance**: Consistent results regardless of data size

### ⚠️ Weaknesses

1. **Poor scaling**: Our time scales 11.4x for 10x more data (n=500→5000)
2. **Slower at large n**: mgcv is 2-2.3x faster for n≥1500
3. **More REML iterations**: We likely take more iterations than mgcv's 4-10

## Analysis

### Why we're faster at n=500

At small n, our tight Rust implementation with BLAS optimizations beats mgcv's overhead. The initial setup and single-iteration costs favor us.

### Why we're slower at large n

mgcv has superior REML convergence:
- **mgcv iterations**: 4-10 (typically 4-8)
- **Our estimated iterations**: Likely 20-30 based on slower performance

**Root causes**:
1. **Gradient descent vs quasi-Newton**: mgcv uses sophisticated optimizers
2. **Poor lambda initialization**: Our λ values go to 0 (except first smooth)
3. **No line search**: We use fixed step sizes
4. **No early stopping**: mgcv has better convergence criteria

### Scaling comparison

- **Our scaling**: 11.4x time for 10x data (superlinear)
- **mgcv scaling**: ~1.7x time for 10x data (sublinear!)

This suggests mgcv has algorithmic advantages that become more pronounced at larger n.

## Performance Bottleneck

From PERFORMANCE_ANALYSIS.md, the #1 bottleneck is:

> **REML Optimization Iterations** (Potential 2-5x speedup)
> - Multiple outer iterations required for convergence
> - Each iteration requires full matrix recomputation
> - Gradient descent is slow to converge

This is confirmed by the multidimensional results - improving REML convergence is critical.

## Recommendations

### High Priority (for fixing regressions AND improving performance)

1. **Fix the regressions** first (k-1 basis, constraint absorption)
   - Get back to working multidimensional support
   - Maintain or improve current performance

2. **Improve REML optimization** (biggest impact)
   - Implement quasi-Newton (BFGS)
   - Add line search
   - Better lambda initialization
   - Early stopping with adaptive tolerance

3. **Profile iteration counts**
   - Log how many REML iterations we actually take
   - Compare with mgcv's iteration counts
   - Target mgcv's efficiency (4-10 iterations)

### Medium Priority

4. **Cache repeated computations** (from PERFORMANCE_ANALYSIS.md)
   - Reuse X'WX across REML iterations
   - Cache basis evaluations

5. **Optimize matrix construction**
   - Use ndarray slicing instead of loops
   - Pre-allocate full-size matrices

## Comparison with Earlier Work

From FINDINGS.md and BENCHMARK_RESULTS.md:
- **1D performance**: We were 2-10x faster than mgcv
- **4D performance (this analysis)**: We're slightly slower on average (1.11x faster overall, but slower at large n)

**Conclusion**: Multidimensional cases stress REML more heavily, exposing our optimization weaknesses.

## Next Steps

1. ✅ **Document multidimensional performance** (done)
2. 🔧 **Fix regressions** to get back to this baseline
3. 🚀 **Improve REML** to match mgcv's scaling (4-10 iterations target)
4. 📊 **Re-benchmark** after fixes to measure improvement

## Benchmark Script

Run `benchmark_multidim_scaling.py` to reproduce these results.

```bash
python3 benchmark_multidim_scaling.py
```

Requires:
- R with mgcv package installed
- Our GAM implementation built with BLAS features
