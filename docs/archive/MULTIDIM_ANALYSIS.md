# Multi-Dimensional GAM Performance Analysis

## Benchmark Results

### Configuration 1: n=6000, dimensions=8, k=10

**Rust:**
- Mean time: 1.1171s ± 0.1267s
- Lambdas: [0.0795, 0.0795, 0.0795, 0.0795, 0.0795, 0.0795, 0.0795, 0.0795]

**R:**
- Mean time: 1.2586s ± 0.2644s
- Lambdas: [5.75, 5.13, 4681.13, 2034.40, 3282.54, 3457.60, 3688.11, 3367.85]
- Iterations: 7.0
- Time per iteration: 179.8ms

**Comparison:**
- **Speedup: 1.13x** (Rust is 13% faster!) ✓
- Rust time: 1.1171s
- R time: 1.2586s

---

### Configuration 2: n=6000, dimensions=10, k=10

**Rust:**
- Mean time: 2.0988s ± 0.0373s
- Lambdas: [0.0366, 0.0478, 0.1481, 0.0372, 0.0323, 0.0734, 90.39, 0.0543, 0.0185, 0.0668]

**R:**
- Mean time: 1.6080s ± 0.0392s
- Lambdas: [5.46, 5.07, 5313.47, 1496.96, 3622.57, 3690.33, 2591.43, 3482.47, 3058.17, 3059.11]
- Iterations: 7.0
- Time per iteration: 229.7ms

**Comparison:**
- **Speedup: 0.77x** (Rust is 30% slower) ✗
- Rust time: 2.0988s
- R time: 1.6080s

---

## Key Observations

### 1. Performance Pattern

- **8 dimensions**: Rust is **1.13x FASTER** than R ✓
- **10 dimensions**: Rust is **0.77x** (30% slower) than R ✗

**Scaling Issue**: Performance degrades faster in Rust as dimensions increase.

### 2. Lambda Values - MAJOR DISCREPANCY ⚠️

**Critical Issue**: The lambda values are on completely different scales:

- **R's lambdas**: Range from ~5 to ~5000
- **Rust's lambdas**: Range from ~0.02 to ~0.15 (mostly), with one outlier at 90.39

This suggests:
1. **Different parameterization** - Rust might be using log(λ) while R uses λ
2. **Different penalty scaling** - The penalty matrices might be scaled differently
3. **Optimization issue** - The optimization might be converging to different local minima

**Evidence of the problem:**
- In 8D case: All Rust lambdas are nearly identical (~0.0795), while R's vary by 3 orders of magnitude
- This suggests Rust may not be properly optimizing each smooth independently

### 3. Iteration Analysis (Cannot compute without Rust iteration counts)

R uses 7 iterations for both cases:
- 8D: 179.8ms per iteration
- 10D: 229.7ms per iteration

We need to add iteration counting to Rust to determine if the bottleneck is:
- **Iteration count** (Rust needs more iterations to converge)
- **Per-iteration time** (Each Rust iteration is slower)

---

## Bottleneck Hypothesis

Based on the scaling behavior:

### Primary Issue: Lambda Scale/Parameterization

The dramatically different lambda values suggest a fundamental issue with how smoothing parameters are handled in multi-dimensional fits. This could affect:
- Convergence speed
- Numerical stability
- Final model quality

### Secondary Issue: Scaling with Dimensions

The fact that Rust degrades from 1.13x to 0.77x when going from 8D to 10D suggests:
- **Hypothesis 1**: Computational complexity grows faster than expected
  - Possible cause: Inefficient handling of multiple penalty matrices
  - Each dimension adds O(k²) to penalty computation
- **Hypothesis 2**: Memory/cache effects
  - Larger design matrices (6000 × 80 vs 6000 × 100) causing cache misses
- **Hypothesis 3**: Optimization inefficiency
  - More dimensions = more parameters to optimize jointly
  - Could be using dense Hessian instead of sparse structure

---

## Action Items

### Immediate Priority: Fix Lambda Scaling Issue

1. **Investigate parameterization**:
   - Check if Rust is working in log(λ) space while R uses λ
   - Compare penalty matrix norms between Rust and R
   - Verify that each smooth is optimized independently

2. **Add debugging output**:
   - Log the actual penalty matrices and their traces
   - Log the REML criterion at each iteration
   - Verify gradient computation

### Secondary: Add Iteration Tracking

Modify Rust code to return iteration count so we can compute per-iteration time:
```rust
result['iterations'] = optimization_iterations
```

### Tertiary: Profile Multi-Dimensional Scaling

Once lambda issue is fixed, profile to identify why 10D is slower:
- Time spent in penalty matrix operations
- Time spent in REML gradient computation
- Memory allocation patterns

---

## Expected Outcomes After Fixes

If we fix the lambda scaling/optimization issue:
- Lambda values should match R's scale and variability
- Performance should improve (correct optimization path is likely faster)
- Scaling to 10D should be more predictable

Target after fixes:
- 8D: Maintain 1.1-1.2x speedup ✓
- 10D: Achieve 0.9-1.1x (within 10% of R) ✓
