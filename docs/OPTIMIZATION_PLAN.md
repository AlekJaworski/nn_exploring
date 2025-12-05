# Optimization Plan for Large n Cases

## Problem Analysis

Current performance for n=2000-7000:
- **mgcvrust**: 0.14-0.34s
- **R's mgcv**: 0.10-0.18s
- **Gap**: R is 1.5-2x faster

## Research Findings from mgcv

### Key Algorithms (Wood et al. 2015, 2017)

1. **Block-wise QR Decomposition** (`discrete=FALSE`)
   - Never forms full model matrix
   - Updates R factor block-by-block
   - Maintains Q'y incrementally
   - Memory: O(p²) instead of O(np)

2. **Discretized Covariates** (`discrete=TRUE`)
   - Discretizes covariate values
   - Pre-computes basis at discrete points
   - Efficient crossproducts via table lookups
   - C-level parallelization

3. **REML Optimization** (Wood 2011)
   - Derivatives without computing score
   - One parallel Cholesky per iteration
   - Efficient caching of decompositions

## Bottlenecks in Current mgcvrust

### Identified Issues

1. **Full Matrix Formation** (reml.rs:63-69)
   ```rust
   let mut x_weighted = x.to_owned();  // O(np) allocation
   for (i, mut row) in x_weighted.rows_mut()...
   ```
   - Allocates n×p matrix
   - For n=5000, p=20: 800KB

2. **Repeated X'WX Computation** (reml.rs:77-78)
   ```rust
   let xtw = x_weighted.t().to_owned();
   let xtwx = xtw.dot(&x_weighted);  // O(np²)
   ```
   - Done every REML iteration
   - No caching across outer loop

3. **Dense Operations**
   - No exploitation of sparsity
   - No BLAS level-3 (GEMM) explicit use
   - ndarray uses BLAS internally but not optimally

4. **No Parallelization**
   - Single-threaded
   - Could parallelize:
     - Matrix crossproducts
     - Multiple smooth evaluations
     - Cholesky decompositions

## Optimization Strategy

### Phase 1: Memory & Computation Efficiency (Target: 1.5x speedup)

**A. Efficient X'WX Without Full Allocation**
```rust
// Instead of:
let x_weighted = x * sqrt(W)
let xtwx = x_weighted.t() * x_weighted

// Do:
fn compute_xtwx_direct(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    // Compute X'WX directly using BLAS SYRK or manual loop
    // Exploits symmetry: only compute upper triangle
}
```

**B. Cache Basis Evaluations**
- Evaluate basis functions once
- Store in gam_optimized.rs::FitCache
- Reuse across REML iterations

**C. Use BLAS GEMM Explicitly**
```rust
#[cfg(feature = "blas")]
use ndarray_linalg::*;

// Use BLAS level-3 for large matrices
```

### Phase 2: QR Caching (Target: 1.3x speedup)

**A. Cache QR Factorization**
```rust
struct REMLCache {
    qr_cached: Option<(Array2<f64>, Array2<f64>)>,
    last_weights: Array1<f64>,
    valid: bool,
}
```

**B. Incremental Updates**
- If weights change < 1%, reuse QR
- Only recompute when necessary

### Phase 3: Algorithmic Improvements (Target: 1.2x speedup)

**A. Better Iteration Strategy**
- Track convergence more carefully
- Adaptive tolerance
- Early stopping when gradient tiny

**B. Smart Initialization**
- Better lambda_0 from data characteristics
- Warm start from previous fits

### Phase 4: Advanced (if needed)

**A. Covariate Discretization** (for n > 10000)
- Discretize X into bins
- Pre-compute basis at bin centers
- Table-based crossproducts

**B. Parallelization**
- Use rayon for parallel iteration
- Multi-threaded BLAS calls

## Implementation Plan

### Step 1: Optimize X'WX (High Impact, Low Risk)

File: `src/reml.rs`

```rust
/// Compute X'WX efficiently without forming X*sqrt(W)
#[cfg(feature = "blas")]
fn xtwx_blas(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    use ndarray_linalg::Lapack;
    // Use BLAS SYRK for symmetric rank-k update
}

fn xtwx_direct(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let (_n, p) = x.dim();
    let mut result = Array2::zeros((p, p));

    // Compute only upper triangle (symmetric)
    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for row in 0..x.nrows() {
                sum += x[[row, i]] * w[row] * x[[row, j]];
            }
            result[[i, j]] = sum;
            if i != j {
                result[[j, i]] = sum;  // Fill lower triangle
            }
        }
    }
    result
}
```

### Step 2: Add Iteration Counting & Profiling

File: `src/smooth.rs`

Add counter and timing:
```rust
pub struct OptimizationStats {
    pub outer_iterations: usize,
    pub total_time_ms: f64,
    pub converged: bool,
}
```

### Step 3: Benchmark & Validate

- Run benchmarks n=1000-7000
- Compare with baseline
- Verify numerical accuracy

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| n=2000, k=20 | 0.149s | <0.100s |
| n=5000, k=20 | 0.343s | <0.180s |
| Memory (n=5000) | ~800KB | <400KB |
| Iterations | ? | Match mgcv (5-6) |

## References

- Wood, S.N., et al. (2015). "Generalized additive models for large datasets." JRSS-C 64(1): 139-155.
- Wood, S.N. (2011). "Fast stable restricted maximum likelihood." JRSS-B 73(1): 3-36.
- Wood, S.N., et al. (2017). "Generalized additive models for gigadata." JASA.
- Li, Z & Wood, S.N. (2020). "Faster model matrix crossproducts." Statistics and Computing.
