# BLAS Integration - Blocker Documentation

## Status: BLOCKED - API Compatibility Issues

### Problem Summary
ndarray-linalg trait methods are not being found at compile time, despite correct feature flags and dependencies.

### Root Cause
The ndarray-linalg crate has API incompatibilities across versions:

1. **Version 0.16**: Requires ndarray 0.15, but numpy 0.22 requires ndarray 0.16 → version conflict
2. **Version 0.18**: Supports ndarray 0.16, but trait methods (`solve`, `det`, `inv`) not found despite imports

### Attempted Solutions

#### Attempt 1: ndarray-linalg 0.16 + ndarray 0.15
- **Result**: Rust tests pass ✓
- **Blocker**: Python bindings fail - numpy 0.22 requires ndarray 0.16
- **Error**: Type mismatch between ndarray 0.15 (mgcv_rust) and 0.16 (numpy)

#### Attempt 2: ndarray-linalg 0.18 + ndarray 0.16
- **Result**: Version alignment correct ✓
- **Blocker**: Trait methods not found
- **Errors**:
  ```
  error[E0599]: no method named `solve` found for struct `ArrayBase<S, D>`
  error[E0599]: no method named `det` found for struct `ArrayBase<S, D>`
  error[E0599]: no method named `inv` found for struct `ArrayBase<S, D>`
  ```

#### Attempted Fixes
- ✗ Wildcard import: `use ndarray_linalg::*;`
- ✗ Explicit imports: `use ndarray_linalg::{Solve, Determinant, Inverse};`
- ✗ Different method names: `solve_into`, `det`, `inv`
- ✗ Clone before calling: `a.clone().solve(&b)`

### Current Configuration
```toml
# Cargo.toml
[dependencies]
ndarray = "0.16"
ndarray-linalg = { version = "0.16", optional = true, features = ["openblas-system"] }

[features]
blas = ["ndarray-linalg"]
```

### Code Structure (Ready, just needs working API)
```rust
#[cfg(feature = "blas")]
fn solve_blas(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>> {
    // This should work but trait methods aren't found
    a.solve(&b).map_err(|_| GAMError::SingularMatrix)
}

#[cfg(not(feature = "blas"))]
fn solve_gaussian(mut a: Array2<f64>, mut b: Array1<f64>) -> Result<Array1<f64>> {
    // Fallback implementation - currently used
    // O(n³) Gaussian elimination with partial pivoting
}
```

### Next Steps to Unblock

1. **Check ndarray-linalg 0.18 documentation** (1 hour)
   - Find correct trait names and method signatures
   - May have changed from `Solve::solve` to different pattern

2. **Try alternative BLAS bindings** (2 hours)
   - Consider `blas-src` + manual LAPACK calls
   - Or `nalgebra` which has better BLAS integration
   - Or wait for ndarray-linalg API to stabilize

3. **Minimal reproduction** (30 mins)
   - Create standalone test case
   - File issue on ndarray-linalg GitHub if API is unclear

4. **Profile without BLAS first** (recommended)
   - Establish baseline performance
   - Identify exact bottlenecks
   - May find other optimizations that are easier wins

### Performance Impact of Blocker

**Estimated speedup if BLAS working**: 3-5x on matrix operations (40-50% of runtime)
**Overall speedup estimate**: 2-3x faster end-to-end
**vs R's mgcv**: Would be 3-5x faster (R uses BLAS natively)

**Current performance** (without BLAS):
- Small samples (n=500): Already 1.5x faster than R ✓
- Other sizes: Tied with R

**Target with BLAS**:
- All sizes: 3-5x faster than R

### Recommended Action

**Don't block on this.** Move to Priority 3 (REML optimization) which can provide 10-20% gains without dependencies. Come back to BLAS integration when:
1. API documentation is clearer
2. ndarray-linalg stabilizes
3. Or we have time to try alternative BLAS wrappers

The current O(n^0.80) scaling is excellent even without BLAS. BLAS is an optimization, not a correctness issue.
