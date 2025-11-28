# Newton Optimization Results (n=5000, d=8)

## Summary

Successfully optimized Rust Newton REML implementation to **beat R's gam()** at large scale (n=5000, d=8). Now 13% faster than gam() through two key optimizations: eliminating zero-step iterations and caching X'WX.

## Performance Comparison

| Method | Iterations | Time (ms) | λ (mean) | vs Rust Original | vs bam() |
|--------|-----------|-----------|----------|------------------|----------|
| **Rust Newton (Original)** | 9 | 1489 | 4.707 | baseline | 9.0x slower |
| **Rust + Zero-step fix** | 7 | 976-1086 | 4.630 | **1.4x faster** | 5.8x slower |
| **Rust + X'WX caching** | 7 | **900-1020** | 4.630 | **1.5x faster** | 5.5x slower |
| R gam(REML) | 7 | 1084 | 4.630 | 1.4x faster | 6.4x slower |
| R bam(REML) | 5 | 165 | 4.630 | **9.0x faster** | baseline |

## Optimization Details

### Optimization 1: Eliminate Zero-Step Iterations

#### Problem Identified

The profiler revealed that iterations 7-9 were taking steps with scale < 1e-9 (effectively zero):

```
Iteration 7: best_step_scale = 0.0000000009
Iteration 8: best_step_scale = 0.0000000019
Iteration 9: gradient = 0.036, converged
```

These "zero steps" wasted ~500ms (3 iterations × ~175ms) without making meaningful progress.

### Solution Implemented

**File:** `src/smooth.rs` lines 440-464

**Changes:**
1. Added minimum step size threshold: `MIN_STEP_SIZE = 1e-6`
2. Reject steps smaller than threshold (effectively zero)
3. When step rejected, check if gradient already small enough (< 0.1)
4. Terminate early if gradient satisfactory rather than trying steepest descent

**Code:**
```rust
const MIN_STEP_SIZE: f64 = 1e-6;

if best_step_scale > MIN_STEP_SIZE {
    // Accept step
    for i in 0..m {
        log_lambda[i] += step[i] * best_step_scale;
    }
} else {
    // Step too small - check if gradient is already acceptable
    if grad_norm_linf < 0.1 {
        // Converge early - no point trying steepest descent
        return Ok(());
    }
    // Otherwise try steepest descent as fallback
}
```

### Results

**Iteration reduction:** 9 → 7 iterations (22% reduction)
**Time improvement:** 1489ms → 976-1086ms (27-35% faster)
**Per-iteration:** 165ms → 139-155ms (6-16% faster per iteration)

**Convergence behavior:**
- Iteration 7 hit step scale = 9.313e-10 (rejected)
- Gradient = 0.088294 < 0.1 threshold
- Early termination triggered
- Saved ~300-500ms by skipping iterations 8-9

### Optimization 2: Cache X'WX and X'Wy

#### Problem Identified

Detailed per-iteration profiling revealed gradient computation was the bottleneck (60-68ms, 45% of time). Investigation showed X'WX and X'Wy were being recomputed every iteration despite X, W, y being constant during optimization (only λ changes).

**Wasted cost:** O(np²) = O(5000 × 64²) = 20M flops × 7 iterations = 140M operations

#### Solution Implemented

**Files:** `src/smooth.rs` lines 232-257, `src/reml.rs` functions modified to accept cached values

**Changes:**
1. Pre-compute X'WX and X'Wy once before optimization loop
2. Add optional `cached_xtwx` and `cached_xtwy` parameters to gradient functions
3. Use references to avoid cloning overhead

**Code (smooth.rs):**
```rust
// OPTIMIZATION: Pre-compute X'WX and X'Wy (don't change during optimization)
let xtwx = compute_xtwx(x, w);
let xtwy = x_weighted.t().dot(&y_weighted);

// Pass to gradient function
let gradient = reml_gradient_multi_qr_adaptive_cached(
    y, x, w, &lambdas, penalties,
    Some(&sqrt_penalties),
    Some(&xtwx),  // ← cached
    Some(&xtwy),  // ← cached
)?;
```

#### Results

**Per-iteration improvement:** 139ms → 113ms (19% faster)
**Gradient time:** 60-68ms → 45-60ms (10-20ms saved)
**Total time:** 1086ms → 960ms (12% faster)

**Now faster than gam():** 960ms vs 1084ms (13% improvement)

## Current Status

✅ **FASTER than gam()** - 960ms vs 1084ms (13% better!)
✅ **Per-iteration optimized** - 113ms vs original 165ms (32% faster)
✅ **Proper convergence** - All λ values match R output (λ ≈ 4.630)
✅ **No wasted iterations** - Early termination prevents zero-step waste
✅ **Gradient optimized** - 45-60ms vs original 60-70ms (caching works!)

❌ **Still 5.5x slower than bam()** - 960ms vs 165ms

### Detailed Per-Iteration Breakdown (113ms total):
- **Gradient: 45-60ms (40-53%)** ← main remaining bottleneck
  - Blockwise QR: ~30-40ms (depends on λ, can't cache easily)
  - Triangular solves: ~10-15ms
- **Hessian: 11-13ms (10%)**
- **Line search: 16-32ms (15-28%)**
- **Other: ~20-30ms (18-26%)**

## Why bam() is Faster

1. **Fewer iterations:** 5 vs 7 (better line search heuristics?)
2. **QR-updating:** Memory-efficient incremental matrix updates
3. **Optimized for large n:** Block-wise computation strategies
4. **Mature BLAS:** Decades of optimization in R's linear algebra

## Future Work

To match bam() performance (~200ms target), need to close 5.5x gap (960ms → 165ms):

### High Priority (Most Impact):

1. **Implement QR-updating for gradient computation** (~30-40ms savings)
   - Current bottleneck: Blockwise QR recomputed each iteration
   - Solution: Incremental R factor updates when λ changes
   - Challenge: R depends on λ, need efficient update formula
   - Reference: Wood (2015) "Large additive models" Section 3.1

2. **Reduce iterations from 7 to 5** (~226ms savings if 2 iterations × 113ms)
   - Better line search heuristics
   - Adaptive step size based on gradient magnitude
   - More aggressive initial steps when far from optimum

### Medium Priority:

3. **Optimize blockwise QR computation**
   - Current: Processes 5 blocks of 1000 rows
   - Could use GPU/SIMD for block processing
   - Investigate faster BLAS routines

4. **Parallelize line search REML evaluations**
   - Currently sequential: try scale 1.0, then 0.5, then 0.25, etc.
   - Could evaluate multiple scales in parallel
   - Potential savings: ~10-15ms per iteration

### Low Priority:

5. **Optimize Hessian computation** (only ~12ms, limited impact)
6. **Better initial lambda estimates** (reduce total iterations)

## Commit History

- **Tag `stable-1d`:** Fast for d=1, baseline for multi-D
- **Commit 91e4aa0:** Zero-step optimization (9→7 iterations, 1489ms→1086ms)
- **Tag `optimized-n5000`:** Matches gam() performance
- **Commit b6a5638:** X'WX caching (1086ms→960ms, now faster than gam())
