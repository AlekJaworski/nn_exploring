# Newton Optimization Results (n=5000, d=8)

## Summary

Successfully optimized Rust Newton REML implementation to match R's `gam()` performance at large scale (n=5000, d=8).

## Performance Comparison

| Method | Iterations | Time (ms) | λ (mean) | vs Rust Before | vs bam() |
|--------|-----------|-----------|----------|----------------|----------|
| **Rust Newton (Before)** | 9 | 1489 | 4.707 | baseline | 9.0x slower |
| **Rust Newton (After)** | 7 | 976-1086 | 4.630 | **1.4x faster** | 5.8x slower |
| R gam(REML) | 7 | 1098 | 4.630 | **1.1x faster** | 6.5x slower |
| R bam(REML) | 5 | 169 | 4.630 | **8.8x faster** | baseline |

## Optimization Details

### Problem Identified

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

## Current Status

✅ **Matches gam() performance** - Same iterations (7), comparable time (~1000ms)
✅ **Proper convergence** - All λ values match R output (λ ≈ 4.630)
✅ **No wasted iterations** - Early termination prevents zero-step waste

❌ **Still slower than bam()** - 5.8x slower (1000ms vs 169ms)

## Why bam() is Faster

1. **Fewer iterations:** 5 vs 7 (better line search heuristics?)
2. **QR-updating:** Memory-efficient incremental matrix updates
3. **Optimized for large n:** Block-wise computation strategies
4. **Mature BLAS:** Decades of optimization in R's linear algebra

## Future Work

To match bam() performance (~200ms target):

1. **Implement QR-updating** (most impactful)
   - Avoid recomputing full Hessian each iteration
   - Use incremental updates for matrix factorizations
   - Reference: Wood (2015) "Large additive models"

2. **Improve line search**
   - Better initial step size heuristics
   - Adaptive backtracking strategy
   - May reduce from 7 to 5 iterations

3. **Profile Hessian computation**
   - Currently O(p³) matrix inverse dominates
   - Consider specialized algorithms for block-diagonal structure
   - Explore parallel computation

## Commit History

- **Before:** Tagged as `stable-1d` (fast for d=1, slow for d>1 at large n)
- **After:** Current commit (matches gam(), ready for QR-updating)
