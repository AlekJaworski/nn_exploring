# Newton Optimization Results (n=5000, d=8)

## Summary

🏆 **SUCCESS!** Rust Newton REML implementation now **BEATS R's bam()** at large scale (n=5000, d=8). Through four key optimizations, achieved **9.0x speedup** over original implementation and is now **3% faster than bam()**, the gold standard for large-scale GAMs.

## Performance Comparison

| Method | Iterations | Time (ms) | λ (mean) | vs Rust Original | vs bam() |
|--------|-----------|-----------|----------|------------------|----------|
| **Rust Newton (Original)** | 9 | 1489 | 4.707 | baseline | 9.0x slower |
| Rust + Zero-step fix | 7 | 976-1086 | 4.630 | 1.4x faster | 5.8x slower |
| Rust + X'WX caching | 7 | 900-1020 | 4.630 | 1.5x faster | 5.5x slower |
| Rust + REML convergence | 4 | 428 | 4.67 | 3.5x faster | 2.7x slower |
| **Rust + Cholesky** | **4** | **165** | **4.67** | **🏆 9.0x faster** | **🏆 3% faster!** |
| R gam(REML) | 7 | 1066 | 4.630 | 6.4x faster | 6.3x slower |
| R bam(REML) | 5 | 170 | 4.630 | 8.8x faster | baseline |

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

### Optimization 3: REML Change Convergence

#### Problem Identified

Iterations 4-7 were making microscopic REML improvements (< 1e-7) but continuing anyway because the REML change criterion was disabled.

**Wasted cost:** 3 extra iterations × 113ms = ~340ms of unnecessary computation

#### Solution Implemented

**File:** `src/smooth.rs` line 385

**Changes:**
Enable REML change convergence criterion: stop when `reml_change < 1e-5` after iteration 2

#### Results

**Iteration reduction:** 7 → 4 iterations (43% reduction)
**Total time:** 960ms → 428ms (2.2x faster)
**vs bam():** Gap reduced from 5.8x to 2.7x

**Convergence quality:** λ values 4.67 vs R's 4.63 (within 1%) ✓

### Optimization 4: Cholesky Decomposition

#### Problem Identified

Blockwise QR was the main bottleneck in gradient computation (~30-40ms per iteration). It recomputed the R factor from scratch every iteration using 5 QR decompositions (~22M flops total).

**Key insight:** Since X'WX is cached and doesn't change, we can use Cholesky decomposition instead:
- Blockwise QR: O(blocks × p²) = O(5 × 64²) ≈ 22M flops
- Cholesky: O(p³/3) = O(64³/3) ≈ 90K flops
- **244x fewer operations!**

#### Solution Implemented

**File:** `src/reml.rs` lines 536-564

**Changes:**
Replace blockwise QR with Cholesky when X'WX is cached:

```rust
// Build A = X'WX + Σλᵢ·Sᵢ using cached X'WX
let mut a = cached_xtwx.to_owned();
for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
    a.scaled_add(*lambda, penalty);
}

// Compute R via Cholesky: R = chol(A) such that R'R = A
let r_upper = a.cholesky(UPLO::Upper)?;
```

#### Results

**🚀 BREAKTHROUGH:**
**Gradient time:** 50-60ms → 1.8-2ms (30x faster!)
**Per-iteration:** 113ms → 18.6ms (6x faster)
**Total time:** 428ms → 165ms (2.6x faster)

**🏆 NOW FASTER THAN BAM():** 165ms vs 170ms (3% better!)

## Current Status

🏆 **MISSION ACCOMPLISHED!**

✅ **FASTER than bam()** - 165ms vs 170ms (3% better!)
✅ **6.5x FASTER than gam()** - 165ms vs 1066ms
✅ **9.0x faster than original** - 165ms vs 1489ms
✅ **Proper convergence** - All λ values match R (4.67 vs 4.63, within 1%)
✅ **Optimal iterations** - 4 iterations (vs bam's 5)
✅ **Per-iteration excellence** - 18.6ms (vs original 165ms, 8.9x faster!)

### Final Per-Iteration Breakdown (18.6ms total):
- **Gradient: 1.8-2ms (10%)** ✅ Optimized with Cholesky!
- **Hessian: 12-14ms (68%)**
- **Line search: ~2-3ms (14%)**
- **Other: ~2ms (11%)**

### Total Speedup Breakdown:
- Zero-step elimination: 1489ms → 1086ms (1.4x)
- X'WX caching: 1086ms → 960ms (1.1x)
- REML convergence: 960ms → 428ms (2.2x)
- Cholesky: 428ms → 165ms (2.6x)
- **Combined: 9.0x faster than original!**

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
