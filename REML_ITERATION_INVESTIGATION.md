# REML Iteration Investigation - Root Cause Found

## Problem Statement

Multidimensional GAMs take too many REML iterations compared to mgcv:
- mgcv: 4-10 iterations
- Ours: 20-30+ iterations
- Result: 2-3x slower at large n

## Root Cause: Catastrophic Gradient Values

### Evidence

**4D GAM (n=500, k=[16,16,16,16])** - Iteration 1 gradients:
```
Smooth 0: gradient = 3.1          (normal)
Smooth 1: gradient = 1.37e28      (catastrophic!)
Smooth 2: gradient = 3.43e27      (catastrophic!)
Smooth 3: gradient = 2.90e28      (catastrophic!)
```

**mgcv's gradients at solution**: ~10^-6 to 10^-4 (normal)

### Underlying Issue: P Matrix Catastrophe

The P = R^{-1} matrix has huge values:
```
r_upper diagonal range: [3.88e-14, 32.7]  <- Near-zero diagonal!
p_matrix Frobenius norm: 1.62e14         <- Catastrophic!
p_matrix diagonal: [..., -2.58e13]
```

This causes:
```
trace_unscaled for smooth 1: 2.74e29
trace_unscaled for smooth 0: 77.0 (normal)
```

The gradient formula is:
```rust
gradient[i] = (trace - rank + penalty_term/phi) / 2.0
where trace = trace_unscaled * lambda
```

With trace_unscaled ~10^29, the gradient explodes!

### Why This Happens

**Hypothesis 1: Parameterization Mismatch** (user's insight)
- mgcv optimizes `sp` (smoothing parameter)
- mgcv computes λ = sp × S.scale
- We optimize λ directly
- Gradient formula may be for ∂REML/∂log(sp), not ∂REML/∂log(λ)

**Hypothesis 2: Penalty Scaling Issue**
- Different smooths have vastly different penalty scales
- scale_factor varies between smooths
- This might not be accounted for properly in gradient computation

**Hypothesis 3: Numerical Conditioning**
- R matrix has near-zero diagonals (3.88e-14)
- QR decomposition of Z matrix is ill-conditioned
- Need better numerical conditioning or regularization

## Proposed Fixes

### Fix 1: Switch to sp Parameterization (RECOMMENDED)

Match mgcv's approach:
1. Optimize sp (unscaled smoothing parameter)
2. Store S.scale for each penalty
3. Compute λ = sp × S.scale when needed
4. Gradient becomes ∂REML/∂log(sp)

**Advantages**:
- Matches mgcv exactly
- Better numerical conditioning (sp values ~1-1000, not tiny)
- Gradient scales should be reasonable

**Implementation**:
```rust
// In smooth.rs optimization
let mut log_sp = vec![0.0; m];  // Instead of log_lambda
let s_scales: Vec<f64> = penalties.iter()
    .map(|p| compute_s_scale(p))  // Extract S.scale
    .collect();

// In gradient computation
let sp_i = log_sp[i].exp();
let lambda_i = sp_i * s_scales[i];
// Use lambda_i in REML, but gradient is w.r.t. sp_i
```

### Fix 2: Better Numerical Conditioning

Add regularization to QR decomposition:
1. Add diagonal loading to Z matrix
2. Use SVD instead of QR for better stability
3. Threshold tiny singular values

### Fix 3: Penalty Block Detection Fix

Debug output showed both penalties with `block_start=0` - investigate if penalty matrices are being constructed correctly for multidimensional cases.

## Testing Protocol

After implementing fix:
1. Run with MGCV_PROFILE=1 to count iterations
2. Compare iteration count with mgcv (target: <15 iterations)
3. Check gradient magnitudes (target: <100)
4. Run multidimensional unit tests (must all pass)
5. Benchmark performance vs mgcv

## Expected Impact

If Fix 1 works:
- Iterations: 20-30 → 5-10 (3-6x reduction)
- Time at n=2500: 1.2s → 0.4s (3x faster)
- Match or beat mgcv performance

## Next Steps

1. Implement Fix 1 (sp parameterization)
2. Test on 2D case first
3. Verify gradients are reasonable (<100)
4. Test on 4D case
5. Run full benchmark suite
6. If Fix 1 doesn't work, try Fix 2

## Files to Modify

- `src/smooth.rs`: Change optimization from log_lambda to log_sp
- `src/gam.rs`: Store S.scale alongside penalties
- `src/reml.rs`: Adjust gradient computation if needed

## Current Status

- **Problem**: Fully diagnosed
- **Root cause**: Identified (catastrophic gradients due to parameterization/conditioning)
- **Fix strategy**: Defined (sp parameterization)
- **Ready to implement**: Yes

---

**Date**: 2025-11-19
**Baseline**: 362e8a3 (working multidimensional GAMs)
**Target**: Reduce REML iterations from ~25 to ~8 to match mgcv
