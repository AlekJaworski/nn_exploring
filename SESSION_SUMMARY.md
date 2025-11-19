# Session Summary: REML Gradient Investigation

**Date**: 2025-11-19
**Branch**: `claude/fix-penalty-gradient-01LhXpn2urqsCEcHTVd7gqWp`
**Status**: Investigation complete, gradient catastrophe NOT resolved

---

## Problem Statement

Multidimensional GAMs take too many REML iterations compared to mgcv:
- **mgcv**: 4-10 iterations
- **Ours**: 20-30+ iterations
- **Impact**: 2-3x slower at large n (n≥1500)

Root cause: Catastrophic gradient values (~10^28) prevent fast convergence.

---

## What We Discovered

### 1. The Gradient Catastrophe (PRIMARY ISSUE)

In multidimensional GAMs, gradients explode for smooths 2, 3, 4:
```
Iteration 1:
  Smooth 0: gradient = 3.1          ✓ normal
  Smooth 1: gradient = 1.37e28      ✗ catastrophic!
  Smooth 2: gradient = 3.43e27      ✗ catastrophic!
  Smooth 3: gradient = 2.90e28      ✗ catastrophic!
```

**Why this matters**: Huge gradients cause the optimizer to take tiny steps or make wild oscillations, requiring many iterations to converge.

### 2. S.scale Formula is Wrong

We tried to implement mgcv's sp parameterization where λ = sp × S.scale, but:
- **Our calculated S.scale**: 0.013 - 0.015
- **mgcv's actual S.scale**: 70 - 173
- **Difference**: 5000x!

**Root cause**: mgcv's stored penalty matrices are already normalized. When we compute `maXX / ||S||_inf` on their matrices, we get 1.0, proving they're pre-scaled. We couldn't find mgcv's actual S.scale formula.

### 3. QR Decomposition Issues

The gradient computation uses QR decomposition of the augmented matrix Z:
```
Z = [√W X; √λ₀ L₀; √λ₁ L₁; ...]
```

When λ values differ greatly or approach zero:
- R matrix gets near-zero diagonal elements (3.88e-14)
- P = R⁻¹ has huge values (Frobenius norm: 1.6e14)
- trace(P'SP) explodes to ~10^29
- Gradient formula: (trace - rank + penalty/φ) / 2 → catastrophic

---

## What We Tried

### ✅ Implemented (didn't fix the issue)

1. **sp parameterization**
   - Changed optimization from log(λ) to log(sp)
   - Added s_scales field to SmoothingParameter
   - Compute λ = sp × S.scale during optimization
   - **Result**: Still catastrophic gradients

2. **Minimum lambda enforcement**
   - Added `.max(0.1)` to prevent λ → 0
   - **Result**: Gradients still ~10^28 even with λ=0.1

3. **Penalty block detection verification**
   - Confirmed penalties are correctly placed in blocks
   - Penalty 0: [0:8], Penalty 1: [8:16] ✓
   - **Result**: Not the source of the problem

4. **Removed penalty normalization**
   - Tried scale_factor = 1.0 (no normalization)
   - **Result**: Made gradients worse

### ❌ Could Not Implement

1. **mgcv's actual S.scale formula** - couldn't reverse-engineer it
2. **Different optimizer** - would require significant refactoring
3. **SVD instead of QR** - would require rewriting gradient computation

---

## Code Changes Made

### src/smooth.rs
```rust
pub struct SmoothingParameter {
    pub lambda: Vec<f64>,
    pub s_scales: Vec<f64>,  // NEW: S.scale factors
    pub method: OptimizationMethod,
}

// NEW: Optimize sp instead of lambda
let mut log_sp: Vec<f64> = self.lambda.iter()
    .zip(self.s_scales.iter())
    .map(|(l, s)| (l / s).ln())
    .collect();

// NEW: Enforce minimum lambda
let lambdas: Vec<f64> = sp_values.iter()
    .zip(self.s_scales.iter())
    .map(|(sp, s_scale)| (sp * s_scale).max(0.1))  // min λ=0.1
    .collect();
```

### src/gam.rs
```rust
// CHANGED: No penalty normalization (scale_factor = 1.0)
let scale_factor = 1.0;
s_scales.push(scale_factor);

// Place unnormalized penalty in appropriate block
penalty_full.slice_mut(s![col_offset..col_offset + num_basis, ...])
    .assign(&smooth.penalty);
```

### src/reml.rs
```rust
// Added debug output for block detection
eprintln!("[BLOCK_DETECT] Penalty {}: detected block_start={}, block_end={}, size={}",
         pen_idx, block_start, block_end, block_end - block_start);
```

---

## Current State

### What Works ✓
- 1D GAMs work perfectly (2-10x faster than mgcv)
- Multidimensional GAMs produce correct results (R² > 0.99)
- All 9 multidimensional unit tests pass
- Faster than mgcv at small n (n=500)

### What Doesn't Work ✗
- **Excessive iterations**: 20-30 vs mgcv's 4-10
- **Slower at large n**: 2-3x slower than mgcv for n≥1500
- **Catastrophic gradients**: ~10^28 for smooths 2+
- **S.scale formula**: Don't know mgcv's actual formula

### Performance Comparison

| n | Our Time | mgcv Time | Speedup | Our Iters (est) | mgcv Iters |
|---|----------|-----------|---------|-----------------|------------|
| 500 | 0.183s | 0.523s | **2.86x faster** | ~25 | 10 |
| 1500 | 0.539s | 0.333s | 0.62x slower | ~25 | 4 |
| 2500 | 1.159s | 0.618s | 0.53x slower | ~25 | 8 |
| 5000 | 2.089s | 0.897s | 0.43x slower | ~25 | 7 |

---

## Recommendations for Next Session

### Option 1: Switch to BFGS Optimizer (RECOMMENDED)

Replace the gradient-based Newton method with BFGS quasi-Newton:
- mgcv uses BFGS, not our gradient descent approach
- BFGS is more robust to numerical issues
- Available in Rust via `argmin` crate or similar

**Pros**: Likely to solve the iteration problem completely
**Cons**: Requires refactoring optimization code (~300 lines)

### Option 2: Extract mgcv's S.scale at Runtime

Call into mgcv from Python/R to get actual S.scale values:
```python
# In Python bindings
sm = mgcv.smoothCon(...)
s_scale = sm.S.scale  # Use mgcv's actual value
```

**Pros**: Would use correct S.scale formula
**Cons**: Requires mgcv dependency, doesn't fix gradient catastrophe

### Option 3: Use SVD Instead of QR

Replace QR decomposition with SVD for better numerical stability:
```rust
// Instead of Z = QR
let svd = Z.svd(true, true)?;
// Use SVD to compute gradients
```

**Pros**: More numerically stable
**Cons**: Slower, still might not fix catastrophic gradients

### Option 4: Accept Current Performance

- 20-30 iterations isn't terrible
- Focus on other optimizations (caching, better BLAS usage)
- Only an issue for large multidimensional GAMs

---

## Key Files to Review

### Documentation
- `GRADIENT_CATASTROPHE_INVESTIGATION.md` - Full technical analysis
- `SP_PARAMETERIZATION_FINDINGS.md` - S.scale investigation
- `REML_ITERATION_INVESTIGATION.md` - Original problem diagnosis

### Test Scripts
- `test_sp_2d.py` - Simple 2D test with profiling
- `benchmark_multidim_scaling.py` - Performance vs mgcv
- `check_mgcv_gradients.R` - mgcv gradient verification

### Core Implementation
- `src/smooth.rs:177-470` - REML optimization (modified)
- `src/gam.rs:251-324` - Penalty construction (modified)
- `src/reml.rs:403-681` - Gradient computation (original issue)

---

## How to Continue

### To test current state:
```bash
git checkout claude/fix-penalty-gradient-01LhXpn2urqsCEcHTVd7gqWp
source .venv/bin/activate
maturin develop --release --features python,blas
MGCV_PROFILE=1 python3 test_sp_2d.py
```

### To implement BFGS:
1. Add `argmin` or `optimization` crate to Cargo.toml
2. Replace `optimize_reml_newton_multi()` in src/smooth.rs
3. Define REML as minimization problem for BFGS
4. Test with profiling enabled

### To merge to master:
**DO NOT MERGE YET** - gradient catastrophe still present. Consider:
- Reverting to commit 362e8a3 (working baseline)
- Or implementing Option 1 (BFGS) first
- Or documenting current limitations

---

## Questions for Next Session

1. **Is 20-30 iterations acceptable?** If yes, we can stop here and optimize elsewhere.

2. **Should we match mgcv exactly?** If yes, implement BFGS or extract S.scale from mgcv.

3. **Is this a priority?** Could focus on other features instead (prediction intervals, different basis types, etc.).

4. **Can we tolerate mgcv dependency?** If yes, could call mgcv for S.scale values.

---

**Bottom Line**: We understand the problem (catastrophic gradients due to QR numerical issues) but couldn't fix it with parameterization changes alone. Need architectural change (BFGS) or acceptance of current iteration count.
