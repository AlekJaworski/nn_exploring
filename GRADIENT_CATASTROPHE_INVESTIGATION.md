# Gradient Catastrophe Investigation - Comprehensive Findings

## Problem
REML optimization takes 20-30 iterations vs mgcv's 4-10, caused by catastrophic gradient values (~10^28).

## Key Findings

### 1. S.scale Formula is Wrong ❌
- **Our S.scale**: 0.013 - 0.015
- **mgcv's S.scale**: 70 - 173  
- **Ratio**: 5000x difference!

**Root cause**: mgcv's stored penalty matrices are ALREADY normalized. When we apply `maXX / ||S||_inf` to them, we get 1.0, proving they're pre-scaled.

Our attempt to compute S.scale as `maXX / ||S||_inf` uses the wrong formula. mgcv's actual S.scale formula is unknown (not documented).

### 2. Penalty Block Detection is Correct ✓
- Penalty 0: block [0:8] ✓
- Penalty 1: block [8:16] ✓  

Block detection code works correctly.

### 3. Gradient Catastrophe Persists
Even after fixes:
- Smooth 0: gradient ≈ 10 (normal)
- Smooth 1: gradient ≈ 2.5×10^28 (catastrophic!)

**The catastrophe is specific to smooth 1+ in multidimensional GAMs.**

### 4. Enforcing Minimum Lambda Helps But Doesn't Fix
Added `lambda.max(0.1)` to prevent lambda→0, but gradients still catastrophic.

## Attempted Fixes

1. ✅ **sp parameterization**: Implemented (λ = sp × S.scale)
2. ✅ **Block detection**: Already working correctly  
3. ✅ **Minimum lambda**: Enforced λ ≥ 0.1
4. ❌ **Penalty normalization**: Removed (made gradients worse)
5. ❌ **S.scale formula**: Couldn't find mgcv's actual formula

## Root Cause Hypothesis

The gradient formula `∂REML/∂log(λ)` computed via QR decomposition may be:
1. **Incorrect for multi-smooth case** - cross-terms between smooths not handled
2. **Numerically unstable** - P = R^-1 has huge values even with λ=0.1
3. **Wrong parameterization** - gradient is for λ but we need it for sp

## Code Changes Made

### src/smooth.rs
- Added `s_scales` field to `SmoothingParameter`
- Changed optimization from `log_lambda` to `log_sp`
- Enforced minimum lambda: `(sp * s_scale).max(0.1)`
- Added extensive debug output

### src/gam.rs  
- Collect s_scales during penalty construction
- Pass s_scales to smoothing parameters
- Added penalty block verification debug output

### src/reml.rs
- Added block detection debug output
- Verified block extraction is correct

## Current Status

**Gradient catastrophe NOT resolved.**

Iterations: Still ~20-30 (target: <10)
Gradients: Still ~10^28 for smooth 1+ (target: <100)

## Recommendations

### Option 1: Use Different Optimizer
Replace QR-based REML with:
- **BFGS** quasi-Newton (like mgcv uses)
- **Nelder-Mead** simplex
- Direct minimization without gradients

### Option 2: Match mgcv Exactly
- Extract mgcv's actual S.scale values
- Use mgcv's penalty matrices directly
- Copy mgcv's gradient computation exactly

### Option 3: Fix Numerical Conditioning
- Use SVD instead of QR  
- Add regularization to Z matrix
- Implement better preconditioning

### Option 4: Simplify
- Remove penalty normalization entirely
- Use single global λ instead of per-smooth
- Optimize in λ space, not sp space

## Files Modified

- `src/smooth.rs`: sp parameterization, minimum lambda
- `src/gam.rs`: s_scales collection, unnormalized penalties  
- `src/reml.rs`: block detection debug
- `test_sp_2d.py`: Test script
- Multiple investigation documents

## Next Steps for User

1. Review findings
2. Choose approach (different optimizer vs exact mgcv matching)
3. Consider if this level of iteration count is acceptable
4. Decide whether to continue investigation

---

**Date**: 2025-11-19  
**Session**: claude/fix-penalty-gradient-01LhXpn2urqsCEcHTVd7gqWp  
**Commit**: Ready to push findings and partial fixes
