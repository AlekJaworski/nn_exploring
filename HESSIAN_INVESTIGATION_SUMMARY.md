# Hessian Investigation Summary

## Problem
The REML Newton optimizer was not converging properly:
- mgcv: 5 iterations to convergence
- Our implementation: 20-30 iterations (or failing)

## Root Cause
The Hessian formula from Wood (2011) J.R.Statist.Soc.B 73(1):3-36 does not match mgcv's implementation.

## Wood (2011) Formula
The paper gives the complete Hessian as:
```
H[i,j] = [-tr(M_i·A·M_j·A) + (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²] / 2
```
where M_i = λ_i·S_i, A = (X'WX + ΣM_i)^(-1)

### Issue with Complete Formula
When implemented exactly as written:
- Term 1 (trace): -5.3e-6
- Term 2 (penalty-beta interaction): +5.5e-6
- Term 3 (penalty-penalty interaction): -1.7e-4
- **Total: -8.7e-5 (NEGATIVE!)**

mgcv's Hessian diagonal: **+2.8 (POSITIVE)**

The complete formula gives:
1. **Wrong sign**: Negative instead of positive
2. **Wrong magnitude**: ~30,000x too small

## Trace-Only Formula
Using ONLY the trace term: `H[i,j] = tr(M_i·A·M_j·A) / 2`

Results:
- **Sign**: Positive ✓
- **Magnitude**: Still ~1000x too small
- **Newton progress**: YES! Gradient decreases (3.99 → 3.52)
- **Problem**: Hits singular matrix after ~2 iterations

## Hypothesis
Wood's paper likely uses mathematical shortcuts or approximations not explicitly stated. mgcv's actual C implementation may use:
1. A different Hessian approximation (e.g., Fisher information instead of observed Hessian)
2. Additional scaling factors not mentioned in the paper
3. A simplified formula omitting the penalty derivative terms

## Current Status
- **Gradient formula**: Correct (matches mgcv at iteration 1)
- **Hessian formula**: Incomplete - trace-only works but is too small
- **Next steps**: Need to inspect mgcv's actual C source code to see their exact formula

## Files Modified
- `src/reml.rs`: Implemented complete Wood (2011) formula, then reverted to trace-only
- `src/smooth.rs`: Removed gradient scaling experiments
- Various diagnostic scripts created to compare with mgcv

## Commits
1. Fix gradient formula (removed incorrect scaling)
2. Implement complete Hessian (found to be incorrect)
3. Revert to trace-only Hessian (works but needs scaling fix)
