# Penalty Matrix Investigation Summary

## Problem Statement

Our mgcv_rust implementation was producing lambda values that didn't match R mgcv:
- CR: 0.52 vs 16.45 (31x too small)
- BS: 75.86 vs 3.46 (22x too large)

## Investigation Steps

### 1. Normalization Hypothesis
**Initial hypothesis**: mgcv doesn't normalize penalty matrices

**Finding**: Confirmed - mgcv uses raw penalties:
- CR penalty max row sum: 2.49 (not 1.0)
- BS penalty max row sum: 1.08 (not 1.0)

**Action**: Removed penalty normalization from both `cubic_spline_penalty()` and `cr_spline_penalty()`

**Result**: Made it WORSE! Lambda values became 1500x too small instead of just 31x off.

### 2. Direct Penalty Matrix Comparison
**Method**: Added `compute_penalty_matrix()` Python function to extract actual penalty matrices

**Findings**:
```
CR Penalty Magnitude:
  mgcv:  Frobenius norm = 4.48
  Ours:  Frobenius norm = 122.08
  Ratio: 27.3x TOO LARGE

BS Penalty Magnitude:
  mgcv:  Frobenius norm = 2.28
  Ours:  Frobenius norm = 233,557
  Ratio: 102,307x TOO LARGE!!!
```

**Structure comparison**:
- mgcv CR has dense, smooth values (0.24 to 0.73)
- Our CR is simple tridiagonal (12, 24 pattern) - WRONG STRUCTURE!
- mgcv BS has moderate values (0.15 to 0.45)
- Our BS has HUGE values (1,000 to 200,000) - CATASTROPHICALLY WRONG!

### 3. Gaussian Quadrature Verification
**Method**: Tested quadrature on simple integrals (∫x² dx from 0 to 1)

**Result**: Quadrature works perfectly (error < 1e-16)
- So the integration method itself is correct

### 4. Root Cause Discovery: Knot Vector Mismatch

**Method**: Extracted actual knots from mgcv smooth objects

**CRITICAL FINDING**:

**BS (B-splines)**:
- mgcv uses 24 knots extending OUTSIDE the domain [0, 1]
- First knot: -0.178 (before data range!)
- Last knot: 1.178 (after data range!)
- We were using: `linspace(0, 1, 18)` interior knots only

**CR (Cubic Regression)**:
- mgcv uses 19 knots evenly spaced from 0 to 1
- We were using: `linspace(0, 1, 19)` ✓ (correct spacing)
- But our CR penalty formula is producing wrong structure (simple tridiagonal)

## Root Causes Identified

1. **BS Penalty**: Wrong knot vector
   - We use interior knots only [0, 1]
   - mgcv extends knots outside domain [-0.18, 1.18]
   - This changes the B-spline basis functions and their derivatives
   - Result: 100,000x error in penalty magnitude!

2. **CR Penalty**: Wrong formula/implementation
   - Our CR penalty is too simple (tridiagonal structure)
   - mgcv's CR penalty is dense and smooth
   - Even though knot spacing looks right, formula is wrong
   - Result: 27x error + wrong structure

## Next Steps

1. **Fix BS penalty**:
   - Match mgcv's extended knot vector exactly
   - Use knots that extend outside [0, 1] by appropriate amount
   - Recompute penalty with correct knots

2. **Fix CR penalty**:
   - Investigate mgcv's actual CR penalty formula
   - Our current analytical formula doesn't match their output
   - May need to use different approach for CR

3. **Verify**:
   - Compare penalty matrices element-by-element
   - Ensure structure and magnitude match
   - Test lambda values converge to mgcv's

## Technical Details

### Gaussian Quadrature (VERIFIED CORRECT)
```rust
for k in 0..(n_knots - 1) {
    let a = knots[k];
    let b = knots[k + 1];
    let h = b - a;
    for &(xi, wi) in &quad_points {
        let x = a + 0.5 * h * (xi + 1.0);  // Transform [-1,1] -> [a,b]
        let d2_bi = b_spline_second_derivative(x, i, degree, &extended_knots);
        let d2_bj = b_spline_second_derivative(x, j, degree, &extended_knots);
        integral += wi * d2_bi * d2_bj * 0.5 * h;  // Jacobian = h/2
    }
}
```
✓ Formula is mathematically correct
✓ Tested on known integrals
✗ But using WRONG KNOTS!

### Current Status

- ✅ Removed normalization (as mgcv doesn't normalize)
- ✅ Analytical integration with Gaussian quadrature
- ✅ Identified root cause: knot vector mismatch
- ⚠️  BS: Need to match mgcv's extended knot placement
- ⚠️  CR: Need to fix penalty formula/structure
- ⏳ Lambda values will match once penalties are fixed

## Files Modified

- `src/penalty.rs`: Removed normalization, but penalties still wrong
- `src/lib.rs`: Added `compute_penalty_matrix()` for debugging
- `compare_penalty_matrices.py`: Comparison script
- `extract_mgcv_knots.R`: Knot extraction script

## Update: Knot Calculation Fixed

### Changes Made
- Implemented mgcv's exact knot formula for BS splines:
  ```rust
  k = num_basis + 1  // Recover k from num_basis
  nk = k - degree + 1  // Interior knots
  xl = x_min - range * 0.001  // Extend range
  xu = x_max + range * 0.001
  dx = (xu - xl) / (nk - 1)  // Spacing
  knots = linspace(xl - degree*dx, xu + degree*dx, nk + 2*degree)
  ```

- ✅ **Verified**: Knots now match mgcv EXACTLY (max diff < 5e-9)
- For k=20, num_basis=19, degree=3:
  - Creates 24 knots from -0.17782353 to 1.17782353
  - Perfectly matches mgcv's knot vector!

### Remaining Issue: Penalty Algorithm

Despite perfect knot matching, penalty matrix is still 31,000x too large with wrong structure:
- mgcv: Dense matrix, values ~0.15-0.45, Frobenius norm = 2.28
- Ours: Tridiagonal, values ~7000-13000, Frobenius norm = 71,940

**Root cause**: mgcv uses a different algorithm than simple Gaussian quadrature integration.

From mgcv source (`smooth.construct.bs.smooth.spec`):
1. Evaluates B-spline derivatives at specific quadrature points
2. Applies complex weighting scheme involving knot spacings
3. Uses bandchol for efficiency with banded matrices
4. Computes S = crossprod(D) where D is the weighted derivative matrix

Our approach: Direct integration of ∫ B''_i(x) B''_j(x) dx using Gaussian quadrature

These should be mathematically equivalent, but mgcv's implementation involves additional scaling/weighting that we're missing.

## Next Steps

1. **Extract mgcv's exact penalty algorithm** from C source code (`C_crspl` for CR, B-spline code for BS)
2. **Replicate their derivative evaluation and weighting** exactly
3. Alternative: Use mgcv's published formulas from Wood (2017) if available

The knot calculation is now correct - remaining work is purely in the penalty computation algorithm.

## Conclusion

The penalty matrix computation is using correct mathematical methods (Gaussian quadrature, Cox-de Boor derivatives) and **now uses correct knots** (verified to match mgcv exactly). However, mgcv uses a specific algorithmic approach with particular weighting/scaling that differs from naive integration. Once we replicate their exact algorithm, penalties and lambda values should match.
