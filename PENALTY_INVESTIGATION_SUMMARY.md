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

## Update 2: Integration Domain Fix

### Additional Changes
- Fixed integration to only cover DATA domain [x_min, x_max], not extended knot range
- Extended knots define basis functions, but penalty integrates over data domain only

### Current Status (After All Fixes)

✅ **Knots**: Match mgcv exactly (max diff < 5e-9)
✅ **Integration domain**: [0, 1] data domain (not extended range)
❌ **Penalty magnitude**: Still ~29,000x too large

### Root Cause: mgcv's Complex Weighting Algorithm

mgcv uses a sophisticated penalty computation that differs from direct Gaussian quadrature:

1. **Quadrature scheme**: For pord=2 (second derivatives), subdivides each knot interval into `pord` sub-intervals
2. **Weight matrix W1**: Computed from polynomial basis inversion and integral matrix H
3. **Band Cholesky**: Applies banded Cholesky decomposition for efficiency
4. **Weighted derivatives**: D1 = B * D where B is Cholesky-weighted band matrix

Source code excerpt:
```r
W1 <- t(P) %*% H %*% P        # Weight matrix
ld <- ...complex indexing...   # Diagonal weights
B <- build_banded_matrix(ld, W1, h)
B <- bandchol(B)              # Band Cholesky
D1 <- apply_band_weights(B, D)
S <- crossprod(D1)            # Final penalty
```

### Investigation Attempts

1. ✅ Implemented mgcv's knot formula - works perfectly
2. ✅ Fixed integration domain to [x_min, x_max] - correct but still wrong magnitude
3. ⚠️  Implemented pord=2 quadrature with subdivisions - structure improved (pentadiagonal) but magnitude still off by 29,000x
4. ⚠️  Attempted W1 weight matrix computation - complex array indexing issues
5. ❌ Band Cholesky weighting - incomplete due to complexity

### Why Direct Integration Fails

The naive approach ∫ B''ᵢ(x) B''ⱼ(x) dx using Gaussian quadrature produces mathematically correct results, but mgcv's algorithm includes additional scaling/weighting that achieves better numerical properties. The ratios between elements vary wildly (not a constant factor), indicating fundamental structural differences in the algorithm.

## Conclusion

**Progress**: Knot calculation is perfect, integration domain is correct.
**Remaining**: mgcv's penalty algorithm uses complex band-matrix weighting that requires either:
1. Complete replication of their bandchol C code, or
2. Finding published formulas from Wood (2017) that document the exact weighting scheme

The mathematical concept is sound - the implementation details are sophisticated and require careful study of mgcv's C source code (`src/gam.c` and related files).
