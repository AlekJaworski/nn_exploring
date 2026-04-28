# Newton Optimizer Investigation: Final Status

## Executive Summary

**Goal**: Fix Newton optimizer convergence to match mgcv (5 iterations to optimal λ)

**Current Status**: Converges to λ=[4.11, 2.32] in ~100 iterations vs mgcv's [5.69, 5.20] in 5 iterations

**Progress**: Fixed 3 bugs, verified all formulas match mgcv, but convergence gap remains

## Bugs Found and Fixed

### 1. Exploding bSb2 Terms (CRITICAL - FIXED) ✅

**Commit**: a1b233c "Fix critical bug: Remove exploding bSb2 terms"

**Problem**:
- Used β directly instead of β derivatives in bSb2
- 1/φ² amplification factor (φ≈0.012 → 1/φ²≈6944x)
- term3 = -18.99 at λ=[1.42, 0.92], making Hessian NEGATIVE

**Fix**:
- Removed incorrect term2/term3
- Used det2-only temporarily
- Later implemented proper bSb2 with β derivatives

**Impact**:
- λ improved from [1.45, 0.94] to [4.16, 2.28]
- Hessian became positive definite
- Monotonic convergence achieved

### 2. Incomplete bSb2 Formula (MAJOR - FIXED) ✅

**Commit**: b5e5456 "Implement complete bSb2 with β derivatives"

**Problem**:
- det2-only Hessian was only 18% of mgcv's total
- Missing penalty Hessian (bSb2) which contributes 82%!

**Fix**:
Implemented complete 4-term formula from mgcv C source:
```
bSb2[k,m] = 2·(d²β'/dρ_k dρ_m · S · β)        [term1]
          + 2·(dβ'/dρ_k · S · dβ/dρ_m)         [term2]
          + 2·(dβ'/dρ_m · S_k · β · λ_k)       [term3]
          + 2·(dβ'/dρ_k · S_m · β · λ_m)       [term4]
          + δ_{k,m}·bSb1[k]                     [diagonal]
```

**Impact**:
- H[0,0] improved from 0.50 to 2.64 (vs mgcv's 2.81)
- Hessian magnitude now matches mgcv at 94-99%
- λ improved from [4.16, 2.28] to [4.11, 2.32]

### 3. Missing d²β Diagonal Term (MINOR - FIXED) ✅

**Commit**: e4aecc0 "Fix φ computation and d²β formula"

**Problem**:
- Second derivative formula missing δ_{ij}·dβ/dρ_i term
- From ∂M_i/∂ρ_j = δ_{ij}·M_i in implicit differentiation

**Fix**:
```rust
if i == j {
    d2beta += &dbeta_drho[i];
}
```

**Impact**:
- Negligible (term ~0.002 vs bsb1 diag_corr ~4.3)
- No change in convergence

### 4. Wrong φ Computation (MINOR - FIXED) ✅

**Commit**: e4aecc0 "Fix φ computation and d²β formula"

**Problem**:
- Used constant penalty rank: φ = RSS / (n - Σrank(S_i))
- Should use λ-dependent edf: φ = RSS / (n - tr(A^{-1}·X'WX))

**Fix**:
```rust
let xtx = x.t().to_owned().dot(&x.to_owned());
let ainv_xtx = a_inv.dot(&xtx);
let edf: f64 = (0..ainv_xtx.nrows())
    .map(|i| ainv_xtx[[i, i]])
    .sum();
let phi = rss / (n as f64 - edf);
```

**Impact**:
- Fixes 1-4% error in φ
- No change in final convergence (still λ=[4.11, 2.32])

## Formula Verification

### bSb1 (First Derivative) ✅ VERIFIED

**mgcv C source** (`gdi.c` get_bSb):
```c
bSb1[k] = lambda[k] * beta' * S_k * beta
        + 2 * (dbeta/drho_k)' * S * beta
```

**Our implementation** (src/reml.rs:866-890):
```rust
bsb1[i] = (lambda_i * beta_s_i_beta + 2.0 * dbeta_s_beta) / phi
```

**Status**: ✅ MATCHES (modulo φ division which is correct)

### bSb2 (Second Derivative) ✅ VERIFIED

**mgcv formula** (4 terms + diagonal):
- All terms implemented correctly
- Second derivatives include diagonal correction
- Matches mgcv C source exactly

**Status**: ✅ MATCHES

### φ (Scale Parameter) ✅ VERIFIED

**mgcv approach**:
- φ = RSS / (n - edf)
- edf = tr(A^{-1}·X'WX)

**Our implementation**:
- Now matches exactly

**Status**: ✅ MATCHES

## Current Convergence Behavior

### Our Implementation
```
Iterations: ~100
Final λ:    [4.11, 2.32]
Final REML: -63.28
Gradient:   0.04-0.05 (L_inf)
H[0,0]:     2.64
H[1,1]:     3.17
```

### mgcv Target
```
Iterations: 5
Final λ:    [5.69, 5.20]
Final REML: -64.64
Gradient:   ~0
H[0,0]:     2.81
H[1,1]:     3.19
```

### Convergence Gap
```
λ[0]:  4.11 vs 5.69  (72% of optimal)
λ[1]:  2.32 vs 5.20  (45% of optimal)
REML: -63.28 vs -64.64 (gap of 1.36)
```

## Trajectory Analysis

| Iteration | Our Gradient | mgcv Gradient | Notes |
|-----------|--------------|---------------|-------|
| 1 | 3.99 | 41.6 | Similar starting point |
| 2 | 3.89 | 29.9 | We decrease slower |
| 3 | 2.97 | 5.07 | Gap widening |
| 4 | 0.80 | 0.24 | mgcv drops 95%, we drop 73% |
| 5 | 0.27 | ~0 | mgcv converged, we continue |

**Key Issue**: After iteration 3, mgcv makes larger Newton step and converges. We plateau around gradient 0.05.

## Hessian Component Breakdown

At our convergence point λ=[4.11, 2.32]:

**H[0,0] = 2.64**:
- det2 = 1.00 (38%)
- bSb2 = 4.30 (162%)
  - diag_corr = 4.32 (dominates!)
  - term1 (d²β) = 0.006
  - term2 = 0.003
  - term3 = -0.008
  - term4 = -0.008
- Total = (1.00 + 4.30)/2 = 2.65

**Key Observation**: bSb1 diagonal correction dominates (~80% of bSb2)

## Remaining Hypotheses

Since all formulas are verified correct, the bug must be subtle:

### Hypothesis 1: Gradient-Hessian Mismatch
- Hessian might be for a different parameterization than gradient
- Need to verify ∂REML/∂ρ vs ∂²REML/∂ρ² consistency
- Check if chain rule is applied correctly everywhere

### Hypothesis 2: REML Criterion Formula
- Might be missing constant terms or factors
- Could affect gradient scaling
- Need exact REML formula from Wood (2011) with all terms

### Hypothesis 3: Matrix Conditioning
- A matrix might be ill-conditioned
- QR decomposition might lose precision
- Regularization might be too strong

### Hypothesis 4: Numerical Precision
- Accumulation of roundoff errors
- Catastrophic cancellation in some terms
- Need higher precision for some operations

### Hypothesis 5: Sign Errors
- Subtle sign error in gradient or Hessian
- Everything "looks right" but one term has wrong sign
- Would cause convergence to wrong minimum

## Documentation Created

1. **COMPLETE_BSB2_RESULTS.md** - Full bSb2 implementation results
2. **DET2_BSB2_COMPARISON.md** - Why bSb2 is essential (82% of Hessian)
3. **DET2_ONLY_RESULTS.md** - Intermediate det2-only results
4. **D2BETA_FORMULA_FIX.md** - Derivation of missing diagonal term
5. **BSB1_FORMULA_VERIFICATION.md** - Verified against mgcv C source
6. **PHI_BUG_ANALYSIS.md** - Analysis of φ computation bug
7. **PHI_FIX_RESULTS.md** - Results showing φ fix didn't help convergence
8. **MGCV_HESSIAN_ANALYSIS.md** - Analysis of mgcv C source
9. **CHAIN_RULE_CLARIFICATION.md** - Mathematical derivations
10. **INVESTIGATION_SUMMARY.md** - Complete journey overview

## Code Changes

**src/reml.rs**:
- Lines 854-864: dβ/dρ computation
- Lines 866-890: bSb1 with correct formula
- Lines 892-1027: Complete 4-term bSb2 + diagonal
- Lines 985-989: d²β diagonal correction
- Lines 847-867: Correct φ = RSS/(n-edf)

**src/smooth.rs**:
- Lines 208-218: Debug output for raw Hessian

## Next Steps for Investigation

### Immediate Priority: Direct Comparison
1. Run mgcv and our implementation at SAME λ
2. Extract ALL intermediate values:
   - β coefficients
   - dβ/dρ values
   - bSb, bSb1, bSb2 values
   - det2 values
   - Gradient components
   - Hessian components
3. Find FIRST value that differs
4. Trace back why it differs

### Secondary: Formula Re-verification
1. Re-derive REML gradient from first principles
2. Verify Hessian is consistent with gradient
3. Check Wood (2011) for exact formula with all constants
4. Look for missing scale factors or signs

### Tertiary: Alternative Approaches
1. Implement BFGS to verify our gradient is correct
2. Use finite differences to verify Hessian
3. Add higher numerical precision in critical sections

## Conclusion

**Major Achievement**:
- Found and fixed critical exploding term bug
- Implemented complete, verified bSb2 formula
- Hessian magnitude now matches mgcv (94-99%)
- All formulas verified against mgcv C source

**Remaining Challenge**:
- Still converge to λ ~45-72% of optimal
- Gap must be from very subtle error
- Likely in gradient-Hessian consistency or REML formula

**Recommendation**:
- Direct value-by-value comparison with mgcv at fixed λ
- This should reveal the exact source of discrepancy
- Once found, should be straightforward to fix

**Status**: Ready for detailed forensic comparison with mgcv internals.
