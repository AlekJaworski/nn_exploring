# Constraint Implementation Investigation - Critical Findings

**Date**: 2025-11-12
**Status**: ⚠️ IMPLEMENTATION INCORRECT - Needs Revision

## TL;DR

My constraint implementation reduces k→k-1 basis functions, but **mgcv does NOT do this**. mgcv keeps all k basis functions and handles constraints during solving via reparameterization. This fundamental difference explains why lambda and deviance still don't match.

## Test Results

### Comparison with mgcv (Same Random Seed)

```
                Rust            mgcv          Ratio
Lambda:         14.454398       1.223084      11.82x  ❌
Deviance:       25.2923         0.7478        33.82x  ❌
Predictions:    correlation 0.9998           ✅
Basis dims:     9 (k-1)         10 (k)        Different!
```

**Key Finding**: Despite implementing constraint, lambda/deviance are **EXACTLY THE SAME** as before constraint implementation (from CR_PENALTY_FIX_SUMMARY.md).

## What I Implemented (WRONG)

### My Approach: Dimension Reduction

```rust
// 1. Create Q matrix (k x k-1) via Gram-Schmidt
let q_matrix = sum_to_zero_constraint_matrix(k)?;  // 10 x 9

// 2. Transform basis: X_constrained = X * Q
let basis = unconstrained_basis.dot(&q_matrix);     // 100 x 9

// 3. Transform penalty: S = Q^T * S * Q
let penalty = q_matrix.t().dot(&S_unconstrained).dot(&q_matrix);  // 9 x 9
```

**Result**:
- ✅ Sum-to-zero constraint is mathematically satisfied
- ✅ Code compiles and runs
- ❌ Changes problem scale (k-1 vs k basis)
- ❌ Lambda off by 11.8x
- ❌ Deviance off by 33.8x

## What mgcv Actually Does (CORRECT)

### mgcv's Approach: Reparameterization

From inspecting mgcv internals:

```R
sm <- smoothCon(s(x, k=10, bs='cr'), ...)

sm$bs.dim                    # = 10 (NOT 9!)
dim(sm$X)                    # = 100 x 10
dim(sm$S[[1]])              # = 10 x 10
dim(sm$C)                    # = 1 x 10 (constraint matrix)
colMeans(sm$X)              # NOT zero! (max = 0.12)
```

**Key observations**:
1. All k=10 basis functions retained
2. Design matrix is 100 x 10
3. Penalty matrix is 10 x 10
4. Constraint matrix C is stored but **not applied to X or S**
5. Basis columns don't sum to zero

### How mgcv Handles Constraints

From mgcv documentation and source code:

```
"The purpose of the wrapper is to allow user transparent
re-parameterization of smooth terms, in order to allow
identifiability constraints to be absorbed into the
parameterization"
```

**Method**: During PiRLS solving, mgcv uses:
- QR decomposition with constraint absorption
- Constraint enforced in **coefficient space**, not basis space
- Effective dimensionality is k-1, but actual matrices are k x k

**Analogy**:
- My approach: Remove one basis function explicitly
- mgcv's approach: Keep all basis functions, constrain coefficients

## Why My Approach Fails

### Scale Mismatch

When I reduce from k=10 to k=9:

```
My penalty: 9x9, Frobenius norm ≈ F_9
mgcv penalty: 10x10, Frobenius norm = 2.837

Ratio: F_9 / F_10 ≈ (9/10)^2 ≈ 0.81

But lambda ratio is 11.82x, not 0.81x!
```

The dimension reduction changes the problem fundamentally, not just by scale.

### Why Predictions Still Match

Despite wrong lambda/deviance, predictions match (correlation 0.9998) because:
- Both methods fit smooth functions
- Both penalize roughness
- The **shape** of the fit is similar
- Only the **penalty scale** differs

## Root Cause Analysis

### Timeline of Discoveries

1. **Initial**: Penalty matrix was 584x too large → Fixed with correct algorithm
2. **After penalty fix**: Lambda 11.8x off, thought it was due to k vs k-1
3. **Implemented constraint**: Reduced k→k-1, but lambda STILL 11.8x off
4. **Tested with mgcv**: Discovered mgcv uses k=10, not k-1=9!

### The Real Issue

The 11.8x lambda discrepancy is **NOT** about constraint implementation. It's about:

1. **Different parameterization**: k vs k-1 basis changes problem scale
2. **Different constraint handling**: Pre-transformation vs runtime enforcement
3. **Possibly different REML/GCV formulas**: Need to check the lambda optimization

## Correct Implementation Path

To match mgcv exactly, need to:

### Option 1: Revert to k=10 (Simple)

```rust
// Don't reduce dimensions
pub fn cr_spline(...) -> Result<Self> {
    let basis = CubicRegressionSpline::with_num_knots(...);  // k=10
    let penalty = compute_penalty("cr", k, ...)?;            // 10x10

    Ok(Self {
        basis: Box::new(basis),
        penalty,
        lambda: 1.0,
        constraint_matrix: None,  // No pre-transformation
    })
}
```

Then handle constraint in PiRLS solver (complex).

### Option 2: Match mgcv's REML Formula

Perhaps the issue is in the REML/GCV criterion calculation, not the constraint. Need to verify:

```rust
// In reml.rs
pub fn reml_criterion(...) -> Result<f64> {
    // Is this formula exactly matching mgcv's?
    // Check Wood (2011) for correct REML formula
}
```

### Option 3: Accept Different Approach

Document that we use k-1 basis (valid alternative), but don't claim to match mgcv exactly.

## Testing Evidence

### Test 1: Basic Constraint Test (Rust)
```
✓ Basis correctly reduced to k-1 = 9
✓ Penalty matrix correctly sized (9x9)
✓ GAM fitted successfully
  Lambda: 6.309573
```

### Test 2: mgcv Comparison (Same Data, Seed=42)
```
Rust:  Lambda 14.45, Deviance 25.29, k-1=9 basis
mgcv:  Lambda 1.22,  Deviance 0.75,  k=10 basis
```

### Test 3: mgcv Internals Inspection
```
mgcv keeps k=10 basis functions ✓
mgcv keeps 10x10 penalty matrix ✓
mgcv stores C but doesn't apply to X/S ✓
mgcv handles constraint during solving ✓
```

## Recommendations

### Immediate Actions

1. **Revert constraint changes**: Go back to k=10 unconstrained basis
2. **Focus on REML/GCV**: Investigate lambda optimization formula
3. **Study mgcv source**: Look at actual PiRLS implementation with constraints

### Long-term Goals

1. **Implement proper constraint handling**: In PiRLS solver, not basis construction
2. **Match mgcv's reparameterization**: Use QR with constraint absorption
3. **Add constraint tests**: Verify coefficients satisfy constraint, not basis

## Lessons Learned

1. ✅ **Test first**: Should have compared with mgcv BEFORE committing
2. ✅ **Verify assumptions**: Assumed k-1 based on docs, didn't check actual behavior
3. ✅ **Look at internals**: mgcv's smoothCon reveals ground truth
4. ⚠️ **Different valid ≠ matching**: My approach is valid math, but not what mgcv does

## Files to Update

1. `src/gam.rs`: Revert constraint application in cr_spline constructors
2. `src/linalg.rs`: Can keep constraint functions (might use later)
3. `IDENTIFIABILITY_CONSTRAINT_SUMMARY.md`: Add note about incorrect approach
4. `CR_PENALTY_FIX_SUMMARY.md`: Update with new findings

## References

- mgcv source: `mgcv::smoothCon`, `mgcv::gam.fit3`
- Wood (2011): "Fast stable restricted maximum likelihood..."
- Wood (2017): GAM book, Chapter 5

---

**Conclusion**: The constraint implementation is mathematically correct but uses a different approach than mgcv. To match mgcv exactly, constraints must be handled during solving with k basis functions, not by pre-reducing to k-1.
