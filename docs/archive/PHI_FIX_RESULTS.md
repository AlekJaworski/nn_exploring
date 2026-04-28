# φ (Scale Parameter) Fix: Results and Analysis

## Fix Implemented

Changed from using penalty matrix rank to correct effective degrees of freedom:

### Before (WRONG):
```rust
let mut total_rank = 0;
for penalty in penalties.iter() {
    total_rank += estimate_rank(penalty);
}
let phi = rss / (n - total_rank) as f64;
```

### After (CORRECT):
```rust
// edf = tr(A^{-1}·X'WX)
let xtx = x.t().to_owned().dot(&x.to_owned());
let ainv_xtx = a_inv.dot(&xtx);
let edf: f64 = (0..ainv_xtx.nrows())
    .map(|i| ainv_xtx[[i, i]])
    .sum();

let phi = rss / (n as f64 - edf);
```

## Verification: φ Error at Optimal λ

From `test_phi_bug.py` at mgcv's optimal λ=[5.69, 5.20]:

| Metric | Correct (mgcv) | Wrong (ours) | Error |
|--------|----------------|--------------|-------|
| **edf** | 14.970 | 16 (fixed) | -6.9% |
| **φ** | 0.009115 | 0.009226 | +1.2% |

**Impact**: Our old φ was 1.2% too large, causing bSb2 to be 1.2% too small.

## Runtime Observations

During Newton iteration (from debug output):

**Iteration 1** (λ ≈ [0.003, 0.003]):
```
edf (correct) = 18.987
old total_rank = 16
φ_correct = 1.214e-2
φ_old = 1.171e-2
ratio = 0.964 (old_phi / correct_phi)
```

**Iteration 2** (λ ≈ [0.079, 0.033]):
```
edf (correct) = 18.825
old total_rank = 16
φ_correct = 1.212e-2
φ_old = 1.171e-2
ratio = 0.966
```

**Key Observation**:
- edf varies with λ (18.987 → 18.825 → ...)
- old total_rank is constant (always 16)
- At small λ: edf > total_rank → old φ too small → bSb2 too large
- At large λ: edf < total_rank → old φ too large → bSb2 too small

## Convergence Results

### Before Fix (with wrong φ):
```
Final λ = [4.11, 2.32]
REML = -63.28
Gradient L_inf = 0.04-0.05
```

### After Fix (with correct φ):
```
Final λ = [4.11, 2.32]  ← UNCHANGED!
REML = -63.28           ← UNCHANGED!
```

### mgcv Optimal:
```
Final λ = [5.69, 5.20]
REML = -64.64
```

## Impact Assessment

**Error magnitude**: 1-4% φ error → 1-4% bSb2 scaling error

**Convergence gap**: ~30-50% error in λ values

**Conclusion**: φ error is TOO SMALL to explain our convergence issues!

The fix is mathematically correct and improves code correctness, but does NOT solve the convergence problem.

## Remaining Hypotheses

Since φ is now correct and bSb1/bSb2 formulas match mgcv, the bug must be elsewhere:

### Hypothesis 1: REML Criterion Sign Error
Perhaps we're minimizing when we should maximize, or vice versa?
- Need to verify exact REML formula and signs

### Hypothesis 2: Gradient Formula Error
The gradient might not match what the Hessian expects
- Need to verify ∂REML/∂ρ formula matches literature

### Hypothesis 3: Numerical Precision Issues
Accumulation of small errors in matrix operations
- Check conditioning of A matrix
- Verify QR decomposition stability

### Hypothesis 4: Missing Terms in REML
mgcv might include additional terms we're not computing
- Cross-reference complete REML criterion from Wood (2011)

## Files Modified

- **src/reml.rs** (lines 847-867): Replaced penalty rank with correct edf computation
- **test_phi_bug.py**: Verification test confirming 1.2% φ error
- **PHI_BUG_ANALYSIS.md**: Detailed analysis of the bug
- **BSB1_FORMULA_VERIFICATION.md**: Verified bSb1 formula against mgcv C source

## Conclusion

✅ **Correctness**: φ now uses proper effective df, fixing a 1-4% error

❌ **Convergence**: No improvement in final λ values

**Next Step**: Need to investigate other potential sources of convergence error, starting with a direct comparison of all REML components (gradient, Hessian, criterion value) against mgcv at the same λ point.
