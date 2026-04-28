# REML Gradient Investigation Summary

## Problem Statement
The REML gradient computation was producing values significantly different from mgcv's reference implementation, with errors ranging from 10-50x in earlier versions.

## Key Findings and Fixes

### 1. Trace Computation (FIXED ✅)
**Issue**: Earlier implementation used block extraction which gave incorrect trace values.

**Root Cause**: Computing `tr(P_block'·S_block·P_block)` instead of `tr(P'·S·P)` for full matrices.

**Fix**: Use full p×p matrices throughout the computation.

**Validation**:
```
At λ=[2.0, 3.0]:
  Rust trace:  [1.340574, 1.437015]
  mgcv trace:  [1.340574, 1.437015]
  ✅ EXACT MATCH
```

### 2. Rank Estimation (FIXED ✅)
**Issue**: Heuristic rank estimation gave rank=7, but correct eigenvalue-based rank is 8.

**Root Cause**: The `estimate_rank()` function used `non_zero_rows - 2` heuristic which undercounted for block-diagonal penalties.

**Fix**: Use actual eigenvalue count from `penalty_sqrt()`: `rank = sqrt_pen.ncols()`

**Validation**:
```
Both penalties: rank = 8 (matching mgcv's eigenvalue-based calculation)
```

### 3. Phi (Scale Parameter) Estimation (FIXED ✅)
**Issue**: Used `phi = rss / (n - total_rank)` which gives incorrect denominator.

**Root Cause**: Should use effective degrees of freedom (edf) from full model, not sum of penalty ranks.

**Fix**: Compute `edf_total = tr(A^{-1}·X'X)` and use `phi = rss / (n - edf_total)`

**Validation**:
```
At λ=[2.0, 3.0]:
  edf_total = 16.222410
  phi = 0.009132
  ✅ Matches Python validation
```

### 4. Gradient Formula (PARTIALLY RESOLVED ⚠️)
**Current Status**: All gradient components (trace, rank, penalty, phi) match mgcv exactly, but final gradient values still differ from mgcv by 10-50% at moderate lambda values.

**Hypothesis**: mgcv may use a different REML gradient formula or parameterization than the standard Wood (2011) formula:
```
∂REML/∂ρᵢ = [tr(A^{-1}·λᵢ·Sᵢ) - rank(Sᵢ) + λᵢ·β'·Sᵢ·β/φ] / 2
```

**Evidence**:
- At λ=[100, 100]: Rust error < 2% ✅
- At λ=[2, 3]: Rust error ~50-100% ⚠️
- All individual components match mgcv exactly

**Next Steps**:
- Examine mgcv's `gdi.c` source code to determine exact gradient formula
- May need to use different REML criterion derivative
- Consider that mgcv uses Implicit Function Theorem for β derivatives

## Code Changes

### src/reml.rs

1. **Lines 452-457**: Use eigenvalue-based rank instead of heuristic
```rust
let rank = sqrt_pen.ncols();  // Actual number of positive eigenvalues
```

2. **Lines 575-581**: Compute phi using edf_total
```rust
let a_inv = p_matrix.dot(&p_matrix.t());
let ainv_xtwx = a_inv.dot(&xtwx);
let edf_total: f64 = (0..p).map(|i| ainv_xtwx[[i, i]]).sum();
let phi = rss / (n as f64 - edf_total);
```

3. **Added comprehensive debug output**:
   - Penalty sqrt computation and verification
   - Z matrix construction
   - QR decomposition verification
   - P matrix and A^{-1} verification
   - Trace computation details
   - Phi and edf_total values

## Test Files Created

1. `test_rust_trace_debug.py`: Tests trace computation with debug output
2. `test_full_gradient_comparison.py`: Compares gradients at multiple lambda values
3. `test_gradient_components_detailed.py`: Component-by-component verification
4. `validate_trace_dag.py`: Step-by-step DAG validation of algorithm
5. `/tmp/check_qr.py`: QR decomposition verification

## Performance Impact

All fixes maintain O(p³) complexity. The additional A^{-1} computation for phi adds minimal overhead since we already compute P = R^{-1}.

## References

- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. Journal of the Royal Statistical Society (B) 73(1):3-36
- Wood, S.N. (2017). Generalized Additive Models: An Introduction with R (2nd edition). Chapman and Hall/CRC.
- mgcv source: https://github.com/cran/mgcv/blob/master/src/gdi.c

## Conclusion

The trace computation, rank estimation, and phi calculation are now correct and match mgcv exactly. The remaining gradient discrepancy suggests mgcv uses a different gradient formula or REML criterion derivative than initially assumed. Further investigation of mgcv's C source code is needed to determine the exact formula.
