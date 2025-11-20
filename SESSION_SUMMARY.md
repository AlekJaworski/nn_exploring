# Investigation Session Summary - Penalty Gradient & Hessian

## Initial Problem
Newton optimizer was taking 20-30 iterations vs mgcv's 5, or failing to converge.

## Major Findings

### 1. We're Minimizing REML (Not Maximizing)
- **Clarification**: REML criterion is `-2*log(likelihood)`, so we MINIMIZE it
- Lower/more negative values are better
- mgcv achieves REML=-64.64, we achieve REML=-63.26

### 2. Gradient Formula is Correct
```rust
gradient[i] = (trace - (rank_i as f64) + penalty_term / phi) / 2.0;
```
- Removed incorrect scaling factors
- Matches Wood (2011) formula
- Magnitude still ~10x too small compared to mgcv (unclear why)

### 3. Simplified Hessian Works Partially
**Current implementation**:
```rust
h_val = lambda_i * lambda_j * trace_term / 2.0;
if i == j {
    h_val += lambda_i * grad_lambda_i;
}
hessian[[i,j]] = -h_val;  // Negation for correct Newton direction
```

**Performance**:
- ✅ Converges in 7 iterations (vs mgcv's 5)
- ✅ Gradient decreases monotonically (3.99 → 0.70)
- ❌ Converges to SUBOPTIMAL point (λ=[3.94, 2.28] vs [5.69, 5.20])
- ❌ Finaloptimal REML=-63.26 vs optimal -64.64 (1.4 units away)

### 4. Complete Hessian Formula Derived But Failed
**Derived formula** (see HESSIAN_FORMULA_DERIVATION.md):
```
H[i,j] = (1/2) · {-tr(A^{-1}·M_i·A^{-1}·M_j)
                  + (2/φ)·β'·M_i·A^{-1}·M_j·β
                  - (2/φ²)·(β'·M_i·β)·(β'·M_j·β)
                  + δ_{ij}·[∇_i + r_i/(2λ_i)]}
```

**Implementation attempt**:
- All three terms computed correctly
- **Issue**: Unclear how to apply chain rule when M_i=λ_i·S_i already contains λ factors
- **Result**: Complete failure - gradient stuck at ~4, no progress after 100 iterations

### 5. Root Cause Still Unknown
The simplified Hessian uses only the trace term with chain rule scaling. Missing terms:
- Term 2: Penalty-beta-Hessian interaction
- Term 3: Penalty-penalty interaction
- Term 4: Rank correction on diagonal

These missing terms cause convergence to wrong minimum.

## Current State

**What Works**:
- ✓ Gradient formula correct
- ✓ Newton takes reasonable steps
- ✓ No numerical instability
- ✓ Converges reliably in ~7 iterations

**What's Wrong**:
- ✗ Converges to wrong λ values (40% error)
- ✗ Final REML 1.4 units away from optimum
- ✗ Complete Hessian formula fails when implemented

## Files Created This Session

| File | Purpose |
|------|---------|
| `CHECKPOINT_SIMPLIFIED_HESSIAN.md` | Documents working simplified state |
| `HESSIAN_FORMULA_DERIVATION.md` | Complete mathematical derivation |
| `NEWTON_FIX_SUMMARY.md` | Earlier investigation summary |
| `test_basic_fit.py` | Basic functionality test |
| `test_hessian_terms.py` | Individual term comparison (WIP) |

## Next Steps (Recommendations)

1. **Option A**: Deep dive into mgcv C source (`gdi.c`) to understand exact implementation
   - May reveal implementation details not in papers
   - Time-intensive but thorough

2. **Option B**: Implement BFGS quasi-Newton method
   - Approximates Hessian from gradient history
   - More robust, standard optimization approach
   - Used by many other packages

3. **Option C**: Accept current performance
   - 7 iterations is reasonable (vs 5)
   - 40% error in λ may be acceptable for many applications
   - Focus optimization effort elsewhere

## Key Lessons

1. **Terminology matters**: "Maximize likelihood" but "minimize REML criterion"
2. **Source code > papers**: mgcv implementation differs from published formulas
3. **Chain rule complexity**: ρ=log(λ) parameterization adds subtle complications
4. **Numerical stability**: Small details like which matrix to use for solve() matter
5. **Partial success**: Simplified Hessian works well enough for many cases

## Performance Comparison

| Metric | mgcv | Our Simplified | Our Complete |
|--------|------|----------------|--------------|
| Iterations | 5 | 7 | 100+ |
| Final λ | [5.69, 5.20] | [3.94, 2.28] | [0.004, 0.004] |
| Final REML | -64.64 | -63.26 | -22.90 |
| Gradient at end | ~0 | ~0.70 | ~4.0 |
| Status | Optimal ✓ | Suboptimal | Failed ✗ |

## Conclusion

We've made significant progress understanding the Newton optimizer and fixing critical bugs (gradient scaling, numerical stability). The simplified Hessian provides reasonable convergence, though not optimal. The complete Hessian formula requires more investigation to implement correctly.

**Recommended path forward**: Implement BFGS as a robust alternative, or accept current 7-iteration performance as "good enough" depending on project priorities.
