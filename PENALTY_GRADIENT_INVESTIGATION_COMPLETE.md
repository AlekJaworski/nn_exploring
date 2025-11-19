# Complete Investigation: Penalty Gradient and Hessian

## Executive Summary

This investigation sought to fix Newton optimizer convergence issues where our implementation takes 20-30 iterations vs mgcv's 3-8, or converges to suboptimal smoothing parameters.

**Key Findings**:
1. ✅ Gradient formula is correct and matches mgcv at iteration 1
2. ✅ Hessian mathematical formula fully derived from first principles
3. ✅ mgcv C source analyzed to understand exact implementation
4. ⚠️  Hessian implementation produces reasonable values but convergence still suboptimal
5. ❌ Root cause of suboptimal convergence not yet identified

## Timeline and Approach

### Phase 1: Gradient Verification
- **Finding**: Gradient at iteration 1 matches mgcv (41.9 vs 42.06)
- **Issue**: Gradient doesn't decrease properly in subsequent iterations
- **Conclusion**: Hessian must be wrong, not gradient

### Phase 2: Hessian Derivation
Derived complete Hessian formula from Wood (2011) JRSSB paper:

**In λ-space**:
```
∂²V/∂λ_i∂λ_j = -tr(A^{-1}·S_i·A^{-1}·S_j)
                + (2/φ)·β'·S_i·A^{-1}·S_j·β
                - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
                + δ_{ij}·r_i/λ_i²
```

**With chain rule to ρ-space** (ρ = log λ):
```
H[i,j] = λ_i·λ_j · ∂²V/∂λ_i∂λ_j + δ_{ij}·∂V/∂ρ_i
```

**Using M_i = λ_i·S_i** (the λ factors cancel!):
```
H[i,j] = (1/2) · {-tr(A^{-1}·M_i·A^{-1}·M_j)
                  + (2/φ)·β'·M_i·A^{-1}·M_j·β
                  - (2/φ²)·(β'·M_i·β)·(β'·M_j·β)
                  + δ_{ij}·[2·∇_i + r_i/λ_i]}
```

### Phase 3: mgcv C Source Analysis

Analyzed `gdi.c` to find exact implementation:

**Key Discovery**: mgcv splits the Hessian into two parts:
```
H_total = det2 + bSb2
```

Where:
- **det2** = ∂²log|A|/∂ρ_k∂ρ_m (log-determinant Hessian, from `get_ddetXWXpS`)
- **bSb2** = ∂²(β'Sβ/φ)/∂ρ_k∂ρ_m (penalty Hessian, from `get_bSb`)

**For Gaussian case**, det2 simplifies dramatically:
- Terms involving weight derivatives (Tk, Tkm) vanish
- Only 2 of 6 terms remain

**Gaussian det2 formula**:
```c
det2[k,m] = δ_{k,m}·λ_m·tr(A^{-1}·S_m) - λ_k·λ_m·tr[(A^{-1}·S_k)·(A^{-1}·S_m)]
```

Using M_i = λ_i·S_i:
```
det2[k,m] = δ_{k,m}·tr(A^{-1}·M_m) - tr[(A^{-1}·M_k)·(A^{-1}·M_m)]
```

**bSb2 formula** from C code:
```
bSb2[k,m] = 2·(d²β'/dρ_k dρ_m · S · β)          [Term 1: second derivatives]
           + 2·(dβ'/dρ_k · S · dβ/dρ_m)           [Term 2: mixed derivatives]
           + 2·(dβ'/dρ_m · S_k · β · sp[k])       [Term 3a: parameter-dependent]
           + 2·(dβ'/dρ_k · S_m · β · sp[m])       [Term 3b: parameter-dependent]
           + δ_{k,m}·bSb1[k]                       [Diagonal correction]
```

### Phase 4: Chain Rule Clarification

**Critical insight**: When using M_i = λ_i·S_i matrices, the chain rule λ_i·λ_j factors CANCEL!

**Derivation**:
```
-tr(A^{-1}·S_i·A^{-1}·S_j) in λ-space
= -tr(A^{-1}·(M_i/λ_i)·A^{-1}·(M_j/λ_j))
= -tr(A^{-1}·M_i·A^{-1}·M_j) / (λ_i·λ_j)
```

After multiplying by chain rule λ_i·λ_j:
```
= -tr(A^{-1}·M_i·A^{-1}·M_j)  [NO λ factors!]
```

**Therefore**: When computing with M matrices, do NOT multiply by λ_i·λ_j!

### Phase 5: Implementation

**Current implementation** (src/reml.rs lines 899-928):
```rust
// Compute M_i = λ_i·S_i
let m_i = penalty_i * lambda_i;
let m_j = penalty_j * lambda_j;

// det2 part: log-determinant Hessian
let trace_a_inv_m_i = if i == j {
    let a_m_i = a_inv.dot(&m_i);
    let mut tr = 0.0;
    for k in 0..p {
        tr += a_m_i[[k, k]];
    }
    tr
} else {
    0.0
};

let trace_term = tr[(A^{-1}·M_i)·(A^{-1}·M_j)];  // Already computed earlier

let det2 = if i == j {
    trace_a_inv_m_i - trace_term
} else {
    -trace_term
};

// bSb2 part: penalty Hessian
let term2 = (2/φ)·β'·M_i·A^{-1}·M_j·β;
let term3 = -(2/φ²)·(β'·M_i·β)·(β'·M_j·β);
let bSb2 = term2 + term3;

// Total Hessian
let h_val = (det2 + bSb2) / 2.0;
hessian[[i,j]] = h_val;  // No negation
```

**Current behavior**:
- ✅ Hessian values are positive and reasonable magnitude
- ✅ det2 ≈ 0.005-0.125 (positive)
- ✅ bSb2 ≈ -0.0001 to -0.065 (negative, but smaller than det2)
- ✅ Total ≈ 0.003-0.030 (positive)
- ❌ Converges to λ=[1.45, 0.94] instead of optimal [5.69, 5.20]
- ❌ Gradient stuck around -2.0, barely decreasing

## Comparison: Current vs Simplified vs mgcv

| Metric | mgcv | Simplified | Current (det2+bSb2) |
|--------|------|------------|---------------------|
| Iterations | 5 | 7 | ~100+ |
| Final λ | [5.69, 5.20] | [3.94, 2.28] | [1.45, 0.94] |
| Final REML | -64.64 | -63.26 | ~-60 (est) |
| Final Gradient | ~0 | ~0.7 | ~-2.0 |
| Hessian Sign | Positive | Mixed | Positive |
| Status | ✅ Optimal | ⚠️  Suboptimal | ❌ Wrong direction |

## Remaining Issues

1. **bSb2 computation incomplete**:
   - Current implementation only has term2 and term3
   - Missing: first derivatives dβ/dρ and second derivatives d²β/dρ²
   - mgcv C code explicitly computes these via implicit differentiation

2. **Possible sign errors**:
   - det2 formula matches C code
   - bSb2 formula may have sign issues
   - Need to verify each term against mgcv values at fixed λ

3. **Diagonal corrections unclear**:
   - Chain rule adds δ_{ij}·∂V/∂ρ_i term
   - Rank correction r_i/(2λ_i) causes huge values at small λ
   - Not clear if these belong in det2, bSb2, or separately

## Key Documents Created

| File | Purpose |
|------|---------|
| `MGCV_HESSIAN_ANALYSIS.md` | Analysis of mgcv C source code |
| `HESSIAN_FORMULA_DERIVATION.md` | Complete mathematical derivation |
| `CHAIN_RULE_CLARIFICATION.md` | Chain rule with M matrices |
| `CHECKPOINT_SIMPLIFIED_HESSIAN.md` | Working simplified state |
| `SESSION_SUMMARY.md` | Previous session overview |
| `test_hessian_terms.py` | Test script for term comparison |

## Recommendations

### Option A: Complete bSb2 Implementation
Implement full bSb2 with explicit β derivatives:
- Compute dβ/dρ_i = -A^{-1}·M_i·β for each i
- Compute d²β/dρ_k∂ρ_m via implicit differentiation
- Implement all 4 terms from C code
- **Effort**: High (complex linear algebra)
- **Confidence**: Medium (many opportunities for errors)

### Option B: Match mgcv Term-by-Term
At fixed λ, compare each term against mgcv:
- Extract mgcv's det2 values
- Extract mgcv's bSb2 values
- Identify which terms are wrong
- **Effort**: Medium (requires mgcv instrumentation)
- **Confidence**: High (direct comparison)

### Option C: BFGS Quasi-Newton
Implement BFGS that approximates Hessian from gradient history:
- Standard optimization method
- More robust than exact Hessian
- Used by many packages
- **Effort**: Medium (well-documented algorithm)
- **Confidence**: High (proven method)

### Option D: Accept Current Performance
Use simplified Hessian (7 iterations, ~40% λ error):
- Reasonable performance for many applications
- Focus optimization effort elsewhere
- **Effort**: None
- **Confidence**: Known limitations

## Technical Lessons

1. **Source code trumps papers**: mgcv implementation differs from published formulas
2. **Chain rule subtlety**: M=λS substitution cancels λ factors - not obvious!
3. **Numerical stability**: Which matrix to use for solve() matters
4. **Gaussian simplification**: Many terms vanish for constant weights
5. **Split computation**: det2 and bSb2 computed separately in mgcv
6. **Rank corrections**: Cause numerical issues at small λ values

## Variable Mapping (C to Math)

| C Variable | Math | Description |
|------------|------|-------------|
| `sp[m]` | λ_m | Smoothing parameter m (NOT log scale!) |
| `S_m` | S_m | Penalty matrix m (q×q) |
| `P` | R^{-1} | Inverse of QR decomposition R matrix |
| `PtSP[m]` | P'·S_m·P | r×r transformed penalty |
| `trPtSP[m]` | λ_m·tr(P'·S_m·P) | Precomputed trace |
| `det2` | ∂²log\|A\|/∂ρ² | Log-determinant Hessian |
| `bSb2` | ∂²(β'Sβ/φ)/∂ρ² | Penalty Hessian |
| `M0` | θ count | Number of non-smoothing parameters |
| `Mtot` | total params | M0 + number of smoothing parameters |

## Next Steps

**Immediate**: Choose between Options A-D based on project priorities.

**If continuing**: Implement Option B (term-by-term comparison) to identify exact discrepancy.

**Testing**: At λ=[5.69, 5.20], mgcv has:
- Hessian[0,0] = 2.813
- Hessian[1,1] = 3.186
- Hessian[0,1] = 0.023

Compare against our values to find the error.
