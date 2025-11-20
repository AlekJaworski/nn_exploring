# det2 vs bSb2: Critical Findings

## Executive Summary

**CRITICAL FINDING**: Our det2-only Hessian is only ~18% of mgcv's total Hessian!

This means **bSb2 contributes ~82% of the total Hessian** - it is NOT negligible!

## Comparison Data

### Our Implementation (det2-only)
At λ ≈ [4.16, 2.28] (our final convergence point):
```
H[0,0] = 0.500
H[1,1] = 0.452
H[0,1] = -0.0055
```

### mgcv (det2 + bSb2)
At λ = [5.69, 5.20] (optimal):
```
H[0,0] = 2.813
H[1,1] = 3.186
H[0,1] = 0.0232
```

### Ratio Analysis
```
Our H[0,0] / mgcv H[0,0] = 0.500 / 2.813 = 17.8%
Our H[1,1] / mgcv H[1,1] = 0.452 / 3.186 = 14.2%
|Our H[0,1]| / mgcv H[0,1] = 0.0055 / 0.0232 = 23.7%

Average: ~18% of mgcv's total
```

**Interpretation**: bSb2 contributes approximately 82% of the total Hessian at optimal λ!

## Why bSb2 is So Large

The penalty term β'·S·β becomes large at optimal λ values:
- At λ=5.69: penalty = λ·β'·S·β is substantial
- The second derivative ∂²(β'·S·β/φ)/∂ρ² scales with penalty magnitude
- Unlike det2 which depends only on penalty matrices S_i

## Implications

### Why We Converge Suboptimally
With only det2:
- Hessian underestimates curvature by 5-7x
- Newton steps are too large
- Converge to λ=[4.16, 2.28] instead of [5.69, 5.20]
- Miss 27-56% of optimal λ values

### Why Gradient Doesn't Reach Zero
- True Hessian H_true = det2 + bSb2 ≈ det2 / 0.18
- We use H_approx = det2 only
- Newton step: Δρ = -H^{-1}·∇
- Our step: Δρ_ours = -(det2)^{-1}·∇ ≈ -5.6·(H_true)^{-1}·∇
- **Steps are 5-6x too large!**
- Overshoot and oscillate, never reaching true minimum

## Conclusion

**We MUST implement proper bSb2 for convergence!**

det2-only was useful for:
- ✅ Proving term2/term3 were wrong (they exploded)
- ✅ Establishing positive definite Hessian
- ✅ Showing improvement over broken implementation

But it's insufficient for optimal convergence because:
- ❌ Missing 82% of total Hessian
- ❌ Newton steps 5-6x too large
- ❌ Converges to wrong minimum

## Next Steps

### Implement proper bSb2

**Step 1**: Compute β derivatives
```rust
// First derivatives: dβ/dρ_i
let mut dbeta_drho = Vec::new();
for i in 0..m {
    let m_i = &penalties[i] * lambdas[i];
    let m_i_beta = m_i.dot(&beta);
    let dbeta_i = -a_inv.dot(&m_i_beta);  // -A^{-1}·M_i·β
    dbeta_drho.push(dbeta_i);
}
```

**Step 2**: Implement 4-term bSb2 from mgcv C code
```rust
for i in 0..m {
    for j in i..m {
        // Term 1: Compute d²β/dρ_i dρ_j (second derivatives via implicit differentiation)
        // Term 2: dbeta_drho[i]' · S · dbeta_drho[j]
        let term2 = dbeta_drho[i].dot(&s.dot(&dbeta_drho[j]));

        // Term 3: dbeta_drho[j]' · S_i · β · λ_i
        let term3 = if i < M0 { 0.0 } else {
            dbeta_drho[j].dot(&penalties[i].dot(&beta)) * lambdas[i]
        };

        // Term 4: dbeta_drho[i]' · S_j · β · λ_j
        let term4 = if j < M0 { 0.0 } else {
            dbeta_drho[i].dot(&penalties[j].dot(&beta)) * lambdas[j]
        };

        // Diagonal correction
        let diag_corr = if i == j { bsb1[i] } else { 0.0 };

        bsb2[[i,j]] = 2.0 * (term1 + term2 + term3 + term4) + diag_corr;
    }
}
```

**Step 3**: Combine det2 + bSb2
```rust
let h_val = (det2 + bsb2) / 2.0;  // Divide by 2 for REML formula
```

### Expected Results

With proper bSb2:
- Hessian will be 5-6x larger (matching mgcv)
- Newton steps will be correctly sized
- Should converge in ~5 iterations (like mgcv)
- Final λ should match [5.69, 5.20]
- Gradient should reach ~0

## References

- `MGCV_HESSIAN_ANALYSIS.md` - C source code analysis
- `DET2_ONLY_RESULTS.md` - det2-only implementation results
- `src/reml.rs` lines 922-935 - Current det2-only code
- mgcv C source: `gdi.c` function `get_bSb`
