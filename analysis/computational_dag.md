# Computational DAG Analysis: Gradient vs Hessian

## Current Gradient Formula (CORRECT - reml_gradient_multi_qr)

```
∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2

where:
  P = RSS + Σⱼ λⱼ·β'·Sⱼ·β
  φ = RSS / (n - r)
  r = Σⱼ rank(Sⱼ)

  ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
  ∂RSS/∂ρᵢ = -2·residuals'·X·∂β/∂ρᵢ
  ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
  ∂P/∂ρᵢ = ∂RSS/∂ρᵢ + λᵢ·β'·Sᵢ·β + 2·Σⱼ λⱼ·β'·Sⱼ·∂β/∂ρᵢ

  ∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ
```

**Gradient Components:**
1. `tr(A⁻¹·λᵢ·Sᵢ)` - determinant term
2. `-rank(Sᵢ)` - rank correction
3. `∂(P/φ)/∂ρᵢ` - penalty quotient (includes implicit β dependencies)
4. `(n-r)·(1/φ)·∂φ/∂ρᵢ` - phi scaling term

## Current Hessian Formula (INCORRECT - reml_hessian_multi_qr)

The Hessian currently computes:
```
H[i,j] = Term1 + Term2 + Term3

Term1: ∂²log|A|/∂ρⱼ∂ρᵢ
Term2: ∂/∂ρⱼ[∂edf/∂ρᵢ·(-log(φ)+1)]
Term3: ∂/∂ρⱼ[∂rss/∂ρᵢ/φ]
```

**PROBLEM:** This decomposition doesn't match the gradient!

The gradient uses:
- `∂(P/φ)/∂ρᵢ` where P = RSS + penalty terms
- `(n-r)·(1/φ)·∂φ/∂ρᵢ`

But the Hessian uses:
- `∂edf/∂ρᵢ·(-log(φ)+1)`
- `∂rss/∂ρᵢ/φ`

These are NOT the same!

## Correct Hessian Formula (Needed)

```
H[i,j] = ∂/∂ρⱼ [∂REML/∂ρᵢ]
       = ∂/∂ρⱼ [(tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ) / 2]
       = [∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)] + ∂²(P/φ)/∂ρⱼ∂ρᵢ + ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ]] / 2
```

**Breaking down each term:**

### Term 1: ∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)]
```
= ∂/∂ρⱼ[λᵢ·tr(A⁻¹·Sᵢ)]
= -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ) + δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ)
```
This is CORRECT in current code (Term 1).

### Term 2: ∂²(P/φ)/∂ρⱼ∂ρᵢ

This is the BIG one! Let me expand:
```
∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ

∂²(P/φ)/∂ρⱼ∂ρᵢ = ∂/∂ρⱼ[(1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ]
                 = ∂/∂ρⱼ[(1/φ)·∂P/∂ρᵢ] + ∂/∂ρⱼ[-(P/φ²)·∂φ/∂ρᵢ]
```

**Part A:** ∂/∂ρⱼ[(1/φ)·∂P/∂ρᵢ]
```
= (1/φ)·∂²P/∂ρⱼ∂ρᵢ - (1/φ²)·∂φ/∂ρⱼ·∂P/∂ρᵢ
```

**Part B:** ∂/∂ρⱼ[-(P/φ²)·∂φ/∂ρᵢ]
```
= -(1/φ²)·∂P/∂ρⱼ·∂φ/∂ρᵢ - (P/φ²)·∂²φ/∂ρⱼ∂ρᵢ + 2·(P/φ³)·∂φ/∂ρⱼ·∂φ/∂ρᵢ
```

### Term 3: ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ]
```
= (n-r)·∂/∂ρⱼ[(1/φ)·∂φ/∂ρᵢ]
= (n-r)·[(1/φ)·∂²φ/∂ρⱼ∂ρᵢ - (1/φ²)·∂φ/∂ρⱼ·∂φ/∂ρᵢ]
```

## Second-Order Derivatives Needed

### ∂²P/∂ρⱼ∂ρᵢ
```
P = RSS + Σₖ λₖ·β'·Sₖ·β

∂P/∂ρᵢ = ∂RSS/∂ρᵢ + λᵢ·β'·Sᵢ·β + 2·Σₖ λₖ·β'·Sₖ·∂β/∂ρᵢ

∂²P/∂ρⱼ∂ρᵢ = ∂²RSS/∂ρⱼ∂ρᵢ
             + δᵢⱼ·β'·Sᵢ·β                    [from ∂λᵢ/∂ρⱼ = δᵢⱼ·λᵢ]
             + λᵢ·∂β'/∂ρⱼ·Sᵢ·β + λᵢ·β'·Sᵢ·∂β/∂ρⱼ  [= 2·λᵢ·∂β'/∂ρⱼ·Sᵢ·β by symmetry]
             + 2·Σₖ[δₖⱼ·λₖ·β'·Sₖ·∂β/∂ρᵢ + λₖ·∂β'/∂ρⱼ·Sₖ·∂β/∂ρᵢ + λₖ·β'·Sₖ·∂²β/∂ρⱼ∂ρᵢ]
```

### ∂²RSS/∂ρⱼ∂ρᵢ
```
RSS = (y - Xβ)'(y - Xβ)
∂RSS/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ  (where r = y - Xβ)

∂²RSS/∂ρⱼ∂ρᵢ = -2·∂r'/∂ρⱼ·X·∂β/∂ρᵢ - 2·r'·X·∂²β/∂ρⱼ∂ρᵢ
              = 2·∂β'/∂ρⱼ·X'·X·∂β/∂ρᵢ - 2·r'·X·∂²β/∂ρⱼ∂ρᵢ
```

### ∂²β/∂ρⱼ∂ρᵢ
```
From A·β = X'y:
∂β/∂ρᵢ = -A⁻¹·∂A/∂ρᵢ·β = -A⁻¹·λᵢ·Sᵢ·β

∂²β/∂ρⱼ∂ρᵢ = -∂/∂ρⱼ[A⁻¹·λᵢ·Sᵢ·β]
            = A⁻¹·∂A/∂ρⱼ·A⁻¹·λᵢ·Sᵢ·β - A⁻¹·λᵢ·Sᵢ·∂β/∂ρⱼ - δᵢⱼ·A⁻¹·λᵢ·Sᵢ·β
            = λⱼ·A⁻¹·Sⱼ·A⁻¹·λᵢ·Sᵢ·β - A⁻¹·λᵢ·Sᵢ·∂β/∂ρⱼ - δᵢⱼ·∂β/∂ρᵢ
```

### ∂²φ/∂ρⱼ∂ρᵢ
```
φ = RSS / (n - r)

∂φ/∂ρᵢ = (1/(n-r))·∂RSS/∂ρᵢ

∂²φ/∂ρⱼ∂ρᵢ = (1/(n-r))·∂²RSS/∂ρⱼ∂ρᵢ
```

## Summary of Discrepancies

| Component | Current Hessian | Should Be |
|-----------|-----------------|-----------|
| Determinant term | ✓ Correct | - |
| Penalty quotient | ✗ Missing | Need ∂²(P/φ)/∂ρⱼ∂ρᵢ |
| Phi scaling | ✗ Wrong form | Need ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ] |
| Second derivatives | ✗ Using old formula | Need ∂²P, ∂²RSS, ∂²β, ∂²φ consistent with gradient |

## Action Plan

1. ✓ Identify discrepancy (this document)
2. Implement correct ∂²β/∂ρⱼ∂ρᵢ (with IFT)
3. Implement correct ∂²RSS/∂ρⱼ∂ρᵢ
4. Implement correct ∂²P/∂ρⱼ∂ρᵢ
5. Implement correct ∂²φ/∂ρⱼ∂ρᵢ
6. Combine into ∂²(P/φ)/∂ρⱼ∂ρᵢ
7. Implement ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ]
8. Assemble final Hessian
9. Validate against numerical differentiation
