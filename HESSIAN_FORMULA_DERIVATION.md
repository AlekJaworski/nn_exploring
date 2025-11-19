# Complete Hessian Formula Derivation

## REML Criterion (Gaussian case)

```
V = (1/2) * [log|X'WX + S| + (n-r)*log(2πφ) + RSS/φ - Σr_i*log(λ_i)]
```

Where:
- S = Σλ_i·S_i (total penalty)
- A = X'WX + S (penalized information matrix)
- φ = RSS/(n-r) (scale parameter)
- r_i = rank(S_i)

## Parameterization

We optimize with respect to ρ_i = log(λ_i), so:
- λ_i = exp(ρ_i)
- dλ_i/dρ_i = λ_i
- d²λ_i/dρ_i² = λ_i

## First Derivatives (Gradient)

```
∂V/∂ρ_i = ∂V/∂λ_i · dλ_i/dρ_i = λ_i · ∂V/∂λ_i
```

Breaking down ∂V/∂λ_i:

1. **Log-determinant term**:
   ```
   ∂log|A|/∂λ_i = tr(A^{-1} · S_i)
   ```

2. **Penalty term**:
   ```
   ∂(RSS/φ)/∂λ_i = -(1/φ²)·RSS·(∂φ/∂λ_i) + (1/φ)·(∂RSS/∂λ_i)

   But for Gaussian, ∂RSS/∂λ_i requires ∂β/∂λ_i = -A^{-1}·S_i·β

   This gives: (β'·S_i·β)/φ
   ```

3. **Log lambda term**:
   ```
   -∂(r_i·log λ_i)/∂λ_i = -r_i/λ_i
   ```

Combined with chain rule:
```
∂V/∂ρ_i = λ_i · [tr(A^{-1}·S_i) - r_i/λ_i + (β'·S_i·β)/φ]
        = [tr(A^{-1}·λ_i·S_i) - r_i + λ_i·(β'·S_i·β)/φ]
```

Dividing by 2 (from REML definition):
```
∇_i = (1/2) · [tr(A^{-1}·M_i) - r_i + (β'·M_i·β)/φ]
```

where M_i = λ_i·S_i.

✅ **This matches our implementation!**

## Second Derivatives (Hessian)

```
∂²V/∂ρ_i∂ρ_j = ∂/∂ρ_j[λ_i · ∂V/∂λ_i]
              = λ_j · ∂/∂λ_j[λ_i · ∂V/∂λ_i]
              = λ_j · [δ_{ij}·∂V/∂λ_i + λ_i · ∂²V/∂λ_i∂λ_j]
```

This has two parts:
- **Diagonal correction**: λ_i · ∇_i when i=j
- **Curvature term**: λ_i·λ_j · ∂²V/∂λ_i∂λ_j

### Computing ∂²V/∂λ_i∂λ_j

**Term 1: Log-determinant Hessian**
```
∂²log|A|/∂λ_i∂λ_j = ∂/∂λ_j[tr(A^{-1}·S_i)]
                    = -tr(A^{-1}·S_i·A^{-1}·S_j)
```

**Term 2: Penalty Hessian** (complex - involves implicit differentiation)
```
∂²(β'·S_i·β/φ)/∂λ_i∂λ_j involves:
- ∂β/∂λ_j = -A^{-1}·S_j·β
- ∂φ/∂λ_j = -(2φ/RSS)·β'·S_j·β
```

After working through the algebra:
```
= (2/φ)·β'·S_i·A^{-1}·S_j·β - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
```

**Term 3: Log lambda Hessian**
```
∂²(r_i·log λ_i)/∂λ_i∂λ_j = δ_{ij}·r_i/λ_i²
```

### Complete Hessian (before chain rule)

```
∂²V/∂λ_i∂λ_j = -tr(A^{-1}·S_i·A^{-1}·S_j)
                + (2/φ)·β'·S_i·A^{-1}·S_j·β
                - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
                + δ_{ij}·r_i/λ_i²
```

### With Chain Rule (ρ-space)

```
H[i,j] = ∂²V/∂ρ_i∂ρ_j
       = λ_i·λ_j · ∂²V/∂λ_i∂λ_j + δ_{ij}·λ_i·∂V/∂λ_i
       = λ_i·λ_j · ∂²V/∂λ_i∂λ_j + δ_{ij}·λ_i·∇_i/λ_i
       = λ_i·λ_j · ∂²V/∂λ_i∂λ_j + δ_{ij}·∇_i
```

Dividing by 2:
```
H[i,j] = (1/2) · {λ_i·λ_j · [-tr(A^{-1}·S_i·A^{-1}·S_j)
                              + (2/φ)·β'·S_i·A^{-1}·S_j·β
                              - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
                              + δ_{ij}·r_i/λ_i²]
                  + δ_{ij}·∇_i}
```

## Simplified Form (using M_i = λ_i·S_i)

```
H[i,j] = (1/2) · {-tr(A^{-1}·M_i·A^{-1}·M_j)
                  + (2/φ)·β'·M_i·A^{-1}·M_j·β
                  - (2/φ²)·(β'·M_i·β)·(β'·M_j·β)
                  + δ_{ij}·[∇_i + r_i/(2λ_i)]}
```

## Current Implementation vs Complete

**Current (INCOMPLETE)**:
```rust
h_val = lambda_i * lambda_j * trace_term / 2.0;
if i == j {
    h_val += lambda_i * grad_i;
}
hessian[[i,j]] = -h_val;
```

Where `trace_term = tr(A^{-1}·M_i·A^{-1}·M_j)`, missing the **negative sign**!

**Should be**:
```rust
// Compute all four terms
let term1 = -tr(A^{-1}·M_i·A^{-1}·M_j);  // Note: NEGATIVE
let term2 = (2/φ)·β'·M_i·A^{-1}·M_j·β;
let term3 = -(2/φ²)·(β'·M_i·β)·(β'·M_j·β);
let term4_diag = if i==j { ∇_i + r_i/(2λ_i) } else { 0 };

hessian[[i,j]] = (term1 + term2 + term3) / 2.0 + term4_diag / 2.0;
```

Then negate for correct Newton direction (if needed based on testing).

## Key Issues in Current Code

1. ❌ Missing **negative sign** on trace term
2. ❌ Missing term2 (penalty-beta-Hessian interaction)
3. ❌ Missing term3 (penalty-penalty interaction)
4. ❌ Missing r_i/(2λ_i) in diagonal correction

This explains why we converge to wrong minimum!
