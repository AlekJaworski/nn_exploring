# Second Derivative d²β Formula Verification

## Problem

Our d²β formula may be missing a diagonal term. Let me re-derive from first principles.

## Starting Point

The normal equations:
```
A·β = X'Wy
```

where `A = X'WX + Σλ_i·S_i` and `M_i = λ_i·S_i`

## First Derivative

Taking ∂/∂ρ_i:
```
∂A/∂ρ_i · β + A · ∂β/∂ρ_i = 0
M_i · β + A · ∂β/∂ρ_i = 0
∂β/∂ρ_i = -A^{-1}·M_i·β  ✓
```

## Second Derivative

Taking ∂/∂ρ_j of the equation `A·(∂β/∂ρ_i) = -M_i·β`:

```
∂A/∂ρ_j · (∂β/∂ρ_i) + A · (∂²β/∂ρ_j∂ρ_i) = -∂M_i/∂ρ_j · β - M_i · (∂β/∂ρ_j)
```

### Compute ∂M_i/∂ρ_j

Since `M_i = λ_i·S_i` and `S_i` doesn't depend on ρ_j:
```
∂M_i/∂ρ_j = (∂λ_i/∂ρ_j)·S_i
```

Now, `ρ_i = log(λ_i)`, so `λ_i = exp(ρ_i)`:
```
∂λ_i/∂ρ_j = ∂exp(ρ_i)/∂ρ_j = δ_{ij}·exp(ρ_i) = δ_{ij}·λ_i
```

Therefore:
```
∂M_i/∂ρ_j = δ_{ij}·λ_i·S_i = δ_{ij}·M_i
```

### Continue the Derivation

Substituting back:
```
M_j · (∂β/∂ρ_i) + A · (∂²β/∂ρ_j∂ρ_i) = -δ_{ij}·M_i·β - M_i · (∂β/∂ρ_j)
```

Rearranging:
```
A · (∂²β/∂ρ_j∂ρ_i) = -δ_{ij}·M_i·β - M_i · (∂β/∂ρ_j) - M_j · (∂β/∂ρ_i)
```

Substituting `∂β/∂ρ_k = -A^{-1}·M_k·β`:
```
A · (∂²β/∂ρ_j∂ρ_i) = -δ_{ij}·M_i·β - M_i·(-A^{-1}·M_j·β) - M_j·(-A^{-1}·M_i·β)
                    = -δ_{ij}·M_i·β + M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β
```

Multiplying by A^{-1}:
```
∂²β/∂ρ_j∂ρ_i = A^{-1}·[-δ_{ij}·M_i·β + M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β]
               = A^{-1}·[M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β] - δ_{ij}·A^{-1}·M_i·β
```

## Final Formula

```
∂²β/∂ρ_j∂ρ_i = A^{-1}·[M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β] - δ_{ij}·A^{-1}·M_i·β
```

Note that `A^{-1}·M_i·β = -∂β/∂ρ_i`, so:
```
∂²β/∂ρ_j∂ρ_i = A^{-1}·[M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β] + δ_{ij}·(∂β/∂ρ_i)
```

## The Bug!

**Current implementation** (src/reml.rs lines 969-981):
```rust
let mut d2beta_term = Array1::zeros(p);
d2beta_term += &m_i_a_inv_m_j_beta;
d2beta_term += &m_j_a_inv_m_i_beta;
let d2beta = a_inv.dot(&d2beta_term);
```

**Missing**: The diagonal correction `+ δ_{ij}·(∂β/∂ρ_i)`!

## Corrected Implementation

```rust
let mut d2beta_term = Array1::zeros(p);
d2beta_term += &m_i_a_inv_m_j_beta;
d2beta_term += &m_j_a_inv_m_i_beta;
let mut d2beta = a_inv.dot(&d2beta_term);

// Add diagonal correction: + δ_{ij}·dβ/dρ_i
if i == j {
    d2beta += &dbeta_drho[i];
}
```

## Why This Matters

At diagonal entries (i==j), the missing term is `dβ/dρ_i`, which can be significant.

For example, if `||dβ/dρ_i|| ≈ 0.1`, this is ~1000x larger than the d²β terms we're seeing (~0.0001).

This could explain why:
- bSb2 term1 (d²β) is tiny (~0.006)
- But we're still converging suboptimally
- The diagonal is where d²β matters most!

## Expected Impact

With this fix:
- d²β diagonal terms will increase significantly
- bSb2 term1 contribution will increase
- Should improve convergence toward optimal λ
- Gradient should decrease faster at later iterations
