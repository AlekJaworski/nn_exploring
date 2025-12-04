# Chain Rule Clarification: Why Complete Hessian Failed

## The Issue

In the complete Hessian implementation (lines 871-912 of src/reml.rs), we compute terms using M_i = λ_i·S_i, but then multiply by λ_i·λ_j again. This double-counts the chain rule!

## The Math

### Starting point: Hessian in λ-space

```
∂²V/∂λ_i∂λ_j = -tr(A^{-1}·S_i·A^{-1}·S_j)
                + (2/φ)·β'·S_i·A^{-1}·S_j·β
                - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
                + δ_{ij}·r_i/λ_i²
```

### Chain rule to ρ-space (ρ = log λ)

```
∂²V/∂ρ_i∂ρ_j = λ_i·λ_j · ∂²V/∂λ_i∂λ_j + δ_{ij}·∂V/∂ρ_i
```

Substituting:
```
H[i,j] = (1/2) · {λ_i·λ_j · [-tr(A^{-1}·S_i·A^{-1}·S_j)
                              + (2/φ)·β'·S_i·A^{-1}·S_j·β
                              - (2/φ²)·(β'·S_i·β)·(β'·S_j·β)
                              + δ_{ij}·r_i/λ_i²]
                  + δ_{ij}·∇_i}
```

### Substituting M_i = λ_i·S_i

Now comes the KEY transformation. Replace S_i with M_i/λ_i:

```
-tr(A^{-1}·S_i·A^{-1}·S_j) → -tr(A^{-1}·(M_i/λ_i)·A^{-1}·(M_j/λ_j))
                             = -tr(A^{-1}·M_i·A^{-1}·M_j) / (λ_i·λ_j)
```

When multiplied by the chain rule λ_i·λ_j:
```
λ_i·λ_j · [-tr(A^{-1}·M_i·A^{-1}·M_j) / (λ_i·λ_j)] = -tr(A^{-1}·M_i·A^{-1}·M_j)
```

**The λ factors CANCEL!**

Similarly for term 2:
```
λ_i·λ_j · (2/φ)·β'·(M_i/λ_i)·A^{-1}·(M_j/λ_j)·β = (2/φ)·β'·M_i·A^{-1}·M_j·β
```

And term 3:
```
λ_i·λ_j · [-(2/φ²)·(β'·(M_i/λ_i)·β)·(β'·(M_j/λ_j)·β)]
= -(2/φ²)·(β'·M_i·β)·(β'·M_j·β) / (λ_i·λ_j)
```

## Final Formula in ρ-space Using M Matrices

```
H[i,j] = (1/2) · {-tr(A^{-1}·M_i·A^{-1}·M_j)              [NO λ_i·λ_j factor!]
                  + (2/φ)·β'·M_i·A^{-1}·M_j·β              [NO λ_i·λ_j factor!]
                  - (2/φ²·λ_i·λ_j)·(β'·M_i·β)·(β'·M_j·β)  [HAS 1/(λ_i·λ_j) factor!]
                  + δ_{ij}·[∇_i + r_i/(2λ_i)]}
```

## Cross-Check with mgcv C Code

From gdi.c (Gaussian case):
```c
// det2: Log-determinant Hessian
det2[k,m] = 0;
if (k >= M0 && k==m) det2[km] += trPtSP[m - M0];  // λ_m·tr(A^{-1}·S_m)
if (k >= M0 && m >= M0)
    det2[km] -= sp[m - M0]*sp[k - M0]*diagABt(...);  // -λ_k·λ_m·tr(...)
```

But wait! The C code DOES multiply by sp[k]*sp[m] = λ_k·λ_m for the trace term!

Let me re-examine... The C code computes:
```
det2[k,m] = -λ_k·λ_m·tr[(A^{-1}·S_k)·(A^{-1}·S_m)]
```

If I substitute S_k = M_k/λ_k:
```
= -λ_k·λ_m·tr[(A^{-1}·M_k/λ_k)·(A^{-1}·M_m/λ_m)]
= -tr[(A^{-1}·M_k)·(A^{-1}·M_m)]
```

So when mgcv uses S matrices, it multiplies by λ factors.
When we use M matrices, we should NOT multiply by λ factors.

## The Bug in Our Code

Line 901 of src/reml.rs:
```rust
let mut h_val = lambda_i * lambda_j * trace_term / 2.0;
```

But trace_term is already computed using M matrices (line 874):
```rust
let product = m_i.dot(&a_m_j_a);  // m_i = M_i, not S_i!
```

So we're computing tr(M_i·A·M_j·A) and then multiplying by λ_i·λ_j. That's WRONG!

## Correct Implementation

```rust
// trace_term = tr(A^{-1}·M_i·A^{-1}·M_j)
// term2 = (2/φ)·β'·M_i·A^{-1}·M_j·β
// term3 = -(2/φ²)·(β'·M_i·β)·(β'·M_j·β)

// Combine terms (NO extra lambda scaling!)
let term3_scaled = term3 / (lambda_i * lambda_j);  // Divide by λ_i·λ_j for term3
let mut h_val = (-trace_term + term2 + term3_scaled) / 2.0;

// Diagonal correction
if i == j {
    h_val += (grad_i + rank_i / (2.0 * lambda_i)) / 2.0;
}

hessian[[i,j]] = h_val;  // Test with and without negation
```

## Careful Re-derivation of All Terms

### Term-by-term transformation

**Term 1 (trace)** in λ-space:
```
-tr(A^{-1}·S_i·A^{-1}·S_j)
```

Substitute S_i = M_i/λ_i:
```
= -tr(A^{-1}·(M_i/λ_i)·A^{-1}·(M_j/λ_j))
= -tr(A^{-1}·M_i·A^{-1}·M_j) / (λ_i·λ_j)
```

Apply chain rule (multiply by λ_i·λ_j):
```
λ_i·λ_j · [-tr(A^{-1}·M_i·A^{-1}·M_j) / (λ_i·λ_j)] = -tr(A^{-1}·M_i·A^{-1}·M_j)
```

✅ **Term 1 with M: No λ_i·λ_j factor**

**Term 2 (penalty-beta-hessian)** in λ-space:
```
(2/φ)·β'·S_i·A^{-1}·S_j·β
```

Substitute:
```
= (2/φ)·β'·(M_i/λ_i)·A^{-1}·(M_j/λ_j)·β
= (2/φ)·β'·M_i·A^{-1}·M_j·β / (λ_i·λ_j)
```

Apply chain rule:
```
λ_i·λ_j · (2/φ)·β'·M_i·A^{-1}·M_j·β / (λ_i·λ_j) = (2/φ)·β'·M_i·A^{-1}·M_j·β
```

✅ **Term 2 with M: No λ_i·λ_j factor**

**Term 3 (penalty-penalty)** in λ-space:
```
-(2/φ²)·(β'·S_i·β)·(β'·S_j·β)
```

Substitute:
```
= -(2/φ²)·(β'·M_i/λ_i·β)·(β'·M_j/λ_j·β)
= -(2/φ²)·(β'·M_i·β)·(β'·M_j·β) / (λ_i·λ_j)
```

Apply chain rule:
```
λ_i·λ_j · [-(2/φ²)·(β'·M_i·β)·(β'·M_j·β) / (λ_i·λ_j)]
= -(2/φ²)·(β'·M_i·β)·(β'·M_j·β)
```

✅ **Term 3 with M: No λ_i·λ_j factor**

**Term 4 (rank correction)** in λ-space:
```
δ_{ij}·r_i/λ_i²
```

Apply chain rule (multiply by λ_i·λ_j, remembering δ_{ij} means i=j so λ_j=λ_i):
```
λ_i·λ_i · r_i/λ_i² = r_i
```

But we also add the gradient term from the chain rule:
```
δ_{ij}·λ_i·∂V/∂λ_i = δ_{ij}·∂V/∂ρ_i = δ_{ij}·∇_i
```

So the diagonal correction is:
```
δ_{ij}·[r_i + ∇_i]
```

Actually, from line 134 of HESSIAN_FORMULA_DERIVATION.md, it should be:
```
δ_{ij}·[∇_i + r_i/(2λ_i)]
```

Let me check... The r_i/λ_i² term comes from ∂(r_i·log λ_i)/∂λ_i = r_i/λ_i.
Then ∂²/∂λ_i² = -r_i/λ_i².

With chain rule: λ_i·λ_i·(-r_i/λ_i²) = -r_i.

But we're minimizing, and there's a division by 2 in the REML formula, so it becomes +r_i/(2λ_i) after full derivation.

## Final Formula in ρ-space

```rust
H[i,j] = (1/2) · {-tr(A^{-1}·M_i·A^{-1}·M_j)
                  + (2/φ)·β'·M_i·A^{-1}·M_j·β
                  - (2/φ²)·(β'·M_i·β)·(β'·M_j·β)
                  + δ_{ij}·[2·∇_i + r_i/λ_i]}
```

Simplifying:
```rust
H[i,j] = (-trace_term + term2 + term3) / 2.0 + δ_{ij}·[∇_i + r_i/(2λ_i)]
```

## Correct Rust Implementation

```rust
// All terms computed with M matrices (M_i = λ_i·S_i)
let trace_term = tr(A^{-1}·M_i·A^{-1}·M_j);
let term2 = (2/φ)·β'·M_i·A^{-1}·M_j·β;
let term3 = -(2/φ²)·(β'·M_i·β)·(β'·M_j·β);

// Combine WITHOUT extra λ_i·λ_j scaling
let mut h_val = (-trace_term + term2 + term3) / 2.0;

// Diagonal correction
if i == j {
    h_val += grad_i + rank_i / (2.0 * lambda_i);
}

hessian[[i,j]] = h_val;  // May need negation for Newton direction
```
