# bSb1 Formula Verification Against mgcv C Source

## mgcv C Code Analysis

### First Part: Compute λ_k·β'·S_k·β

```c
for (p1=Skb,rSoff=0,i=0;i<*M;i++) {
   /* form S_k \beta * sp[k]... */
   bt=1;ct=0;mgcv_mmult(work,rS + rSoff ,beta,&bt,&ct,rSncol+i,&one,q);
   for (j=0;j<rSncol[i];j++) work[j] *= sp[i];  // sp[i] = λ_i
   bt=0;ct=0;mgcv_mmult(p1,rS + rSoff ,work,&bt,&ct,q,&one,rSncol+i);

   /* now the first part of the first derivative */
   for (xx=0.0,j=0;j<*q;j++,p1++) xx += beta[j] * *p1;
   bSb1[i + *M0] = xx;  // = β'·(λ_i·S_i·β)
}
```

Result: `bSb1[i] = λ_i·β'·S_i·β`

### Second Part: Add 2·(dβ/dρ_k)'·(S·β)

```c
/* Now finish off the first derivatives */
bt=1;ct=0;mgcv_mmult(work,b1,Sb,&bt,&ct,&Mtot,&one,q);
// work[k] = (dβ/dρ_k)'·(S·β)
for (i=0;i<Mtot;i++) bSb1[i] += 2*work[i];
```

Result: `bSb1[i] += 2·(dβ/dρ_i)'·(S·β)`

### Complete mgcv Formula

```
bSb1[k] = λ_k·β'·S_k·β + 2·(dβ/dρ_k)'·(S·β)
```

This is the **first derivative of β'·S·β** with respect to ρ_k = log(λ_k).

## Our Implementation

From `src/reml.rs` lines 866-890:

```rust
// β'·S_i·β
let s_i_beta = penalty_i.dot(&beta);
let beta_s_i_beta: f64 = beta.iter().zip(s_i_beta.iter())
    .map(|(b, sb)| b * sb)
    .sum();

// 2·dβ/dρ_i'·S·β where S = Σλ_j·S_j
let mut s_beta_total = Array1::zeros(p);
for (lambda_j, penalty_j) in lambdas.iter().zip(penalties.iter()) {
    let s_j_beta = penalty_j.dot(&beta);
    s_beta_total.scaled_add(*lambda_j, &s_j_beta);
}
let dbeta_s_beta: f64 = dbeta_drho[i].iter().zip(s_beta_total.iter())
    .map(|(db, sb)| db * sb)
    .sum();

bsb1.push((lambda_i * beta_s_i_beta + 2.0 * dbeta_s_beta) / phi);
```

Our formula: `bsb1[i] = (λ_i·β'·S_i·β + 2·dβ/dρ_i'·S·β) / φ`

## Comparison

| Component | mgcv | Our Implementation | Match? |
|-----------|------|-------------------|--------|
| First term | λ_k·β'·S_k·β | λ_i·β'·S_i·β | ✅ |
| Second term | 2·(dβ/dρ_k)'·(S·β) | 2·dβ/dρ_i'·S·β | ✅ |
| Division by φ | **NOT HERE** | **/ φ** | ⚠️ |

## CRITICAL FINDING: Where does φ appear?

### In REML Criterion

The REML criterion has the term:
```
-½·β'·S·β/φ
```

So the derivative is:
```
∂(-½·β'·S·β/φ)/∂ρ_k = (-1/2φ)·∂(β'·S·β)/∂ρ_k
```

And the second derivative:
```
∂²(-½·β'·S·β/φ)/∂ρ_k∂ρ_m = (-1/2φ)·∂²(β'·S·β)/∂ρ_k∂ρ_m
```

### mgcv's Approach

mgcv computes:
- `bSb` = β'·S·β (the raw penalty)
- `bSb1[k]` = ∂(β'·S·β)/∂ρ_k (derivative of raw penalty)
- `bSb2[k,m]` = ∂²(β'·S·β)/∂ρ_k∂ρ_m (second derivative of raw penalty)

Then **elsewhere in the REML calculation**, it multiplies by -1/(2φ).

### Our Approach

We compute:
- `bsb1[i]` = ∂(β'·S·β/φ)/∂ρ_i = (1/φ)·∂(β'·S·β)/∂ρ_i
- `bsb2[i,j]` = ∂²(β'·S·β/φ)/∂ρ_i∂ρ_j = (1/φ)·∂²(β'·S·β)/∂ρ_i∂ρ_j

We're dividing by φ **inside the bSb computation** rather than later.

## Is Our Approach Correct?

**YES, as long as:**
1. We use the correct φ estimate
2. We don't divide by φ again elsewhere
3. The formula matches mgcv's formula times 1/φ

Let me verify: mgcv's bSb1 times 1/φ should equal our bsb1:
```
mgcv: bSb1[k] = λ_k·β'·S_k·β + 2·(dβ/dρ_k)'·(S·β)
ours: bsb1[k] = (λ_k·β'·S_k·β + 2·(dβ/dρ_k)'·(S·β)) / φ
      ✅ Correct! Our formula = mgcv's formula / φ
```

## Question: What φ value do we use?

Need to check in our code what value of `phi` we're using in the bsb1/bsb2 computation.

From REML theory:
```
φ = ||y - Xβ||²/(n - effective_df)
```

But during Newton iteration, φ changes because β changes with λ!

**Hypothesis**: We might be using the wrong φ value (e.g., from previous iteration or initial estimate).

## Next Steps

1. ✅ Verified bSb1 formula matches mgcv (modulo φ scaling)
2. ⚠️ Need to check: What φ value are we using?
3. ⚠️ Need to check: Does φ update correctly during Newton iteration?
4. ⚠️ Need to verify: mgcv's φ at each iteration vs ours
