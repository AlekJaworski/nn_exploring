# CRITICAL BUG: Wrong φ (Scale Parameter) Computation

## Current Implementation (WRONG!)

From `src/reml.rs` lines 847-852:
```rust
// Compute total rank for φ
let mut total_rank = 0;
for penalty in penalties.iter() {
    total_rank += estimate_rank(penalty);
}
let phi = rss / (n - total_rank) as f64;
```

This computes:
```
φ = RSS / (n - Σrank(S_i))
```

Where `rank(S_i)` is the rank of penalty matrix i (independent of λ!).

## Why This Is Wrong

### 1. φ Should Depend on λ!

The scale parameter φ estimates the residual variance **after accounting for smoothing**.

The effective degrees of freedom (edf) **changes with λ**:
- Small λ → less penalty → more flexible fit → higher edf
- Large λ → more penalty → smoother fit → lower edf

But `rank(S_i)` is constant - it doesn't change with λ!

### 2. Correct Formula

From Wood (2011) and mgcv documentation:

```
φ = RSS / (n - edf)
```

where:
```
edf = tr(A^{-1}·X'WX)
```

and A is the penalized information matrix:
```
A = X'WX + Σλ_i·S_i
```

**Key point**: edf depends on λ through A^{-1}!

### 3. Impact on Hessian

Since we divide bSb1 and bSb2 by φ:
```rust
bsb1.push((lambda_i * beta_s_i_beta + 2.0 * dbeta_s_beta) / phi);
```

If φ is wrong, then:
- bSb1 is wrong by a factor
- bSb2 is wrong by the same factor
- Hessian is wrong
- Newton steps are wrong size
- Convergence to wrong λ!

### 4. Example: Our Case

At λ ≈ [4.11, 2.32]:
- Our φ uses `total_rank` = rank(S_1) + rank(S_2) ≈ 8 + 8 = 16
- `n = 100`
- Our φ ≈ RSS / (100 - 16) = RSS / 84

At mgcv's optimal λ = [5.69, 5.20]:
- Correct edf = tr(A^{-1}·X'WX) ≈ different value!
- Should be: φ = RSS / (100 - edf)

If edf ≠ 16, then our φ is systematically wrong!

## Hypothesis: This Explains Convergence Issues

**If our φ is too large** (edf < total_rank):
- We divide bSb2 by too-large φ
- Hessian bSb2 contribution becomes too small
- Hessian underestimates curvature
- Newton steps too large
- Converge to wrong minimum ✓ **Matches our symptoms!**

**If our φ is too small** (edf > total_rank):
- Opposite problem
- Steps too small
- Slow convergence

## Verification Needed

1. Compute correct edf = tr(A^{-1}·X'WX) at each iteration
2. Compare against our `total_rank`
3. Check if the ratio explains our convergence error

## How mgcv Computes φ

Need to find in mgcv C source how they compute the scale parameter.

From Wood (2011) Section 2.1.6:
> The Pearson estimate of the scale parameter is:
> φ̂ = ||y - μ̂||² / (n - edf)

From Wood's book Section 4.8.4:
> edf is given by the trace of the influence/hat matrix

For penalized regression:
> F = A^{-1}·X'W (the "influence matrix")
> edf = tr(F·X) = tr(A^{-1}·X'WX)

## Correct Implementation

We have access to `a_inv` (A^{-1}) and `x` matrix. We need to:

```rust
// Compute effective degrees of freedom
// edf = tr(A^{-1}·X'WX)
let xtwx = x.t().dot(&x.mapv(|_| 1.0));  // Simplified for W=I
let influence = a_inv.dot(&xtwx);
let edf: f64 = (0..influence.nrows())
    .map(|i| influence[[i, i]])
    .sum();

// Correct φ computation
let phi = rss / (n as f64 - edf);
```

## Action Items

1. ✅ Document the bug
2. ⚠️ Verify mgcv's φ computation from C source
3. ⚠️ Implement correct edf = tr(A^{-1}·X'WX)
4. ⚠️ Test convergence with corrected φ
5. ⚠️ Compare our edf vs total_rank at each iteration

## Expected Impact

If this is the root cause:
- Correcting φ should fix Hessian scaling
- Newton steps should become correct size
- Should converge to optimal λ = [5.69, 5.20]
- Should match mgcv's trajectory

**This could be THE bug we've been looking for!**
