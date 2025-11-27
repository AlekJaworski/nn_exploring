# Fellner-Schall Algorithm: Theory and Implementation Notes

## Sources

**Primary Papers:**
1. [Wood (2011) "Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models"](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2010.00749.x) JRSSB 73(1):3-36
2. [Wood & Fasiolo (2017) "A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models"](https://onlinelibrary.wiley.com/doi/full/10.1111/biom.12666) Biometrics 73(4):1071-1081
3. [arXiv version of 2017 paper](https://arxiv.org/abs/1606.04802)

**Implementation:**
- R package `mgcv` by Simon Wood
- `bam()` function with `method="fREML"` uses Fellner-Schall

## Key Concepts from Literature

### 1. Penalty Matrix Structure

For a smooth term with `k` basis functions:
- **Penalty matrix S**: (k × k) matrix penalizing roughness
- **Penalty rank**: Number of penalized parameters (non-zero eigenvalues of S)
- **Null space dimension**: Unpenalized polynomial part

For cubic regression splines (bs="cr"):
- k = number of basis functions
- rank = k - 2 (penalizes 2nd derivatives)
- null.space.dim = 2 (linear + constant unpenalized)
- **rank + null.space.dim = k**

### 2. Effective Degrees of Freedom (EDF)

For the full model:
```
EDF = tr(X(X'WX + Σ λᵢSᵢ)⁻¹X'W)
    = tr((X'WX + Σ λᵢSᵢ)⁻¹X'WX)   [by cyclic property of trace]
```

For individual smooth term i:
```
EDFᵢ = tr((X'WX + Σ λⱼSⱼ)⁻¹Sᵢ) ???
```

**Key question**: What trace quantity does Fellner-Schall actually compute?

### 3. Fellner-Schall Update (from general literature)

Original Fellner (1986) and Schall (1991) for variance components:
- Simple explicit updates that increase restricted likelihood
- Faster than EM algorithm
- Requires only 1st and 2nd derivatives (not 3rd/4th)

Wood's generalization:
- Handles penalties linear in multiple smoothing parameters
- Covers tensor products and adaptive smoothers
- Alternates coefficient estimation with smoothing parameter updates

**Generic form** (needs verification from paper):
```
λᵢ_new = λᵢ_old × f(trace_i, rank_i)
```

Where f() is some function of a trace quantity and penalty rank.

### 4. Our Current Implementation Questions

**What we do:**
```rust
trace = tr((X'WX + Σ λⱼSⱼ)⁻¹ · Sᵢ)
ratio = trace / rank_i
λᵢ_new = λᵢ_old × ratio^damping
```

**Problems identified:**
1. After penalty normalization (S → c·S where c ≈ 0.000078):
   - trace scales: tr(A⁻¹·(cS)) = c·tr(A⁻¹·S)
   - But rank stays constant (algebraic rank)
   - Ratio becomes ≈ 0.13 instead of ≈ 1.0

2. At R's optimal λ ≈ 1.2:
   - We observe trace ≈ 1.07
   - rank = 8
   - ratio = 0.13 → algorithm thinks λ too high → decreases λ

3. Without normalization:
   - ratio ≈ 1.0 (correct)
   - But numerical accuracy breaks (correlation 0.58)

## Critical Unknowns

### Need to find from Wood (2011) or Wood & Fasiolo (2017):

1. **Exact update formula**: What is f() in λᵢ_new = λᵢ_old × f(trace, rank)?

2. **Trace definition**: Is it tr((X'WX + ΣλⱼSⱼ)⁻¹Sᵢ) or something else?

3. **Expected ratio**: What should trace/rank converge to?
   - Is it always 1.0?
   - Does it depend on penalty scaling?
   - Is "rank" the algebraic rank or something else?

4. **Penalty normalization**:
   - Does mgcv normalize penalties before or after Fellner-Schall?
   - If before, how do they adjust the target ratio?
   - Our normalization: S → S · (||X||²∞ / ||S||∞)

5. **EDF vs trace relationship**:
   - How does tr(A⁻¹Sᵢ) relate to EDFᵢ?
   - Are they the same thing?
   - If not, what's the connection?

6. **Convergence criterion**:
   - Do they check |Δλ| < ε?
   - Or |ΔREML| < ε?
   - Or something else?

## Observations from R's mgcv

### Test case: n=500, d=3, k=10 per smooth

**R's bam() results:**
- λ = [1.23, 1.64, 470.0]
- EDF per smooth ≈ [1.0, 0.99, 0.97] (very penalized!)
- Total EDF ≈ 22.3
- Converges in ~7 iterations

**Our results (with normalization):**
- λ = [10⁻⁷, 10⁻⁷, 10⁻⁷] (lower bound)
- Converges in 22 iterations
- Correlation >0.999 (fits are correct!)
- trace ≈ 1.07, rank = 8, ratio ≈ 0.13

**Key insight**: Despite wrong λ values, our fits match R! This suggests:
- Either fit insensitive to λ in range [10⁻⁷, 10]
- Or some other compensation occurring

### Penalty matrix after smoothCon

From R inspection:
```R
sm <- smoothCon(s(x, bs="cr", k=10), ...)[[1]]
sm$rank           # 8
sm$null.space.dim # 2
sum(diag(sm$S[[1]]))  # ≈ 6.29  (after their normalization!)
max(sm$S[[1]])    # ≈ 0.946
```

So R's penalty trace is ≈ 6.29, which matches our normalized penalty trace!
**This means mgcv DOES apply normalization before Fellner-Schall.**

## Hypothesis

If mgcv normalizes penalties and still gets ratio ≈ 1.0, they must be using a different "effective rank" that accounts for the normalization.

**Possible adjustments:**
1. effective_rank = algebraic_rank × (penalty_trace_after / penalty_trace_before)
2. effective_rank = something derived from EDF target
3. effective_rank = function of penalty matrix properties post-normalization

## Findings from mgcv Source Code (v1.8-42)

### Key Discovery: fREML uses Newton, not Fellner-Schall!

Examined `/tmp/mgcv/R/fast-REML.r` and `/tmp/mgcv/R/bam.r`:

**`bam(..., method="fREML")` implementation:**
1. Calls `fast.REML.fit()` (line 1352 in bam.r)
2. This function uses **Newton optimization** (lines 1549-1614 in fast-REML.r)
3. NOT the Fellner-Schall algorithm from Wood & Fasiolo (2017)!

**Newton optimization approach:**
```R
# Newton loop (fast-REML.r:1549-1614)
for (iter in 1:200) {
  # Compute gradient and Hessian
  grad <- t(L)%*%best$reml1
  hess <- t(L)%*%best$reml2%*%L

  # Newton step: -H^{-1} * grad
  eh <- eigen(hess)
  step <- - eh$vectors%*%((t(eh$vectors)%*%grad)/eh$values)

  # Line search with halving
  while (trial$reml > best$reml) step <- step/2

  # Update log smoothing parameters
  rho <- rho + step
}
```

**Convergence criterion:**
- Checks gradient: `abs(grad) < reml.scale * conv.tol`
- Checks REML change: `abs(best$reml - trial$reml) < reml.scale * conv.tol`
- Default `conv.tol = .Machine$double.eps^0.5 ≈ 1.5e-8`

### Implications for Our Implementation

1. **We implemented Fellner-Schall, but mgcv uses Newton!**
   - This explains different iteration counts (22 vs 7)
   - Newton converges faster but needs 2nd derivatives
   - Fellner-Schall is simpler but slower

2. **Wood & Fasiolo (2017) "extended Fellner-Schall":**
   - Not yet in mgcv 1.8-42 (released 2022)
   - Might be in newer versions or planned for future
   - Or maybe "fREML" name is historical, now uses Newton

3. **Penalty normalization in mgcv:**
   - Happens in `Sl.setup()` and related functions
   - Penalties are transformed/reparameterized
   - But actual optimization uses Newton, not ratio-based updates

### What This Means for Our Work

**Option 1: Switch to Newton optimization (like mgcv)**
- Pros: Faster convergence (7 iterations instead of 22)
- Cons: Need to compute Hessian (more complex), need 2nd derivatives

**Option 2: Keep Fellner-Schall but fix it**
- Pros: Simpler, matches original literature
- Cons: Need to solve the penalty normalization issue
- Must find correct "effective rank" after normalization

**Option 3: Hybrid approach**
- Use Fellner-Schall for initial iterations
- Switch to Newton for final refinement

## Next Steps (Revised)

1. **Decision point:** Newton vs Fellner-Schall?
   - If Newton: Study `fast.REML.fit()` implementation carefully
   - If Fellner-Schall: Need to solve normalization problem

2. **For Fellner-Schall fix:**
   - Still need Wood (2011) paper for theory
   - May need to examine older mgcv versions
   - Or implement without normalization and handle accuracy differently

3. **Test hypothesis:** Compare our Fellner-Schall vs mgcv Newton
   - Same test problem
   - Track iterations, λ values, REML scores
   - See if Newton really is that much faster

## References

- Wood S.N. (2011) Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society Series B*, 73(1):3-36.
- Wood S.N., Fasiolo M. (2017) A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4):1071-1081.
- Fellner W.H. (1986) Robust estimation of variance components. *Technometrics*, 28:51-60.
- Schall R. (1991) Estimation in generalized linear models with random effects. *Biometrika*, 78:719-727.
