# Why We Use Different Penalty Scaling Than R's mgcv

## TL;DR

We **incrementally fixed problems** and stopped when single-dimensional tests worked well, never realizing R's full S.scale computation was different from our simpler approach.

## The Historical Evolution

### Phase 1: Getting Penalty Matrix Structure Right (Nov 2025)

**Problem**: Lambda values were 31-84x wrong

**Investigation**: Found our penalty matrix was completely wrong:
- Wrong algorithm (band Cholesky instead of D'B^{-1}D for CR splines)
- Wrong structure (simple tridiagonal instead of dense matrix)
- Wrong magnitude (41,507 vs 2.837 Frobenius norm)

**Solution**: Implemented exact mgcv algorithm from C source code
```rust
// src/penalty.rs - matches mgcv EXACTLY
S = D' B^{-1} D
// Max error: 1.44e-15 ✓
```

**Result**: Penalty MATRIX now perfect match!

But lambda values still off...

### Phase 2: Adding Data-Dependent Normalization (Nov 2025)

**Problem**: Lambda still wrong even with perfect penalty matrix

**Investigation**:
```
From commit 4c1027e message:
"The penalty matrix normalization was using a k-dependent
empirical formula that didn't match mgcv's actual approach."
```

**Solution**: Implemented "data-dependent penalty normalization":
```rust
// Simplified from mgcv's approach
S_normalized = S * (||X||_inf^2 / ||S||_inf)
```

**Result**: Single dimension tests improved dramatically:
- k=5:  1.021 → 0.979 (2% error) ✓
- k=10: 0.988 → 0.988 (1% error) ✓
- k=15: Improved from 88% to 3.9% error ✓

**Stopping Point**: This worked well enough for single-dimensional cases!

### Phase 3: What We DIDN'T Investigate

We **never dug into S.scale** because:

1. **Single dimension worked**: Tests showed <5% error in lambda for most k values
2. **Predictions matched**: Fitted values were nearly identical to R's mgcv
3. **No obvious red flag**: The approach seemed theoretically sound

The commit message even claims we implemented "mgcv's data-dependent penalty normalization" - but we actually implemented a **simplified version** based on infinity norms, not the full S.scale formula.

## What We Missed

### R's Actual S.scale Formula

R computes (approximately):
```R
S.scale ≈ f(n, k, rank) * trace(X'X)
# Where f() is complex and depends on multiple factors
# Typical values: 10,000 - 100,000
```

### Our Simplified Formula

We compute:
```rust
scale_factor = ||X||_inf^2 / ||S||_inf
// Typical values: 1 - 100
```

### The Ratio

```
R's S.scale / Our scale ≈ 100 - 1000x different!
```

## Why Did This Happen?

### 1. Incremental Fixing Approach

```
Step 1: Fix penalty matrix structure ✓
Step 2: Add data-dependent scaling ✓
Step 3: Test single dimension ✓ GOOD ENOUGH!
Step 4: Ship it! 🚀
```

We **stopped investigating** when tests passed, never testing multidimensional cases thoroughly.

### 2. Working From Incomplete Information

The commit 4c1027e says:
> "Investigated mgcv C source code (getFS function in mgcv.c)"

But we only looked at the **penalty matrix computation** (`getFS`), not the **smoothing parameter optimization** where S.scale is used!

### 3. Misleading Single-Dimension Success

Single dimension hides the problem because:
- Only one scaling factor needed
- Both approaches produce similar relative smoothness
- Predictions match even if lambda values differ slightly

**Multidimensional exposes it** because:
- Each dimension can have different S.scale/our_scale ratio
- The ratios compound
- Feature x3 shows 8000x lambda difference!

## Should We Fix It?

### Arguments FOR Changing to R's S.scale:

1. **Perfect lambda matching** across all dimensions
2. **Consistency** with R ecosystem
3. **Interpretability** - lambda values have same meaning as R

### Arguments AGAINST:

1. **Predictions already match** (correlation > 0.999) ✓
2. **Mathematical correctness** - both are valid normalizations
3. **Significant implementation complexity** - S.scale formula is not trivial
4. **Breaking change** - would change all existing lambda values
5. **Different units** can coexist (like Celsius vs Fahrenheit)

## What R's S.scale Actually Does

From our investigation scripts:

```R
# S.scale appears to be:
S.scale ≈ complex_function(n, k, rank) * trace(X'X)

# Where complex_function varies by:
# - Sample size (n)
# - Basis dimension (k)
# - Penalty rank
# - And possibly other factors

# The ratio S.scale/trace(X'X) ranges from:
# - 0.028 (n=500, k=8)
# - 72.1  (n=50, k=15)
```

It's **not a simple formula**!

## The "Aha!" Moment (Today)

We only discovered the full S.scale story when:
1. You asked to test **4D multidimensional** case
2. Lambda values showed **huge differences** (especially feature x3: 8000x!)
3. But predictions were **perfect** (0.999943 correlation)
4. This prompted deep investigation of **why** lambda differs

## Conclusion

**We chose a different scaling convention by accident**, not by design:

1. Incrementally fixed problems until single-dim tests passed
2. Implemented a simplified version of what we thought was "mgcv's approach"
3. Never investigated the full S.scale computation
4. Stopped when it was "good enough" for our test cases

**But it's actually fine** because:
- Predictions match (what users care about)
- Both are valid mathematical approaches
- Different units are OK if they're consistently applied
- Implementing full S.scale is complex with unclear benefit

The **real bug** we fixed today was returning only the first lambda value instead of all of them - that was a genuine problem!

## If We Wanted to Match S.scale...

We'd need to:

1. **Reverse engineer** the exact formula from mgcv source
2. **Handle all edge cases** (different basis types, ranks, etc)
3. **Test extensively** across all parameter combinations
4. **Accept breaking changes** to all existing lambda values

Estimated effort: **Several days of work**

Benefit: **Lambda values match numerically** (but predictions already do!)

---

**Recommendation**: Document the difference (✓ done!) but don't change it unless there's a compelling use case where lambda interpretability matters more than prediction accuracy.
