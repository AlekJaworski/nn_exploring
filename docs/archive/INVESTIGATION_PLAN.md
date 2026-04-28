# Investigation Plan: Making mgcv_rust Match R mgcv

## Current Discrepancies

| Basis | Our λ  | mgcv λ | Ratio | Status |
|-------|--------|--------|-------|--------|
| BS    | 75.86  | 16.47  | 4.6x  | Better, but still off |
| CR    | 0.011  | 16.47  | 0.0007x | Still broken! |

## Questions to Answer

### 1. Penalty Matrix Scaling in mgcv
- Does mgcv normalize penalties differently per basis type?
- Do they use design-matrix-relative scaling: `norm(S) / norm(X)²`?
- Do they have basis-specific scaling factors?

### 2. Why is CR Still 1500x Off?
- CR uses analytical natural cubic spline formulas (should be correct!)
- But lambda is identical to the old broken values
- Suggests the CR formula itself might be wrong or using wrong scale

### 3. Extended Knot Vector
- Are we creating the extended knot vector correctly?
- mgcv uses specific boundary knot repetition schemes
- Could affect the penalty integral limits

### 4. Penalty Rank and Constraints
- mgcv may apply absorption/reparameterization
- Removes null space (polynomial terms) before optimization
- Could affect effective penalty scaling

## Proposed Investigation Steps

1. **Extract actual penalty matrix from R mgcv** (if possible with rpy2)
2. **Compare matrix values element-by-element** with our implementation
3. **Check mgcv source code** for normalization/scaling details
4. **Test with different k values** - does the ratio stay constant?
5. **Profile: does lambda × penalty give same effective smoothing?**

## What This Would Give You

If we get lambda values matching mgcv exactly:
- ✅ Drop-in replacement for R mgcv in Python
- ✅ Same smoothing parameter estimates
- ✅ Directly comparable results
- ✅ Can use established mgcv tuning guidelines

## Recommendation

Let me investigate the mgcv source code and try to extract actual penalty
matrices from R for comparison. This should reveal the missing piece.

Then you can test with confidence that the implementation is truly mgcv-compatible.
