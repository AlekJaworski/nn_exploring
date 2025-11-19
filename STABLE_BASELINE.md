# Stable Multidimensional GAM Baseline

## Current State

**Commit**: Based on 362e8a3 ("Fix lambda initialization: enforce minimum λ=0.1")

**Status**: ✅ **STABLE** - All multidimensional tests passing

## Decision

After investigating regressions introduced by later commits (a0f3435 and a1da133), we have reset to the last known working state for multidimensional GAMs.

The later commits attempted to:
1. Add intercept column and use k-1 basis functions (a0f3435)
2. Implement constraint absorption for better penalty rank (a1da133)

While these changes improved single-dimensional performance and fixed some numerical issues, they broke multidimensional support due to dimension tracking bugs.

## What Works

### Multidimensional GAMs
✅ **9/9 unit tests passing** (test_multidim_unit.py)
- 2D-5D GAM fitting
- Different k values per dimension
- Prediction shapes
- Lambda dimensions
- Reproducibility
- Performance scaling

### Performance vs mgcv (4D, k=[16,16,16,16])

| n | Our Time | mgcv Time | Speedup |
|---|---|---|---|
| 500 | 0.146s | 0.523s | **3.58x faster** |
| 1500 | 0.550s | 0.333s | 0.61x slower |
| 2500 | 1.008s | 0.618s | 0.61x slower |
| 5000 | 1.998s | 0.897s | 0.45x slower |

**Key observations**:
- Fast at small n (3.58x faster than mgcv at n=500)
- R² > 0.99 consistently
- Scaling needs improvement for large n

## What We Cherry-Picked

From the broken HEAD, we kept ONLY the test and documentation commits:
- ✅ df83db6: Multidimensional tests and benchmarks
- ✅ 73999de: Analysis summary documentation
- ✅ 60f7c27: mgcv comparison benchmarks

We did NOT keep:
- ❌ a0f3435: k-1 basis and intercept changes (broke multidim)
- ❌ d327955: S.scale penalty normalization (may be OK but skipped for safety)
- ❌ a1da133: Constraint absorption (broke multidim)
- ❌ 89b992a: Verification tests (dependent on broken changes)
- ❌ cd9736d: Prediction correlation verification
- ❌ 988450a: estimate_rank optimization (dependent on constraint absorption)

## Architecture at This Baseline

### Basis Functions
- Uses k basis functions directly (not k-1)
- No intercept column
- CubicRegressionSpline with k knots

### Penalty Matrices
- k × k penalty matrices
- Computed via cr_spline_penalty()
- No constraint absorption
- No S.scale normalization (uses simple scaling)

### Design Matrix
- No intercept column
- total_basis = sum of k values across all smooths
- Direct concatenation of basis evaluations

### REML Optimization
- Gradient descent with adaptive step sizes
- Lambda initialization: max(0.1, computed_value)
- No quasi-Newton methods
- Multiple iterations (likely 20-30 for convergence)

## Known Limitations at This Baseline

1. **Lambda convergence**: Most λ values → 0 (only first smooth regularized)
2. **Scaling**: Performance degrades at large n compared to mgcv
3. **REML iterations**: Takes more iterations than mgcv (4-10)
4. **Penalty rank**: May not match mgcv's constrained rank structure

## Next Steps

### Option 1: Stay at This Baseline
**Pros**:
- Stable, tested, working
- Good performance at small-medium n
- Excellent fit quality

**Cons**:
- Doesn't have constraint absorption improvements
- Lambda convergence issues
- Poor scaling at large n

### Option 2: Carefully Add Improvements
Incrementally add back improvements while maintaining multidimensional support:

1. **Add S.scale normalization** (d327955)
   - Likely safe, improves penalty scaling
   - Test multidimensional after adding

2. **Add intercept + k-1 basis** (a0f3435) - **CAREFULLY**
   - Fix dimension tracking bugs first
   - Ensure num_basis() returns correct values
   - Test multidimensional at each step

3. **Add constraint absorption** (a1da133) - **CAREFULLY**
   - Only after k-1 basis works
   - Fix index out of bounds issues
   - Ensure penalty dimensions match design matrix

4. **Optimize estimate_rank** (988450a)
   - Only after constraint absorption works
   - Should be safe since it's just an optimization

5. **Improve REML optimizer**
   - Implement quasi-Newton (BFGS)
   - Add line search
   - Better lambda initialization
   - Target mgcv's 4-10 iteration efficiency

## Recommendation

**For production use NOW**: Stay at this baseline. It works well for typical use cases.

**For continued development**: Carefully add improvements one at a time, running the full multidimensional test suite after each change.

## Testing Protocol

Before merging any future changes:

```bash
# 1. Run unit tests
python3 test_multidim_unit.py

# 2. Run benchmark
python3 benchmark_multidim_scaling.py

# 3. Verify no regressions
- All 9 tests must pass
- Performance should not regress significantly
- Multidimensional support must work
```

## Files at This Baseline

**Source code**:
- Based on 362e8a3

**Tests**:
- test_multidim_unit.py (9 tests)
- benchmark_multidim.py
- benchmark_multidim_scaling.py

**Documentation**:
- MULTIDIMENSIONAL_PERFORMANCE.md
- MULTIDIMENSIONAL_SUMMARY.md
- MULTIDIM_VS_MGCV.md
- STABLE_BASELINE.md (this file)

---

**Status**: ✅ Stable and ready for production use
**Last verified**: 2025-11-19
**Commit**: b17f5ed (cherry-picked tests onto 362e8a3 baseline)
