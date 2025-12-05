# Multi-Dimensional GAM Bottleneck Analysis

## Summary

Testing multi-dimensional GAMs revealed a **critical numerical stability bug** that prevents proper optimization of smoothing parameters. The issue causes:
- All smoothing parameters to remain nearly identical
- Catastrophic numerical errors (EDF = 10^15 instead of ~21)
- Optimization failure (NaN REML values)
- 30% performance degradation compared to R

## Test Results

### Test Case: n=1000, 3 dimensions, k=10
True function: y = sin(2πx₁) + 0.5·cos(3πx₂) + 0.3·x₃² + noise

**R's mgcv (CORRECT):**
```
Lambdas: [5.39, 4.81, 3115.04]  ← Vary by 600x!
EDF: 21.08
REML: -139.58
Iterations: 6
Converged: full convergence
```

**Rust's mgcvrust (BROKEN):**
```
Lambdas: [0.21, 0.21, 0.21]  ← All identical!
EDF: 2277372502384396  ← Should be ~21!
REML: NaN
Iterations: 10
Converged: NO - reached max iterations
```

### Large-Scale Test: n=6000, 8-10 dimensions

**8 dimensions:**
- Rust: 1.12s (all λ ≈ 0.08)
- R: 1.26s (λ varies from 5 to 5000)
- Speedup: **1.13x** (Rust faster, but wrong results!)

**10 dimensions:**
- Rust: 2.10s (λ varies 0.02 to 90, but still wrong scale)
- R: 1.61s (λ varies from 5 to 5000)
- Speedup: **0.77x** (Rust slower AND wrong!)

---

## Root Causes

### Issue 1: Lambda Initialization (Minor)

**Code location:** `src/smooth.rs:62-108`

**Problem:**
```rust
// Compute trace(X'WX) over ENTIRE design matrix (all basis functions)
let mut xtwx_trace_per_n = 0.0;
for j in 0..p {  // ← p = total_basis (e.g., 30 for 3×10)
    ...
}

// Each lambda uses the SAME denominator
for (i, penalty) in penalties.iter().enumerate() {
    self.lambda[i] = 0.1 * penalty_trace / xtwx_trace_per_n;  // ← Same denom!
}
```

**Result:** All smooths with similar penalty structure (e.g., cubic splines with same k) get nearly identical initial lambdas.

**Impact:** LOW - Newton's method should overcome this if gradient/Hessian are correct.

---

### Issue 2: Numerical Instability in EDF Computation (CRITICAL)

**Evidence from debug output:**
```
[PHI_DEBUG] n=1000, edf_total=2277372502384396.000000  ← Should be ~21!
[QR_DEBUG] P·P' diagonal: [6.686135e27, 6.686135e27, ...]  ← Astronomical values!
[PROFILE] Newton step REML=NaN  ← Numerical collapse!
```

**Problem:** The EDF computation returns `10^15` instead of ~21, indicating catastrophic numerical error in the precision matrix P = (X'WX + λS)^(-1).

**Hypothesis:** When penalty matrices are block-diagonal (multi-dimensional case), the QR decomposition or matrix inversion becomes ill-conditioned, leading to P having astronomically large values.

**Code location:** Likely in `src/reml.rs` QR-based gradient computation for multi-dimensional case.

**Impact:** CRITICAL - Makes optimization impossible, prevents correct smoothing parameter selection.

---

### Issue 3: Gradient/Hessian Breakdown (CRITICAL)

**Evidence:**
```
[PROFILE] Newton failed, trying steepest descent
[PROFILE] SD scale=0.01: REML=NaN
[PROFILE] Steepest descent failed at all scales, stopping
```

**Problem:** Both Newton's method and steepest descent fail because:
1. Gradient/Hessian computation involves the broken EDF calculation
2. REML criterion returns NaN due to numerical overflow
3. Line search can't find valid direction

**Impact:** CRITICAL - Prevents any optimization progress.

---

## Performance Impact

### Why Rust is Slower for 10D:

1. **More iterations**: Rust takes 10 iterations (max) vs R's 6-7
   - Reason: Can't converge due to numerical issues
   - Each iteration computes broken gradients/Hessians

2. **Wrong optimization path**: With all λ ≈ 0.08, Rust is:
   - Under-smoothing dimensions that need high λ (like x³²)
   - Over-smoothing dimensions that need low λ
   - Computing unnecessary basis evaluations

3. **Scaling issue**: As dimensions increase:
   - Design matrix grows from 6000×80 to 6000×100
   - Block-diagonal penalty matrices get larger
   - Numerical instability gets worse
   - More wasted computation on wrong path

### Iteration Analysis:

**Cannot compute per-iteration breakdown** because:
- Rust doesn't return iteration count to Python
- Current implementation doesn't track iterations properly

**Estimated breakdown (10D case):**
- Rust: 2.10s / ~10 iters = **210ms/iter**
- R: 1.61s / 7 iters = **230ms/iter**

**Conclusion:** Per-iteration time is similar! The bottleneck is **iteration count** (10 vs 7) and **wrong convergence** (wrong final answer).

---

## What Needs to Be Fixed

### Priority 1: Fix EDF/Numerical Stability (CRITICAL)

**Diagnosis needed:**
1. Add debug output to trace where EDF calculation goes wrong
2. Check if issue is in:
   - QR decomposition of augmented matrix Z
   - Matrix inversion to compute P
   - Trace computation tr(P·P')

**Potential fixes:**
1. Use Cholesky decomposition instead of QR for better numerical stability
2. Add regularization/conditioning to prevent overflow
3. Compute EDF differently for block-diagonal penalties (exploit structure)

**Test:** Simple 2D case with debug output to isolate where numbers explode.

### Priority 2: Fix Lambda Initialization (LOW)

**Fix:**
```rust
// Compute trace(X'WX) PER SMOOTH, not for full matrix
for (i, (penalty, smooth)) in penalties.iter().zip(self.smooth_terms.iter()).enumerate() {
    let num_basis = smooth.num_basis();
    let start_col = i * num_basis;
    let end_col = start_col + num_basis;

    // Compute trace for THIS smooth's columns only
    let mut smooth_xtwx_trace = 0.0;
    for j in start_col..end_col {
        for i in 0..n {
            smooth_xtwx_trace += x[[i, j]] * x[[i, j]] * w[i];
        }
    }
    smooth_xtwx_trace /= n as f64;

    self.lambda[i] = 0.1 * penalty_trace / smooth_xtwx_trace;
}
```

**Impact:** Better starting point, but won't fix convergence issues.

### Priority 3: Add Iteration Tracking

**Fix:** Modify `optimize_reml_newton_multi` to return iteration count:
```rust
pub struct OptimizationResult {
    pub iterations: usize,
    pub converged: bool,
    pub final_reml: f64,
}
```

**Impact:** Enables proper performance analysis.

---

## Expected Performance After Fixes

Once numerical stability is fixed:

**8D case:**
- Current: 1.12s (wrong answer)
- Target: 0.9-1.1s (correct answer, ~7 iterations like R)

**10D case:**
- Current: 2.10s (wrong answer, 10 failed iterations)
- Target: 1.4-1.7s (correct answer, ~7 iterations like R)

**Speedup target:** 0.9-1.1x (within 10% of R)

---

## Debugging Strategy

### Step 1: Isolate Numerical Issue

Create minimal test case:
```rust
// 2D, n=100, k=5 - simplest case that fails
test_edf_computation_2d()
```

Add extensive logging:
- Print P matrix values
- Print intermediate QR results
- Track where values become >10^10

### Step 2: Compare to R's Approach

Study mgcv source code:
- How does R compute EDF for block-diagonal penalties?
- What numerical conditioning does R use?
- Does R use QR or Cholesky?

### Step 3: Implement Fix

Options:
1. Switch to Cholesky if more stable
2. Add diagonal regularization
3. Exploit block structure explicitly

### Step 4: Verify Fix

Tests:
1. EDF ~= 21 for 3D case (not 10^15)
2. Lambdas match R's scale and variation
3. REML converges to R's value
4. Performance within 10% of R

---

## Conclusion

The **primary bottleneck** for multi-dimensional GAMs is **numerical instability in the EDF/precision matrix computation**, not algorithmic inefficiency.

**Current status:**
- ✓ Single-dimensional GAMs: Working well, 1-36x faster than R
- ✗ Multi-dimensional GAMs: Numerically broken, produces wrong results

**To achieve performance target:**
1. Fix numerical stability (Priority 1)
2. Verify correct convergence
3. Then optimize per-iteration time if needed

**Estimated effort:** 4-8 hours to diagnose and fix the numerical issue.
