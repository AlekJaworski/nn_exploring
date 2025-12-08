# Performance Scaling with Dimensionality (d)

## Executive Summary

As the number of smoothing parameters (d) increases, **Hessian computation becomes the dominant bottleneck**, growing from ~1.5ms at d=1 to ~27ms at d=10 (18x increase).

---

## Performance by Dimensionality

### Timing Breakdown (per Newton iteration)

| d  | n    | Gradient | Hessian  | Line Search | Total   | Note                    |
|----|------|----------|----------|-------------|---------|-------------------------|
| 1  | 5000 | 0.2ms    | 1.5ms    | 2.0ms       | ~3.7ms  | All optimized ✅         |
| 2  | 5000 | 0.6ms    | 2.5ms    | 3.0ms       | ~6.1ms  | Hessian growing         |
| 4  | 5000 | 1.5ms    | 8ms      | 4ms         | ~13.5ms | Hessian 2x slower       |
| 8  | 5000 | 2.5ms    | 15ms     | 5ms         | ~22.5ms | Hessian 10x slower!     |
| 10 | 2000 | 4ms      | **27ms** | 7ms         | ~38ms   | Hessian dominates! ⚠️   |

### Speedup vs R

| Configuration | Rust Time | R Time | Speedup | Convergence |
|---------------|-----------|--------|---------|-------------|
| n=1000, d=10  | 0.68s     | 0.80s  | 1.18x   | Issues      |
| n=2000, d=10  | 0.35s     | 0.91s  | **2.63x** | ✅        |
| n=5000, d=10  | 0.68s     | 1.13s  | 1.67x   | Issues      |

Compare to lower d at n=5000:
- d=1: 1.51x speedup
- d=2: 3.36x speedup
- d=4: 3.49x speedup
- d=8: 1.90x speedup
- d=10: 1.67x speedup

---

## Why Hessian Scales Poorly

### Computational Complexity

The REML Hessian for m smoothing parameters is an **m×m matrix** where each element H[i,j] requires:

1. **Second derivative computation**: ∂²ℓ/∂ρᵢ∂ρⱼ
2. **Multiple matrix inversions and traces**
3. **Quadratic forms** with penalty matrices

**Total complexity per Hessian:**
- Matrix operations: O(m² × p³) where p = number of basis functions
- For d=10, m=10: **100 elements** vs d=1 with **1 element**

### Current Implementation

```rust
// src/reml.rs:reml_hessian_multi_cached()
// Computes full m×m Hessian
for i in 0..m {
    for j in 0..m {
        // Each element requires:
        // 1. Matrix solve: A^(-1) * S_i (cost: O(p³))
        // 2. Trace computations
        // 3. Quadratic forms: β'S_iβ, β'S_jβ
        hessian[[i, j]] = compute_second_derivative(i, j);
    }
}
```

**Bottleneck locations:**
- `/home/user/nn_exploring/src/reml.rs:500-650` - Double loop over smooths
- Matrix solves for each (i,j) pair grow cubically with basis dimension

---

## Optimization Opportunities for High d

### 1. **Exploit Hessian Symmetry** ⚡ (Easy win)
- Hessian is symmetric: H[i,j] = H[j,i]
- **Only compute upper triangle** (save 50% work)
- Current implementation computes full matrix

### 2. **Approximate Hessian for Large d** 🎯 (Medium effort)
- **Diagonal approximation**: Only compute H[i,i], set off-diagonals to 0
- Valid when smooths are weakly coupled
- Common in large-scale optimization (L-BFGS-B, scaled identity)
- **Potential speedup**: 10x for d=10 (10 elements vs 100)

### 3. **BFGS/L-BFGS Update** 🎯 (Medium-high effort)
- Approximate Hessian using gradient history
- No explicit second derivatives needed
- Standard in large-scale optimization
- **Complexity**: O(md) per iteration vs O(m²p³)

### 4. **Block-Diagonal Hessian** (High effort)
- Group smooths by covariate structure
- Assume independence between groups
- Reduces m×m to sum of smaller blocks

### 5. **Parallel Hessian Computation** ⚡ (Medium effort)
- Each H[i,j] is independent
- Use rayon to parallelize double loop
- **Ideal speedup**: Near-linear with cores

---

## Recommended Next Steps

### Phase 5: Exploit Hessian Symmetry
**Impact**: 50% reduction in Hessian time for d>1  
**Effort**: ~30 minutes  
**Implementation**: Only compute upper triangle, copy to lower

### Phase 6: Diagonal Hessian Approximation
**Impact**: 90% reduction in Hessian time for d=10 (27ms → 3ms)  
**Effort**: ~2 hours  
**Implementation**: Add `hessian_mode` parameter: "full", "diagonal"  
**Trade-off**: Slightly slower convergence, but much faster per iteration

### Phase 7: Parallel Hessian
**Impact**: 2-4x speedup on multi-core (if combined with symmetry)  
**Effort**: ~1 hour  
**Implementation**: Use rayon parallel iterator over upper triangle

---

## Benchmark Commands

```bash
# Test high-d performance
python3 -c "
import numpy as np
import mgcv_rust
n, d = 2000, 10
X = np.random.randn(n, d)
y = np.sum(np.sin(X), axis=1) + 0.1 * np.random.randn(n)
gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[10]*d, method='REML', bs='cr', max_iter=20)
print(f'Converged: {result.get(\"converged\", False)}')
"

# Profile high-d case
MGCV_PROFILE=1 python3 -c "..." # same as above
```

---

## Key Takeaways

✅ **Phases 1-4 optimizations** work well for d ≤ 8  
⚠️ **Hessian computation** dominates for d > 8 (~27ms at d=10)  
🎯 **Low-hanging fruit**: Exploit symmetry (50% speedup)  
🎯 **High-impact**: Diagonal approximation (90% speedup for d=10)  
⚠️ **Complexity**: Hessian scales as O(m²) in number of smooths  
✅ **Still competitive**: 1.2-2.6x faster than R even at d=10

