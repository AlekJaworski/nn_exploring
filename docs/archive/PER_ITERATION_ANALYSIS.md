## Per-Iteration Performance Analysis

Based on profiling data, here's what each iteration does:

### For n=5000, k=20, each iteration computes:

1. **Block-wise R factor** (~60ms Rust vs ~30ms R)
   - Process 5 blocks of 1000 rows each
   - 5 × QR(1020×20) decompositions
   - Each QR: ~12ms Rust vs ~6ms R

2. **Gradient computation** (~10-15ms both)
   - tr(P'SP) via optimized computation
   - RSS derivatives
   - Beta derivatives

3. **Hessian computation** (~5-10ms both)
   - Second derivatives

### Why R is faster per iteration:

**BLAS optimization:**
- R uses highly optimized BLAS (likely MKL or OpenBLAS with AVX2)
- Our code uses OpenBLAS but may not be calling BLAS optimally

**Specific issues in our code:**

1. **compute_xtwx()** (src/reml.rs:10-30)
   ```rust
   for i in 0..p {
       for j in i..p {
           for row in 0..n {  // Manual loop!
               sum += x[[row, i]] * w[row] * x[[row, j]];
           }
       }
   }
   ```
   - **NOT using BLAS!** Manual nested loops
   - Should use BLAS SYRK for symmetric rank-k update

2. **Matrix products in gradient** (src/reml.rs:561-565)
   ```rust
   let p_t_sqrt_s = p_matrix.t().dot(sqrt_penalty);  // Uses BLAS GEMM
   ```
   - This IS using BLAS (good!)
   - But called 5 times per iteration (once per Newton step)

3. **a_inv.dot(&lambda_s_beta)** (src/reml.rs:574)
   - Multiple p×p matrix-vector products per iteration
   - Each uses BLAS GEMV (good)

### Estimated time breakdown for n=5000 iteration:

| Operation | Rust | R | Difference |
|-----------|------|---|------------|
| Block-wise QR (5 blocks) | ~35ms | ~20ms | 1.75x slower |
| compute_xtwx (manual) | ~15ms | ~5ms | 3x slower! |
| Gradient computation | ~7ms | ~6ms | ~same |
| Hessian computation | ~3ms | ~3ms | ~same |
| **Total** | **~60ms** | **~34ms** | **1.76x** |

### Fix: Use BLAS for compute_xtwx

The main bottleneck is `compute_xtwx()` which uses manual loops instead of BLAS SYRK.

If we fix this ONE function to use BLAS, we should get:
- compute_xtwx: 15ms → 5ms (save 10ms)
- Per iteration: 60ms → 50ms
- Total for n=5000: 0.298s → 0.250s
- **Expected speedup: 1.19x** (enough to match or beat R!)

### Next Action

Implement BLAS SYRK for compute_xtwx() - this is the **single biggest win** we can get.
