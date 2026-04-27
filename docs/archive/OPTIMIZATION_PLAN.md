# Optimization Plan: Closing the Gap with R's bam()

## Current State (2026-02-12)

Branch: `validation-framework-v2`, after Fellner-Schall rewrite.

### Benchmark Results (1 warmup + median of 5 runs, seed=42)

| Config               | Newton (ms) | FS (ms) | R gam (ms) | R bam (ms) | Best Rust | vs bam     | vs gam     |
|----------------------|-------------|---------|------------|------------|-----------|------------|------------|
| n=500, d=1, k=10     | **5.1**     | 21.0    | 21.0       | 11.0       | 5.1(N)    | 2.1x WIN   | 4.1x WIN   |
| n=1000, d=1, k=10    | **9.7**     | 37.4    | 24.0       | 11.0       | 9.7(N)    | 1.1x WIN   | 2.5x WIN   |
| n=1000, d=2, k=10    | **20.4**    | 72.8    | 38.0       | 17.0       | 20.4(N)   | 1.2x LOSE  | 1.9x WIN   |
| n=1000, d=4, k=10    | **47.2**    | 198.3   | 117.0      | 34.0       | 47.2(N)   | 1.4x LOSE  | 2.5x WIN   |
| n=2000, d=1, k=10    | **14.0**    | 35.2    | 51.0       | 17.0       | 14.0(N)   | 1.2x WIN   | 3.7x WIN   |
| n=2000, d=2, k=10    | **36.3**    | 90.8    | 96.0       | 28.0       | 36.3(N)   | 1.3x LOSE  | 2.6x WIN   |
| n=2000, d=4, k=10    | **85.6**    | 336.8   | 274.0      | 45.0       | 85.6(N)   | 1.9x LOSE  | 3.2x WIN   |
| n=2000, d=8, k=8     | 205.8       | **138.2**| 1003.0    | 84.0       | 138.2(FS) | 1.6x LOSE  | 7.3x WIN   |
| n=5000, d=1, k=10    | **38.3**    | 65.7    | 112.0      | 23.0       | 38.3(N)   | 1.7x LOSE  | 2.9x WIN   |
| n=5000, d=2, k=10    | **74.0**    | 118.9   | 223.0      | 38.0       | 74.0(N)   | 1.9x LOSE  | 3.0x WIN   |
| n=5000, d=4, k=8     | 115.7       | **108.6**| 543.0     | 57.0       | 108.6(FS) | 1.9x LOSE  | 5.0x WIN   |
| n=5000, d=8, k=8     | 365.6       | **311.8**| 2764.0    | 134.0      | 311.8(FS) | 2.3x LOSE  | 8.9x WIN   |
| n=10000, d=1, k=10   | **81.0**    | 104.8   | 203.0      | 32.0       | 81.0(N)   | 2.5x LOSE  | 2.5x WIN   |
| n=10000, d=2, k=10   | **145.5**   | 197.6   | 494.0      | 50.0       | 145.5(N)  | 2.9x LOSE  | 3.4x WIN   |
| n=10000, d=4, k=8    | 256.3       | **196.3**| 1107.0    | 86.0       | 196.3(FS) | 2.3x LOSE  | 5.6x WIN   |

**vs gam(): 15/15 wins, geometric mean 3.5x faster.**
**vs bam(): 3/15 wins, geometric mean 1.52x slower.**

### Time Breakdown (n=5000, d=2, k=10, p=20, total ~102ms)

| Component              | Time  | %   |
|------------------------|-------|-----|
| Basis evaluation       | 38ms  | 37% |
| REML optimization (2x) | 51ms  | 50% |
| PiRLS iterations       | 11ms  | 11% |
| Overhead               | 2ms   | 2%  |

---

## Root Causes of bam() Speed Advantage

### 1. Redundant Newton Restart (our bug)

Our outer loop (`gam_optimized.rs:275`) runs PiRLS + Newton repeatedly. Newton
converges internally (4 iterations) but **resets lambda to 1.0 every call**
(`smooth.rs:133`). For Gaussian identity link, the weights don't change between
outer iterations, so the second call produces identical results. This doubles
our REML time for no benefit.

**Files:** `src/smooth.rs:131-136`, `src/gam_optimized.rs:275-355`

### 2. Basis Evaluation Cost (37% of total)

We spend 38ms evaluating basis functions into a full n x p design matrix. bam()
never materializes the full design matrix — it stores **compressed marginal
matrices** (m unique rows, m << n) plus integer index arrays mapping each
observation to its compressed row. The basis evaluation is done once during
setup and the compressed form is reused.

**Files:** `src/gam_optimized.rs:28-62` (FitCache::new)

### 3. O(n*p^2) X'WX Computation

We compute X'WX as `X.t().dot(&(diag(W) * X))`, which is O(n * p^2). bam()'s
C-level `XWXd()` function uses **scatter-gather**:

1. For each column pair (i,j), scatter weighted observations into m-sized
   buckets via integer index lookups: O(n)
2. Then do BLAS dgemv on the compressed m x p matrix: O(m * p)
3. Total: O(n*p + m*p^2) where m ~ 1000 << n

For n=10000, p=32: naive = 10M ops, scatter-gather = 1M ops (10x reduction).

**Reference:** Wood, Li, Shaddick & Augustin (2017) JASA, mgcv `src/discrete.c`

### 4. Nested vs Interleaved Iteration

We use nested iteration: `outer { full_PiRLS; full_Newton_on_sp }`. bam() uses
interleaved iteration via `Sl.fitChol`: one Newton step on smoothing parameters
per PiRLS step, with step-halving for stability. The smoothing parameters and
coefficients converge jointly, typically in 7-9 total iterations.

**Reference:** mgcv `R/smooth.r` (Sl.fitChol), `R/bam.r` (bgam.fitd)

### 5. Pivoted Cholesky vs Eigendecomposition

We compute `penalty_sqrt` via eigendecomposition at the start of each Newton
call. bam() works directly on the p x p penalized system `X'WX + S` using
pivoted Cholesky with diagonal preconditioning, avoiding eigendecomposition
entirely.

---

## Optimization Roadmap

### Phase 1: Low-Hanging Fruit (estimated 1.5-2x speedup)

#### 1a. Fix redundant Newton outer loop
- For Newton algorithm, run exactly **one outer iteration**: PiRLS once, Newton
  to convergence, final PiRLS with optimal lambda. No restart.
- For Fellner-Schall, keep the outer loop (FS does one step per call by design).
- **Expected:** ~25% total speedup (eliminates duplicate REML optimization)
- **Difficulty:** Easy
- **Files:** `src/gam_optimized.rs`

#### 1b. Interleave sp optimization with PiRLS (bam-style)
- Instead of full Newton convergence, do **one Newton step per PiRLS iteration**
  with step-halving.
- Check joint convergence of both lambda and deviance.
- For Gaussian identity link, X'WX doesn't change, so this reduces to the
  current approach but without the restart.
- **Expected:** 10-20% speedup for non-Gaussian, negligible for Gaussian
- **Difficulty:** Medium
- **Files:** `src/gam_optimized.rs`, `src/smooth.rs`

#### 1c. Cache basis evaluation across predict calls
- Already done in FitCache. No action needed.

### Phase 2: Discretized Computation (estimated 2-5x speedup at large n)

#### 2a. Compressed storage format
- For each smooth term's marginal, store only unique (or binned) covariate
  values in a compressed matrix (m x k where m << n).
- Store integer index array mapping each observation to its compressed row.
- For continuous covariates with many unique values, bin to ~1000 grid points
  (1D), ~100 per dimension (2D), ~25 per dimension (3D+).
- **Data structures:** `CompressedBasis { values: Array2<f64>, indices: Vec<u32> }`
- **Files:** New `src/discrete.rs`

#### 2b. Scatter-gather X'WX kernel
- Implement the scatter-gather pattern from bam's `XWXd()`:
  ```
  For each (block_a, block_b) pair:
    For each column j of block_b:
      1. Scatter: temp[idx[i]] += w[i] * X_b_compressed[idx_b[i], j]  for i in 0..n
      2. Gather: X'WX[a_cols, j] = X_a_compressed.t() @ temp             (BLAS dgemv)
  ```
- Use Rayon for parallelism over block pairs.
- **Expected:** 2-5x speedup on X'WX for n >= 5000
- **Difficulty:** Hard
- **Files:** New `src/discrete.rs`

#### 2c. Efficient eta = X*beta computation
- Instead of materializing X then computing X*beta (O(n*p)):
  ```
  For each smooth term j:
    eta_j = X_j_compressed @ beta_j          (m_j x k_j times k_j = m_j vector)
    eta[i] += eta_j[idx_j[i]]  for i in 0..n  (scatter via index)
  ```
- Total: O(sum(m_j * k_j) + n * d) vs O(n * p)
- **Files:** New `src/discrete.rs`

### Phase 3: Solver Improvements

#### 3a. Pivoted Cholesky with diagonal preconditioning
- Replace eigendecomposition-based penalty handling with direct pivoted Cholesky
  on the p x p penalized system.
- Apply diagonal preconditioning: `d = sqrt(diag(A))`, solve `(A/dd') * (d*x) = f/d`.
- Use LAPACK `dpstrf` for pivoted Cholesky.
- **Expected:** Small speedup (eigendecomp is already fast for small p)
- **Difficulty:** Medium
- **Files:** `src/smooth.rs`, `src/reml.rs`

#### 3b. Implicit function theorem for d(beta)/d(rho)
- bam's `Sl.iftChol` computes derivatives of beta w.r.t. log(lambda) using the
  implicit function theorem, avoiding re-solving the system.
- Enables efficient REML gradient/Hessian computation from the Cholesky factor.
- **Expected:** Modest speedup on REML gradient computation
- **Difficulty:** Hard
- **Files:** `src/reml.rs`

### Phase 4: Parallelism

#### 4a. Rayon parallelism for scatter-gather
- The XWXd block computation is embarrassingly parallel.
- Use Rayon `par_iter` over independent block pairs.
- **Expected:** Near-linear speedup with core count for the X'WX computation
- **Difficulty:** Easy (once Phase 2 is done)
- **Files:** `src/discrete.rs`

#### 4b. Parallel basis evaluation
- Evaluate basis functions for each smooth term in parallel.
- **Expected:** Modest speedup (basis eval is already fast per-term)
- **Difficulty:** Easy
- **Files:** `src/gam_optimized.rs`

---

## Priority Order

1. **Phase 1a** — Fix redundant Newton restart. Immediate ~25% win, trivial change.
2. **Phase 2a+2b** — Discretized storage + scatter-gather X'WX. This is the
   single biggest optimization needed to match bam() at large n. Should be done
   as a unit.
3. **Phase 2c** — Efficient eta computation. Natural extension of 2a/2b.
4. **Phase 1b** — Interleaved iteration. Important for non-Gaussian families.
5. **Phase 3a** — Pivoted Cholesky. Polish, not critical.
6. **Phase 4a** — Rayon parallelism. Easy win once discretization exists.
7. **Phase 3b** — IFT for gradients. Advanced optimization.

## Target Performance

After Phase 1a: ~1.2x slower than bam() (from 1.5x)
After Phase 2: ~1.0x (parity with bam for n <= 10000)
After Phase 4: potentially faster than bam for large n (Rayon > R's OpenMP)
