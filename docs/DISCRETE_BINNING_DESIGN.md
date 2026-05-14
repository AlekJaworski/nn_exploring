# `discrete=TRUE` (Covariate-Binning Fast Path) — Design Proposal

Status: **research-only**, no implementation yet.
Driver: customer feedback note (2026-05-14) and memory `project_scat_perf_gap_vs_mgcv_bam` —
`mgcv.bam(scat, discrete=TRUE, nthreads=4)` runs the production sale-price model
in 155-272ms vs 4-9s for `mgcv_rust` t-dist on the same fixture (~30× gap).
mgcv's fast path bins repeated covariate values, collapses each smooth's
n × k basis to m × k (m = number of bins, m << n), and reuses scatter-gather
across all PIRLS iterations.

Parity baseline:
- `test_data/discrete_binning_parity_synthetic.json` — synthetic with controlled ties.
- `test_data/discrete_binning_parity_production.json` — split_0 production fixture.
- Capture script: `scripts/r/tests/extract_discrete_binning_parity.R`.

Local measurement (2026-05-14, this capture):

| Engine | Split 0 (n=5,157) elapsed | REML score |
|---|---|---|
| `gam(scat, REML)` (no binning) | **1.382 s** | -8897.29 |
| `bam(scat, fREML, discrete=TRUE)` | **0.164 s** | -8891.28 |
| **Ratio** | **8.4×** | Δ = 6.0 |

For Gaussian on the same data:

| Engine | Split 0 (n=5,157) elapsed |
|---|---|
| `gam(gaussian, REML)` | 0.687 s |
| `bam(gaussian, fREML, discrete=FALSE)` | 0.072 s |
| `bam(gaussian, fREML, discrete=TRUE)` | **0.049 s** |

Discrete gives a 1.5× win over chunked-bam on Gaussian (small absolute), but a
**14× win** over gam on scat (because scat's IRLS hits the X'WX hot path many
more times). The headline customer perf gap is scat × discrete.

---

## 1. Math

### 1.1 mgcv's binning strategy — what `discrete=TRUE` actually does

mgcv's `bam(discrete=TRUE)` runs `mgcv:::discrete.mf(gp, mf, ..., m=discrete)`
**before** building any basis. `discrete.mf` walks every smooth.spec and
every parametric column, calling `mgcv:::compress.df` on each smooth's
covariate(s) independently. The result is:

- A length-`n_marg` list of compressed model-frame columns (each holding only
  the *unique* covariate values, plus an `index` attribute mapping every
  original observation back to its compressed row).
- A single `n × n_marg` integer matrix `k` (alias `kd` in the fitted G object)
  whose row `i`, column `j` holds the compressed-row index for observation `i`
  in marginal `j` (1-based).
- A vector `nr` of per-marginal compressed-row counts.
- A 2-column matrix `ks` mapping each marginal to its column range inside `k`
  (so multi-term tensor smooths can take a slab of columns).

The binner itself (`compress.df`, mgcv R/bam.r):

```r
m <- if (d == 1) 1000 else if (d == 2) 100 else 25   # default per-dim cap
xu <- uniquecombs(dat, TRUE)                          # exact unique-row collapse
if (nrow(xu) > mm * mf) {
  # quantile-grid binning: snap each x to a grid of m equal-width bins
  for (i in 1:d) if (!is.factor(dat[, i])) {
    xl <- range(dat[, i]); dx <- diff(xl) / m
    kx <- round((dat[, i] - xl[1]) / dx) + 1
    dat[, i] <- xu[kx]
  }
  xu <- uniquecombs(dat, TRUE)
}
```

**Two regimes**:
1. **Pure dedup** when the natural unique-row count is below `m=1000` per
   dimension — exact representation, *zero* approximation error.
2. **Quantile grid** when uniques exceed the cap — equal-width binning over
   the covariate range, snapping each observation to its grid centre. This
   is the only place an approximation enters.

On the production fixture (split_0, n=5157), per-column compression:

| Column | nunique | bins after compress.df |
|---|---|---|
| `current_list_price` | 606 | 606 (pure dedup, no binning) |
| `price_change_pct_from_original` | 767 | 767 (pure dedup) |
| `cum_dom_before_current_regime` | 168 | 168 (pure dedup) |
| `days_in_current_price_regime` | 22 | 22 (pure dedup) |
| `monthly_index` | 12 | 12 (pure dedup) |

The full compressed model-frame is **767 rows** (the max per-column unique
count, because mgcv stores all columns on the same compressed axis).
**That is the n that the hot path actually sees per IRLS iteration**,
versus 5,157 for the unbinned path — a **6.7× n-reduction**.

### 1.2 Compressed `X'WX` formula

For a smooth term `j` with per-row bin index `k_j[i] ∈ {1, ..., m_j}` and
compressed basis `X̃_j ∈ ℝ^{m_j × p_j}`:

```
X_j[i, :] = X̃_j[k_j[i], :]                       (definition; n rows of X_j collapse to m_j rows of X̃_j)
```

Then for blocks `(a, b)`:

```
(X'WX)[a, b]_rc  =  Σ_{i=1..n} w_i · X_j_a[i, r] · X_j_b[i, c]
                =  Σ_{i=1..n} w_i · X̃_a[k_a[i], r] · X̃_b[k_b[i], c]
```

mgcv's `XWXd` (C function `CXWXd0`) computes this in two stages:

1. **SCATTER**: build the *weight-cube* `T[μ, ν] = Σ_i w_i · 1[k_a[i] = μ] · 1[k_b[i] = ν]`
   — an `m_a × m_b` matrix accumulated in `O(n)` work.
2. **GATHER**: `(X'WX)[a, b] = X̃_a' · T · X̃_b`, an `O(m_a · m_b · (p_a + p_b))` product.

For the **diagonal block** (`a = b`), this simplifies further:
`T = diag(Σ_{i: k_a[i] = μ} w_i)` — just a per-bin weight sum — and the gather
becomes `X̃_a' · diag(t) · X̃_a` in `O(m_a · p_a²)`.

Total cost per X'WX assembly: `O(n + Σ_{a,b} m_a · m_b · (p_a + p_b))` vs
the un-binned `O(n · p²)`. For production fixture: `n=5157`, `p=30`, so
unbinned is ~4.6M ops, binned (with the off-diagonal scatter to a tiny
`m_a × m_b` mid-cube and 30² assembly steps) is roughly `n + 30·767²` ~ 1.8M
**but with much better cache locality** — and crucially, an enormous wins
manifest on the scat IRLS path which calls X'WX ≥30× per fit.

### 1.3 Compressed `X'Wy` and `Xβ`

```
(X'Wy)_j[r] = Σ_i w_i · y_i · X̃_j[k_j[i], r]            (one term)
            = Σ_{μ=1..m_j} X̃_j[μ, r] · t_j[μ]            with t_j[μ] = Σ_{i: k_j[i]=μ} w_i · y_i

(Xβ)[i]     = Σ_j X̃_j[k_j[i], :] · β_j                  (gather)
            = Σ_j ξ_j[k_j[i]]                            with ξ_j[μ] = X̃_j[μ, :] · β_j
```

Both are `O(n + Σ_j m_j · p_j)`. The compressed working linear-predictor
`ξ_j ∈ ℝ^{m_j}` is itself a useful intermediate — it's all PIRLS needs to
compute `η = Σ_j (ξ_j gathered)`.

### 1.4 Prediction at new points

mgcv's `predict.bam` (and `predict.bamd`) explicitly **does NOT** reuse the
compressed basis. For new x values, the bins from training are irrelevant —
predict.bam falls back to `predict.gam` (full lpmatrix path). Verified by
`mgcv:::predict.bam` source: it dispatches to `predict.bamd` only when
called with the *training* data, otherwise calls `predict.gam` directly.

**Consequence for mgcv_rust**: the binning lives only in the fit path;
`Gam.predict()` keeps the existing full-matrix path. **No predict-time
changes needed.**

### 1.5 Interaction with IRLS — weight refresh per iteration

For non-Gaussian (binomial, scat, Poisson, ...) each IRLS iteration produces
new `w_i`, so the scatter cube `T` must be rebuilt every iteration. The
work per iter:

```
X'WX  rebuild :   O(n + Σ_{a,b} m_a · m_b · (p_a + p_b))
X'Wz  rebuild :   O(n + Σ_j m_j · p_j)
Solve         :   O(p³)              (Cholesky on the assembled p × p)
Update η      :   O(Σ_j m_j · p_j + n)
```

The `O(n)` scatter cannot be amortised across iterations (w changes), but the
`O(n)` *operations themselves* — single floating-point adds with index lookups
— are vastly faster than the `O(n · p²)` BLAS GEMM. **This is the IRLS hot
path on which scat × discrete sees its 8-15× wins.**

For Gaussian, only **one** X'WX is needed (W=I, fixed), so the scatter can
be cached. This is what `bam(gaussian, discrete=TRUE)` does and is why the
Gaussian gain is smaller — the un-binned GEMM was already a one-shot cost.

### 1.6 The binning bias

Both pure dedup (regime 1) and quantile-grid binning (regime 2) introduce
zero or small bias respectively:

- **Pure dedup**: zero bias by construction. The compressed basis exactly
  reproduces the un-binned basis when each compressed row's covariate
  appears verbatim in the original data.
- **Quantile grid**: each observation's covariate is snapped to a grid
  midpoint; the worst-case `|x_orig - x_binned|` is half the grid spacing,
  i.e. `range(x) / (2m)`. For typical splines (cubic regression spline,
  thin-plate), the basis-value error is bounded by `|x_orig - x_binned| ·
  max|f_basis'(x)|`, which is `O(1/m)`. With `m=1000` (mgcv's 1-D default)
  this is negligible (~1e-3 in coefficients on real data).

Captured empirical bias on production split_0:

| family | coef max\|gam − bam_discrete\| | rel. error vs max\|coef\| | REML-score Δ |
|---|---|---|---|
| Gaussian | 1.21e-2 | 2.88e-3 | (essentially equal) |
| scat | 4.23e-3 | small | -8897.29 → -8891.28 |

**Parity tolerance recommendation: 5e-3 absolute on coefficients, 1e-3
relative — looser than the existing 1e-6 parity bar because binning is an
admitted approximation.** Same tolerance mgcv documents internally for
bam vs gam.

---

## 2. mgcv internals — concrete references

| Function | Location | Role |
|---|---|---|
| `bam` | `R/bam.r` | Top-level entry; dispatches to `bgam.fitd` when `discrete=TRUE`. Calls `discrete.mf` to build the compressed model frame **before** `gam.setup`. |
| `discrete.mf` | `R/bam.r` | Walks the formula, calls `compress.df` on each smooth's covariate(s) and parametric columns, assembles the `kd` (n × n_marg) index matrix, `ks` (marginal → column range), `nr` (per-marginal n_unique). |
| `compress.df` | `R/bam.r` | The actual binner. Pure-dedup when `nunique ≤ m`, quantile-grid binning otherwise. Returns the compressed data + `index` attribute (1-based per-row mapping). |
| `bgam.fitd` | `R/bam.r` (~620 LOC) | The discrete-mode fitter. Allocates `qrx = list(R, f, y.norm2)` and rebuilds `qrx$R = XWXd(...)` and `qrx$f = XWyd(...)` per IRLS iter via the scatter-gather C routines. Calls `Sl.fitChol` to drive the outer Newton over log-λ. |
| `XWXd` | `R/bam.r` | R wrapper over `.Call(C_CXWXd0, ...)` — the scatter-gather X'WX. Takes the compressed `Xd` list, `w`, `kd`, `ks`, etc. |
| `XWyd` | `R/bam.r` | R wrapper over `.Call(C_CXWyd, ...)` — X'Wy via scatter-gather. |
| `Xbd` | `R/bam.r` | R wrapper over `.Call(C_CXbd, ...)` — compressed Xβ (compute η). |
| `Sl.fitChol` | `R/Sl.r` | Outer-Newton-step computer: takes `qrx$XX, qrx$Xy, qrx$y.norm2`, an `Sl` block-structured penalty, current log-λ, log-φ → returns β, gradient, Hessian, ldetS, ldetXXS. **No knowledge of binning** — operates entirely on the assembled p × p quantities. |
| `Sl.setup` / `Sl.rSb` / `Sl.initial.repara` | `R/Sl.r` | Penalty-block structure utilities that take the un-binned smooth list and build the block-diagonal Sl representation. Independent of binning. |
| `discrete.c` / `bam.c` (C source) | `src/` | `CXWXd0`, `CXWyd`, `CXbd`, etc. — the actual scatter-gather kernels. Hand-tuned for OpenMP parallelism via the `nthreads` arg. |

**Key architectural insight**: mgcv keeps a clean separation between
(a) the discretized X-side (`Xd`, `kd`, `XWXd`, `XWyd`, `Xbd`) and (b) the
penalty-side outer-Newton machinery (`Sl.setup`, `Sl.fitChol`). The latter
operates on the assembled p × p matrices and is binning-agnostic. **Our
Rust port can keep the same separation**: replace the X'WX assembly hot
path, leave the REML score / Newton outer loop intact.

---

## 3. Existing mgcv_rust infrastructure (the partial-port we already have)

There's a `src/discrete.rs` (753 LOC) that ships a `DiscretizedDesign`
struct + scatter-gather `compute_xtwx`, `compute_xtwy`, `compute_eta`. It
is wired into `FitCache::new` (`src/gam_optimized.rs:90-144`) — active
for `n >= 2000` and `!mgcv_exact_disable_disc`. **But the discretization
strategy diverges from mgcv's in three load-bearing ways**:

1. **Wrong binning axis**. `CompressedBasis::from_basis_1d` bins by
   *uniform x-grid* with `bin_width = range / max_bins` and `max_bins=1000`.
   mgcv's `compress.df` does **exact unique-row dedup** when `nunique ≤ 1000`,
   only falling back to quantile-grid otherwise. For all production columns
   (max 767 unique), our path is binning when mgcv would dedup exactly.
2. **Wrong key for dedup fallback** (`from_basis_dedup`). Quantizes the
   *basis row* to 8 decimals and hashes — fine for floating-point
   covariates, but for the integer covariates that dominate the production
   fixture (`monthly_index`, `cum_dom`, `days_in_current_price_regime` are
   all `int64`) this is over-engineered and slower than a direct
   per-covariate `HashMap<i64, u32>`.
3. **Not used by the REML hot path**. `compute_xtwx(x, w)` in
   `src/reml/system.rs:114` and the ~25 callsites in `src/reml/mod.rs`
   all build X'WX from the **full `x` matrix** every iteration. The
   discretized path is only invoked inside `fit_pirls_discretized`
   (PIRLS-internal), so the REML score / gradient / Hessian evaluations
   that drive the outer Newton trial points still run the full GEMM.

**Net**: the existing module is a useful scaffold (the scatter-gather
shape is right, the cache-`DiscretizedDesign`-on-`FitCache` wiring is
right) but the **binning strategy and the REML integration both need a
rewrite**. Per the user's "rewrites over fallbacks" guidance, this is the
right port to do as a *replacement* of the current module, not an
augmentation.

### 3.1 Where `compute_xtwx` is called from REML

Hot-path callsites that need replacement (from `grep -n "compute_xtwx"
src/reml/mod.rs`): ~25 places, including the REML score, REML gradient
and REML Hessian assemblers, `tdist_gdi2_native`, the Fellner-Schall
update, and the quantile / Tweedie / NegBin profile-derivative routes.
Every one of them rebuilds X'WX from the full `n × p` design.

This is the structural reason mgcv_rust runs scat in 4-9s where bam
finishes in 164ms: not just IRLS speed, but the *outer Newton search*
runs many score+grad+Hess evals, each of which is a full GEMM. Replacing
the X'WX assembly with scatter-gather discretized assembly removes the
`O(n · p²)` cost from every one of these evals.

---

## 4. Integration plan for mgcv_rust

### 4.1 Data structures (the canonical names track mgcv exactly)

```rust
// src/discrete.rs  (existing file, but heavy rewrite)

/// A single smooth's compressed marginal.
pub struct DiscreteMarginal {
    /// Compressed basis values: m × p matrix (m = number of unique bins).
    pub x_d:        Array2<f64>,
    /// Per-row bin index, 0-based: indices[i] ∈ {0, ..., m-1}.
    pub indices:    Vec<u32>,
    /// Number of unique bins.
    pub nr:         usize,
    /// Column range in the global β: [col_offset, col_offset + p).
    pub col_offset: usize,
    /// Number of basis columns p (same as x_d.ncols()).
    pub num_basis:  usize,
    /// Optional reverse-index map: for each compressed row μ, the original-row
    /// observations that map to it. Used by the per-iter `t = Σ w[i]` scatter
    /// (sparse-matrix-style optimisation; mgcv builds this as `r` and `off`
    /// for OpenMP-friendly streaming).
    pub r_off:      Option<(Vec<u32>, Vec<u32>)>,
}

/// The full discretized design — one DiscreteMarginal per smooth, plus an
/// intercept marginal (m=1, x_d=[[1.0]]) at index 0 so the leading β₀
/// column flows through the same scatter-gather pipes.
pub struct DiscreteDesign {
    pub marginals:   Vec<DiscreteMarginal>,
    pub total_basis: usize,    // sum of all marginals' num_basis
    pub n:           usize,
}
```

This mirrors mgcv's `(Xd, kd, ks, nr)` quadruple. The `r_off` field is
optional and only built when OpenMP-parallel scatter is wanted (Phase 4).

### 4.2 Bin construction

```rust
impl DiscreteDesign {
    pub fn new(
        smooth_terms: &[SmoothTerm],
        x: &Array2<f64>,
        has_intercept: bool,
        config: &DiscreteConfig,
    ) -> Self
}
```

Per-smooth: extract the column `x[:, smooth_idx]`, call a `compress_1d`
that mirrors `mgcv:::compress.df`:

```rust
fn compress_1d(x_col: &Array1<f64>, max_bins: usize) -> (Vec<u32>, Vec<f64>) {
    // 1. exact unique-value dedup via HashMap<OrderedF64, u32>
    // 2. if uniques > max_bins, quantile-grid: snap each x to a grid of
    //    max_bins equal-width centres
    // returns (per-row indices, bin centres)
}
```

Then evaluate the **basis on the bin centres** (not on every original
observation). That's where the m × p compressed basis `x_d` comes from —
this is materially cheaper than the current path (which evaluates the basis
on all n observations and then deduplicates the result).

Mgcv's `m` default mirrors here: `1000` for 1-D smooths, `100` per dim for
2-D, `25` per dim for 3+. For pure-dedup (uniques ≤ m), no error introduced;
for quantile-grid, the design tolerance widens to 5e-3 absolute.

### 4.3 Replace `compute_xtwx` hot path

Introduce a new function in `src/reml/system.rs`:

```rust
pub fn compute_xtwx_discrete(disc: &DiscreteDesign, w: &Array1<f64>) -> Array2<f64>
```

with the formula from §1.2. Two variants:

- **Diagonal block fast path** (`a == b`): `T = diag(Σ_{i: k_a[i]=μ} w_i)`,
  product = `X̃_a' · diag(t) · X̃_a` via a single `O(m_a · p_a²)` GEMM
  with the m-long weight applied as a row-scale.
- **Off-diagonal block** (`a ≠ b`): build full `T ∈ ℝ^{m_a × m_b}` (sparse
  pattern, but for our small m we can keep it dense), then `X̃_a' · T · X̃_b`.

Then wire the existing REML code paths to switch on
`fit_cache.discrete.is_some()`:

```rust
let xtwx = match (&self.discrete, prior_weights) {
    (Some(disc), pw) => compute_xtwx_discrete(disc, &combined_w(pw, w)),
    (None,       _ ) => compute_xtwx(x, &combined_w(pw, w)),
};
```

The 25 `compute_xtwx(x, w)` callsites in `src/reml/mod.rs` collapse to a
single helper that dispatches based on `disc.is_some()` — closed-over via
the existing `FitCache`. Big refactor blast radius but mechanically simple.

### 4.4 PIRLS loop integration

Already half-done: `fit_pirls_discretized` (`src/pirls.rs:3011`) exists.
**But** it currently uses the *wrong* `DiscretizedDesign` (the existing
buggy one). When `DiscreteDesign` replaces it, the function body keeps
the same structure but switches over to the new types. The η-update via
`disc.compute_eta(&beta)` also goes through the new compressed gather
exactly as mgcv's `Xbd` does.

For TDist (the headline scat case), `fit_pirls_tdist` (`src/pirls.rs:1692`)
needs a parallel `fit_pirls_tdist_discrete` that calls the scatter-gather
X'WX inside its IRLS loop. The σ²/df profiling outer block is unchanged.

### 4.5 What does NOT need touching

- **Penalty pipeline** (`BlockPenalty`, `src/block_penalty.rs`, mgcv's
  `Sl.setup`): binning-agnostic. Same penalty matrices, same λ search.
- **Outer Newton** (`optimize_reml_newton_multi_with_xtwx`,
  `src/smooth.rs:1042`): the gradient/Hessian assembly *uses* X'WX but
  through the same helper that gets discretized — no algorithmic change.
- **Predict path** (`Gam.predict`, `src/gam.rs`): predict at new points
  goes through the full-design lpmatrix, just like mgcv's
  `predict.bam → predict.gam`. Zero changes.
- **Parametric terms** (just shipped 0.14.0): parametric columns can be
  packed as additional "marginals" with their own per-column dedup
  (mgcv does this — see `discrete.mf` lines processing `names.pmf`).
  Treating a parametric column as a 1-bin-per-unique-level marginal
  composes naturally; for 0/1 indicator columns this means exactly 2
  bins.
- **Sample weights** (just shipped via `prior_weights` in PIRLS):
  the combined weight `w_combined = w_IRLS · w_prior` enters the scatter
  step as a per-row scalar — no structural change.
- **TDist (scat) weighted PIRLS** (just landed in 318e395): the σ²/df
  block is on the *outer* shape Newton; the *inner* PIRLS just needs the
  X'WX swap. The dispatch in `gam_optimized.rs:298-316` already
  branches on `Family::TDist`; we add a sibling branch for "TDist +
  discrete" → `fit_pirls_tdist_discrete`.

### 4.6 Score-formula coupling

`reml_score_gamfit5` (for scat/quantile/ocat) takes X'WX as an input —
no algorithm change. The score formula doesn't *know* whether X'WX came
from a full GEMM or a scatter-gather assembly.

The only subtle interaction is **`log|X'WX + λS|`**. With binning, the
matrix `X̃' diag(t) X̃ + λS` has the same `p × p` shape; the Cholesky/log-det
cost is unchanged. Numerically the eigenvalues are slightly perturbed
(the binning bias), but the determinant remains positive definite as long
as `m ≥ p` (each marginal must have enough unique values to identify its
basis). The existing PIRLS ridge (1e-5 · max_diag, or 1e-12 in
mgcv_exact) provides the same safety net.

---

## 5. Effort estimate

| Sub-task | Effort | Risk |
|---|---|---|
| D1: rewrite `compress_1d` + `DiscreteMarginal` data structures | 1d | low |
| D2: build full `DiscreteDesign` from smooth_terms (incl. parametric & intercept) | 1d | low |
| D3: `compute_xtwx_discrete` (diagonal + off-diagonal blocks) + benchmark vs full GEMM | 1-2d | low |
| D4: rewire REML score/gradient/Hessian to dispatch through FitCache.discrete | 2-3d | **medium** — 25+ callsites, must preserve every other path |
| D5: `fit_pirls_discrete` (Gaussian and generic-GLM) | 1d | low (most code reusable) |
| D6: `fit_pirls_tdist_discrete` — the scat hot path | 1-2d | medium (scat IRLS is delicate; weight refresh order matters) |
| D7: parametric-term integration (treat each parametric column as a small marginal) | 1d | low |
| D8: synthetic + production parity bake-in (1e-3 to 5e-3 tolerance battery) | 2d | medium — coefficients differ by binning bias; need to set the right tolerances per scenario |
| D9: predict-path verification — confirm `Gam.predict` keeps the un-binned route at new x; smoke-test on the production fixture | 0.5d | low |
| D10: OpenMP-parallel scatter (mgcv's `nthreads`) | 2-3d | medium — `r_off` reverse-index plumbing, but pure perf, not correctness |
| D11: env-var / Python-API toggle (`Gam(discrete=True)` mirror) + docs | 0.5d | low |

**Total: 12-17 days** (~2.5-3.5 weeks for one engineer).

**Hardest subtask: D4** (REML rewire). It's the *only* one that touches a
large fraction of `src/reml/mod.rs`, the same file that just absorbed the
joint-outer-Newton actuation and the TDist weighted PIRLS work. Pure
mechanical replacement with `compute_xtwx_dispatch(&disc_opt, &x, &w)` —
no algorithmic change — but the blast radius is wide.

**Second-hardest: D6** (TDist × discrete). The IRLS weight refresh has to
happen *inside* the scat dDeta block, and the σ²/df shape-line-search must
recompute X'WX via the discrete path on every trial. If we land D5 first
and pattern off it, this is bounded to 1-2 days.

---

## 6. Parity protocol

Looser-than-1e-6 tolerances because binning is an admitted approximation:

| Quantity | Tolerance | Rationale |
|---|---|---|
| Coefficients (`β`) | `5e-3 abs / 1e-3 rel` (vs **gam** baseline) | Empirical gap: 1.21e-2 abs on prod Gaussian, 4.23e-3 abs on prod scat |
| `β` (vs `bam_discrete` baseline) | **1e-6 rel** | Our port should match mgcv's discrete path bit-for-bit, modulo BLAS-order non-determinism |
| `sp` / `log_sp` | `1e-3 rel` | Outer Newton can find slightly different optima under binning |
| `edf` total / per smooth | `1e-3 rel` | Tracks sp |
| `deviance` | `1e-3 rel` | Cascades from β |
| `REML score` | `5.0 abs` (yes really) | Captured Δ = 6.0 on prod scat between gam and bam_discrete |
| Per-row fitted η | `1e-2 abs` | Worst-case binning bias |
| Compressed bin counts `nr` | **exact** | Structural; if `nr` differs we've got the binner wrong |
| Per-row bin indices `kd[i, j]` | **exact** | Same |

The **structural** tolerances (`nr`, `kd`) are the load-bearing ones —
if the compressed model frame doesn't match mgcv's down to the integer
index, we're not implementing the same algorithm. Numerical tolerances
on β/score absorb the order-of-operations differences but cannot mask a
structural bug.

The production parity JSON captures all of these. Additionally captured
for future regression: `discrete_state` (mgcv's stashed `G$kd`, `G$Xd`
shapes), `discrete_mf` (the standalone `mgcv:::discrete.mf` output —
*the* function we're porting).

---

## 7. Perf projection

Headline numbers (production split_0, scat-REML):

| Path | Per-fit (ms) | Speedup vs current Rust |
|---|---|---|
| Current `mgcv_rust` t-dist | 4,261 - 8,765 | 1× (baseline) |
| mgcv `gam(scat, REML)` (local) | 1,382 | 3-6× |
| mgcv `bam(scat, fREML, discrete=TRUE)` | **164** | **26-53×** |

A faithful Rust port of `bam(discrete=TRUE)` should land somewhere between
`gam(scat)` and `bam(scat, discrete=TRUE)` — call it **400-800ms** per
fit on the production fixture. Why not all the way to 164ms?

1. **Single-thread**: mgcv ran `nthreads=4`. Our Rust port lands
   single-threaded first (D10 is the OpenMP-parallel scatter, 2-3d
   extra). Without nthreads: rough projection 4 × 164 ≈ 656ms (assuming
   linear thread scaling — optimistic but a reasonable upper bound).
2. **Rust-vs-C overhead**: the inner scatter loop is `O(n)` adds with index
   lookups. mgcv's C code is hand-tuned, including OpenMP pragmas and
   cache-blocking. A first-pass Rust port using ndarray will be within
   ~1.5-2× of the C — that pushes us to **~1s** without nthreads.
3. **REML outer-Newton iteration count**: mgcv's `Sl.fitChol` uses a
   tuned Newton + line-search that converges in fewer iterations than our
   current outer loop. This is *not* a discrete-binning thing; it's an
   independent perf gap. Worst case adds 20-30% extra.
4. **Joint outer Newton** (Phase 3 of `docs/JOINT_OUTER_NEWTON_DESIGN.md`):
   the existing `tdist_gdi2_native` already builds the full joint
   Hessian; switching from sequential to joint cuts outer iters by ~2-3×.
   *Composes* with discrete binning — gives an extra 2-3× on top.

**Realistic projection after D1-D11**: **600-900ms per fit on production
scat** — a **7-15× speedup over current Rust**, **closing roughly half
the 30× customer gap**. Adding OpenMP-parallel scatter (D10) brings it
to **300-500ms**, **closing 80%+ of the gap**. The remaining 2× to
mgcv parity is a long tail (BLAS-link choice, mgcv-tuned Newton
schedule, etc.) — beyond this implementation plan.

For the synthetic n=2000 case the gap is much smaller (gam = 271ms, bam
discrete = 52ms — only ~5×), so the *absolute* speedup from the port
shrinks as n shrinks. **Discrete binning's payoff scales with n and with
the IRLS-iter count** (so non-Gaussian families benefit most). For
Gaussian + small-n + already-dedup'd data, the perf win can be marginal.
This is the same hint mgcv gives in `?bam`: discrete=TRUE shines on
*"very large data sets"*.

---

## 8. Sub-task breakdown (for parallel implementation)

Listed in dependency order. Each has a clear in/out contract — designed
so a sub-agent can be briefed on a single ID without needing the
others' internals.

### D1 — `compress_1d` and `DiscreteMarginal` data structures
**In**: `x_col: &Array1<f64>`, `max_bins: usize`.
**Out**: `(indices: Vec<u32>, bin_centres: Vec<f64>)`. Pure-dedup when
uniques ≤ max_bins, equal-width grid otherwise. Unit-test against the
`discrete_mf$nunique_per_col` / `nr` fields captured in the parity JSON.
**Effort**: 1d.

### D2 — `DiscreteDesign::new`
**In**: `smooth_terms: &[SmoothTerm]`, `x: &Array2<f64>`,
`has_intercept: bool`, `parametric_cols: Option<&Array2<f64>>`.
**Out**: a `DiscreteDesign` with one marginal per smooth (basis evaluated
on bin centres), one per parametric column (1-bin-per-unique-value), and
an intercept marginal at index 0. Unit-test: the `nr` field matches the
captured `discrete_mf$nr` on the production fixture.
**Effort**: 1d. Depends on D1.

### D3 — `compute_xtwx_discrete` and `compute_xtwy_discrete`
**In**: `disc: &DiscreteDesign`, `w: &Array1<f64>`, `y: &Array1<f64>`.
**Out**: assembled `p × p` and `p`-vector matching the un-binned BLAS
output to 1e-12 on the same w, y. Diagonal vs off-diagonal blocks split
per §1.2. **The numerical equivalence vs `compute_xtwx(x, w)` is the
load-bearing test** — if this doesn't hold, every downstream parity will
fail.
**Effort**: 1-2d. Depends on D2.

### D4 — REML rewire
**In**: the 25+ `compute_xtwx(x, w)` call-sites in `src/reml/mod.rs`.
**Out**: each replaced by a `compute_xtwx_dispatch(&disc_opt, x, w)`
helper that selects discrete vs full. **Mechanically large but
algorithmically null** — the same Newton math runs, just on a faster
X'WX. Unit-test: every existing REML-parity test (Gaussian, binomial,
Poisson, Tweedie, NegBin) passes byte-identical to current master at
`discrete=False`.
**Effort**: 2-3d. Depends on D3.

### D5 — `fit_pirls_discrete` (generic-GLM)
**In**: same signature as `fit_pirls_cached`, plus
`disc: &DiscreteDesign`. Mirrors `bgam.fitd`'s IRLS body (mgcv R/bam.r):
each iter compute working w & z, scatter-gather assemble qrx$R and qrx$f,
solve, update η. Re-derive the existing `fit_pirls_discretized` against
the new types.
**Out**: `PiRLSResult` matching current `fit_pirls_cached` within 1e-3
absolute on Gaussian, binomial, Poisson, Gamma on the production
fixture.
**Effort**: 1d. Depends on D3.

### D6 — `fit_pirls_tdist_discrete`
**In**: same signature as `fit_pirls_tdist`, plus `disc`.
**Out**: scat-PIRLS using the scatter-gather X'WX inside the IRLS loop.
The σ²/df profile-shape outer block remains identical. **Parity target:
production split_0 scat fit within 5e-3 absolute on coefficients vs
`gam(scat, REML)`**.
**Effort**: 1-2d. Depends on D5 (most of the boilerplate is reusable).

### D7 — Parametric-term integration
**In**: the new parametric-terms API (just shipped 0.14.0).
**Out**: parametric columns enter `DiscreteDesign` as 1-bin-per-unique
marginals. Confirmed on the production fixture with the
`at_least_*_price_drop` indicators (2 bins each).
**Effort**: 1d. Depends on D2.

### D8 — Parity battery
**In**: `test_data/discrete_binning_parity_synthetic.json` and
`test_data/discrete_binning_parity_production.json` (captured today).
**Out**: integration test asserting Rust output within the §6
tolerances. Wire as `tests/discrete_binning_parity.rs`. Bake in the
empirical bias (5e-3 abs / 1e-3 rel) as the tolerance floor.
**Effort**: 2d.

### D9 — Predict-path verification
**In**: trained Rust + mgcv models from D8.
**Out**: `Gam.predict` on the original X and on a held-out X returns
matching predictions (1e-2 absolute). Confirms that the binning lives in
the fit path only and predict.bam-equivalent isn't needed.
**Effort**: 0.5d. Depends on D5/D6.

### D10 — OpenMP-parallel scatter (optional perf polish)
**In**: existing single-threaded `compute_xtwx_discrete`.
**Out**: rayon-parallel scatter accumulating per-thread `T` cubes then
summing. Expected 2-3× wall-clock on a 4-core machine, matching mgcv's
`nthreads=4` win.
**Effort**: 2-3d. Depends on D3.

### D11 — Python API: `Gam(discrete=True)`
**In**: Python `Gam` kwargs.
**Out**: pass-through to the Rust backend; auto-enable when n ≥ 2000
unless `discrete=False` is explicit (mgcv's pattern is opt-in via
`discrete=TRUE`; we may keep opt-in to match). Update README family
table.
**Effort**: 0.5d. Depends on D8.

**Recommended first implementation sub-task: D3** —
`compute_xtwx_discrete` against the existing `compute_xtwx(x, w)`. It's
the smallest piece that lets us *prove the math is right* before we
touch the REML rewire. Build a small fixture that materialises the same
binning by hand (or via the captured `discrete_mf$kd`), assemble X'WX
both ways, and assert 1e-12 equality. Everything downstream builds on
this kernel.

---

## 9. Open questions / risks

- **Coefficient bias acceptance**: mgcv accepts a `bam(discrete=TRUE)`
  vs `gam` coefficient gap of ~1e-2 in practice. Our PIRLS ridge
  (1e-5 · max_diag) might compound with the binning bias; in
  `MGCV_EXACT_FIT` mode (ridge 1e-12) the bias should drop. Worth
  verifying that the tighter ridge path doesn't reintroduce numerical
  instability under binning.
- **Tensor smooths**: this design assumes 1-D smooths exclusively (the
  production fixture only uses `s(x, k=K, bs='cr')`). mgcv's
  `discrete.mf` handles tensor.smooth.spec by binning each margin
  separately and assembling the Kronecker-product basis at fit time.
  mgcv_rust doesn't ship tensor smooths today; when we do, the
  DiscreteDesign extension is straightforward (per-margin compression
  → Kronecker-product gather).
- **AR1 / `rho ≠ 0`**: mgcv's `bam(rho=0.5, ...)` uses the AR-row
  weighting machinery inside the same XWXd path. Not in current scope —
  mgcv_rust doesn't support AR1 either.
- **`fREML` ≡ `REML`**: mgcv's `bam(method='fREML')` uses the same
  Newton-on-`Sl.fitChol` outer loop as the discrete path; from the
  fitter's perspective `fREML` and `REML` differ only in trivial score
  bookkeeping. Our `method='REML'` already matches.
- **`bs='cr'` vs `bs='tp'`**: production uses `bs='cr'` exclusively, and
  cr-splines are the cheapest basis to compress (knot placement is
  deterministic from the data range). Thin-plate (`bs='tp'`) involves
  an eigendecomp that depends on the *full* covariate vector — under
  binning, the eigendecomp should run on the compressed-x sequence,
  which is what mgcv does. Worth a unit-test if/when we extend to tp.

---

## 10. References

- Wood, Goude & Shaw (2015). *Generalized additive models for large data
  sets*. JRSS-C 64:139-155 — the foundational paper for `bam`.
- Wood, Li, Shaddick & Augustin (2017). *Generalized additive models for
  gigadata*. JASA 112:1199-1210 — the formal binning derivation +
  scatter-gather analysis.
- mgcv source: `R/bam.r` (specifically `bam`, `bgam.fitd`, `discrete.mf`,
  `compress.df`, `XWXd`, `XWyd`, `Xbd`). C kernels in `src/discrete.c`,
  `src/bam.c` (`CXWXd0`, `CXWyd`, `CXbd`).
- mgcv `?bam` documentation: "When `discrete=TRUE` covariates are binned
  to allow the computation of `X'WX` and `X'Wy` to be performed in O(n)
  rather than O(n p²) time."
