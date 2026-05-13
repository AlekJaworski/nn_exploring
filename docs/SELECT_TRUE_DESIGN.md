# `select=TRUE` (Null-Space Shrinkage) — Design Proposal

Status: **research-only**, no implementation yet.
Driver: site #4 in `docs/NEIGHBOURHOODS_INTEGRATION_GAPS.md` — `time_to_sell_analysis/algorithm/predict.py:513` uses `select=TRUE` for automatic variable selection on the ordered-categorical model.

Reference: **Marra & Wood, 2011, "Practical variable selection for generalized additive models"** (CSDA 55:2372–2387).

Parity data captured in `test_data/select_parity_basic.json` (script: `scripts/r/tests/extract_select_parity.R`).

---

## 1. Math

### 1.1 What `select=TRUE` actually does

A smooth's penalty `S_j` (e.g., the integrated-squared-second-derivative penalty for cubic regression splines) penalises *wiggliness* but has a **null space** — the set of basis-coefficient vectors `β ∈ N(S_j)` for which `β' S_j β = 0`. For a 2nd-derivative penalty, the null space is exactly the space of **linear functions** of the covariate (rank-deficiency = 2 before centring, = 1 after sum-to-zero centring).

This means however large `λ_j` grows, the smooth never collapses to zero — it collapses to the **best-fitting straight line**. `select=TRUE` fixes this by adding a *second* penalty per smooth that targets the null space:

```
S_j^total(λ_j, λ_j*) = λ_j · S_j + λ_j* · S_j^null
```

where `S_j^null` is the projector onto the null space of `S_j` (rank = null-space dimension). As `λ_j* → ∞`, the null-space coefficients are shrunk to zero, and together with `λ_j → ∞` the smooth becomes identically zero → the term is dropped automatically.

### 1.2 Construction of `S_j^null`

From mgcv's `smoothCon` (`R/mgcv.r`, lines ~432–481 of the deparsed body), the construction has two branches:

**Case A — already-canonical diagonal-S smooths** (e.g. when the smooth comes in with `S = diag(S11, …, S11, 0, …, 0)`, ranked block + null-block layout): just append

```
S_j^null = diag(0, …, 0, 1, …, 1)
```

where the trailing 1-block has dimension `p − rank(S_j)`. No eigendecomp needed.

**Case B — general S** (cr-splines, thin-plates, etc., where `S` is dense): compute total `St = Σ_k S_j^k` (sum over existing penalties of smooth `j`), do an eigendecomposition

```
St = U Λ U',     ind = (λ_i < eps^0.66 · max(λ))
U_0 = U[:, ind]
S_j^null = U_0 U_0'      (projector onto null space — rank = sum(ind))
```

Then mgcv appends `S_j^null` as the next element of `m$smooth[[j]]$S`, and `length(m$smooth[[j]]$S)` jumps from 1 to 2. Each smooth now contributes **two** smoothing parameters to the outer-loop search, not one.

### 1.3 Effect on REML

Total penalty becomes `S_total(λ) = Σ_j λ_j S_j + Σ_j λ_j* S_j^null`. The REML score has the same structure as before — `Dp + log|H| − log|S|+`. Two changes:

1. `log|S|+` is computed on the *combined* `S_total` whose null space (under any λ_j* > 0) shrinks. `Mp` (the total-penalty null-space dimension) reflects only what's still un-penalised after the extra terms — for site #4, where every smooth has its null space penalised, `Mp` reduces to the global intercept dimension (= 1).
2. The outer Newton search dimension doubles: `dim(log_sp) = 2 · n_smooth`.

### 1.4 Captured parity evidence

From `test_data/select_parity_basic.json` (data: `y = sin(2π x1) + ε`, `x2` has no effect):

```
no-select sp (len 2): [4.83, 3.45e+04]
select    sp (len 4): [4.80, 0.0217, 2.56e+04, 2.96e+05]
```

Interpretation:
- `sp[1] ≈ 4.8` (x1, wiggliness penalty) — unchanged.
- `sp[2] ≈ 0.02` (x1, null-space penalty) — small, because x1 *does* have a non-zero linear trend (the null space contains the linear component, and we want to keep it).
- `sp[3] ≈ 2.6e+04` (x2, wiggliness penalty) — already huge in no-select.
- `sp[4] ≈ 3.0e+05` (x2, null-space penalty) — **huge**, suppressing the residual linear trend.

Per-smooth edf and coef L2 norm:
```
smooth x1: no-select edf=8.14  → select edf=8.14   (unchanged)
smooth x2: no-select edf=1.30  → select edf=0.38   (collapsing to zero)
smooth x2: no-select |β|=0.028 → select |β|=0.014  (≈ halved)
```

x2 is being driven out — automatic variable selection working as advertised.

### 1.5 Penalty-matrix structure

```
m_select$smooth[[1]]$S:  list of 2  →  S_orig (9×9, rank 8) + S_null (9×9, rank 1)
m_select$smooth[[2]]$S:  list of 2  →  S_orig (9×9, rank 8) + S_null (9×9, rank 1)
```

(`9 = k − 1 = 10 − 1` for the centred basis; null-space rank is 1 because the centring removed the constant, leaving only the linear function in the null space.)

---

## 2. mgcv internals — concrete references

| Function | Location | Role |
|---|---|---|
| `gam.setup` | `R/gam.r` | Passes `select` through as `null.space.penalty=select` when calling `smoothCon`. |
| `smoothCon` | `R/mgcv.r:~432–481` | **Where the extra penalty is built.** Two branches: diagonal-S fast path (append `diag(0…1…)`) or general eigendecomp (`Sf = U_0 U_0'`). Updates `m$smooth[[j]]$S`, `rank`, `S.scale`, and zeroes `null.space.dim`. |
| `null.space.dimension(d, m)` | `R/mgcv.r` | Combinatorial formula for the null-space dim of a d-dim m-th-order thin-plate penalty. Used as a *spec lookup*; not directly the projector. |
| `Sl.setup` | `R/Sl.r` | Block-structure setup for the post-`smoothCon` total penalty. Detects diagonal/disjoint blocks per smooth and splits each smooth's `m` penalties into independent `Sl[[b]]` blocks where possible — important for outer-loop diagonal-Hessian shortcuts. |
| `gam.fit3` outer loop | `R/gam.fit3.r` | Outer Newton over `log(sp)`. The doubled `sp` length under `select=TRUE` is transparent here — the optimizer just sees a longer vector. |

---

## 3. Existing Rust penalty infrastructure

Two layers carry per-smooth penalty state:

1. **`BlockPenalty`** (`src/block_penalty.rs`). Single non-zero k×k block at `(offset, offset)` of a notional p×p matrix. Used everywhere in PIRLS / REML for the `scaled_add_to`, `dot_vec`, `quadratic_form`, `scale` operations on each penalty. Effort is O(k²) instead of O(p²).
2. **`Vec<BlockPenalty>`** in `src/pirls.rs::fit_pirls` and `src/reml/*.rs`. The penalty list has one `BlockPenalty` per smooth currently; `lambda[i]` pairs 1-to-1 with `penalties[i]`.

**Reparametrization** (`src/reparam.rs`):

- `gam_reparam_core(rs, sp, deriv, …)` takes square-root factors `rs[i]` and smoothing params `sp[i]`. Currently the assumption is `rs.len() == sp.len()` (one root per smoothing param). Under `select=TRUE` this still holds — each *block* becomes its own entry in `rs` — so the reparam machinery does not need structural changes.

**Where the extra penalty would slot in:**

The cleanest fit is **per-smooth penalty lists**, mirroring mgcv's `m$smooth[[j]]$S` of length 2:

```rust
pub struct SmoothTerm {
    ...
    pub penalty: Array2<f64>,             // existing S_orig
    pub null_space_penalty: Option<Array2<f64>>, // NEW — S_null when select=TRUE
    pub lambda: f64,                      // for S_orig
    pub lambda_null: Option<f64>,         // NEW — for S_null
}
```

And the PIRLS / REML call sites that today iterate `&[BlockPenalty]` of length `n_smooth` would instead iterate a list of length `n_blocks` where `n_blocks = n_smooth + n_smooth_with_select` (one extra block per smooth under `select=TRUE`). Each block keeps its own offset/size, and the outer-loop log-sp vector grows from `n_smooth` to `n_smooth + n_select_extras`.

A flatter alternative: keep `Vec<BlockPenalty>` as the canonical representation everywhere and add a `smooth_idx: usize` field to `BlockPenalty` so we can recover which smooth a block belongs to. This is closer to mgcv's `Sl.setup` output.

---

## 4. Integration plan — sub-tasks

| # | Task | Effort | Risk |
|---|---|---|---|
| S1 | API surface: add `select: bool` to `Gam(...)` / formula spec. Plumb through `GamBuilder` → `SmoothTerm` construction. | 0.5d | Low. |
| S2 | Null-space penalty construction. Helper `fn build_null_space_penalty(s: &Array2<f64>, tol: f64) -> Option<Array2<f64>>` that follows the mgcv branch logic: detect diagonal-S layout (cheap path → `diag(0…1…)`), otherwise eigendecomp `S` and form `U_0 U_0'`. **Port verbatim from mgcv smoothCon:432–481.** | 1d | Medium — must match mgcv's `eps^0.66` tolerance for null-space detection. |
| S3 | Restructure penalty plumbing: per-smooth → flat `Vec<BlockPenalty>` with offsets, where each smooth contributes 1 or 2 blocks. Update all sites in `src/pirls.rs`, `src/reml/*.rs`, `src/lib.rs` that assume `penalties.len() == n_smooth`. | **2d** | **High** — touches ~20 call sites. Easy to miss one. Need targeted unit tests for each. |
| S4 | Outer-loop dimension change. Confirm Newton optimizer in `src/newton_optimizer.rs` / `src/reml/system.rs` handles longer log-sp vectors transparently. Update Mp computation (`src/lib.rs:1320-1330`) — with null-space penalty, null-space dim drops to 0 for each select-smooth; Mp becomes just the global intercept dimension. | 1d | Medium — Mp is load-bearing for REML score normalization. |
| S5 | REML gradient / Hessian. The Tk and `log|S|+`-derivative paths in `src/reml/system.rs` and `src/reml/tk_kkt.rs` already handle multiple penalty blocks — extending to per-smooth block pairs should be a no-op once S3 lands. Verify with FD. | 0.5d | Low (probably; S3 covers most of it). |
| S6 | Parity test against `test_data/select_parity_basic.json` — assert `len(sp) == 2 · n_smooth`, individual `sp` values within rtol 1e-4, edf within rtol 1e-5, coefficients within rtol 1e-6. Specifically verify the **null-space λ blows up for x2** (sentinel value of e.g. `sp[3] > 1e4`). | 0.5d | Low. |

**Total: ~5.5d.**
**Hardest part: S3 (per-smooth → flat penalty list)** — invasive refactor touching every code path that loops `n_smooth`. The other steps are localized.

---

## 5. Parity protocol

Asserted against `test_data/select_parity_basic.json`. The structural assertions are load-bearing.

1. **`len(sp) == 2 · n_smooth`** when `select=True`. Hard equality, not a tolerance. If we get this wrong the rest is meaningless.
2. **`sp` order**: the captured layout is `[S1_orig, S1_null, S2_orig, S2_null, …]`. Document and match.
3. **Null-space λ grows for unused predictors.** Sentinel: `select.sp[3] > 1e4` (the x2 null-space slot) — this is the variable-selection signature.
4. **Coefficients** — rtol 1e-6 against `select.coefficients`.
5. **Per-smooth edf** — rtol 1e-5. **Important**: x2's edf should drop from ~1.3 (no-select) to ~0.38 (select) — that's the selection in action and a precise number to match.
6. **REML score** — rtol 1e-6.
7. **Penalty-matrix block structure** — for each smooth, `len(S_list) == 2`, `S[1]` has expected rank (`p − rank(S[0])`), and `S[1]` is symmetric (it's a projector U_0 U_0' so this should hold to machine precision).
8. **No-regression on `no_select` fit** — the `select=False` reference path must continue producing the captured baseline coefficients to rtol 1e-6. This catches accidentally-on-by-default refactor mistakes.

---

## 6. Open questions / risks

- **Tolerance for null-space detection.** mgcv uses `eps^0.66` of the largest eigenvalue. Picking a tighter/looser tolerance changes the rank of `S_j^null`. Document and match exactly.
- **Already-diagonal smooths.** For random-effect smooths (`bs="re"`, where `S = I`), there is no null space and `select=TRUE` is a no-op. Need to handle this branch explicitly (mgcv's `smoothCon` does — when `nsm == 1` and `S = diag(S11, …, S11, 0, …, 0)`, it skips the eigendecomp).
- **Interaction with `ocat`** (site #4). Once both features exist, the outer-search vector becomes `[log_λ_orig (per smooth); log_λ_null (per smooth); θ]`. The Hessian assembly has to handle all three blocks. This is mentioned in `OCAT_DESIGN.md` §6 too.
- **`reparam.rs` assumption.** It currently treats `rs[i]` and `sp[i]` as 1-to-1. Confirm the call site passes per-block roots/sp under select=TRUE; this is structural rather than algorithmic but easy to miss.
- **Gradient FD-check at multiple `select=TRUE` configurations** before shipping. The doubled-λ space has not been exercised by any existing parity test in the repo (`tests/parity/` shows no `select=TRUE` cases).
