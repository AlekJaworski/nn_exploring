# Plan: routing `smooth.rs` outer Newton through the discrete fast path

**Status:** scoping doc, no implementation yet.
**Authored by:** Plan subagent 2026-05-14 (`a6e5ca6a6064468c0`).
**Prerequisite:** D3 + D4 + D6 landed (head `89eed8c`). m_max cap heuristic in flight.

## Context recap

After D6 (`89eed8c`), `cache.discrete: Option<DiscreteDesign>` exists and the
**inner** PIRLS routes through `fit_pirls_tdist_discrete`. The **outer** Newton
in `smooth.rs::optimize_reml_newton_multi_with_xtwx` receives a `cached_xtwx`
that was built via `compute_xtwx_dispatch(cache.discrete.as_ref(), …)` from
`gam_optimized.rs:844, 916`. But every downstream function
(`reml_joint_gh_gamfit4_tdist_analytic` → `tdist_gdi2_native`,
`reml_hessian_mgcv_exact_ift`, `tk_kkt_hessian_analytical`, etc.) re-uses the
full `n × p` design `x` for **everything except** the seed `X'WX`. `smooth.rs`
itself never sees a `DiscreteDesign`.

## 1. Inventory of remaining full-`x` usage

### 1a. In `src/smooth.rs` directly

| line | call site | category | notes |
|---|---|---|---|
| `smooth.rs:374-388` (`compute_xtwy_helper`) | `x.t().dot(&y_weighted)` | **X'Wy-shaped, NOT caught by D4** | Should call `compute_xtwy_dispatch`. Trivial swap. |
| `smooth.rs:1197` | `compute_xtwx_dispatch(None, x, &w_local)` | **D4 swept, `None` passed** | Needs `Some(disc)`. |
| `smooth.rs:1201, 1335` | `compute_xtwy_helper(x, &w_local, &y_local)` | **X'Wy-shaped, NOT caught by D4** | Trivial swap. |
| `smooth.rs:1216, 1337` | `compute_xtwx_cholesky(&xtwx_local)` | Operates on `p × p`, not on `x` | No change needed. |
| `smooth.rs:1401-1410` (call into `reml_joint_gh_gamfit4_tdist_analytic`) | passes `x` | Tk·KK' & joint Hessian path | Must thread `disc` into the analytic. |
| `smooth.rs:1416-1494, 1516-1559` (closed-form/IFT gradient + Hessian) | pass `x` | Heavy: per-row Tk·KK', `add_weighted_xtx_inplace`, `xa = x.dot(&a_inv)`, etc. | Same problem — needs disc threaded into these reml fns. |
| `smooth.rs:1429-1438` (`reml_gradient_finite_diff`) | passes `x` | Inner re-evaluates score → score → `compute_xtwx_dispatch(None, …)` chain | Once dispatch carries disc the whole FD subtree benefits. |
| `smooth.rs:2619` | `compute_xtwx_dispatch(None, x, w)` | Fellner-Schall path | Swap `None` → `Some(disc)`. |
| `smooth.rs:2679, 2690` | `x.dot(beta)` (residual / RSS) | **Pure read of η at converged β** | Replace with `compute_eta_discrete(disc, beta)`. |
| `smooth.rs:988` (`initialize_lambda_adaptive`) | `x[[i, j]] * x[[i, j]] * w[i]` scan | One-shot at fit start | Negligible; leave. |

### 1b. Reached transitively from `smooth.rs` (still full-`x`)

These live in `src/reml/mod.rs` and `src/reml/tk_kkt.rs`. They are the **actual
hot loops** that need either (a) discrete-aware variants or (b) a way to consume
`compute_eta_discrete` / `B = X̃' diag(t) X̃`-style assemblies:

| fn | call site → file:line | shape | cost driver |
|---|---|---|---|
| `tdist_gdi2_native` | `reml/mod.rs:1152` | `add_weighted_xtx_inplace(&mut ai, x, &wi, …)` ×ntot, `x.t().dot(&v)` ×O(ntot²), `eta = x.dot(beta)`, `eta1[k] = x.dot(b1[k])`, `eta2[k,j] = x.dot(b2[k,j])` | **DOMINANT.** ~O(n·p²·ntot²) per call. |
| `tk_kkt_hessian_analytical` | `reml/tk_kkt.rs:208` | `xa = x.dot(&a_inv)` (n·p²), `eta1 = x.dot(&b1)`, per-row `B_k = Xᵀ diag(Tk) X` (n·p² per k) | Active for InvGauss/Binomial/QuasiBinomial, also gated by `MGCV_TK_GRAD`. |
| `reml_gradient_mgcv_exact_ift_inner_at_beta` and `_ift` | `reml/mod.rs:1767, 2050` | `eta = x.dot(beta)`, `xtwx = x.t().dot(&wx)`, `eta1 = x.dot(&b1)`, `xa = x.dot(&a_inv)` | Used for non-canonical-link non-Gaussian (InvGauss+log, etc.) — secondary. |
| `reml_hessian_mgcv_exact_ift` | `reml/mod.rs:2285` | same shapes as above + `add_weighted_xtx_inplace` | Secondary. |
| `reml_gradient_mgcv_exact_closed_form` and `_hessian_…_closed_form` | `reml/mod.rs:667-797` | `fitted = x.dot(beta)`, working-RSS form mostly via `xtwx`/`xtwy` | Already 70 % on `p × p` — small remaining win. |
| `reml_gradient_multi_qr_…` family | `reml/mod.rs:2959, 3052, 3315, …` | `x_dbeta = x.dot(&dbeta_drho)` for each smooth, ditto for second derivative | Default (non-mgcv-exact) Gaussian-style path. |

## 2. Architectural option assessment

Three plausible routes:

**A. Thread `Option<&DiscreteDesign>` through public APIs of the gradient/Hessian functions and `optimize_*`.**
Pros: explicit, lifetime-safe, no global state, type system enforces "did you forget to plumb it?". Cons: blast radius is wide — every `reml_*` function above gains a parameter, every call site updates.

**B. Stash `Option<&DiscreteDesign>` in `SmoothingParameter` (like `family_cell`).**
Pros: zero API change to `reml_*` free functions if we instead bake disc into a *new* set of `_discrete` variants that the existing functions dispatch into. Cons: borrow lifetime issues — `DiscreteDesign` lives on `FitCache` which lives on `gam_optimized`. Would need `Arc<DiscreteDesign>`. Already have precedent with `family_cell` as `Arc<Mutex<_>>`.

**C. New `compute_*_discrete` helper family — discrete-specific kernels for the four hot shapes:**
- `η = X·v` → already have `compute_eta_discrete`.
- `B_k = Xᵀ diag(d) X` → **new kernel**, scatter-gather variant of `add_weighted_xtx_inplace`.
- `xa = X · A⁻¹` (n × p) → cannot avoid the n × p output directly. **No win**; ergo for `xa` we either keep dense or accept that "Tk·KK' is fundamentally `O(n·p)`-bound".

**Recommendation:** B + C, applied incrementally. Add `pub disc: Option<Arc<DiscreteDesign>>` to `SmoothingParameter`, plumbed from `gam_optimized.rs` exactly like `family_cell`. Land new `_discrete` kernels for the cheap shapes. The hot-loop reml functions get **internal** dispatch.

Avoid option A: a 30-site API sweep through `reml_*` functions risks the same long PR that D4 was, but on much more sensitive code.

## 3. Joint outer Newton interaction (the trap)

`tdist_gdi2_native` is the engine of 0.15.0's joint Newton actuation. Its math
derives `b1[t]` for the **two θ-axes** (df, σ²) from `-0.5 · A⁻¹ · X' · det_th[t]`
(mod.rs:1194). Note `det_th[t]` is a length-`n` vector that depends on per-row
`eta`, and `det2_th[t]` is also length-`n` and **changes per outer Newton iter**.

- `x.t().dot(&det_th[t])` is **always** per-iter, per-trial-(df,σ²). It has a
  clean scatter-gather analogue: `Σ_i X̃[k[i],:] · v[i]` = `X̃ᵀ · t` with
  `t[μ] = Σ_{i: k[i]=μ} v[i]`. Cost `O(n + m·p)` vs `O(n·p)` — a real win at
  `n=5157, m=767`.
- `add_weighted_xtx_inplace(&mut ai, x, &wi, 1.0)` is the **N×P²** assembly
  that already has a discrete analogue. `compute_xtwx_discrete` only assumes
  nonnegativity in the diagonal-block fast path; a small refactor lifts that.
- `eta = x.dot(beta)`, `eta1[k] = x.dot(b1[k])`, `eta2[k,j] = x.dot(b2[k,j])`
  — pure `compute_eta_discrete`. The function already exists.

**What would break naively:** the symmetry between b1, η1, b2, η2 in the joint
Newton actuation relies on `η_i = X_i β`. Mixing discrete `compute_eta_discrete`
for `η` with raw `x.t().dot(det_th[t])` for `Xᵀv` injects a `O(1/m)` skew that
compounds over the b1→η1→a1→η2 chain. **The invariant: every `Xᵀv` and every
`Xv` operation in `tdist_gdi2_native` must be replaced together — partial
conversion violates the X̃-consistency the joint Hessian needs.**

**Track to defer:** `tk_kkt_hessian_analytical`'s `lev_uw` computation. Its
`xa = x.dot(&a_inv)` produces an n×p object used in `tr(C_k · S_j · A⁻¹)`
traces, and there's no obvious scatter-gather win for that trace shape.

## 4. Effort estimate

| Sub-task | Where | LOC | Hours |
|---|---|---|---|
| **T1.** Plumb `Option<Arc<DiscreteDesign>>` through `SmoothingParameter` and `optimize_with_beta_xtwx_and_pirls_callback`. | smooth.rs +20, gam_optimized.rs +10 | 30 | 1 |
| **T2.** New kernel `compute_xtv_discrete(disc, v) -> Array1<f64>`. | discrete.rs | 90 | 2 |
| **T3.** New kernel `add_weighted_xtx_discrete(out, disc, d, scale)`. | discrete.rs | 60 | 2 |
| **T4.** Refactor `tdist_gdi2_native` to accept `disc` and dispatch internally. | reml/mod.rs | ~80 | 3 |
| **T5.** Pass `disc` from smooth.rs:1401 and smooth.rs:1416-1494 into `reml_*` analytics. | reml/mod.rs, smooth.rs | 20 | 1 |
| **T6.** Swap `compute_xtwy_helper` → `compute_xtwy_dispatch` at smooth.rs:374, 1201, 1335; `x.dot(beta)` → `compute_eta_discrete` at smooth.rs:2679, 2690. | smooth.rs | 15 | 0.5 |
| **T7.** Unit tests + parity sanity. | tests/test_discrete_smooth_newton.rs (new) | 200 | 2 |

**Total: ~500 LOC, ~11.5 hours, 1.5 subagent-days.**
**Decomposable: T1+T6 = smallest viable PR (~1 hour). T2+T3+T4+T5+T7 = the hot loop, 1 subagent-day.**

## 5. Risk callouts — load-bearing invariants

1. **X̃-consistency in `tdist_gdi2_native`.** Every `Xᵀv` and every `Xb` in lines 1188-1288 must use the *same* compressed representation. Reviewer must check that every appearance of `x` in this function is replaced in the same commit.
2. **σ² / df sync via `family_cell`.** Share via `Arc`, not clone per-trial. No `Mutex` — disc is read-only inside the Newton.
3. **Joint Hessian preconditioning** (smooth.rs:1608-1638). Stress-test on a fixture where smooths approach saturating-λ in the quantile-grid regime.
4. **`compute_xtwy_helper`** is easy to miss because the function name doesn't shout "X'Wy". Two callers at 1201 and 1335 must be swept.
5. **`reml_gradient_finite_diff`** and **`reml_hessian_finite_diff`** automatically benefit once the score's X'WX/X'Wy path is on discrete — but the FD assembly doesn't re-use cached xtwx between perturbations. Separate perf-only follow-up.
6. **The non-blas branch.** Keep the discrete-feature gating *additive* to the existing blas gating, not orthogonal.

## 6. Recommended first sub-task (the "smallest first" PR)

**T1 + T6 only.** Specifically:

- Add `pub discrete: Option<Arc<DiscreteDesign>>` to `SmoothingParameter`.
- Set it from `gam_optimized.rs` right next to `smoothing_params.family_cell = …` (line 791).
- Replace the four `None` passes in `smooth.rs` (`1197, 2619` and inside the helper at `374`) and the two `x.dot(beta)` reads at `2679, 2690` with `self.discrete.as_ref().map(|a| a.as_ref())` / `compute_eta_discrete(disc, beta)`.
- Add a unit test in `tests/` confirming that for the production fixture, the converged `lambda` from the discrete path matches the dense path to 1e-3 rel.

**Estimated landing: 1-1.5 hours, ~50 LOC.** This alone delivers a measurable
speedup on the production fixture because every outer Newton iter that the
joint analytic path *does not* cover (Fellner-Schall fallback, Quantile, NegBin
profile) currently does a full-`x` X'WX in `compute_xtwy_helper`. **Crucially,
it does NOT touch `tdist_gdi2_native`** — that's T4, which is the heavy lift
and where a YOLO swap will hurt.

## 7. Verification strategy

1. **Unit kernel parity** — pure-dedup to 1e-12, quantile-grid to 5e-3 abs.
2. **End-to-end converged-`lambda` parity** on existing parity JSONs. Required: max `|Δβ|` ≤ 5e-3 abs / 1e-3 rel, `Δ REML` ≤ 5.0 abs.
3. **Perf battery** on `data/sale_price_fixtures/*.parquet` split_0. **Account for the in-flight m_max cap landing**: rerun BEFORE baseline once m_max cap is committed, then measure AFTER. Use `MGCV_PROFILE=1` to break out grad/Hessian time.

## YOLO-mode warnings

- **"Just swap every `x.dot(…)` for `compute_eta_discrete(disc, …)` inside `tdist_gdi2_native`"** — without also swapping every `x.t().dot(…)`, the b1/b2 derivatives are computed against the dense X but applied through compressed η. Pure-dedup masks this; quantile-grid does not.
- **"Compute `xa = X · A⁻¹` once and cache it across Newton iters"** — `A⁻¹` changes every iter.
- **"Plumb `disc` only into the joint path and leave `reml_gradient_mgcv_exact_closed_form` dense"** — that's actually the right *first* move (T4 only does TDist's hot path) but reviewers must keep an inventory of which gradient functions are reachable from which family/score-formula gate.
- **"Disc is shared, lock it"** — `Arc<DiscreteDesign>` (no Mutex) is correct; copying the `family_cell` pattern blindly adds an unnecessary mutex on the hot path.

## Critical files for implementation

- `/home/alex/vibe_coding/nn_exploring/src/smooth.rs`
- `/home/alex/vibe_coding/nn_exploring/src/reml/mod.rs`
- `/home/alex/vibe_coding/nn_exploring/src/discrete.rs`
- `/home/alex/vibe_coding/nn_exploring/src/reml/tk_kkt.rs`
- `/home/alex/vibe_coding/nn_exploring/src/gam_optimized.rs`
