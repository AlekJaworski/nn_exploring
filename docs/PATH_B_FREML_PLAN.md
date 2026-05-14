# Path B: Parallel fREML Optimizer — Implementation Plan

**Status:** scoping / planning (read-only). Authored by Plan subagent `a45b41c1c5e0a4df4` on 2026-05-14.
**Triggered by:** `docs/MGCV_BAM_ALGORITHM_INVESTIGATION.md` — the 55× perf gap to `mgcv.bam(scat, fREML, discrete=TRUE)` is algorithmic (fREML on cached XX) not kernel-level.
**Prerequisite:** Path C was ruled out as infeasible (`a2207e606cd9686cf` feasibility report); mgcv itself does not have O(p³) scat-joint-REML on cached XX. Decoupling θ from λ IS the architectural pivot.

**Reference source:** `/tmp/mgcv-src/mgcv/R/bam.r:430` (`bgam.fitd`), `/tmp/mgcv-src/mgcv/R/fast-REML.r:1585` (`Sl.fitChol`), `/tmp/mgcv-src/mgcv/R/fast-REML.r:1740` (`fast.REML.fit`).

## TL;DR — second design pass (reuse audit)

The first-pass estimate of **8-11 subagent-days** assumed Path B was a clean port from mgcv R source. A reuse audit reveals we already have ~80% of the supporting infrastructure. **Revised estimate: 4-5 subagent-days** if we land 3 small refactor PRs first that expose existing helpers. See §8 (Reuse audit) and §9 (Refactor plan) below — those are the practical roadmap.

## 1. Components to port (with current-state audit)

| mgcv component | LOC (R) | LOC (Rust est) | Current Rust analog | Gap |
|---|---|---|---|---|
| `Sl.setup` | ~200 (R/fast-REML.r:68) | n/a (reuse) | `BlockPenalty` in `src/block_penalty.rs:16` + smooth construction in `src/smooth.rs` | Need to expose: per-block `rank`, per-block `start/stop` (offset already there), `lambda` field, **per-block `Skb = S_k β` term-multiply** (`Sl.termMult`/`Sl.mult`), and a per-block "linear singleton vs multi-S" tag. |
| `ldetS` | ~250 (R/fast-REML.r:762) | ~120 | Partial: `pseudo_determinant` and `log_det_s = log_lambda_sum + log_pseudo_det_sum` in `src/reml/mod.rs:441,576`. | Need standalone `ldet_s(sl, rho) → {ldet, ldet1[k], ldet2[k,j]}` with **first AND second derivatives**. |
| `Sl.fitChol` | ~95 (R/fast-REML.r:1585) | ~250 | None | Closed-form REML step on cached `XX = X'WX`, `f = X'Wy`, `yy = y'Wy`. Returns `{β, grad, hess, step, db = dβ/dρ, PP = (XX+S)^-1, ldetS, ldetXXS}`. |
| `Sl.iftChol` | ~80 (R/fast-REML.r:1405) | ~120 | None (we have `compute_b1_ift` at `src/reml/mod.rs:1575` but consumes our `RemlSystem` not a Cholesky factor) | `dβ/dρ_k = -(XX+S)^{-1} dS/dρ_k · β`, plus `bSb1`, `bSb2`, `rss2`. |
| `d.detXXS` | ~40 (R/fast-REML.r:1329) | ~80 | Related trace machinery in `trace_product_dense` (`src/reml/mod.rs:971`) | `d1[k] = tr(S_k PP)` and `d2[k,j] = -tr(PP S_k PP S_j)`. |
| `Sl.addS` | ~25 (R/fast-REML.r:1016) | ~20 | `BlockPenalty::scaled_add_to` (`src/block_penalty.rs:71`) | Trivial wrapper for singletons. |
| `Sl.initial.repara` | ~70 (R/fast-REML.r:517) | ~150 | `src/reparam.rs` has the related machinery. | **Decision in B1**: apply mgcv-style initial reparam to `XX` and `f` before `fitChol`? Recommend yes. |
| `bgam.fitd` outer body | ~340 (R/bam.r:430–768) | ~250 | None for "IRLS-once" pattern. `fit_pirls_tdist` runs PIRLS to convergence. | Brand-new driver. |

**Total new Rust LOC: ~1000–1200, plus ~150 LOC of tests.**

## 2. Integration points in `mgcv_rust`

- **New module:** `src/reml/fastreml.rs` (sibling to `mod.rs`, `system.rs`, `tk_kkt.rs`). Houses `compute_sl_fitchol_step`, `compute_ldet_s_with_derivs`, `compute_d_det_xxs`, `compute_sl_ift_chol`. Pure functions.
- **New driver:** `src/pirls.rs::fit_pirls_fastreml` (next to `fit_pirls_tdist`). Owns the outer loop: IRLS-once → X'WX/X'Wy → `Sl.fitChol` → step control → repeat.
- **Method enum:** extend `OptimizationMethod` in `src/smooth.rs:495–499` with `FastREML` variant.
- **Dispatch:** `src/gam_optimized.rs::fit_optimized_with_scale_method` (line 477) adds a new branch that bypasses the inner Newton entirely.
- **Python API:** `python/mgcv_rust/_fitter.py:306` already accepts `method='fREML'` (currently routes to REML). Wire to `FastREML` in `_make_native()` / PyO3 binding.
- **Discrete cache:** New driver checks `cache.discrete.is_some()` and uses `compute_xtwx_discrete` / `compute_xtwy_discrete` (D3 kernels, already in `src/discrete.rs:395,539`). Falls back to GEMM when absent.
- **Extended families:** scat / NegBin / Tweedie θ becomes an **outer 1-D Newton** (mgcv `estimate.theta` in `efam.r:5`) applied AFTER each `Sl.fitChol` step. Scale `log.phi` stays inside `fitChol`.

## 3. Discrete-binning interaction

`Sl.fitChol` only ever touches `XX` (p×p), `f` (p), `yy` (scalar). The compressed assembly happens once per outer iter. Our `compute_xtwx_discrete`/`compute_xtwy_discrete` kernels drop in directly. **No predict-path change needed.**

## 4. Sub-task breakdown (8–11 subagent-days)

Ordered by dependency. Each task is a self-contained PR with unit tests.

**B1 — Penalty interface gaps (1 day, ~200 LOC).**
- Add `BlockPenalty::term_mult(beta) → Array1<f64>` (matches `Sl.termMult` per-term).
- Add `BlockPenalty::log_det(rho, deriv=2) → (f64, grad_scalar, hess_scalar)` for singleton blocks.
- Multi-`S` blocks: defer, stub.
- Tests: `term_mult` vs `BlockPenalty::dot_vec` consistency; `log_det` vs `pseudo_determinant + rank * rho`.

**B2 — `ldet_s` with first AND second derivatives (1 day, ~120 LOC).**
- `compute_ldet_s(&[BlockPenalty], &[f64] rho) → (f64 ldet, Array1 ldet1, Array2 ldet2)`.
- Singleton blocks: `ldet1[k] = rank_k`, `ldet2[k,k] = 0`. Trivial.
- Tests: FD parity at 1e-6 abs on a 3-smooth fixture.

**B3 — `Sl.fitChol` core (2 days, ~250 LOC).**
- `compute_sl_fitchol_step(sl, XX, f, rho, yy, log_phi, phi_fixed, nobs, mp, gamma) → SlFitCholResult { beta, grad, hess, step, db, pp, ldet_s, ldet_xxs }`.
- Sub-helpers: `Sl.iftChol` (B3a, ~120 LOC), `d.detXXS` (B3b, ~80 LOC).
- Tests: `grad` and `hess` vs FD at 1e-6 abs.

**B4 — `fit_pirls_fastreml` driver, Gaussian & exponential families (2 days, ~250 LOC).**
- Loop body per `bam.r:430-768`. Step-length control per `bam.r:749-756`.
- Tests: parity vs `bam(method='fREML', discrete=TRUE)` on gaussian fixture, coef tol 5e-3 abs.

**B5 — Extended-family θ outer 1-D Newton (2 days, ~200 LOC).**
- After each `Sl.fitChol` step, call `estimate_theta_outer_1d` for TDist / NegBin / Tweedie. Mirrors mgcv `estimate.theta` (efam.r:5).
- TDist: 2-D Newton on `(log σ², log(df-2))` at fixed β.
- Tests: fitChol+θ-outer parity vs `bam(family=scat, fREML, discrete=TRUE)`. Coef tol 5e-3, df tol 5%, σ² tol 1%.

**B6 — Python API plumbing + dispatch (1 day, ~150 LOC).**
- Wire `Gam(method='fREML')` end-to-end.
- Add `OptimizationMethod::FastREML` enum variant. Branch in `gam_optimized.rs::fit_optimized_with_scale_method`.
- Tests: end-to-end smoke on production fixture.

**B7 — Parity battery on existing fixtures (1 day).**
- Run mgcv parity battery comparing `Gam(method='fREML')` against `mgcv::bam(method='fREML', discrete=TRUE)` (NOT `gam(method='REML')`).
- Investigate 7 known weighted-fixture REML failures (`project_edf_off_by_one_finding`) — may auto-sidestep on fREML path.

**B8 — Production bench + OpenMP scatter prep (1 day).**
- Run `scripts/python/bench_binning_baseline.py --binning auto --method fREML`. Target: 5–20× vs current REML.
- Profile `compute_xtwx_discrete` under the new driver; assess OpenMP scatter (D10) for next phase.

**Total:** 8–11 subagent-days. Sequenced strictly: B1 → B2 → B3 (B3a, B3b in parallel after B3 scaffold) → B4 → B5 (independent of B4 once B3 lands) → B6 → B7 → B8.

## 5. Test posture

- **Unit (per task):** finite-difference checks at 1e-6 abs on small fixtures (n=200, p=20, 3 smooths). `tests/test_fastreml_unit.rs`.
- **Parity (B7):** end-to-end vs `mgcv::bam(method='fREML', discrete=TRUE)`. Tolerance 5e-3 abs on β. Use **bam-fREML** as reference, NOT gam-REML.
- **Production bench (B8):** `scripts/python/bench_binning_baseline.py --method fREML`. Target: 5–20× speedup; with OpenMP scatter another 2–3×.
- **Regression guard:** existing `method='REML'` path is touched only by the dispatch fork. Run existing REML parity battery to confirm no regression.

## 6. Risks & deferred items

- **Stage-failure inheritance:** 7 known weighted-fixture REML failures may auto-sidestep on fREML path (different β path). Investigate in B7 — do not block on fixing.
- **Joint Newton actuation (0.15.0):** untouched. Path B is strictly additive. Users keep `method='REML'` for unbinned scat (joint Newton win), `method='fREML'` for binned production.
- **Multi-`S` blocks (tensor smooths):** B1/B2/B3 ship singleton-only. Audit at B1 — grep `src/smooth.rs` for tensor constructors.
- **`L` matrix + `rho.0` offsets:** mgcv supports both; production uses neither. Skip in v1, document as known limitation.
- **Parity tolerance:** `bam(fREML)` β differs from `gam(REML)` β by ~1e-3 due to IRLS-once vs IRLS-to-convergence algorithmic difference. Test against bam-fREML.

## 7. Open architectural questions

1. **API surface:** `method='fREML'` kwarg OR auto-detect from `discrete=True`? Recommend mirroring mgcv (explicit `method='fREML'`).
2. **Non-extended families under fREML:** Gaussian / Poisson / Binomial. Same driver as scat but skip B5 (no θ to estimate). Single driver with `family.has_extra_params()` branch.
3. **Long-term REML deprecation:** with fREML shipped, does REML stay forever? **PM call.** Recommendation: keep both at least one major release cycle.
4. **Dispatch default:** when user passes `discrete=True` and `method='REML'`, silently upgrade to fREML or honor their choice? Recommend honor — fail loudly if behavior surprises.

## 8. Reuse audit — what we already have

A second design pass on 2026-05-14 mapped each fitChol-needed building block to existing Rust code. Most pieces are already shipped; the genuinely-new code is small.

| Need (from §1) | Status | Where | Gap |
|---|---|---|---|
| Block-penalty structure | ✅ | `src/block_penalty.rs::BlockPenalty` (~250 LOC) | Add `log_det_with_derivs(rho) → (val, d1, d2)` (trivial: `rank·rho, rank, 0` for singletons). ~30 LOC. |
| Penalty term-multiply | ✅ | `BlockPenalty::dot_vec` (line 82) — exactly `S_k · β` | None. |
| Penalty add-to | ✅ | `BlockPenalty::scaled_add_to` (line 71) | None. |
| log\|S\|+ value | ✅ | `pseudo_determinant` + `log_lambda_sum` pattern (`src/reml/mod.rs:441,576`) | Extend to return first AND second derivatives (rank for singletons, 0 for the second). ~50 LOC. |
| log\|S\|+ derivs | ❌ | — | New: `compute_ldet_s_with_derivs(sl, rho) → (val, d1, d2)`. Singleton-only is enough for v1. ~50 LOC. |
| Cholesky factorisation | ✅ | `src/reml/system.rs::compute_xtwx_cholesky` (line 75) | None. |
| IFT-based dβ/dρ | ✅ partial | `src/reml/mod.rs::compute_b1_ift` (line 1575) | Different signature (takes `RemlSystem` not Cholesky factor); refactor or add a fitChol-shaped sibling. ~80 LOC. |
| log\|XX+S\| derivs | ✅ partial | `trace_product_dense` (line 971) + per-block trace machinery already used in `reml_hessian_*` | Wrap into `compute_d_det_xxs(sl, pp) → (d1, d2)`. ~50 LOC. |
| **Cache-once, eval-many pattern** | ✅✅ | **`src/reml/mod.rs::TweedieThetaCache` (line 529)** — this is the architectural twin of mgcv's `bgam.fitd` outer pattern. | Already does: build XX/f/log|H|/log|S|/tr_A once, evaluate score at multiple θ values reusing the cache. Generalise from Tweedie-only to family-agnostic. ~80 LOC. |
| `assemble_reml_system` with cached XX | ✅ | `src/reml/mod.rs:558` — signature already accepts `xtwx_local: &Array2<f64>` | Make `x` parameter optional (only needed for `fitted`); fitChol path supplies cached XX and computes fitted from compressed gather. ~30 LOC. |
| Compressed X'WX / X'Wy | ✅ | D3 kernels in `src/discrete.rs` (lines 395, 539, 576) | None — drop-in for fastreml driver. |
| Dispatch helper | ✅ | `src/reml/system.rs::compute_xtwx_dispatch` (line 253), `compute_xtwy_dispatch` (line 270) | None. |
| `OptimizationMethod` enum | ✅ | `src/smooth.rs:495` | Add `FastREML` variant. ~5 LOC. |
| Family interface | ✅ | `src/pirls.rs::Family` enum + per-family weight/dev derivations | None. |
| Single-step IRLS (for outer driver) | ❌ | `fit_pirls_tdist` (line 1811) loops to convergence | Refactor: extract `tdist_irls_step(...)` from the loop body. ~100 LOC refactor, zero behavior change. |
| scat ν profile | ✅ | `profile_df`, `profile_df_weighted` in `src/pirls.rs:1645,1672` | Currently called inside PIRLS; for fastreml driver, call outside the IRLS step. Same fn, different placement. |
| TDist σ²/df joint optimization | ✅ partial | `tdist_gdi2_native` joint Newton (0.15.0) | NOT used by fastreml — fastreml decouples θ from λ. Existing code untouched. |
| Newton-on-θ outer loop | ✅ | `OuterSearchVector` + `newton_1d_with_halving` / `newton_2d_with_halving` in `src/reml/search_vector.rs` (0.15.0 foundation) | Reuse as-is for the outer 2-D Newton on (log σ², log df). |
| **The kernel: Sl.fitChol step** | ❌ | — | New: `compute_sl_fitchol_step(sl, xx, f, yy, rho, log_phi, ...) → {β, grad, hess, step, db, pp}`. The genuinely-new piece. ~250 LOC. |
| **The driver: fastreml outer loop** | ❌ | — | New: `fit_pirls_fastreml(...)` — IRLS-once → fitChol → θ-outer → step control → repeat. ~150 LOC after the refactor. |

**Net new code:** ~600 LOC (down from ~1200 first-pass estimate).
**Refactor-and-reuse code:** ~250 LOC of changes to existing modules.
**Total impact:** ~850 LOC, of which ~250 is the true math kernel.

## 9. Refactor plan (3 prep PRs before B-tasks)

Land these three small refactors first to expose existing helpers. Each is byte-identical-behavior and ships its own unit tests.

### R1 — Generalize `TweedieThetaCache` → `OuterLinearCache<Family>` (0.5 day)

Pull `TweedieThetaCache` (`src/reml/mod.rs:529`) out of its Tweedie-specific naming. The struct already holds: `fitted`, `bsb`, `log_det_h`, `log_det_s`, `mp`, `tr_a`. Add: `ldet_s_d1: Array1<f64>`, `ldet_s_d2: Array2<f64>` (per-λ first/second derivatives of log|S|+). Rename `score_at_p` to a family-dispatched `score_at_theta(theta: &FamilyExtras)`.

Justifies itself: the existing Tweedie-θ-FD path (used by 0.15.0's Tweedie profile) becomes the first user of the generic API; fastreml's outer θ Newton becomes the second.

**Test:** the existing Tweedie θ-FD parity tests must pass byte-identical.

### R2 — Extract `irls_step()` from `fit_pirls_tdist` (0.5 day)

`fit_pirls_tdist` (`src/pirls.rs:1811`) is a 300-line loop with one IRLS step per iteration. Pull the body into `tdist_irls_step(y, eta_in, w_prior, sigma2, df, ...) → IrlsStepResult { working_w, working_z, eta_out, dev_components }`. The existing function becomes a thin loop:

```rust
for iter in 0..max_iter {
    let step = tdist_irls_step(y, &eta, ...);
    let xtwx = compute_xtwx_dispatch(disc, x, &step.working_w);
    // ... solve, update beta, eta
    if step.converged() { break; }
}
```

No behavior change. Fastreml's outer driver calls `tdist_irls_step` once per outer iter instead of looping.

**Test:** parity battery byte-identical (524/7 still passes).

### R3 — Penalty extensions: `BlockPenalty::log_det_with_derivs` (0.25 day)

Add to `BlockPenalty`:

```rust
pub fn log_det_singleton_with_derivs(&self, rho: f64) -> (f64, f64, f64) {
    let rank = self.estimate_rank() as f64;
    (rank * rho + self.log_pseudo_det(), rank, 0.0)
}
```

Plus a free function:
```rust
pub fn compute_ldet_s_with_derivs(sl: &[BlockPenalty], rho: &[f64])
    -> (f64, Array1<f64>, Array2<f64>);
```

That iterates the blocks and stacks. ~80 LOC total.

**Test:** FD parity at 1e-6 abs on a 3-smooth fixture.

### After R1+R2+R3 lands, the B-task sequence collapses:

| Task | Old estimate | New estimate | Reason |
|---|---|---|---|
| B1 (penalty interface) | 1 day | **subsumed by R3** | already shipped |
| B2 (ldet_s with derivs) | 1 day | **subsumed by R3** | already shipped |
| B3 (Sl.fitChol core) | 2 days | **1.5 days** | helpers in place |
| B4 (fastreml driver) | 2 days | **0.5 day** | R1 + R2 do the heavy lifting; driver is a thin loop calling existing pieces |
| B5 (θ outer Newton) | 2 days | **0.5 day** | newton_2d_with_halving reusable from 0.15.0; profile_df already exists |
| B6 (Python API) | 1 day | **0.25 day** | mostly enum-add + wire |
| B7 (parity) | 1 day | 1 day | unchanged |
| B8 (bench + OpenMP scope) | 1 day | 0.5 day | bench script already exists; OpenMP scoping not impl |

**Revised total: ~4.5 subagent-days (after R1+R2+R3 = ~1.25 days of refactor prep) = ~5.75 days end-to-end.**

The refactor PRs are also **standalone wins**: R1 makes the Tweedie code reusable, R2 enables future fREML-for-quantile, R3 just tidies a scattered computation. They earn their keep even if fastreml weren't shipping.

### Risk: the "single IRLS step" semantics

Our `fit_pirls_tdist` runs PIRLS to **convergence** per outer iter; mgcv's `bgam.fitd` runs **one** IRLS step per outer iter. After R2, the function exists in both forms (looped via existing `fit_pirls_tdist`, one-shot via `tdist_irls_step`). The semantic difference is genuine — mgcv's design assumes the outer λ Newton can handle a not-fully-converged β. We need parity testing to confirm our scat IRLS step is numerically compatible with the one-step regime. If it's not (e.g. needs warm-start specifically tuned), B4 inherits an extra ~0.5 day of stability work.

## Critical Files for Implementation

- `src/block_penalty.rs`
- `src/reml/mod.rs`
- `src/pirls.rs`
- `src/gam_optimized.rs`
- `src/smooth.rs`
- `src/discrete.rs`
- `python/mgcv_rust/_fitter.py`
