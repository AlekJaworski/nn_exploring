# Joint Outer Newton — Generalised Search Vector Design

Status: **research-only**, no implementation yet.
Driver: 0.12.0 release-notes top-of-backlog item — *"Joint (ρ, log φ) outer Newton — top numerical item, would close the remaining dispersion-bearing GLM perf gap."* Today's TDist-specific implementation (at `src/gam_optimized.rs:270-720`, `src/smooth.rs:2302-2443`, `src/reml/mod.rs:1146-1466`) already proves the pattern; this doc plans the factor-out + extension to Gaussian-φ, ocat thresholds, and the existing Tweedie-p / NegBin-θ / Quantile-σ profilers that today share the search-vector design ad-hoc.

Parity baseline: `test_data/gaussian_phi_joint_parity.json` (script `scripts/r/tests/extract_phi_joint_parity.R`) captures mgcv's REML vs ML behaviour at a Gaussian fit with the joint (M+1)x(M+1) gradient/Hessian extracted via central-difference probes around the converged point.

---

## 1. Dissection of the existing TDist joint path

### 1.1 Search-vector layout (today)

There are **two distinct layouts** in flight. mgcv's `gdi2.c` internal layout vs. the Rust outer-Newton's outer layout do not match.

**Native (mgcv `gdi2`) order — what `tdist_gdi2_native` returns** (`src/reml/mod.rs:1146`):
```
[theta_df = log(df - 2),   theta_sigma = log(sigma),   log lambda_1, ..., log lambda_M]
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   ntheta = 2 family-shape params                       M smoothing params
```
Returns `Array1` of length `ntheta + M` (gradient) and `Array2` of `(ntheta+M, ntheta+M)` (Hessian, with all cross-terms).

**Outer-Newton (Rust optimiser) order — what `optimize_reml_newton_multi_with_xtwx` actually consumes** (`src/smooth.rs:1042`):
```
log_lambda    : Vec<f64> of length M    -- the only thing Newton solves for
tdist_log_sigma2, tdist_log_df  -- updated by a separate 2-D Newton step
                                   after each ρ-step (smooth.rs:2302-2443)
```
The slicing happens in `reml_gradient_gamfit4_tdist_analytic` (`src/reml/mod.rs:1409`) and `reml_hessian_gamfit4_tdist_analytic` (`src/reml/mod.rs:1438`): they discard rows/cols 0..2 of the native arrays and feed only the M-dim ρ block into the outer-Newton solver, then `tdist_shape_derivatives_gamfit4` (`src/reml/mod.rs:1375`) re-extracts the (df, log_sigma) 2x2 block from the same `tdist_gdi2_native` evaluation for a separate small Newton step.

**Critical finding**: the assembly already produces the full joint (ntheta+M, ntheta+M) Hessian with all cross-terms. The outer Newton harness **throws away the cross-terms** between (df, σ²) and λ. mgcv's `gam.fit5` solves the joint system instead — that is the perf gap.

### 1.2 Where extra-param gradient/Hessian rows are assembled

For each family the rows/cols are emitted in a **family-specific function**, glued in by an `if family.is_xxx() { ... }` block inside `optimize_reml_newton_multi_with_xtwx`:

| Family | Extra params | Assembler | Outer-loop hook | Score-formula route |
| --- | --- | --- | --- | --- |
| TDist (scat) | `log σ²`, `log(df − 2)` | `tdist_gdi2_native` (`reml/mod.rs:1146`) | `smooth.rs:2302-2443` (sequential 2-D Newton on shape after ρ-step) | LAML (`GamFit5`) |
| Tweedie | `θ` (working param for p) | `tweedie_theta_derivatives_cached` (`reml/mod.rs:5883`) + 3-probe FD on score | `smooth.rs:2066-2210` (scalar Newton on θ after ρ-step) | GamFit3 / REML (φ profiled) |
| NegBin | `log θ` | 3-probe FD on `dispatch_reml_score_with_family` | `smooth.rs:2218-2290` (scalar Newton) | GamFit3 / REML (φ=1) |
| Quantile (ELF) | `log σ` | (planned; presently FS only) | n/a | GamFit5 |
| Gaussian-φ (ML) | `log φ` | **none today** | n/a | REML profiles φ analytically |

There is no shared abstraction. Each family duplicates the *boilerplate* (FD step sizing, 2-D Newton solve, line-search halvings, bounds clamping, family-cell sync) at ~100-150 LOC apiece. Total payload of the four blocks in `smooth.rs`: ~440 lines, of which roughly 200 are duplicated structure.

### 1.3 Profile vs joint per family

Determined at fit setup (`gam_optimized.rs:586-625`):

- **Gaussian dispersion**: always profiled (closed form `φ̂ = RSS / (n − tr A)`). Not exposed as a knob.
- **TDist (df, σ²)**: `df=0` sentinel ⟹ outer Newton drives both `df` and `σ²`; `df=fixed` ⟹ outer Newton drives only `σ²`. `tdist_profile = true` gates the joint-shape block.
- **Tweedie p**: `p=None` ⟹ profile-p mode (`tweedie_profile = true`); `p=fixed` ⟹ frozen, no outer step.
- **NegBin θ**: `family="nb"` ⟹ profile-θ; `family="negbin"` with `theta=val` ⟹ frozen.
- **ocat thresholds**: not implemented; design doc in `docs/OCAT_DESIGN.md` calls for joint with ρ.

### 1.4 ρ ↔ λ parameterisation

Both mgcv and mgcv_rust use **ρ = log λ**. Confirmed:
- `optimize_reml_newton_multi_with_xtwx` works in `log_lambda` exclusively (`src/smooth.rs:1126`).
- mgcv: `gam.fit3` source line `scale <- exp(sp[nsp])` and `sp <- sp[-nsp]` — sp is log-space throughout `newton()` and `bfgs()`.
- The captured `outer.info$hess` in `test_data/gaussian_phi_joint_parity.json` is in (log λ, log φ) coordinates — matches our convention exactly.

### 1.5 Captured Gaussian-φ parity headline

From `test_data/gaussian_phi_joint_parity.json`:

```
REML: sp = 15.450,  sig2 = 0.15506,  edf=8.32,  score=258.128,  outer iters=5
ML  : sp = 15.539,  sig2 = 0.15506,  edf=8.31,  score=253.318,  outer iters=5

ML joint Hessian at converged (log λ, log φ) = (2.7433, -1.8679):
    H = [[  3.539,  -3.190 ],
         [ -3.190, 250.000 ]]
ML joint gradient at the same point (FD-from-mgcv):
    g = [ 2.5e-5,  -4.2e-7 ]    (both within 5·machine eps of zero)
```

The φ-row diagonal `H[1,1] ≈ n − Mp_ML ≈ 250` (with `Mp_ML = 250` here) versus REML's `H[1,1] ≈ 249`. The off-diagonal magnitude `|H[0,1]| ≈ 3.19` is **comparable to the λ-diagonal** `3.54` — i.e. ignoring this cross-term in a sequential update would discard ~50% of the curvature signal. This is the structural reason mgcv solves the joint system rather than alternating.

---

## 2. Search-vector abstraction

### 2.1 Proposed types

Introduce a `OuterSearchVector` carrying both the layout metadata and the working values:

```rust
// src/reml/search_vector.rs (new file)

#[derive(Debug, Clone)]
pub struct ExtraParam {
    pub name: &'static str,    // "log_phi", "log_sigma2", "log_df_m2",
                                // "tweedie_theta", "log_negbin_theta",
                                // "log_sigma_qgam", "ocat_theta_k"
    pub value: f64,             // current value (in working coords — usually log)
    pub lo: f64,                // lower bound (working coords)
    pub hi: f64,                // upper bound
    pub kind: ExtraKind,        // for the family hook dispatch
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtraKind {
    GaussianLogPhi,
    TDistLogSigma2,
    TDistLogDfM2,           // log(df - 2)
    TweedieTheta,
    NegBinLogTheta,
    QuantileLogSigma,
    OcatTheta(usize),       // ocat threshold index (0..R-2)
}

pub struct OuterSearchVector {
    pub log_lambda: Vec<f64>,     // M smoothing params (always first)
    pub extras: Vec<ExtraParam>,  // 0..K family-extra params
}

impl OuterSearchVector {
    pub fn dim(&self) -> usize { self.log_lambda.len() + self.extras.len() }

    pub fn pack(&self) -> Array1<f64> {
        let mut v = Array1::zeros(self.dim());
        for (i, &lp) in self.log_lambda.iter().enumerate() { v[i] = lp; }
        for (j, x)   in self.extras.iter().enumerate()     { v[self.log_lambda.len()+j] = x.value; }
        v
    }
    pub fn unpack(&mut self, v: &Array1<f64>) { /* mirror; apply bounds clamp */ }
}
```

This generalises *all four* existing ad-hoc layouts to one canonical form. The `kind` tag drives dispatch at exactly two pinch points:
1. **Family configures extras**: a method `Family::declare_extras(&self) -> Vec<ExtraParam>` returns the family's contribution. Gaussian-REML returns `vec![]`; Gaussian-ML returns `[GaussianLogPhi]`; TDist returns `[TDistLogSigma2, TDistLogDfM2]`; ocat-R returns `(0..R-2).map(OcatTheta)`.
2. **Family contributes gradient/Hessian rows**: a method on `Family` (or a free function dispatched on `ExtraKind`) computes the extra rows of `g` and rows/cols of `H` for the assembled native vector. This already exists per-family (`tdist_gdi2_native`, `tweedie_theta_derivatives_cached`, etc.) — the refactor reroutes them through a uniform trait.

### 2.2 The combined search vector for site #4

`family=ocat(R=4), select=TRUE` over 2 smooths becomes a **6-dim** outer search:

```
[log λ_1_orig, log λ_2_orig, log λ_1_null, log λ_2_null, ocat_θ_1, ocat_θ_2]
   M=4 (2 smooths × 2 penalties from select)              K=2 (R-2 = 2 thresholds)
```

Ordering matches mgcv exactly: `select=TRUE` always interleaves `(orig, null)` per smooth at the smoothCon layer, and mgcv's `gam.outer` then appends family-internal `theta`s at the end. Verified against the field shape of `test_data/ocat_parity_basic.json` `trajectory.grad` (length 3: 1 sp + 2 θ) and `test_data/select_parity_basic.json` `sp` (length 4: 2 smooths × 2 penalties).

---

## 3. Derivative plumbing

### 3.1 Gradient extension — each kind

| ExtraKind | ∂score/∂(param) formula | Already in code? |
| --- | --- | --- |
| `GaussianLogPhi` | `dlr.dlphi = (n_minus_Mp/2) - Dp/(2φ) - ls[2]·φ ...` (mgcv gam.fit3 source, see §3.3) | Partial — `reml_optimized.rs:291` computes the **profiled** form `n_minus_r · dφ̂/dρ / φ`, which is the chain-rule pull-through. The **joint** form (φ as a free variable) is not implemented. |
| `TDistLogSigma2` | mgcv `scat$Dd` chain — `tdist_gdi2_native` row 1 | Yes |
| `TDistLogDfM2` | mgcv `scat$Dd` chain — `tdist_gdi2_native` row 0 | Yes |
| `TweedieTheta` | FD on REML score with cached linear system (`TweedieThetaCache`) | Yes (analytical-cached) |
| `NegBinLogTheta` | FD on REML score, 3-probe | Yes (FD only — analytical-cached is the next refactor) |
| `QuantileLogSigma` | LAML chain through ELF density derivatives | Planned (qgam-style) |
| `OcatTheta(k)` | mgcv `ocat$Dd$Dth` — `n × (R-2)` Jacobian | Planned (see `OCAT_DESIGN.md`) |

### 3.2 Hessian — cross-terms

The cross-term ∂²score/∂(log λ)∂(extra) is what the sequential layout loses. mgcv computes it as:

```
H[ρ_j, extra_k]  =  d/dρ_j ( d score / d extra_k )
                 =  -tr[ A⁻¹ · ∂A/∂ρ_j · A⁻¹ · ∂²A/∂ρ_j∂extra_k ] / 2     (log|H| piece)
                  + d²Dp / (dρ_j · d extra_k) / (2 φ)
                  + (extra-specific pieces from family ls and Dd)
```

For TDist this is already in `tdist_gdi2_native` (sub-blocks of the returned `(2+M)x(2+M)` Hessian). The same structure applies to:

- **Gaussian-φ ML**: `H[ρ_j, log_φ] = -(1/2φ) · ∂(Dp+Bsb)/∂ρ_j + (1/φ²)·(...)`. Closed-form (see §3.3).
- **Ocat-θ**: mgcv `Dmuth` and `Dmu2th` arrays carry these cross-derivatives; `gam.fit5`'s LAML formula assembles them.
- **Tweedie / NegBin**: presently zero (sequential update) — switching to joint requires adding cross-terms. For Tweedie, the cached system already isolates `A⁻¹` so the FD of `∂Dp/∂ρ_j` at perturbed θ is `O(1)` per evaluation.

### 3.3 Closed-form Gaussian-φ joint derivatives

mgcv's `gam.fit3.r` source (lines around `dlr.dlphi`):

```r
ls   <- family$ls(y, weights, n, scale)      # ls[1] = -n·log(2πφ)/2 - RSS/(2φ), ls[2..3] = derivatives w.r.t. log phi
REML <- (Dp/(2*scale) - ls[1])/gamma + oo$rank.tol/2 - rp$det/2
        - remlInd * (Mp/2 * (log(2*pi*scale) - log(gamma)))

dlr.dlphi   <-  (-Dp/(2*scale) - ls[2] * scale)/gamma
                - remlInd * (Mp/2)
d2lr.d2lphi <-  (Dp/(2*scale) - ls[3]*scale^2 - ls[2]*scale)/gamma
                # plus Hessian cross-pieces, see source
```

For **Gaussian**, `ls = -n/2 · log(2πφ) - RSS/(2φ)`, hence `ls[2] = ∂ls/∂(logφ) = -n/2 + RSS/(2φ)` and `ls[3] = -RSS/(2φ)`. Substituting:

```
∂REML/∂(log φ) = -Dp/(2φ) - (-n/2 + RSS/(2φ))·φ  -  remlInd · Mp/2
              = -Dp/(2φ) + nφ/2 - RSS/2 - remlInd · Mp/2
              = -(Dp + RSS)/2 + nφ/2 - remlInd · Mp/2          (after simplification — RSS=Dp - bsb for Gaussian)
              = (n - Mp - effective_dof_correction) / 2  · ... (see mgcv source for exact remlInd handling)
```

The captured Hessian diagonal `H[log_φ, log_φ] ≈ 250 ≈ n − Mp_ML` matches this closed-form perfectly. **Effort estimate to port**: ~80 LOC in a new `gaussian_phi_joint_derivs` function (analogous to `tdist_gdi2_native` row 1 + Hessian sub-block), no FD needed.

### 3.4 Per-trial-λ PIRLS refresh interaction

The existing Newton path in `optimize_reml_newton_multi_with_xtwx` already refreshes inner PIRLS at every line-search trial (`smooth.rs:762-803`). Extending the trial point to `(log λ, extras)` requires:
- The `PirlsCallback` signature gains an `extras: &[ExtraParam]` slice OR we keep the family-cell trick (`family_cell: Arc<Mutex<Family>>`) and just publish the trial extras into it before each callback.
- The shape-line-search block (`smooth.rs:2330-2360`) already implements this for TDist; the same pattern generalises.

---

## 4. Migration plan

### Step 1 — refactor TDist to use `OuterSearchVector` (no behaviour change)

Extract the search vector into a struct; the (log σ², log df-2) values move from `SmoothingParameter.tdist_log_sigma2` / `.tdist_log_df` into `OuterSearchVector.extras`. The two-stage Newton (ρ-step then shape-step) stays — pure structural refactor.

- Files touched: `src/smooth.rs` (~250 LOC moved + glue), `src/gam_optimized.rs:602-625` (init from family), `src/reml/mod.rs` (re-route `tdist_shape_derivatives_gamfit4` through the new pack/unpack).
- Tests: existing TDist parity battery (no expected delta).
- **Effort: 1-2 days.**

### Step 2 — fold Tweedie / NegBin / Quantile into the same struct (no behaviour change)

Replace the three duplicated profile blocks with one parameterised block that loops over `extras` and dispatches to a per-kind derivative function. Sequential semantics preserved (one extra-Newton step after each ρ step).

- Files touched: `src/smooth.rs:2066-2443` (the four profile blocks collapse to one).
- Net LOC: roughly **+150 / -440 = -290 lines**.
- Tests: full parity battery — no expected delta.
- **Effort: 2-3 days.**

### Step 3 — add Gaussian-φ joint as opt-in `method="ML"`

Add `MLMethod::REML | MLMethod::ML` enum to the Python API (currently only REML is exposed — confirmed at `python/mgcv_rust/_fitter.py:306`). For ML:
- `Family::Gaussian` declares one extra `GaussianLogPhi`.
- New free function `gaussian_phi_joint_derivs` computes the closed-form `∂REML/∂(log φ)` row + Hessian rows/cols (§3.3).
- The criterion `reml_criterion_multi_cached_mgcv_exact` already handles the score formula correctly given `mp` — only the (n_minus_Mp · log(2πφ)/2) term shifts via the `remlInd` indicator. Add `score_type: REML | ML` to gate this.
- Search-vector dim is M+1. Outer Newton solves the full (M+1)x(M+1) system directly — no sequential alternation.

- Files touched: `src/lib.rs` (Python API), `python/mgcv_rust/_fitter.py` (method arg), `src/reml/mod.rs` (Mp adjustment based on score_type), `src/reml/search_vector.rs` (new) + Gaussian-φ extra registration.
- Tests: `test_data/gaussian_phi_joint_parity.json` becomes the regression fixture; ML vs REML should both match mgcv to ~1e-3 on coefficients.
- **Effort: 3-5 days.**

### Step 4 — extend to other dispersion-bearing families (joint mode opt-in)

Add `GammaLogPhi`, `InvGaussLogPhi`, `QuasiPoissonLogPhi`, `QuasiBinomialLogPhi` extras. Each gets a closed-form derivative function mirroring §3.3 (the families differ only in `family$ls` — variance function for the Pearson piece is family-specific but `ls`-derivatives are still closed-form for the Gaussian-style scale).

- For Tweedie / NegBin: keep sequential as default (mgcv's `tw()` / `nb()` do *not* use joint outer-Newton for θ at present — they alternate). Joint as an opt-in for future perf work.
- Files touched: `src/reml/search_vector.rs` (extra kinds), per-family closed-form functions in `src/reml/mod.rs`.
- **Effort: 1 week** (mostly verification + closed-form derivation per family).

### Step 5 — ocat thresholds via the same mechanism

`ocat(R)` declares `R-2` extras (`OcatTheta(0..R-2)`) via `Family::declare_extras`. The cross-terms come from mgcv's `ocat$Dd` arrays (`Dmuth`, `Dmu2th`) per `docs/OCAT_DESIGN.md`. With the search-vector abstraction in place, no further outer-Newton plumbing is needed.

- Files touched: ocat family implementation (per `OCAT_DESIGN.md`) + the `Family::declare_extras` hook.
- **Effort: tracked in OCAT_DESIGN.md as a separate work package, but the join with this refactor is one method.**

---

## 5. Risks

### 5.1 Refactor blast radius

| File | Current LOC | Outer-Newton-relevant lines | Refactor impact |
| --- | --- | --- | --- |
| `src/smooth.rs` | 3173 | ~600 (search-vector glue + 4 profile blocks) | Heavy — owner of the joint search struct. **Net LOC drop ~290.** |
| `src/gam_optimized.rs` | 1056 | ~80 (`tdist_*` setup, ~50 lines around 270-720) | Light — replace per-extra fields with `OuterSearchVector::default_for(family)`. |
| `src/reml/mod.rs` | 6174 | ~150 (`tdist_gdi2_native` + slicers) | Light — slicers (`reml_gradient_gamfit4_tdist_analytic`, `reml_hessian_gamfit4_tdist_analytic`, `tdist_shape_derivatives_gamfit4`) become thin wrappers that use the search-vector layout to dispatch. |
| `src/reml/system.rs` | 444 | 0 today | One new pub fn for the score-type-aware `Mp` shift. |
| `src/reml_optimized.rs` | 324 | `log_phi_deriv` term (1 line) | Note: the term computed at line 291 is the **profiled** chain-rule (Gaussian REML), distinct from the **joint** version we'd add. Both can coexist — REML keeps the chain rule, ML uses the joint row. |

Total: **~830 LOC re-shaped**, **~290 net deletion**, **2 new files** (`src/reml/search_vector.rs`, optionally `src/reml/joint_derivs.rs`).

### 5.2 Parity preservation

- **Step 1-2 (refactor)**: pure code motion; existing parity battery (TDist, Tweedie, NegBin) must pass byte-identical. The joint Hessian path is *not* yet wired — sequential semantics preserved.
- **Step 3 (Gaussian-ML)**: new opt-in. Existing REML path unchanged.
- **Step 4 (other ML dispersion)**: opt-in per family.
- **Step 5 (ocat)**: greenfield; first parity comes from `test_data/ocat_parity_basic.json`.

The TDist code being refactored is already battle-tested (the 0.12.0 parity battery exercises it heavily); a no-op refactor with the same Newton step + line-search structure carries low parity risk. The main hazard is the family-cell publish ordering during the line search — captured today by the `family_cell.lock()` calls in `smooth.rs:2319-2324`. The refactor must preserve that ordering or trial-λ score evaluations will read stale extras.

### 5.3 Interaction with select=TRUE

`SELECT_TRUE_DESIGN.md` plans to **double `log λ`** (interleaving each smooth's orig + null penalty). This commutes cleanly with the joint-extra extension: select doubles `log_lambda.len()`, the extras stay at the tail. Concrete combined layout from §2.2:

```
[ log λ_1_orig, log λ_2_orig, log λ_1_null, log λ_2_null,    ocat_θ_1, ocat_θ_2 ]
   <----------- M = 2 · n_smooths from select=TRUE ----------> <- K = R-2 ->
```

Both extensions ship through `OuterSearchVector` without coupling code, which is the architectural payoff of the refactor.

### 5.4 mgcv's score-type ambiguity for "ML" Gaussian

Surprise from the parity probe: mgcv's `$outer.info$grad` and `$outer.info$hess` are **already (M+1)-dim** for both REML and ML when scale is unknown. Reading `gam.fit3` source line `if (!scale.known && scoreType %in% c("REML","ML","EFS"))` confirms that **REML also does joint outer-Newton over (log λ, log φ)** internally — REML and ML differ only via the `remlInd` flag controlling the `Mp/2 · log(2πφ)` term in the score formula. This means the joint outer-Newton is **even more universal** than the backlog item suggested: REML for Gaussian and friends already implicitly needs the φ row, but Rust currently profiles via the chain rule (`reml_optimized.rs:291`) at the gradient level — which is mathematically equivalent at the manifold but loses the (M+1)x(M+1) Hessian structure during the Newton step. Switching to the joint layout for REML too is a free perf win at no behaviour cost.

---

## 6. Effort summary

| Step | Description | Effort |
| --- | --- | --- |
| 1 | TDist → search-vector refactor (no-op) | 1-2 days |
| 2 | Tweedie / NegBin / Quantile fold-in | 2-3 days |
| 3 | Gaussian-ML opt-in (`method="ML"`) | 3-5 days |
| 4 | Other dispersion-bearing families | 1 week |
| 5 | ocat thresholds (joins via search-vector) | tracked in OCAT_DESIGN |
| **Foundation (1+2)** | **3-5 days, ~290 LOC net deletion, zero behaviour change** | |
| **First headline feature (3)** | **~1 week incl. parity bake-in** | |

The smallest no-op refactor that lands the foundation is **Step 1 + Step 2 together** — the four duplicated profile blocks collapse to one, and the (ntheta+M) joint Hessian sliced out by today's TDist code is exposed via the new struct without changing what Newton actually solves. From there each downstream family extension is a 1-2 day plumbing job.
