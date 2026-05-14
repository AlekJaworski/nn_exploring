# B4 Design Doc — fREML Outer Driver

**Author:** Plan subagent (read-only).
**Status:** ready to brief B4 impl. Assumes R1+R2+R3 landed at `a1f3921`, and B3 (`compute_sl_fitchol_step`) is being implemented in parallel with the signature in `docs/PATH_B_FREML_PLAN.md` §4.

---

## 1. mgcv → mgcv_rust call-graph map (`bgam.fitd`, bam.r:430–895)

The driver mirrors `bgam.fitd` only for `method="fREML"` (the NCV branch is out of scope). Mapping is line-anchored to bam.r master.

| mgcv section (bam.r line) | Purpose | Rust analog | Status |
|---|---|---|---|
| 442–476 init: extended.family pre-init, scale guess `log.phi=log(var(y)*.1)` | Per-family seeding | inline in driver; reuse `crate::pirls::Family` enum and the seed logic in `gam_optimized.rs:672–688` (sample-variance σ²) | **inline new** (~20 LOC) |
| 478–497 AR1 weighting | unused in v1 — error if `rho!=0` is requested | n/a | **defer** (document as known limitation) |
| 499–534 family init / valideta / mustart | Initial η, μ, dev | reuse `Family::link`/`Family::inverse_link` exposed on `crate::pirls::Family`; for scat use sample-variance σ² seed identical to `gam_optimized.rs:672–688` | **inline new** (~20 LOC) |
| 541 `Sl <- Sl.setup(G,...)` | Block penalty struct | `cache.penalties: Vec<BlockPenalty>` (built in `FitCache::new`, `gam_optimized.rs:174–195`) | **reuse** |
| 544 `Mp <- ncol(X) - rank` | null space dim | already computed at `gam_optimized.rs:617–622` for mgcv_exact; lift to `FitCache` or compute inline at driver entry | **lift** (~5 LOC) |
| 546 `theta <- family$getTheta()` | extended family init params | for scat: read `(df, sigma2)` from `Family::TDist`; for tweedie `p`; for negbin `theta`. mirror `gam_optimized.rs:665–688` seeding | **reuse via SmoothingParameter seeding** |
| 563–768 main fitting loop | OUTER fREML loop, ≤ `control$maxit` (≈200) | **NEW driver** `fit_pirls_fastreml` in `src/pirls.rs` | **B4 core** |
| 567–660 "form X'WX, X'Wy from one IRLS step" | one IRLS step + scatter-gather X'WX/X'Wy + `qrx$XX`/`qrx$Xy` | for scat: `tdist_irls_step` (R2, `pirls.rs:1868`) gives `(w, z)`; for exponential families: a NEW helper `exp_family_irls_step` (see §5, refactor R4); X'WX via `compute_xtwx_dispatch`, X'Wy via `compute_xtwy_dispatch` (`reml/system.rs:253,270`) | **mostly reuse; one new helper for non-scat** |
| 586–604 inner step control on `dev0 + bSb0 < dev + bSb` | β half-stepping when penalised deviance got worse | NEW small helper — `beta_halving_step` (see §3, line C8) | **inline new** (~25 LOC); independent of B3 |
| 614–630 efam θ estimation via `estimate.theta` | outer 1-D Newton on family.theta | **B5** target — `estimate_theta_outer_1d`. For B4 we wire the **callsite** and pass through a closure / `Option` so B4 can land with theta-frozen for unit tests | **call-out only** |
| 632–651 working w, z from `dDeta` / `mu.eta` | per-family IRLS pair | scat → `tdist_irls_step`; exponential families → reuse the per-row IRLS arithmetic that `fit_pirls_cached` (pirls.rs) inlines today | **see §5 R4** |
| 657 `XWXd(...)` | discrete X'WX | `compute_xtwx_discrete` (`discrete.rs:395`) via `compute_xtwx_dispatch` | **reuse** |
| 659 `XWyd(...)` | discrete X'Wz | `compute_xtwy_discrete` (`discrete.rs:539`) via `compute_xtwy_dispatch` | **reuse** |
| 664–665 `Sl.initial.repara(XX)`, `Sl.initial.repara(f)` | rotate XX, f into stable basis | **DEFERRED** for v1 (see §6 Q3). B3 result will land in raw-Z basis; document non-parity to mgcv's rotated coefficient frame | **defer**, see §6 |
| 678 convergence: `abs(dev-devold)/(0.1+abs(dev)) < eps` AND `abs(Nstep[n.sp+1])<eps*(abs(log.phi)+1)` | outer-loop convergence test | inline in driver | **inline new** (~15 LOC) |
| 687–693 `initial.sp(qrx$R, G$S, G$off, XX=TRUE)` | initial λ from X'WX scale | reuse `gam_optimized.rs::initialize_lambda_smart` (line 398) — applied per smooth at iter 1 only | **reuse** |
| 705–757 `Sl.fitChol` call + step control on Nstep | B3's surface | **B3** — `compute_sl_fitchol_step(sl, xx, f, rho, yy, log_phi, phi_fixed, nobs, mp, gamma) → SlFitCholResult { beta, grad, hess, step, db, pp, ldet_s, ldet_xxs }` | **B3 dependency** |
| 749–756 `Nstep` step length control: `sum(grad*Nstep) > dev*1e-7 ⇒ Nstep/2` | check if proposed Newton step on log-λ is uphill on REML; halve if so | **NEW** `fastreml_step_blending` (see §3 C12) — different from existing `newton_2d_with_halving` because the test is `g·step > 0` not Armijo | **inline new** (~30 LOC) |
| 759 `Sl.initial.repara(prop$beta, inverse=TRUE)` | rotate β back to natural-param frame | skip (we don't rotate); document parity tolerance | **defer** |
| 767 `crit = (dev/(exp(log.phi)*gamma) - prop$ldetS + prop$ldetXXS)/2` | bam-fREML score for trace output | reuse from B3 (`SlFitCholResult { ldet_s, ldet_xxs }`); just compute inline | **inline new** (~5 LOC) |
| 782–895 result assembly (`object$edf`, `Vp`, `Ve`, `gcv.ubre`, `sp`) | post-processing | NEW `FastRemlResult` struct + post-loop assembly. EDF from `F = PP·XX`, β, λ = exp(ρ), σ² = exp(log.phi). Wire into `Gam` fields the same way `gam_optimized.rs:1000+` does | **inline new** (~50 LOC) |

**Gaps & honesty:**
- `Sl.initial.repara`: skipped in v1. mgcv rotates β into the natural-parameter frame for the linear system; without this rotation, B3 gives β in raw-Z basis. Coef-by-coef parity will differ by an orthogonal transform but **predictions and EDF are basis-invariant**, so the parity test (B7) on `predict` and `gcv.ubre` will still hold.
- `AR1 rho`: bam.r:478–497. Out of scope. Driver should error if `rho != 0`.
- `coef` (user-supplied warm start), `in.out` (user-supplied initial sp/scale), `nei` (NCV): all out of scope for v1.
- `G$L`, `G$lsp0` (linear reparam of sp): out of scope. mgcv production doesn't use them.

---

## 2. Driver function signature

After reading bam.r:430–768 carefully, the right v1 shape is:

```rust
// src/pirls.rs (next to fit_pirls_tdist)

#[cfg(feature = "blas")]
pub struct FastRemlConfig {
    pub max_outer_iter: usize,      // default 200, matches mgcv control$maxit
    pub tol: f64,                   // default 1e-7 (matches mgcv control$epsilon)
    pub gamma: f64,                 // mgcv gamma correction; default 1.0
    pub phi_fixed: bool,            // true ⇒ scale known (binomial/poisson); false ⇒ estimated
    pub log_phi_init: Option<f64>,  // None ⇒ seed from log(var(y)*.1) per bam.r:475
    pub theta_callback: Option<TwithetaCallback>, // B5 hook — None means skip θ Newton (only Gaussian-like families)
}

#[cfg(feature = "blas")]
pub struct FastRemlResult {
    pub beta: Array1<f64>,
    pub lambda: Vec<f64>,                  // λ = exp(ρ)
    pub log_phi: f64,
    pub sigma2: f64,                       // = exp(log_phi)
    pub pp: Array2<f64>,                   // (X'WX + S)^-1 — Bayesian Vp / scale
    pub edf: Array1<f64>,                  // diag(PP · X'WX)
    pub gcv_ubre: f64,                     // crit at convergence (REML proxy)
    pub iterations: usize,
    pub converged: bool,
    pub final_weights: Array1<f64>,        // last-step IRLS w (for diagnostic)
    pub family_out: crate::pirls::Family,  // converged family-shape params propagated back
    pub db_drho: Array2<f64>,              // dβ/dρ (from B3's SlFitCholResult.db)
    pub grad: Array1<f64>,                 // last fREML grad on ρ
    pub hess: Array2<f64>,                 // last fREML Hessian on ρ
}

#[cfg(feature = "blas")]
pub fn fit_pirls_fastreml(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,                    // dense design — used when discrete=None AND for the IRLS step's η=Xβ
    prior_weights: Option<ArrayView1<f64>>,
    sl: &[BlockPenalty],                   // block penalties (one per smooth)
    family: crate::pirls::Family,          // carries θ; refresh between iters
    discrete: Option<&DiscreteDesign>,     // None ⇒ dense XWX/XWy; Some ⇒ scatter-gather kernels
    config: &FastRemlConfig,
) -> Result<FastRemlResult> { ... }
```

**Rationale for not bypassing X entirely:** mgcv's `bgam.fitd` keeps `G$Xd` (compressed design) and never materialises X dense — but our existing FitCache *does* hold both `cache.design_matrix` and `cache.discrete`, and the IRLS step needs η = Xβ which the dense path computes via `x.dot(&beta)` and the discrete path via `compute_eta_discrete(disc, &beta)`. Carrying both keeps the signature symmetric with `fit_pirls_tdist`/`fit_pirls_tdist_discrete` and lets B4 land as a thin dispatch:
- `discrete: None` → fallback dense (parity tests, unit tests on n=200)
- `discrete: Some(_)` → production hot path

**TwithetaCallback** (forward-declaration for B5):
```rust
pub type TwithetaCallback<'a> = &'a mut dyn FnMut(
    /*y*/ ArrayView1<f64>,
    /*mu_hat*/ ArrayView1<f64>,
    /*prior_w*/ Option<ArrayView1<f64>>,
    /*family_in*/ crate::pirls::Family,
    /*log_phi*/ f64,
) -> Result<crate::pirls::Family>;  // returns family with refreshed θ
```

Keeping it as a callback (rather than dispatching on `family.is_extended()` internally) means B4 can land with `theta_callback=None` and unit-test against Gaussian/exponential families before B5 is ready.

---

## 3. Outer-loop pseudo-impl

```rust
// src/pirls.rs::fit_pirls_fastreml — pseudocode, ~200 LOC.
// Comments map each block to its mgcv source line and Rust helper.

pub fn fit_pirls_fastreml(
    y: ArrayView1<f64>,
    x: ArrayView2<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    sl: &[BlockPenalty],
    mut family: crate::pirls::Family,
    discrete: Option<&DiscreteDesign>,
    config: &FastRemlConfig,
) -> Result<FastRemlResult> {
    let n = y.len();
    let p = x.ncols();
    let m = sl.len();

    // C1 — Mp = p - Σ rank_k.   bam.r:544.   gam_optimized.rs:617-622 pattern.
    let mp: usize = p - sl.iter().map(|s| s.estimate_rank()).sum::<usize>();

    // C2 — Initial η, μ. bam.r:514-528.
    //  For Gaussian: η = y (identity link is safe seed).
    //  For non-Gaussian: η = link(μ_init) where μ_init follows family.initialize().
    let mut beta = Array1::<f64>::zeros(p);
    let mut eta: Array1<f64> = compute_eta(x, discrete, &beta);   // 0-vector
    // For non-identity links seed eta from family.initialize_eta(y), TBD per family

    // C3 — Initial log φ seed. bam.r:475, 696-702. Reuse gam_optimized.rs:672-688.
    let mut log_phi = config.log_phi_init.unwrap_or_else(|| {
        let y_var = sample_variance(y);
        (y_var * 0.05).ln()
    });

    // C4 — Initial smoothing params via initial.sp pattern. bam.r:687.
    //   Reuse src/gam_optimized.rs::initialize_lambda_smart per-block at iter 1.
    let mut rho: Vec<f64> = sl.iter().map(|s| {
        crate::gam_optimized::initialize_lambda_smart(&y.to_owned(), &x.to_owned(), s).ln()
    }).collect();

    // C5 — Storage for step control. bam.r:545 (Nstep=0), 549-560 (coef0/b0/dev0).
    let mut nstep = vec![0.0_f64; m + if config.phi_fixed { 0 } else { 1 }];
    let mut prev_dev = f64::INFINITY;
    let mut prev_crit = f64::INFINITY;
    let mut prop: Option<SlFitCholResult> = None;
    let mut last_w = Array1::<f64>::ones(n);
    let mut last_z = y.to_owned();
    let mut converged = false;
    let mut iter = 0;

    // C6 — Main fitting loop. bam.r:563.
    for outer in 0..config.max_outer_iter {
        iter = outer + 1;

        // C7 — IRLS-once. bam.r:567-651.
        //   - eta = X·β  (or compute_eta_discrete)                       → discrete.rs:576
        //   - working pair (w, z): per-family
        //       * Scat   ⇒ tdist_irls_step(...).                         → pirls.rs:1868
        //       * Exp fam⇒ exp_family_irls_step(...) (refactor R4, §5).
        eta = compute_eta(&x, discrete, &beta);
        let (w, z) = irls_step_dispatch(y, eta.view(), prior_weights, family, /*log_phi*/ log_phi);
        last_w = w.clone();
        last_z = z.clone();

        // C8 — INNER step control on β (bam.r:586-604). Halve β if pen-dev went up.
        //   Only fires for iter > c_iter; skipped on iter 1.
        //   NEW small helper `beta_halving_step` keeps this tidy.
        if outer >= 1 {
            beta_halving_step(&mut beta, &eta, /* prev β/η/μ/dev */ ..., sl, &rho, &family);
            // re-evaluate w, z if β was halved
        }

        // C9 — Form X'WX and X'Wy via dispatch. bam.r:657-659.
        //   - dense: compute_xtwx(x, w), compute_xtwy(x, w, z)            → reml/system.rs
        //   - discrete: compute_xtwx_discrete / compute_xtwy_discrete     → discrete.rs:395, 539
        let xx = compute_xtwx_dispatch(discrete, &x.to_owned(), &w);     // reml/system.rs:253
        let f  = compute_xtwy_dispatch(discrete, &x.to_owned(), &w, &z); // reml/system.rs:270
        let yy: f64 = w.iter().zip(z.iter()).map(|(wi, zi)| wi * zi * zi).sum();  // bam.r:653

        // C10 — Skip Sl.initial.repara in v1 (see §6 risk). Log a one-time note.

        // C11 — Inner convergence test BEFORE Sl.fitChol. bam.r:678.
        let crit = current_crit_estimate(&prop, prev_crit, &y, &z, &w, log_phi);
        if outer > 2 {
            let rel = (prev_dev - prev_crit).abs() / (0.1 + prev_crit.abs());
            let log_phi_step_ok = config.phi_fixed
                || nstep.last().map_or(0.0, |s| s.abs()) < config.tol * (log_phi.abs() + 1.0);
            if rel < config.tol && log_phi_step_ok {
                converged = true;
                break;
            }
        }

        // C12 — Sl.fitChol step. bam.r:733 (REML branch).
        //   B3 surface — pure function on cached XX, f, ρ, yy.
        //   Returns Newton step on (ρ, log φ) — ρ if phi_fixed=true, else ρ ⊕ log φ.
        //
        //   The `step` returned is mgcv's `prop$step` — the proposed delta on
        //   (ρ, log φ). On the first iter (Nstep == 0) we accept it unconditionally;
        //   subsequently we use the bam.r:749-756 uphill check.
        let proposal = compute_sl_fitchol_step(           // → src/reml/fastreml.rs (B3)
            sl,
            xx.view(),
            f.view(),
            &rho,
            yy,
            log_phi,
            config.phi_fixed,
            n,
            mp,
            config.gamma,
        )?;

        // C13 — Step length control on (ρ, log φ). bam.r:749-756.
        //   if Nstep was zero (first iter): accept proposal.step verbatim.
        //   else: if g·Nstep > dev*1e-7 (uphill on REML at proposed dir), halve Nstep.
        //   When the halved step is taken, the proposal must be re-evaluated at the
        //   halved point — i.e. compute_sl_fitchol_step is called AGAIN with
        //   rho_halved. This is what makes step control nonlinear.
        let (new_rho, new_log_phi, accepted_nstep, refreshed_prop) =
            fastreml_step_blending(                       // NEW — see §5 R5
                &proposal,
                &nstep,
                &rho,
                log_phi,
                config.phi_fixed,
                /* eval_at: re-call compute_sl_fitchol_step with halved rho */
                &|trial_rho, trial_log_phi| compute_sl_fitchol_step(
                    sl, xx.view(), f.view(),
                    trial_rho, yy, trial_log_phi,
                    config.phi_fixed, n, mp, config.gamma,
                ),
                prev_dev,
            )?;
        rho = new_rho;
        log_phi = new_log_phi;
        nstep = accepted_nstep;
        prop = Some(refreshed_prop);

        // C14 — Update β from proposal. bam.r:759 (we skip Sl.initial.repara inverse).
        beta = prop.as_ref().unwrap().beta.clone();

        // C15 — Family-shape θ estimation (B5 callsite). bam.r:614-630.
        //   For Gaussian/exponential families: theta_callback = None → no-op.
        //   For scat/negbin/tweedie: callback runs 1-D Newton on family.theta
        //     given the new μ̂ from β. Returns updated `family`.
        if let Some(cb) = config.theta_callback.as_mut() {
            let mu_hat = compute_mu(x, discrete, &beta, family);
            family = cb(y, mu_hat.view(), prior_weights, family, log_phi)?;
        }

        // C16 — Track for convergence test next iter. bam.r:606-612, 767.
        let crit_new = (working_dev(&y, &beta, &x, &w, family)
                        / (log_phi.exp() * config.gamma)
                        - prop.as_ref().unwrap().ldet_s
                        + prop.as_ref().unwrap().ldet_xxs) / 2.0;
        prev_dev = working_dev(&y, &beta, &x, &w, family);
        prev_crit = crit_new;

        // C17 — Non-finite β guard. bam.r:761-765.
        if beta.iter().any(|b| !b.is_finite()) {
            return Err(GAMError::OptimizationFailed("non-finite β".to_string()));
        }
    } // end main loop

    // C18 — Post-processing. bam.r:782-895.
    let prop = prop.expect("at least one outer iter ran");
    let xx_final = compute_xtwx_dispatch(discrete, &x.to_owned(), &last_w);
    let edf: Array1<f64> = (prop.pp.dot(&xx_final)).diag().to_owned();

    Ok(FastRemlResult {
        beta,
        lambda: rho.iter().map(|r| r.exp()).collect(),
        log_phi,
        sigma2: log_phi.exp(),
        pp: prop.pp,
        edf,
        gcv_ubre: prev_crit,
        iterations: iter,
        converged,
        final_weights: last_w,
        family_out: family,
        db_drho: prop.db,
        grad: prop.grad,
        hess: prop.hess,
    })
}
```

**Key dependencies on B3's interface:**
- `proposal.beta` (so we don't need to re-solve)
- `proposal.step` (the proposed Δ on (ρ, log φ))
- `proposal.grad` (length `m + 0 or 1`, used by step blending uphill check)
- `proposal.pp` (final Vp/scale)
- `proposal.db` (dβ/dρ for output)
- `proposal.ldet_s`, `proposal.ldet_xxs` (for `crit` and parity tests)

**Note on `phi_fixed` toggling:** bam.r:617-628 dynamically toggles φ-estimation: in iters 1–4 it lets `estimate.theta` jointly optimise (φ, family.theta); after iter 4 it switches `phi.fixed=TRUE` and lets the outer ρ Newton drive log φ. **For v1 we keep `phi_fixed` constant per fit** — the joint-θ-φ inner phase is a B5 concern. Document.

---

## 4. Integration points to wire up

| File | Change | Effort |
|---|---|---|
| `src/smooth.rs:495–499` | Add `OptimizationMethod::FastREML` variant | ~3 LOC |
| `src/lib.rs:368–369, 578–579` | Add `"fREML"` arm mapping to `OptimizationMethod::FastREML`. Note the old `Some("fREML")` arm at lib.rs:627 is `--algorithm` for FS — keep separate | ~6 LOC |
| `src/lib.rs:332` | `#[pyo3(signature)]` — already accepts `method` string; no schema change | 0 LOC |
| `src/gam_optimized.rs::fit_optimized_full` | New dispatch branch: when `opt_method == FastREML`, bypass `optimize_with_beta_and_xtwx` / inner Newton entirely and call `fit_pirls_fastreml` directly. Need to translate `FitCache` fields → `fit_pirls_fastreml` args (sl=`&cache.penalties`, x=`&cache.design_matrix`, discrete=`cache.discrete.as_ref()`) | ~60 LOC |
| `src/gam_optimized.rs` populate result | Translate `FastRemlResult` → existing `Gam` fields (`coefficients`, `smoothing_params.lambda`, `fitted_values`, `deviance`, `pp` if any) | ~30 LOC |
| `python/mgcv_rust/_fitter.py:264–266` | Update docstring — `"fREML"` is now a real distinct method, not a REML alias. Existing acceptance at construction (line 306) doesn't need code change | ~3 LOC docstring |
| `python/mgcv_rust/_fitter.py:_make_native()` | Confirm the `method` string flows through to `inner.fit_optimized_full` without translation. Should be untouched if `lib.rs` accepts `"fREML"` | 0 LOC |
| `src/reml/mod.rs` exports | Re-export `fit_pirls_fastreml`, `FastRemlConfig`, `FastRemlResult`. Not strictly needed (pirls.rs path) but tidy | ~3 LOC |
| `tests/test_fastreml_driver.rs` | New unit tests: Gaussian fixture parity, scat fixture parity, dense=discrete byte-identical on pure-dedup data | ~150 LOC tests |

**Total integration LOC outside `fit_pirls_fastreml` itself: ~100 LOC.**

---

## 5. Identified refactor-prep tasks (potential R4/R5)

### R4 — Extract `exp_family_irls_step` (recommended, ~0.5 day)

Today's exponential-family IRLS pair `(w, z)` is **inlined** in three places:
- `pirls.rs::fit_pirls_cached` inner loop body
- `pirls.rs::fit_pirls_discretized` inner loop body
- `gam_optimized.rs` callback at line 833

The pattern is uniform: `z = η - offset + (y - μ)/μ_eta_val`, `w = (prior · μ_eta_val²)/V(μ)`, matching bam.r:646–650.

**Why B4 benefits:** without R4, B4 has to either (a) duplicate the IRLS arithmetic inline (~40 LOC), or (b) call into `fit_pirls_cached` and discard everything except one iteration (wasteful and confusing). With R4, the driver is a clean 3-line dispatch:

```rust
let (w, z) = match family {
    Family::TDist { df, sigma2 } => tdist_irls_step(y, eta, pw, sigma2, df, false),
    _ => exp_family_irls_step(y, eta, pw, family),
};
```

**Effort:** byte-identical refactor of ~120 LOC across three callsites into one helper. ~0.5 day.

**Recommendation: LAND R4 BEFORE B4.** Saves ~0.5 day in B4 (avoids duplicating IRLS math) plus ~0.5 day of inevitable drift cleanup later. Net win: 0.5 day. **Concrete gain >½ day → flag.**

### R5 — Extract `fastreml_step_blending` (judgement call, ~0.25 day)

The step blending at bam.r:749–756 is genuinely different from `newton_2d_with_halving` (`reml/search_vector.rs:94`): mgcv uses `g·Nstep > dev*1e-7` (gradient-direction uphill check) not Armijo (`f_new ≤ f_old + c1·step·grad`). Also the step is on `(ρ, log φ)` of arbitrary length `m+1`, not 2.

I recommend **inlining it in `fit_pirls_fastreml` for v1** rather than extracting now — it's ~30 LOC and the abstraction isn't yet earned (no second caller). If a future Path D wants the same control on a different driver, extract then.

### No other refactors required.

The other helpers needed (`compute_eta`, `compute_mu`, `working_dev`, `beta_halving_step`, `current_crit_estimate`) are all small private helpers internal to the new driver — no abstraction value to lifting.

---

## 6. Risks & open questions

### Risk: B3 interface gaps

**Risk 1 — `proposal.grad` length.** B3 must return `grad` of length `m` when `phi_fixed=true` and `m+1` when `phi_fixed=false` (the trailing entry is `∂REML/∂log φ`). The driver's step-blending needs `grad·Nstep` over the **same combined dimension** as `Nstep`. Verify B3 spec matches.

**Risk 2 — `proposal.step` is the FULL Newton step, not a unit-norm direction.** bam.r:1670–1675 takes the raw `-H⁻¹ g` then caps at `4·u/|u|_∞`. B4's uphill check at bam.r:753 also operates on the **raw** step, not a normalised one. So B3 must return `step = -H⁻¹ g` (possibly capped per bam.r:1675). Confirm.

**Risk 3 — `proposal.pp` lives in the same basis as XX.** Since we skip `Sl.initial.repara`, B3's PP is in raw-Z coords; we don't need to inverse-rotate. But if B3 internally rotates for stability (it should, per `docs/PATH_B_FREML_PLAN.md` §1 row 8), it must rotate back before returning. Confirm.

**Risk 4 — extended-family deviance vs working RSS.** bam.r:611 adds `sum(rSb²)` to `dev` to get pen-deviance for step control. B4 needs the same — and the rSb (`= rS·β` = Cholesky-root-S times β) is per-block in mgcv. Our `BlockPenalty::quadratic_form(β)` gives β'S_kβ; we use that to form `bsb = Σ λ_k β'S_kβ` and add to working RSS. Should suffice.

### Open question: shape of `FastRemlConfig`

Recommendation:
- `max_outer_iter: usize` (default 200, mgcv parity)
- `tol: f64` (default 1e-7)
- `gamma: f64` (default 1.0 — only changes if user passes gamma kwarg)
- `phi_fixed: bool` — derived from family at the dispatch site (binomial/poisson/negbin → true; gaussian/gamma/tdist/tweedie → false)
- `log_phi_init: Option<f64>` — None → sample-variance seed; Some → user/in.out override
- `theta_callback: Option<TwithetaCallback>` — None for Gaussian/binomial/poisson; Some(closure) for scat/negbin/tweedie (B5)

### Open question: γ (gamma correction) integration

mgcv's `bgam.fitd` threads `gamma` into the REML score (bam.r:767: `(dev/(exp(log.phi)*gamma) - ldetS + ldetXXS)/2`) and into B3 (bam.r:734: `gamma=gamma`). It's a scalar that inflates the effective scale. For B4 v1: pass `gamma` straight through to `compute_sl_fitchol_step` (B3 spec already takes a `gamma` arg per PATH_B plan §1 row 3), and propagate into the inner crit/dev computations. No interaction with discrete kernels.

### Open question: how step control differs from mgcv's REML path

mgcv's `fast.REML.fit` (fast-REML.r:1740, the *non-bgam* outer loop) uses true step-halving with score acceptance (`trial$reml > best$reml ⇒ halve`). bam.r's `bgam.fitd` uses a different control: only ONE inner trial via `g·Nstep > dev*1e-7`, no nested while-loop. **Implement bam.r:749–756 semantics exactly** — it's the regime our parity test (`bam(method='fREML', discrete=TRUE)`) targets.

### Surprises found in bgam.fitd
1. **`c.iter`** is dynamic (bam.r:559–560): if caller passes `coef`, c.iter=1 (start step control at iter 1); otherwise c.iter=2 (skip until iter 2). For B4 v1 we don't accept `coef`, so c.iter=2 hard-coded.
2. **β-halving** (bam.r:586–604) is **separate** from the Nstep-halving (bam.r:749–756). The first guards inner-loop divergence on penalised deviance; the second guards outer-loop divergence on REML. We need both.
3. **θ is re-estimated in iters 1–4 with `scale<0` even when method=REML**, then frozen (bam.r:617–628). This is the φ–θ joint mini-loop that drops out after iter 4. B4 v1 punts on this; B5 will revisit.
4. **`prop$db` rotation** (bam.r:800–801): mgcv inverse-rotates dβ/dρ into the original coord frame. If we skip Sl.initial.repara entirely (forward & inverse), `prop$db` stays in raw-Z and we report it as such. Document the parity caveat.

---

## 7. Estimated B4 LOC + days post-refactor

| Component | LOC est. | Notes |
|---|---|---|
| `fit_pirls_fastreml` body | ~180 | per §3 pseudocode |
| `FastRemlConfig` / `FastRemlResult` structs | ~40 | |
| `beta_halving_step` helper | ~30 | inner step control C8 |
| `fastreml_step_blending` helper (inline) | ~30 | C13 |
| `current_crit_estimate`, `compute_eta`, `compute_mu`, `working_dev` helpers | ~50 | small internal utilities |
| Dispatch in `gam_optimized.rs` | ~60 | new branch wrapping result |
| Result → `Gam` field wiring | ~30 | |
| `lib.rs` PyO3 method-string match | ~6 | |
| `OptimizationMethod::FastREML` enum variant + matches updates | ~15 | smooth.rs callers — most are exhaustive matches |
| Python docstring update | ~3 | |
| Unit tests (`tests/test_fastreml_driver.rs`) | ~150 | Gaussian parity, dense=discrete byte-identical on pure-dedup, error path on `rho!=0`, error path on `coef`/`in.out` |
| **Total** | **~590 LOC** | |

vs PATH_B plan §9 estimate of **150 LOC, 0.5 day** — that estimate was optimistic. It didn't budget for:
- the **two** distinct step-control mechanisms (β-halving + Nstep-halving)
- the IRLS dispatch (scat vs exp.fam, requiring R4)
- the result struct + post-processing
- dispatch + Python wiring
- meaningful unit tests

**Revised B4 estimate: ~600 LOC, 1.0–1.25 days** (post-R4 refactor).

Budget breakdown:
- R4 refactor (recommended prerequisite): 0.5 day
- B4 core driver + tests: 1.0 day
- B4 dispatch + Python wiring: 0.25 day

**Total to "Gaussian fREML end-to-end + dense/discrete parity": 1.75 days from current HEAD.** This is double the §9 estimate but matches the actual surface area in `bgam.fitd`. Scat/extended-family parity comes with B5; B4 alone validates Gaussian and the structural correctness of the loop.
