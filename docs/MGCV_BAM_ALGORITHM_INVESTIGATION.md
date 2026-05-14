# mgcv `bam(discrete=TRUE)` Algorithm Investigation

**Date:** 2026-05-14
**Triggering question:** why does `mgcv.bam(scat, fREML, discrete=TRUE, nthreads=4)` finish in 164ms when `mgcv_rust` (with D3+D4+D6 + m_max cap) takes ~9,000ms on the same fixture?
**Conclusion:** The gap is **not primarily about scatter-gather X'WX kernels** (which we did correctly). The gap is **algorithmic**: mgcv's outer optimization (`fast.REML.fit` + `Sl.fitChol`) is O(p³) per Newton iter and never touches the full design matrix again after a single X'WX assembly per IRLS step. Our outer Newton (`tdist_gdi2_native`) is O(n·p²·ntot²) per call because it builds the joint Hessian from the full design matrix every iteration.

## Source studied

- `R/bam.r:430` — `bgam.fitd`: the discrete-binned IRLS fit driver
- `R/fast-REML.r:1585` — `Sl.fitChol`: closed-form REML score + grad + Hess + Newton step on cached `XX`
- `R/fast-REML.r:1740` — `fast.REML.fit`: the fREML outer optimizer (Newton on log λ)

mgcv version: 1.9-4 (CRAN tarball, dated current).

## Mental model: mgcv's `bgam.fitd` loop

```r
for iter in 1..maxit:
    # ── 1. PIRLS step ─ ONE IRLS update per outer iter, not until convergence ──
    if iter > 1:
        eta <- Xbd(Xd, coef, kd, ks, ts, ...)   # compressed gather, O(n + Σ m_j p_j)
    mu  <- linkinv(eta)
    
    # Extended-family theta estimate (df, etc.) — SEPARATE 1D opts, NOT joint
    if iter > 1 and efam:
        theta <- estimate.theta(theta, family, y, mu, ...)
        family$putTheta(theta)
    
    # IRLS working weights + working response from family
    dd <- dDeta(y, mu, G$w, theta, family, 0)
    w  <- dd$Deta2 * 0.5
    z  <- (eta - offset) - dd$Deta.Deta2
    
    # ── 2. ONE compressed X'WX + X'Wy assembly per outer iter ──
    qrx$R  <- XWXd(Xd, w, ...)        # compressed X'WX, O(n + Σ m_a m_b (p_a + p_b))
    qrx$f  <- XWyd(Xd, w, z, ...)     # compressed X'Wy, O(n + Σ m_j p_j)
    qrx$y.norm2 <- sum(w * z^2)
    
    # ── 3. fREML Newton step on log λ ──
    #     ENTIRELY on the cached p×p XX. No reference to the full design.
    prop <- Sl.fitChol(Sl, qrx$XX, qrx$Xy, rho=lsp[1:n.sp], yy=qrx$y.norm2, log.phi=log.phi, ...)
    # Sl.fitChol returns:
    #   - beta: PLS solve on (XX + λS)
    #   - grad: REML gradient (from IFT identities on cached XX)
    #   - step: Newton step on log λ (and log φ if !phi.fixed)
    #   - hess: REML Hessian
    
    # Step length control on the proposed Newton step
    if sum(prop$grad * Nstep) > dev * 1e-7:
        Nstep <- Nstep / 2
    else:
        coef <- prop$beta
        Nstep <- prop$step
        lsp0  <- lsp
    
    # Step control on β/η: halve if penalized deviance is going uphill
    while not improving:
        coef <- coef0/2 + coef/2; eta <- eta0/2 + eta/2; ...
end
```

**Key facts:**

1. **One X'WX rebuild per outer iter.** Total: typically 5–15 for the whole fit.
2. **REML gradient/Hessian comes from cached `XX` only** (`Sl.fitChol`). Cost: O(p³) per call.
3. **No nested PIRLS-to-convergence**: the IRLS update is single-step. β is solved once per outer iter via Cholesky on `XX + λS`.
4. **Extended-family θ (df, etc.) is updated SEPARATELY** via `estimate.theta` — a 1D optimization. NOT in the joint Newton.
5. **All numerics on n happens via the compressed Xd → XWXd/XWyd/Xbd pipeline.** The outer Newton never sees n directly.

## Mental model: `mgcv_rust` today (post-D3+D4+D6+cap, head `70318f8`)

```rust
loop {  // outer Newton on (log λ, log σ², log df) — joint
    pirls_to_convergence(...);  // inner IRLS: MANY iters, each rebuilds X'WX
    grad_and_hess = tdist_gdi2_native(...);  // O(n·p²·ntot²) per call
    line_search_with_armijo_halving(...);    // many trial points, each re-runs PIRLS
}
```

**Per-iter cost:**

- `tdist_gdi2_native` is `O(n·p²·ntot²)` where `ntot = M + 2` (smooths + σ² + df)
- Line-search PIRLS refresh is `O(inner_iters · n · p²)`
- Outer Newton iters: 16 on production split_0 (vs mgcv ~5–10)

Sum: ~9 seconds. Profile breakdown (split_0):
- `tdist_gdi2_native`: 56% (~4.2s, 16 iters × ~260ms)
- Line-search PIRLS refresh: 31% (~2.3s, 64 callback calls × ~36ms)
- Eigendecomp / step-solve: 10%
- Setup: 2%

## The gap, decomposed

| Source of slowdown | mgcv-bam | mgcv_rust | Speedup possible from match |
|---|---|---|---|
| Outer iters | 5–10 | 16 | 1.5–3× |
| X'WX rebuilds per outer iter | 1 | ~5 (PIRLS) + line-search trials | 5–10× |
| Cost per REML score/grad/Hess eval | O(p³) Sl.fitChol | O(n·p²·ntot²) tdist_gdi2_native | n/(p·ntot²) ≈ 5000/(30·16) ≈ 10× per eval |
| θ (df, σ²) opt | Separate 1D estimate.theta | Joint Newton dim (M+2) | Marginal — joint may be slightly slower at conv |

**Stacked, this gives the observed 55×:**
- 3× from outer-iter count
- 10× from per-eval cost
- The remaining ~2× from line-search/PIRLS-refresh efficiency

So D3+D4+D6 + m_max cap landing zero perf win is **NOT** a bug. It's the right work for the wrong architecture. The `tdist_gdi2_native` path is fundamentally untouched by discrete kernels, and IT is the dominant cost.

## What `Sl.fitChol` actually computes (the key trick)

`Sl.fitChol(Sl, XX, f, rho, yy, log.phi)` evaluates the REML score and its derivatives **using only** `XX`, `f`, `yy`, `rho`, and the penalty structure `Sl`. The math:

1. **β** solves `(XX + λS) β = f` (Cholesky on `XXp = XX + λS`).
2. **log|XXp|** is `2 Σ log diag(R)` (from the Cholesky factor R).
3. **log|S|** is computed via `ldetS(Sl, rho, ...)` — penalty structure only, no design.
4. **RSS + bSb** is `yy - 2 b'f + b'XXp b = yy - b'f` (identity for the minimizer).
5. **REML score** is `0.5 (log|XX+S| - log|S|+ + (n-Mp) log φ + (RSS + bSb)/φ)`.
6. **dβ/dρ** via the IFT: `dβ/dρ_k = -(XX+S)^{-1} dS/dρ_k · β`.
7. **d²/dρ²** via second-order IFT.

All step costs are `O(p³)` (the Cholesky factorization and back-substitution dominate). The crucial fact: **the design matrix X enters only through `XX = X'WX` (the cached p×p input)**. Once `XX` is computed (once per IRLS iter, via the compressed `XWXd`), the entire REML optimization happens in p×p space.

For the extended-family case (scat = t-dist), the REML derivatives w.r.t. (σ², df) also fit this framework — they're computed via `family$dDeta` evaluated at per-row residuals (cost O(n) but cheap scalar work, NOT n·p²).

## Implications for `mgcv_rust`

We must choose between three paths:

### Path A — Port mgcv's `Sl.fitChol`-style algorithm wholesale

Replace `tdist_gdi2_native` and the joint outer Newton with:
1. Single IRLS step per outer iter (like mgcv)
2. REML grad/Hess from cached `XX` via a `compute_sl_fitchol_step(xx, f, yy, rho, sl, ...)` analog
3. Separate 1D optimisation for extended-family θ

**Pros:** matches mgcv's perf within a constant factor. Closes most of the 55× gap.

**Cons:** large refactor. ~1500 LOC across `smooth.rs`, `reml/mod.rs`, `pirls.rs`. Loses our 0.15.0 joint Newton actuation (which was a real win on the non-binned scat path). The joint actuation gave ~2× speedup over sequential; switching to mgcv's sequential approach trades that 2× for the bigger architectural win.

### Path B — Add fREML as a parallel optimizer to existing REML

Keep the joint-Newton REML path for cases where binning doesn't help (small n, no ties), and add a new `fREML` path that mirrors mgcv's `Sl.fitChol` for cases where binning DOES help (large n with ties, the customer's scenario).

**Pros:** preserve 0.15.0's win on unbinned fits. New path is purely additive. User opts in via `method='fREML'` (mirrors mgcv API exactly).

**Cons:** maintenance burden of two paths. ~2000 LOC.

### Path C — Investigate whether our outer Newton CAN reuse cached `XX`

Specifically: can `tdist_gdi2_native`'s O(n·p²·ntot²) cost be reduced to O(p³·ntot²) by computing derivatives from cached `XX` + family-evaluated per-row residual quantities? The math: scat's REML score for a fixed β depends on `eta = Xβ` (O(n) per η-eval) and per-row likelihood derivatives `dDeta`. The full design X is needed for `dβ/dρ` (which needs `(XX+S)^{-1} X' diag(w_θ) X`), but maybe via IFT we can derive `dβ/dρ` from `XX` and the per-row stuff without rebuilding the full `X' diag(...) X`.

**Pros:** preserves joint Newton, drops cost from n·p² to p³.

**Cons:** Math needs to be derived from scratch. Unclear if the joint-Newton machinery survives the substitution. Could be a research project; could be a 200-LOC fix. Won't know until we try.

## Recommendation

**Path B (parallel fREML optimizer)** is the safest investment:
- Predictable scope (~2 weeks of careful work)
- Doesn't risk regressing 0.15.0's joint Newton wins
- Direct API parity with mgcv (`method='fREML'`, `discrete=TRUE`)
- Customer can opt in for production workloads

**Path A** is cleaner long-term but risky in the short term; we'd be deleting tested code (the 0.15.0 joint Newton) and rebuilding the surrounding numerics.

**Path C** is research and shouldn't block product progress.

## Next concrete steps (if Path B)

1. Implement `Sl::setup` analog — penalty block structure (we already have most of this in `block_penalty.rs`)
2. Implement `compute_sl_fitchol_step(xx, f, yy, rho, sl, log_phi, ...)` — the closed-form REML step on `p × p`
3. New driver `fit_pirls_fastreml(family, ...)` that loops: IRLS-step → fitChol → step-control → repeat
4. Parity tests: compare `fit_pirls_fastreml` output to `fit_pirls_tdist` on small fixtures; expect coef diffs in the 1e-3 range (different algorithm, different convergence)
5. Hook to Python via `Gam(method='fREML')`
6. Re-bench production fixtures; expect 5–20× speedup
7. OpenMP scatter in `XWXd` (D10 from the design doc) for the last factor of ~2–3×

Estimated effort: 8–12 subagent-days, with strong test posture (mgcv parity battery on each PR).

## What NOT to do next

- **T1+T6 from `docs/SMOOTH_NEWTON_DISCRETE_PLAN.md` is the wrong target now.** It would put `tdist_gdi2_native` on discrete kernels (saving ~28% of wall), but the real win comes from not calling `tdist_gdi2_native` at all per outer iter. Skip the plan; pivot to Path B.
- **Don't push more D-tasks from the original binning design.** The kernels are correct (verified at 1e-12 in unit tests). The problem is they're plugged into the wrong outer algorithm.

## Memory updates needed

- [[scat-perf-gap-vs-mgcv-bam]] — update with the architectural finding: the gap is fREML vs joint-Newton-REML, not scatter-gather vs GEMM
- new memory: `feedback_methodical_before_impl` — verify architectural assumptions BEFORE sinking subagent-days into impl work that targets the wrong layer
