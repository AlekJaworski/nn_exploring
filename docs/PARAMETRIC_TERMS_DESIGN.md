# Parametric (linear, unsmoothed) terms — design notes

Status: **scaffold landed (Ergo-N-parametric)**. End-to-end path runs; full
mgcv-parity is **not** closed yet. See "What's still TODO" at the bottom.

## Motivation

Three of the four call sites in the neighbourhoods integration
(`at_least_{1,2,3}_price_drop`) pass 0/1 indicators that should enter the
formula additively — `y ~ s(x) + dummy`, not `y ~ s(x) + s(dummy)`. Fitting a
binary covariate as a smooth wastes basis dimensions on a function that has
only two possible inputs, and the resulting `f(0)` / `f(1)` shift is identical
to a single linear coefficient on `dummy` after identifiability — just much
more expensive and numerically rougher.

## API decision — Option A: `predictor_basis_map["col"] = "parametric"`

The existing per-predictor basis-type plumbing already runs

```
Gam.predictor_basis_map  →  Gam._build_bs_list  →  bs_list (positional list)
                         →  _NativeGAM.fit(bs_list=…)
                         →  Rust dispatch in lib.rs::fit_auto_optimized
```

— so wiring `"parametric"` (or `"linear"`) into this enum is one new arm in
the Rust match plus one new `SmoothTerm` constructor. A separate
`parametric_terms: list[str]` kwarg would duplicate the same plumbing and
introduce a second source of truth that has to be kept in sync with
`predictors`. The dict-keyed form means *presence* in the map is itself the
spec; no length matching, no ordering ambiguity.

This is also closer to mgcv's mental model: mgcv users think of `bs="re"`,
`bs="cr"`, `bs="tp"` as a single enum-valued choice per term. Parametric
terms in mgcv are formally a different formula-level thing — they don't have
a `bs` — but for the user-facing API the basis-map slot is the right place
because:

1. The neighbourhoods code already builds the dict; it would set `bs_map[col]
   = "parametric"` rather than maintaining a second list.
2. The Rust side already special-cases `is_random_effect` to skip centering
   and pc-anchoring (see `FitCache::new` in `src/gam_optimized.rs:60-77`).
   Adding `is_parametric` follows the same pattern.
3. We keep all per-term knobs (`k`, `bs`, `pc`) in a single, documented
   surface (the constructor kwargs of `Gam`).

The user-visible string we accept is `"parametric"`. We do **not** also accept
`"linear"` — one canonical name is less surface area to maintain.

## Math — why a parametric term has no penalty block

In a penalised GAM the score being optimised (Gaussian, REML) is

```
−log L(β, λ)  =  (1 / 2σ²) ||y − Xβ||² + ½ Σⱼ λⱼ β'Sⱼβ
                  +  ½ log|X'X/σ² + Σⱼ λⱼSⱼ|  −  ½ log|Σⱼ λⱼSⱼ|₊
```

where `Sⱼ` is the per-smooth penalty matrix, and `|·|₊` is the
pseudo-determinant over the range space (mgcv's `det1` / `det2`). A
parametric column `xₚ` enters `X` exactly the same way as a smooth's basis
columns, but its corresponding `Sⱼ` is the **zero matrix**. Consequences:

- The penalty term `λⱼ β'Sⱼ β = 0` regardless of `λⱼ` ⇒ that coefficient is
  unpenalised, retaining its full degree of freedom.
- The total-penalty determinant `|Σⱼ λⱼSⱼ|₊` is unaffected — the parametric
  column sits in the null space and is excluded from the pseudo-determinant
  by construction.
- The log-marginal-likelihood gradient `∂/∂log λⱼ` for that smooth is
  identically zero (the term doesn't appear in the score), so the outer
  optimiser leaves its λ wherever it was initialised. We **freeze** it at
  `λ=1` to keep the gradient/Hessian routines well-conditioned, but it
  doesn't matter numerically.
- The PIRLS Hessian `X'WX + Σⱼ λⱼSⱼ` retains a full diagonal contribution
  from the column's `X'WX` part — the column is still in the design matrix,
  it just isn't shrunk toward zero.

In `mgcv` itself, parametric formula terms (`y ~ s(x) + dummy`) accumulate
into the model's `pterms` / `paraPen` slots and the design-matrix block they
contribute is treated separately from the `mgcv::smooth.construct` path —
no penalty is generated for them at all. Our `BlockPenalty` representation
(`src/block_penalty.rs`) already gracefully handles zero blocks — see
`reparam.rs:476` where `norm < 1e-15` is `continue`d, and
`embed_penalty_sqrt` which returns a `p × 0` square root when the block's
rank is 0. So we can keep a uniform "every term has one BlockPenalty" view
while still expressing "no penalty" as `S = 0` of size 1×1.

## Mgcv-side mechanics (summary of what we're matching)

In mgcv's R source:

- **Formula parsing** (`mgcv::interpret.gam`, `mgcv:::gam.setup`): smooth
  terms in `s(...)` / `te(...)` go through `smooth.construct` and produce
  per-smooth design columns + penalty matrices. Plain terms (`y ~ x + dummy`)
  go through R's `model.matrix(parametric_formula, data)` and contribute
  columns to the leading "parametric block" of the design matrix.

- **Design layout**: `[1 | parametric_cols | smooth_1_cols | smooth_2_cols |
  …]`. The intercept and the parametric columns are unpenalised by default
  — mgcv's `Sl` (penalty list) only covers the smooth blocks.

- **Identifiability**: parametric columns are NOT sum-to-zero centred — the
  centring constraint applies only to smooths (otherwise constants would
  collapse into the intercept and lose their interpretation).

- **EDF**: a parametric term's effective degree of freedom is exactly 1 (one
  unpenalised column ⇒ one parameter retained). mgcv's `m$edf` vector
  records `1.0` for each parametric column.

- **Coefficient labelling**: mgcv keeps the original column name (`dummy`)
  whereas smooths get labels like `s(x).1`, `s(x).2`, …. Our wrapper
  surfaces the parametric coefficient by `predictor` name — see
  `Gam.parametric_coef_(name)` (followup; not in this scaffold).

## Integration points in `src/`

In rough order from API surface to numerical core:

1. `src/lib.rs:520-548` (`fit_auto_optimized`): the per-term `match
   term_bs { … }` block. Adding the `"parametric"` arm here is the entry
   point.
2. `src/gam.rs:14-145` (`SmoothTerm` struct + constructors): adds a new
   `SmoothTerm::parametric(name, x_data)` constructor that returns a smooth
   with a one-column identity basis, a 1×1 zero penalty, and
   `is_random_effect = true` so the existing "skip centering" branch in
   `FitCache::new` covers it. (We could add a separate `is_parametric` flag,
   but the *behaviour* — no centering, no pc-anchoring — is the same as
   random effects from `FitCache::new`'s perspective.)
3. `src/basis.rs`: a new `ParametricBasis` impl of `BasisFunction` whose
   `evaluate(x)` returns the n×1 column `x` as-is, and `num_basis() = 1`.
4. `src/block_penalty.rs`: **no changes**. The existing `BlockPenalty` with
   a 1×1 zero block is already handled correctly by `reparam.rs`'s
   norm-threshold short-circuit (`reparam.rs:476`) and by
   `embed_penalty_sqrt`'s rank-0 fast path (`reparam.rs:532`).
5. `src/reparam.rs`: no behavioural change needed — verified by inspection
   that the zero-block paths already short-circuit. A regression test
   should confirm this is true across the full optimiser path.

On the Python side:

1. `python/mgcv_rust/_fitter.py:538-548` (`_build_bs_list`): no change. The
   helper already passes any string in `predictor_basis_map` through.
2. `python/mgcv_rust/_fitter.py:478-504` (`_resolve_ks`): treat
   `predictor_basis_map[name] == "parametric"` the same way as `"re"` — use
   placeholder `k=1`, the Rust side ignores it. (One small change.)

## Parity protocol — what to check beyond the final coefficients

Final coefficient agreement (`coef(m)["dummy"]`) is the obvious sanity
check, but the user explicitly wants to verify the implementation is
**structurally** right. So the parity script captures, and downstream tests
should diff:

1. **Design-matrix layout** (column count, column ordering). With
   `y ~ s(x) + dummy` and `k=10`, mgcv produces `1 (intercept) + 1 (dummy)
   + 9 (s(x) after sum-to-zero) = 11` columns. Our scaffold currently puts
   columns in `[1 | s(x) | dummy]` order because we don't yet move
   parametric blocks to the leading position — this is a known parity gap
   (item #1 in TODO).

2. **Per-term EDF**. mgcv reports `m$edf` as a vector with the s(x) EDF
   (some value < 9) and `1.0` for dummy.

3. **Smoothing parameters λ**. mgcv reports one λ — for s(x) — because
   parametric terms have no smoothing parameter. Our `get_all_lambdas`
   currently returns one entry per smooth-term slot in
   `smooth_terms`, so the parametric placeholder will surface a `λ=1`
   entry that should be filtered out by the wrapper (TODO).

4. **σ² (Gaussian scale)**. mgcv's `m$sig2` should match our
   `get_scale_parameter()` to ~1e-6 relative.

5. **Fitted values** (`fitted(m)` vs our `predict(X_train)`). The strictest
   end-to-end check.

6. **Parametric coefficient `coef(m)["dummy"]`**. Should match our
   coefficient for the parametric column to ~1e-6 absolute.

The R script `scripts/r/tests/generate_parametric_parity.R` captures all of
these (`m$edf`, `m$sp`, `m$sig2`, `m$coefficients` with `attr(…, "names")`,
`m$fitted.values`, and `model.matrix(m)` for the design layout).

## What's still TODO to close parity

The scaffold lands API + minimal Rust dispatch + smoke test. The following
items are needed before we claim mgcv parity on parametric terms:

- [ ] **Column ordering**: move parametric columns to the leading position
  in the design matrix so the layout matches mgcv's `[1 | param | smooth_1
  | …]`. Currently we emit `[1 | smooth_1 | param]` because we iterate
  `smooth_terms` in user order. Predict-time and `get_term_indices()` need
  to agree.
- [ ] **No sum-to-zero on parametric columns**: covered by reusing
  `is_random_effect = true` (FitCache::new skips centering when this flag
  is set). Verified that downstream code paths (PIRLS, reparam) treat the
  zero-penalty block correctly, but needs a regression test that compares
  byte-for-byte against mgcv's design matrix.
- [ ] **EDF reporting for parametric terms**: should surface as `1.0` in
  `get_edf_per_smooth()`. Our current EDF formula is `tr(A⁻¹ X'WX_block)`
  which should naturally give 1.0 for an unpenalised column, but worth
  verifying numerically.
- [ ] **λ filtering in the wrapper**: `Gam.get_lambdas()` should drop
  entries for parametric placeholders so users see one λ per actual smooth.
- [ ] **Coefficient labelling**: expose `Gam.parametric_coef_(name)` that
  picks out the parametric column's coefficient by `predictor` name.
- [ ] **Multi-parametric / mixed**: currently we test one parametric + one
  smooth; a fuller battery should cover (a) parametric-only models
  (`y ~ x`, no smooths), (b) multiple parametric columns, (c) parametric
  + multiple smooths in arbitrary order.
- [ ] **Non-Gaussian families**: the scaffold only smoke-tests Gaussian.
  Binomial parametric terms should also work — the math doesn't depend on
  the family — but needs a parity check.
- [ ] **`auto_k` interaction**: ignore parametric terms in the auto-k loop
  (they have no k to grow). Trivial fix once the scaffold is in.
