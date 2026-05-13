# `ocat` (Ordered Categorical) Family — Design Proposal

Status: **research-only**, no implementation yet.
Driver: site #4 in `docs/NEIGHBOURHOODS_INTEGRATION_GAPS.md` — `time_to_sell_analysis/algorithm/predict.py:513` calls `mgcv.bam(..., family=ocat(R=4), select=TRUE)`. Replacing that with `mgcv_rust.Gam` requires (a) the `ocat` family and (b) `select=TRUE` (covered in `SELECT_TRUE_DESIGN.md`).

Parity data captured in `test_data/ocat_parity_basic.json` (script: `scripts/r/tests/extract_ocat_parity.R`).

---

## 1. Math

### 1.1 Likelihood

For an ordered categorical response `Y ∈ {1, 2, …, R}` with linear predictor `η = X β` (identity link in mgcv — `μ = η`), the cumulative-logit model is

```
P(Y ≤ k) = F(α_{k+1} − η),     F(z) = 1 / (1 + exp(−z))   (logistic CDF)
P(Y = k) = F(α_{k+1} − η) − F(α_k − η),   k = 1, …, R
```

with thresholds `α_1 = −∞ < α_2 < α_3 < … < α_R < α_{R+1} = +∞`.

mgcv's **identifiability fix** (forced in `ocat()`): `α_2 = −1` (a constant). The free parameters are `α_3, …, α_R` (length R − 2), and order is enforced by the *log-gap* reparameterisation

```
θ ∈ ℝ^{R-2},     α_{k+2} = α_2 + Σ_{j=1..k} exp(θ_j),   k = 1, …, R − 2
```

so `θ_j = log(α_{j+2} − α_{j+1})` is the log of consecutive threshold gaps. `exp(·)` enforces `α_{k+1} > α_k` automatically — no inequality constraints required.

For `R = 4`: free params `θ_1, θ_2` (length 2) parametrise `α_3, α_4`. `α_2 = −1` is fixed.

### 1.2 Saturated log-likelihood, deviance

`ls` returns zero for `ocat` (mgcv's `ls = function(y, w, theta, scale) list(ls = 0, lsth1 = 0, …)`). The deviance per observation is

```
d_i = −2 w_i log[ F(α_{y_i+1} − η_i) − F(α_{y_i} − η_i) ]
```

with sign attribute `s_i = sign((α_{y_i+1} + α_{y_i})/2 − η_i)` carried for `residuals(..., type="deviance")`.

### 1.3 Derivatives `Dd(y, mu, theta, level)`

mgcv computes a struct with:
- `D`        — n-vector of `d_i`
- `Dmu`      — n-vector of `∂D/∂μ` (and identity link gives `∂D/∂η` directly)
- `Dmu2 = EDmu2` — n-vector of `∂²D/∂μ²` (also acts as expected-information weight)
- `Dth`      — n × (R−2) matrix of `∂D/∂θ`
- `Dmuth`    — n × (R−2) of `∂²D/∂μ∂θ`
- `Dmu2th`   — n × (R−2) of `∂³D/∂μ²∂θ`
- (higher orders `Dmu3, Dmu4, Dth2, Dmuth2, Dmu2th2, Dmu3th` for level=2 — full Newton-Hessian path).

These are constructed from auxiliary scalars `a, b, c, d = derivatives of F at α_k − η` and the difference quantities `a = a_1 − a_0`, etc., where subscript 0 / 1 indexes the two threshold breakpoints bounding `y_i`.

### 1.4 PIRLS weights / working response

Standard extended-family IRLS:
- Working weight  `w_i = 0.5 · Dmu2_i`     (Fisher-information scaling; identity link)
- Working response `z_i = η_i + (y_i_pseudo − μ_i) / dμ_dη_i` reduces, since `dμ/dη = 1`, to `z_i = η_i − Dmu_i / Dmu2_i`.

(In `gam.fit5.r` the weighted least-squares step solves `(X' W X + S_λ) β = X' W z` with `W = diag(0.5 · Dmu2)`.)

### 1.5 Joint estimation of (β, θ)

mgcv treats `θ` as **family-internal nuisance parameters** profiled inside the LAML / `gam.fit5` extended-family path:

1. Inner PIRLS loop iterates `β` at fixed `(θ, λ)` using IRLS weights from `Dd`.
2. Outer Newton loop iterates `(log λ, θ)` jointly by maximising the LAML score. The gradient/Hessian dimension at the outer level is `M + n.theta` (length of `sp` + length of `θ`).

Evidence from our captured trajectory (`test_data/ocat_parity_basic.json`):
```
trajectory.grad : 3-vector  (1 sp + 2 θ)
trajectory.hess : 3×3 matrix
score.hist : 4 iterations to convergence
```

### 1.6 Score formula

`ocat` is an *extended* family — its REML uses the **`gam.fit5` LAML form**, identical to `Family::TDist` and `Family::Quantile` in the existing Rust code:

```
LAML = Dp/2 − ls + log|H|/2 − log|S|+/2 − Mp/2 · log(2π)
```

(No `Dp/(2σ²)` and no `−Mp/2 · log(2πσ²)` — there is no GLM dispersion φ; `ls = 0`.)

In Rust this corresponds to `ScoreFormula::GamFit5` (already implemented at `src/pirls.rs:56`).

---

## 2. mgcv internals — concrete references

All references against mgcv 1.9-4.

| Function | Location | Role |
|---|---|---|
| `ocat()` | `R/family.r` (`mgcv:::ocat`) | Family object factory. Stores `θ` in a closure env; exposes `getTheta(trans)`, `putTheta`, `Dd`, `dev.resids`, `aic`, `initialize`, `preinitialize`, `residuals`, `predict`, `rd`. |
| `Dd` (closure inside `ocat`) | same | Computes `D, Dmu, Dmu2, Dth, Dmuth, Dmu2th, …` to whatever derivative `level` is asked. |
| `gam.fit5` | `R/gam.fit3.r` | Extended-family fitter. Outer Newton over (log λ, θ). |
| `gam.fit5.post.proc` | `R/gam.fit3.r` | Glues family threshold params into final fit object — `m$family$getTheta()` is what consumers call. |
| `initialize` (expression inside `ocat`) | same | Picks `mustart` such that initial `μ_i = (α_{y_i+1} + α_{y_i})/2` (the midpoint between flanking thresholds). |
| `preinitialize` | same | If `n.theta > 0`, computes a heuristic initial `θ` from category frequencies (`tabulate(y)/n`). |
| `predict` | same | Maps converged (β, θ) to per-category probabilities. |

---

## 3. Existing Rust family infrastructure

All families are variants of `pub enum Family` in `src/pirls.rs:70`. Each is a unit/tuple variant; per-family methods on `impl Family { … }` cover:

| Method | Site | Notes |
|---|---|---|
| `variance(mu)` | `src/pirls.rs:168` | V(μ) for IRLS weight construction. |
| `link / inverse_link / d_inverse_link` | `src/pirls.rs:193`, `:208`, `:232` | Link and its derivative. |
| `dvar / d2var / d3var` | `:252`, `:269`, `:322` | Variance derivatives used by full-Newton α correction (non-canonical links) and Tk gradient. |
| `d2link / d3link / d4link` | `:284`, `:302`, `:338` | Link derivatives. |
| `score_formula` | `:381` | Returns `GamFit3` or `GamFit5`. |
| `is_canonical_link` | `:391` | Toggles Fisher scoring vs full Newton. |
| `saturated_log_likelihood` | `:418` | `ls[1]` term. |
| `estimate_phi_mgcv` | `:700` | Family-specific φ̂ solver. |
| Top-level `fit_pirls` dispatch | `src/pirls.rs:1194` | Dispatches to `fit_pirls_gaussian_fast`, `fit_pirls_tdist`, `fit_pirls_quantile`, or the generic GLM path. |

**Where `Family::Ocat` slots in:**

```rust
pub enum Family {
    ...
    /// Ordered categorical (mgcv's ocat(R=K)). Identity link.
    /// Threshold params θ (length R-2) live alongside the smooth coefs β
    /// and are optimised jointly by the outer Newton loop in σ²/θ space.
    Ocat {
        r: u8,              // number of categories
        theta: SmallVec<[f64; 4]>,   // log-gaps, length R-2; profiled
    },
}
```

The existing `Family::TDist { df, sigma2 }` and `Family::Quantile { tau, sigma }` are the closest precedents — they also carry profiled state in the enum variant and follow the `GamFit5` LAML path with a dedicated `fit_pirls_*` function.

**Dedicated PIRLS function**: `fit_pirls_ocat(...)` mirroring `fit_pirls_tdist` at `src/pirls.rs:1692`. Returns extended `PirlsResult` carrying converged `θ`, `Dmu`, `Dmu2`, and the threshold derivatives needed by the outer Newton.

**Outer optimisation**: the existing `src/reml/system.rs` / `src/newton_optimizer.rs` Newton solver already handles a vector of log-sp; extending to also include `θ` means concatenating `[log_λ; θ]` into a single search vector. For tdist this is already done — same plumbing.

---

## 4. Integration plan — sub-tasks

| # | Task | Effort | Risk |
|---|---|---|---|
| O1 | Add `Family::Ocat { r, theta }` enum variant + identity-link methods (variance returns 1, derivatives all zero). | 0.5d | Low — boilerplate. |
| O2 | Implement `dd_ocat(y, eta, theta, level)` — the derivative struct, ported from mgcv `Dd` closure. Includes `Fdiff(a, b)` and `abcd(x, level)` helpers. | 2d | **Medium-high** — most error-prone step. Lots of numerical-stability tricks (branch on sign of `b > 0` to pick `exp(b)` vs `exp(-b)`). Need exact bit-for-bit port. |
| O3 | `fit_pirls_ocat` — inner IRLS loop at fixed `(θ, λ)`. Pattern after `fit_pirls_tdist` at `src/pirls.rs:1692`. PIRLS weights `w = 0.5 · Dmu2`, working response `z = η − Dmu/Dmu2`. | 1d | Low if O2 is correct. |
| O4 | Outer-loop integration — extend Newton search vector to `[log_λ; θ]`, add gradient/Hessian rows for `θ` derived from `Dth, Dmuth, Dmu2th, Dth2, Dmuth2, Dmu2th2`. Wire `theta_init` heuristic from mgcv's `preinitialize`. | 2d | **High** — outer Newton needs new code path; line search must handle the longer vector. |
| O5 | Per-category probability prediction — port `ocat.prob(theta, lp)` from mgcv. Add `Gam.predict_proba()` method on Python side. | 0.5d | Low. |
| O6 | Parity test against `test_data/ocat_parity_basic.json` — assert thresholds, per-category fitted probs, edf, deviance all within 1e-6 of mgcv. | 0.5d | Low. |

**Total: ~6.5d.**
**Hardest part: O4 (joint outer Newton over (λ, θ))**, because the gradient assembly path in `src/reml/system.rs` currently assumes only `log λ` as outer variables; extending it to add `θ` rows requires re-deriving Tk-equivalent terms for the `Dmuth` cross-derivatives.

---

## 5. Parity protocol

Not just final-coef comparison — capture intermediate state because numerical algorithms can produce right final answers with wrong intermediate steps.

Asserted against `test_data/ocat_parity_basic.json`:

1. **Final coefficients** — `coef(m)` vs `gam.beta_` (rtol 1e-6).
2. **Thresholds** — both `θ_raw` (free, log-gaps) and `θ_alpha` (cumulative) within rtol 1e-6.
3. **Per-category fitted probabilities** — `n × R` matrix, max-abs error 1e-7. This is the **load-bearing** check; coefficients can be off if the threshold parameterisation is wrong, and the only way to detect is to check the *probabilities* themselves.
4. **edf** — `summary(m)$s.table[, "edf"]` vs Rust `edf_per_smooth()` (rtol 1e-5).
5. **Deviance** — `m$deviance` vs Rust deviance (rtol 1e-6).
6. **REML score** — `m$gcv.ubre` vs `Gam.reml_score()` (rtol 1e-6).
7. **Working residuals** — `residuals(m, type="working")` vs Rust working residuals (rtol 1e-5). Detects bugs in the `Dmu/Dmu2` derivatives even if the final β is correct.
8. **Outer-loop trajectory** — `outer.info$grad` (final), `outer.info$score.hist` (per-iter REML) should match Rust optimizer's per-iteration scores within rtol 1e-4.

If any of (3) / (7) fail while (1) / (5) pass, the bug is in the derivative struct (`Dd`) — exactly the kind of bug that a coef-only parity test would miss.

---

## 6. Open questions / risks

- **`select=TRUE` interaction.** Site #4 uses *both* `ocat` and `select=TRUE`. Once `ocat`'s outer Newton handles `[log λ; θ]`, the `select` path will double the `log λ` portion → outer search vector becomes `[log λ_orig; log λ_null; θ]`. The Newton gradient / Hessian assembly must scale to this; not a blocker but worth knowing before starting O4.
- **`discrete=TRUE`.** Site #4 also uses `discrete=TRUE`. The discretized PIRLS path in `src/discrete.rs` and `src/pirls.rs:2832` will need an ocat-aware specialization, or we accept a slower non-discretized fallback for this single call site.
- **Initial-θ heuristic.** mgcv's `preinitialize` heuristic from category tabulation matters for convergence on small-n data. Worth porting verbatim rather than starting at `θ = (-1, -1, …)`.
- **Derivative cross-checks.** Before shipping, finite-diff every analytical derivative in `Dd` against a numerical reference at a few random `(η, θ)` points — past mgcv-port experience (BSB2, Tk·KK') shows that the analytical-derivative subtree is where bugs hide.
