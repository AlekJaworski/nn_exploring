# mgcv_rust

A Rust port of R's [`mgcv`](https://cran.r-project.org/web/packages/mgcv/) package — Generalized Additive Models with automatic smoothing-parameter selection (REML / LAML) and PIRLS fitting — with first-class Python bindings.

The numerical core targets **byte-for-byte parity with `mgcv`** on common families, and the Python API is designed to feel like `sklearn`: a single `Gam` class, named-predictor `DataFrame` inputs, `fit()` / `predict()` / `score()`, posterior CI, and subset views for marginal analysis.

| | |
|---|---|
| **Latest release** | `0.11.0` |
| **Parity** | 554 / 0 / 0 on the mgcv comparison battery |
| **Python tests** | 211 passed, 1 xfailed |
| **Headline perf** | `2d_gaussian_additive_n50000_k15_cr` — 394 ms → 97 ms (4.05× vs. mgcv) |
| **Python wrapper** | `mgcv_rust.Gam` (canonical), `GAMFitter` deprecated alias |

## Why another GAM library?

- **R-equivalent answers, Python ergonomics.** If you have R / `mgcv` users producing models and Python services consuming them, this gives both sides the same numbers from the same fit. The `serialize()` method produces a portable artefact for deployment.
- **Fast.** Aggregate time across 80 parity fixtures: 2.0 s. On the largest Gaussian fixture (`n=50,000`, `k=15`), `mgcv_rust` is ~4× faster than R's `mgcv`.
- **Real CIs, paired diffs.** `predict_ci(X)` returns `(mean, lo, hi)` so you never have to "+= intercept" by hand. `predict_diff(from_X, to_X, level=...)` gives a paired-posterior CI for the difference between two predictions — strictly tighter than the naive `predict_ci` difference.
- **Frozen serving view.** `GamPredictor(gam)` is an `__slots__`-locked, inference-only wrapper with strict input validation (`feature_names_in_` enforcement) and a `check_against(gam, X_sample)` round-trip assertion for deployment safety.

## Installation

The Python package is built from this repo via [maturin](https://www.maturin.rs/):

```bash
# Clone, then in a venv:
pip install maturin
maturin develop --features python,blas,blas-system --release
```

After that, `import mgcv_rust` works.

For a Rust-only consumer, add to your `Cargo.toml`:

```toml
[dependencies]
mgcv_rust = { git = "https://github.com/AlekJaworski/nn_exploring" }
```

## Quick start

```python
import numpy as np
import pandas as pd
from mgcv_rust import Gam

rng = np.random.default_rng(0)
n = 500
X = pd.DataFrame({
    "x0": rng.uniform(-2, 2, n),
    "x1": rng.uniform(0, 5, n),
})
y = np.sin(X["x0"]) + 0.3 * (X["x1"] - 2.5)**2 + rng.normal(0, 0.1, n)

gam = Gam(family="gaussian").fit(X, y)

gam.predict(X[:5])                  # response-scale predictions
mean, lo, hi = gam.predict_ci(X[:5])  # 95% CI, response scale
gam.score(X, y)                     # adjusted R²
print(gam.summary())                # mgcv-style block
```

That's it for the simple case. Read on for the curated tour, or jump
to [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) for a
step-by-step tutorial with worked examples.

---

## Tour

### 1. Build the model

`Gam()` accepts pandas / polars / numpy. If you pass a `DataFrame`, the column names become the predictor names; otherwise you can pass them explicitly.

```python
gam = Gam(
    family="gaussian",           # gaussian, binomial, poisson, gamma, tweedie, nb, t-dist, scat, ...
    link=None,                   # None → canonical link for the family
    k_default=10,                # default basis dimension per smooth (mgcv default)
    term_k_mapping={"x0": 25},   # per-predictor overrides
    method="REML",
)
gam.fit(X, y)
```

After `fit`, sklearn-style attributes are populated:

```python
gam.coef_              # full β vector (intercept first)
gam.intercept_         # link-scale
gam.intercept_response_ # response-scale (= mean(y) for identity link)
gam.feature_names_in_  # predictor names, fitting order
gam.n_features_in_
gam.lambda_            # per-smooth smoothing parameter
gam.edf_               # per-smooth effective degrees of freedom
gam.k_, gam.bs_        # basis dimension / basis type per smooth
gam.vcov_              # posterior covariance of β̂
```

### 2. Predict on different scales

```python
gam.predict(X)                       # response (default — same as mgcv predict(type="response"))
gam.predict(X, scale="link")         # linear predictor, no inverse link (mgcv type="link")

result = gam.predict(X, type="terms")  # decomposition
result.intercept                       # scalar, link scale
result.contributions                   # DataFrame (n × m_smooths), link-scale, centered on training
result.total                           # response-scale ≡ gam.predict(X)
```

Invariant: `gam.predict(X) == inv_link(intercept + result.contributions.sum(axis=1))`.

### 3. Confidence intervals

```python
mean, lo, hi = gam.predict_ci(X, level=0.95)       # response, 95% CI
mean, lo, hi = gam.predict_ci(X, scale="link")     # link-scale CI
mean, lo, hi = gam.predict_ci(X, level=0.5)        # 50% CI (narrower)
```

`mean` matches `gam.predict(X, scale=scale)` exactly. The CI is built from `n_samples` (default 1000) draws of `β ~ N(β̂, vcov)`.

The legacy `predict_ci(X, alpha=0.05) → (lo, hi)` form still works (emits a `DeprecationWarning`) so existing callers don't break.

### 4. Differences between predictions, with paired CI

```python
# Per-row diff: to[i] - from[i]
diff = gam.predict_diff(from_X, to_X)

# With paired posterior CI — much narrower than predict_ci differences
diff, lo, hi = gam.predict_diff(from_X, to_X, level=0.95)

# One row broadcast against many
diff = gam.predict_diff(baseline_row, candidates, broadcast="from")
diff = gam.predict_diff(candidates, target_row, broadcast="to")
```

Identity-link only — for non-identity links the response-scale difference isn't linear in β, so the closed-form CI argument doesn't transfer.

### 5. Subset / marginal views

`gam[name]` returns a view that includes only the listed smooths (and optionally the intercept). All predict methods inherit it.

```python
gam[["x0"]].predict(X)                              # marginal effect of x0, response scale
gam[["x0"]].predict(X, scale="deviation")           # link-scale, intercept zeroed (sum-to-zero on train)
gam[["x0", Gam.INTERCEPT]].predict(X)               # marginal effect + intercept
gam[["x0"]].partial_effect("x0").plot()             # what the smooth looks like
```

All sklearn attributes filter to the active features:

```python
gam[["x0"]].coef_            # just the x0 block (+ intercept if active)
gam[["x0"]].feature_names_in_  # ["x0"]
```

### 6. Plotting, summary, score

```python
gam.plot()                  # figure with one subplot per smooth (CI bands shaded)
gam.plot("x0")              # single-smooth axes
df = gam.partial_effect("x0", level=0.95)  # underlying data: columns x0, effect, lo, hi

print(gam.summary())
# Gam summary  family=gaussian  link=identity  n_obs=500
#   intercept (link)     = -0.0123
#   intercept (response) = -0.0123
#   scale (σ²)           = 0.0101
#   deviance             = 4.8732
#   R² (adj)             = 0.9612
#   smooths:
#     s(          x0)  k= 10  edf=  6.13  λ=2.345e-02
#     s(          x1)  k= 10  edf=  3.07  λ=4.890e-01

gam.score(X, y)              # adjusted R² for regression; accuracy@0.5 for binomial
gam.predict_proba(X)         # binomial only: (n, 2) [[P(0), P(1)]]
```

### 7. Serving with `GamPredictor`

For deployment paths, wrap the fitted `Gam` in a `GamPredictor`. Same API surface, plus strict column validation and a round-trip assertion.

```python
from mgcv_rust import GamPredictor

predictor = GamPredictor(gam)
predictor.check_against(gam, X[:50])   # raises if predictions diverge
predictor.predict(X_serve)             # ValueError if any expected column is missing
predictor[["x0"]].predict(X_serve)     # subset view returns another GamPredictor

# `__slots__` blocks attribute leaks — once built, the bound Gam's
# coef/vcov/schema are the contract.
```

This structurally closes two production bug classes:

- **Column index drift** when the serving DataFrame's columns differ from training: `feature_names_in_` is enforced on every `predict` call (re-ordered → realigned; missing → `ValueError`).
- **x-grid clamping** in interpolation-based predictors: `GamPredictor` recomputes the basis at the requested X via `evaluate_lpmatrix`, so there's no pre-computed grid to clamp against.

## Families and links

| Family | Default link | Other links | Notes |
|---|---|---|---|
| `gaussian` | identity | — | bs-spline / cubic-regression / random-effects bases supported |
| `binomial` | logit | — | `predict_proba()` available |
| `poisson` | log | — | |
| `gamma` | inverse | `log` | log-link via `link="log"` |
| `tweedie` | log | — | `tweedie_p=` (default 1.5); fixed-p |
| `nb` / `negbin` | log | — | `nb` profiles θ; `negbin` is fixed-θ |
| `t-dist` / `scat` | identity | — | df profiled if not given (`df ∈ [2, 100]`) |
| `quasipoisson`, `quasibinomial` | log / logit | — | dispersion-aware |
| `quantile` | identity | — | see `mgcv_rust.fit_quantile` / `fit_quantile_lss`; OOS presets documented in [`docs/qgam_oos_presets.md`](docs/qgam_oos_presets.md) |

## Architecture (Rust side)

| File | Responsibility |
|---|---|
| `src/gam.rs` | `GAM` struct + `fit` / `predict` entry points |
| `src/pirls.rs` | Penalized IRLS inner loop |
| `src/reml/` | Outer-loop REML / LAML optimization |
| `src/smooth.rs` | Basis functions (cubic-regression, B-spline, random effects, tensor products) |
| `src/penalty.rs` | Penalty-matrix construction |
| `src/lib.rs` | PyO3 bindings — `PyGAM`, `compute_penalty_matrix`, `newton_pirls_py` |

Build features:
- `python` — enable PyO3 bindings.
- `blas` / `blas-system` — link against system BLAS for matmul-heavy paths.

## Parity and performance

Run the parity battery against R/`mgcv`:

```bash
pytest tests/parity/ -q
# 554 passed, 0 failed, 0 xfailed, 0 skipped
```

Microbench:

```bash
python3 scripts/python/bench_step_blend.py 5
```

Headline (`2d_gaussian_additive_n50000_k15_cr`, identity link):

| | Time |
|---|---|
| R `mgcv` | ~394 ms |
| `mgcv_rust` 0.11.0 | **97 ms** |

## Status and limitations

- Joint `(ρ, log φ)` outer Newton not yet implemented — the binding constraint for closing the remaining performance gap on dispersion-bearing GLMs. Tracked in `mgcv_rust - Backlog - Next Numerical Steps` (Obsidian).
- `predict_diff` is identity-link only; non-identity raises with a workaround pointing at `get_posterior_samples`.
- Auto-`k` tuning is **opt-in** via `Gam(auto_k=True)`. Default is a single fit at `k_default=10` (mgcv's default), with `term_k_mapping` overrides — closer to mgcv's "tune k, run k.check" convention and avoids hidden multi-fit costs.
- sklearn `BaseEstimator` mixin (for `Pipeline` / `GridSearchCV`) is not yet wrapped — soft-dep, deferred.

## References

- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.
- Wood, S.N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *J. R. Stat. Soc. B*, 73(1), 3–36.

## License

MIT — see `LICENSE`.
