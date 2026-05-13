# Getting started with `mgcv_rust`

A walkthrough of the `mgcv_rust.Gam` Python API. We'll fit a synthetic
regression model, inspect it, predict on new data with proper
confidence intervals, do marginal-effect analysis with subset views,
then package the fit as a `GamPredictor` for deployment.

If you're already familiar with R's `mgcv` or `scikit-learn`, this
should read as a thin combination of the two.

## Contents

1. [Install](#install)
2. [Fit your first GAM](#fit-your-first-gam)
3. [What's in a fitted model?](#whats-in-a-fitted-model)
4. [Predicting on three scales](#predicting-on-three-scales)
5. [Confidence intervals](#confidence-intervals)
6. [Differences between predictions, with paired CI](#differences-between-predictions-with-paired-ci)
7. [Subset / marginal views](#subset--marginal-views)
8. [Plot, summary, score](#plot-summary-score)
9. [Classification (binomial)](#classification-binomial)
10. [Serving with `GamPredictor`](#serving-with-gampredictor)
11. [Common patterns and gotchas](#common-patterns-and-gotchas)

---

## Install

```bash
# In a fresh venv:
pip install maturin numpy pandas matplotlib  # matplotlib is optional but recommended
maturin develop --features python,blas,blas-system --release
python -c "import mgcv_rust; print(mgcv_rust.Gam)"
```

## Fit your first GAM

```python
import numpy as np
import pandas as pd
from mgcv_rust import Gam

rng = np.random.default_rng(0)
n = 500
X = pd.DataFrame({
    "age":   rng.uniform(20, 80, n),
    "score": rng.uniform(0, 100, n),
})
# True relationship: smooth in age, U-shape in score, additive
y_true = 0.05 * (X["age"] - 50)**2 + 0.001 * (X["score"] - 50)**2
y = y_true + rng.normal(0, 1.0, n)

gam = Gam(family="gaussian").fit(X, y)
```

That's it. `Gam()` works with sensible defaults — `family="gaussian"`,
canonical link, `k_default=10` (mgcv default), REML smoothing parameter
selection.

## What's in a fitted model?

`Gam` exposes sklearn-style underscore attributes after `fit`:

```python
gam.feature_names_in_    # array(['age', 'score'], dtype=object)
gam.n_features_in_       # 2
gam.coef_                # full β vector (length = 1 + sum(k_i - 1))
gam.intercept_           # link-scale (= response-scale for identity link)
gam.intercept_response_  # response-scale (= mean(y_train) for identity link)
gam.lambda_              # one smoothing parameter per smooth
gam.edf_                 # effective degrees of freedom per smooth
gam.k_                   # basis dim per smooth (after auto-cap)
gam.bs_                  # basis type per smooth ('cr', 'bs', 're', ...)
gam.vcov_                # posterior covariance of β̂
```

For a quick overview, use `gam.summary()`:

```python
print(gam.summary())
# Gam summary  family=gaussian  link=identity  n_obs=500
#   intercept (link)     = 9.473
#   intercept (response) = 9.473
#   scale (σ²)           = 0.9712
#   deviance             = 478.04
#   R² (adj)             = 0.9847
#   smooths:
#     s(         age)  k= 10  edf=  3.05  λ=4.231e-04
#     s(       score)  k= 10  edf=  2.81  λ=8.762e-04
```

## Predicting on three scales

```python
gam.predict(X)                     # response scale (default)
gam.predict(X, scale="link")       # linear predictor, no inverse link
```

For Gaussian/identity link the two are identical. For binomial/logit
they differ — `scale="link"` returns the log-odds.

To decompose into per-smooth contributions:

```python
terms = gam.predict(X, type="terms")

terms.intercept       # scalar, link scale
terms.contributions   # DataFrame (n × 2), columns ['age', 'score']
terms.total           # 1-D, response scale, ≡ gam.predict(X)
```

The contributions are link-scale and **centered on training data**
(mgcv's sum-to-zero identifiability constraint), so each column
averages to ~0 on the training X.

## Confidence intervals

```python
mean, lo, hi = gam.predict_ci(X, level=0.95)
```

Three things to know:

- `mean` matches `gam.predict(X)` *exactly*. No "+= intercept" needed
  by the caller.
- The CI is built by sampling `β ~ N(β̂, vcov)`. Bump `n_samples` (default
  1000) for tighter quantile estimates.
- For `scale="response"` (default), the link-scale quantiles are
  inv-linked. For monotonic links (identity / log / logit) this is the
  correct interval; for non-monotonic links it would be biased.

```python
mean, lo, hi = gam.predict_ci(X, level=0.5)    # 50% interval (narrower)
mean, lo, hi = gam.predict_ci(X, scale="link") # link-scale interval
```

The legacy `predict_ci(X, alpha=0.05) → (lo, hi)` still works (emits
a `DeprecationWarning`) for back-compat.

## Differences between predictions, with paired CI

When you want the effect of *moving* from one X to another (e.g.
"what's the marginal impact of increasing `age` by 10 years for this
customer?"), `predict_diff` is the right tool:

```python
to_X = X.copy()
to_X["age"] = X["age"] + 10
diff = gam.predict_diff(X, to_X)                  # to[i] - from[i]
diff, lo, hi = gam.predict_diff(X, to_X, level=0.95)
```

The CI here is computed from **paired posterior samples** — one β draw
is shared between `from_X` and `to_X`, so the (correlated) coefficient
uncertainty cancels. The resulting interval is dramatically tighter
than naively differencing two `predict_ci` calls.

Three broadcast modes:

```python
# All rows of X relative to a single baseline
diff = gam.predict_diff(baseline_row_df, X, broadcast="from")

# All rows of X relative to a single target
diff = gam.predict_diff(X, target_row_df, broadcast="to")

# Row-wise pair
diff = gam.predict_diff(X_before, X_after)   # broadcast="none" (default)
```

Identity-link only — for non-identity links the response-scale
difference isn't linear in β, so the closed-form CI argument doesn't
hold. The method raises `NotImplementedError` with a pointer to
`get_posterior_samples` if you want to do it by hand.

## Subset / marginal views

`gam[name]` returns a view that masks the lpmatrix down to a chosen
set of smooths. All predict methods inherit the mask:

```python
# Marginal effect of `age` only, response scale:
gam[["age"]].predict(X)

# Marginal effect, link scale with intercept zeroed (the "centered smooth"):
gam[["age"]].predict(X, scale="deviation")

# Include the intercept explicitly:
gam[["age", Gam.INTERCEPT]].predict(X)

# CI on the marginal:
mean, lo, hi = gam[["age"]].predict_ci(X, scale="deviation")
```

sklearn attributes filter to the active features:

```python
gam[["age"]].feature_names_in_  # array(['age'])
gam[["age"]].coef_              # intercept (if active) + age block
gam[["age"]].n_features_in_     # 1
```

Subset views are cheap shallow copies — they don't refit. The
underlying coefficients and vcov are shared.

## Plot, summary, score

`partial_effect()` returns the data behind a smooth's curve:

```python
df = gam.partial_effect("age", grid_n=100, level=0.95)
# columns: age, effect, lo, hi
```

`plot()` is a matplotlib-only convenience:

```python
ax = gam.plot("age")     # single smooth
fig = gam.plot()         # all smooths
```

`score(X, y)` follows the sklearn "higher is better" convention:

- For regression families: adjusted R².
- For binomial / quasibinomial: classification accuracy at the 0.5
  threshold.

```python
gam.score(X, y)    # 0.985
```

## Classification (binomial)

```python
# Binary y in {0, 1}
rng = np.random.default_rng(1)
X = pd.DataFrame({"x0": rng.uniform(-2, 2, 500), "x1": rng.uniform(-1, 1, 500)})
eta = 0.8 * X["x0"] - 1.2 * X["x1"]
p = 1.0 / (1.0 + np.exp(-eta))
y = (rng.uniform(0, 1, 500) < p).astype(float)

clf = Gam(family="binomial").fit(X, y)

clf.predict(X)         # P(y=1)
clf.predict_proba(X)   # (n, 2) [[P(0), P(1)]] — sklearn convention
clf.score(X, y)        # accuracy at 0.5 threshold
clf.predict_ci(X)      # (mean, lo, hi) all response-scale (probabilities)
```

## Serving with `GamPredictor`

For deployment, wrap the fitted `Gam` in a `GamPredictor`. Same
predict / predict_ci / predict_diff API, plus:

- `__slots__` to block attribute creep.
- Strict `feature_names_in_` validation — any predict-time column drift
  raises `ValueError` (missing) or silently re-aligns (reordered).
- `check_against(gam, X_sample)` for a round-trip assertion at load
  time.

```python
from mgcv_rust import GamPredictor

predictor = GamPredictor(gam)
predictor.check_against(gam, X_train.sample(50))  # asserts predictions match
predictor.predict(X_serve)                        # raises on missing columns
```

Subset views on a `GamPredictor` return another `GamPredictor` (not a
subset `Gam`), so the frozen contract carries through:

```python
sub_pred = predictor[["age"]]      # GamPredictor
isinstance(sub_pred, GamPredictor)  # True
```

## Common patterns and gotchas

### Per-smooth `k`

`k_default=10` is mgcv's default. Override per smooth via
`term_k_mapping`:

```python
gam = Gam(
    term_k_mapping={"age": 25, "score": 12},
).fit(X, y)
```

The library caps each smooth's `k` at `n_unique(x_j) - 1` automatically.

### Random effects

Treat a categorical "group" predictor as a random effect:

```python
gam = Gam(
    predictor_basis_map={"cluster_id": "re"},
).fit(X, y)
```

`k` is ignored for `re` terms — it's set to the number of unique levels.

### Tweedie / negbin / scat

```python
Gam(family="tweedie", tweedie_p=1.5)  # fixed p
Gam(family="negbin", negbin_theta=2.0)  # fixed θ
Gam(family="nb")                       # profile θ
Gam(family="t-dist", df=None)          # profile df ∈ [2, 100]
```

### Polars / numpy inputs

`fit` and all predict methods accept pandas, polars, or numpy. Column
names are inferred from `DataFrame` columns; for numpy you can either
pass `predictors=("a", "b", ...)` to `Gam()` or let it default to
`x0, x1, ...`.

### Serialize for downstream consumers

If your serving environment can't install the Rust extension,
`gam.serialize()` returns a dict suitable for grid-based predictors.
**Prefer `GamPredictor` when possible** — it recomputes the basis at
predict time instead of interpolating, avoiding x-grid clamping bugs.

### Where to look next

- [`README.md`](../README.md) for the high-level overview.
- The `Gam` source in `python/mgcv_rust/_fitter.py` is the canonical
  reference for the Python API.
- The Rust core lives in `src/`; entry points are `src/lib.rs` (PyO3
  bindings) and `src/gam.rs` (the `GAM` struct).
- Parity tests in `tests/parity/`. Ergonomics tests in
  `tests/ergonomics/`.
