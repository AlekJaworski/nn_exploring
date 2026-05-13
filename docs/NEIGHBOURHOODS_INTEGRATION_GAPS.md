# Neighbourhoods Integration — mgcv_rust Gaps

Context: porting `mgcv.bam(...)` calls in the neighbourhoods worker (R via rpy2) to `mgcv_rust.Gam`. Targeting the `sale_price_prediction` workflow (slow path: ~230s qgam aggregation observed in QA trace 2026-05-13).

Call sites in `gitlab/2025/neighbourhoods/worker/src/workers/ml_worker/`:

| # | File:line | family | method | weights? | smooths | parametric terms | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `sale_price_prediction/algorithm/train_mean_gam.py:48` | `scat(link='identity')` | `fREML` | yes (`train_df['weight']`) | 5× `s(x, k=K, bs='cr')` | 3× `at_least_{1,2,3}_price_drop` (0/1 ints) | also `discrete=True, nthreads=4` |
| 2 | `time_to_sell_analysis/algorithm/predict.py:171` | `scat(link='identity')` | `fREML` | yes | 4× `s(x, k=K, bs='cr')` | 3× `at_least_{1,2,3}_price_drop` | active-listings GAM |
| 3 | `time_to_sell_analysis/algorithm/predict.py:275` | `scat(link='identity')` | `fREML` | **no** | 4× `s(x, k=K, bs='cr')` | 3× `at_least_{1,2,3}_price_drop` | pending-listings GAM |
| 4 | `time_to_sell_analysis/algorithm/predict.py:513` | `ocat(R=4)` | `fREML` | no | 3× `s(x, k=K, bs='cr'\|'cs')` | none | `discrete=True, select=True`; ordered category outcome |

## Features needed in mgcv_rust for a clean swap

### Required for #1–#3 (the `scat` calls)

- **Sample weights** — `Gam.fit(X, y, sample_weight=...)` or `Gam(... weights=...)`. Currently used in #1, #2.
- **Parametric (linear, unsmoothed) terms** — 0/1 indicator columns `at_least_{1,2,3}_price_drop` enter the formula as plain additive terms, not `s(...)`. Need a way to mark a DataFrame column as parametric so it isn't smoothed.
- **Explicit basis selection per term** — current code pins `bs='cr'`. Confirm whether `term_k_mapping={"col": k}` alone selects `cr`, or if we need `term_bs_mapping={"col": "cr"}`.
- **`fREML` ≡ `REML`?** — current code asks for `fREML`. If `method="REML"` in mgcv_rust returns the same coefficients (it should, fast-REML is an outer-loop accelerator only), document that explicitly.

### Optional but nice-to-have for #1

- **`discrete=True` analogue** — mgcv's discrete covariate binning for `bam()`. mgcv_rust may already be doing the equivalent internally; if so, mention that in family-table notes.
- **Multi-thread / `nthreads`** — mgcv_rust is already Rust-parallel internally; document the parallelism model so we know `nthreads=4` is a no-op equivalent.

### Required for #4 (the `ocat` call)

- **`ocat` (ordered categorical)** family with `R=` levels. Not in the family table.
- **`select=True`** — `mgcv`'s smoothness selection that allows smooth terms to be shrunk to zero (i.e., extra null-space penalty). If supported under another name, document.

## R-formula quirks we'd want to handle

Both #1 and #2 build their formula strings dynamically from `get_*_k_values()` — a Python dict of `{col: k}`. That maps cleanly to `term_k_mapping`. The parametric terms are static (always 3 indicator columns) — if mgcv_rust can't auto-detect "this column is binary, don't smooth it", we'd need an explicit `parametric_terms=["at_least_1_price_drop", ...]` kwarg.

## Prediction-side

Workers call `ro.r.predict(model, newdata=df, type='response')`. With mgcv_rust this is `gam.predict(df, scale='response')` — no gap.

## Adjacent (qgam) — not in scope of mgcv replacement but worth noting

`sale_price_prediction/algorithm/train_quantile_gam.py:53` calls `qgam.qgam(..., qu=0.95)`. The README mentions `family="quantile"` and `mgcv_rust.fit_quantile / fit_quantile_lss`. Worth a separate gap-doc for the qgam swap — that's where the 230s slow path actually lives.
