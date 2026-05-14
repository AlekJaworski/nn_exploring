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

## Reference fixtures

Captured 2026-05-13 from a real `/sale_price_prediction` call (KBB subject in Gilbert AZ, ss_cluster ≈ 5,200 listings → ~6,400 per-day-expanded training rows). Each parquet is the exact DataFrame fed into `fit_sale_price_gam` per CV fold.

Location: `data/sale_price_fixtures/`

| File | n_rows |
|---|---|
| `split_0_train.parquet` | 5,157 |
| `split_1_train.parquet` | 5,071 |
| `split_2_train.parquet` | 5,059 |
| `split_3_train.parquet` | 5,072 |
| `split_4_train.parquet` | 5,241 |
| `entire_dataset_train.parquet` | 6,400 |

### Schema (11 columns)

| Column | Dtype | Role |
|---|---|---|
| `sale_to_list_price_ratio` | float64 | **target** |
| `current_list_price` | int64 | smooth |
| `price_change_pct_from_original` | float64 | smooth |
| `cum_dom_before_current_regime` | int64 | smooth |
| `days_in_current_price_regime` | int64 | smooth |
| `monthly_index` | int64 | smooth (k clamped to ≤5) |
| `at_least_1_price_drop` | int64 (0/1) | parametric (mgcv only) |
| `at_least_2_price_drops` | int64 (0/1) | parametric (mgcv only) |
| `at_least_3_price_drops` | int64 (0/1) | parametric (mgcv only) |
| `weight` | float64 | per-row weight (mgcv only) |
| `listing_number` | object | id; not in the formula |

### Per-fold timings observed

`mgcv_rust 0.13.0`, `Gam(family='scat', method='REML', term_k_mapping={col: 5..7}, predictor_basis_map={col: 'cr'})`, no weights, no parametric terms. Container is `python:3.11-trixie`, `--features python,blas-system`, single-process worker (`--concurrency=1`).

| Fold | n_rows | mgcv_rust fit time |
|---|---|---|
| split_0 | 5,157 | 7.84 s |
| split_1 | 5,071 | 3.71 s |
| split_2 | 5,059 | 4.16 s |
| split_3 | 5,072 | 7.73 s |
| split_4 | 5,241 | 5.36 s |
| entire_dataset | 6,400 | 5.03 s |
| **mean** | — | **~5.6 s** |

R baseline (from a QA trace on similar data): `mgcv::bam(family=scat, method='fREML', discrete=TRUE, nthreads=4)` finished each fold in ~1.4 – 2.1 s. So mgcv_rust appeared ~3× slower per fit — but the comparison isn't apples-to-apples (mgcv had extra parametric terms and weights, *and* used `discrete=TRUE` covariate binning + 4 threads).

For a fair bench the dev probably wants to compare `Gam(scat, REML, no-weights)` to `bam(scat, fREML, no-discrete, nthreads=1, no-weights, no-parametric)`. The fixtures above let them reproduce the workload exactly.

### Minimal repro

```python
import pandas as pd
from mgcv_rust import Gam

SMOOTH = ['current_list_price', 'price_change_pct_from_original',
          'cum_dom_before_current_regime', 'days_in_current_price_regime',
          'monthly_index']

df = pd.read_parquet('data/sale_price_fixtures/split_0_train.parquet')
k = {c: min(7, df[c].nunique()) for c in SMOOTH}
k['monthly_index'] = min(5, df['monthly_index'].nunique())

gam = Gam(
    family='scat', method='REML',
    term_k_mapping=k,
    predictor_basis_map={c: 'cr' for c in SMOOTH},
).fit(df[SMOOTH], df['sale_to_list_price_ratio'].to_numpy())
```

### qgam fixture

The same `entire_dataset` frame plus 5 fold predictions feeds `train_quantile_gam` (now `mgcv_rust.fit_quantile`). The qgam fit at q=0.95 ran in **0.13 s** on 6,400 OOS rows — vs ~5 s for `qgam::qgam` locally and **230 s** on QA. No reproducibility issue there — this swap is a clear win.
