# mgcv_rust — captured fit failures on synthetic test data

Captured 2026-05-17→18 while swapping `mgcv::bam` / `qgam::qgam` for
`mgcv_rust.Gam` / `mgcv_rust.fit_quantile` in the neighbourhoods codebase.

Two distinct failure clusters, both from **synthetic test fixtures** (not from
production data — real-data fits at `~/vibe_coding/nn_exploring/data/sale_price_fixtures/`
are stable on both 0.16.1 and 0.16.2, see `BLOWUP_REPORT.md` in the gitlab worktree).

> **The directory name `mgcv_rust_lapack_failures/` is now slightly historical** —
> it was created when the original cluster was a `Lapack(LapackComputationalFailure)`
> error on 0.16.1. 0.16.2 fixed those four cases; the remaining cluster (`em_nb_cases/`)
> fails with a different error (`non-finite gradient`) on 0.16.2.

## Layout

```
mgcv_rust_lapack_failures/
├── README.md                       this file
├── saleprice_synthetic/            sale_price_prediction worker test fixtures
│   ├── case_20x5_main_fixture/     primary fixture (20 listings × 5 obs)
│   ├── case_10x5/                  parametrized (10 × 5)
│   ├── case_25x4/                  parametrized (25 × 4)
│   ├── case_50x3/                  parametrized (50 × 3)
│   └── summary.json
└── em_nb_cases/                    test_gam_confidence_intervals fixtures
    ├── case00_seed123_family_nb/   ... through case19_seed142_family_nb
    ├── case20_seed143_family_gaussian/ ... through case39_seed162_family_gaussian
    └── summary.json
```

Each `caseNN/` contains:
- `train.parquet` — the exact (X, y) handed to `.fit()`
- `params.json` — predictors / family / link / method / k / n_obs / seed / shift_amount
- `error.txt` — full exception + python traceback (present only when fit fails)

---

## Cluster A — `saleprice_synthetic/` (FIXED IN 0.16.2)

**Origin:** the `training_dataframe` pytest fixture in
`worker/tests/ml_worker/sale_price_prediction/test_gam_training.py:18-37`,
plus the `(n_listings, obs_per_listing)` parametrize at line 98-102.

**0.16.1 behavior** — all four cases raised:
```
ValueError: Invalid parameter: clamped_newton_step: eigendecomposition failed:
            Lapack(LapackComputationalFailure { return_code: 7 })
```
inside `mgcv_rust._fitter._single_fit` → native `clamped_newton_step`.
`return_code: 7` is LAPACK `info > 0` from the symmetric eigensolver (DSYEVR
or a relative) — Hessian goes singular before Newton-step clamping rescues it.

**0.16.2 behavior** — **all four cases fit successfully** (re-verified
2026-05-18). The parquet inputs are preserved here as a regression artifact in
case a future version reintroduces it.

**Fit signature (sale_price mean GAM):**
- `mgcv_rust.Gam(family='t-dist', method='fREML', discrete=True, predictors=[5 smooths + 3 parametric])`
- 5 cubic-regression smooths: `current_list_price, price_change_pct_from_original,
  cum_dom_before_current_regime, days_in_current_price_regime, monthly_index`
- 3 parametric indicators: `at_least_{1,2,3}_price_drop`
- Per-row weights = 1/n_obs_per_listing (uniform within fixture)

---

## Cluster B — `em_nb_cases/` (FIXED 2026-05-19)

**Origin:** the `gam_test_case` parametrize fixture in
`src/tests/pipelines/em/test_r_model.py:140-167` (`generate_data` + `PARAMS`).

**Root cause** — `Family::d_inverse_link` (log-link families) did not apply
the same `eta.min(20.0)` upper clamp that `inverse_link` does. So when the
fREML inner IRLS over-shot β and produced an iterate with `eta > 20` on any
row, `inverse_link` returned `mu = exp(20)` (finite, clamped) while
`d_inverse_link` returned `exp(eta) = inf`. The working weight
`w = (dμ/dη)² / V(μ) = inf²/finite = inf` (or NaN) propagated through
`X'WX`, `X'Wz`, and the fREML gradient, tripping `clamped_newton_step`.

**Fix** — `src/pirls.rs:Family::d_inverse_link` now clamps `eta` symmetrically
with `inverse_link` for the log-link families (Poisson, QuasiPoisson, GammaLog,
Tweedie, InverseGaussian, NegBin).

**Regression coverage** —
- `tests/parity/fixtures/1d_em_nb_seed{123..142}_n1000_k4_cr.json` —
  parity vs mgcv under REML (where mgcv's `gam` and our Rust core agree).
- `tests/usability/test_em_nb_freml_smoke.py` — fREML smoke at k=4 on
  all 20 captured seeds, verifying the original non-finite-gradient
  failure mode does not return. Seed 137 is currently xfailed: fREML
  no longer crashes but converges to a degenerate β in the flat
  η > 20 plateau region — a deeper follow-up.

**Earlier 0.16.2 behavior (pre-fix snapshot)** — **20/20 `nb`-family seeds
fail, 20/20 `gaussian`-family seeds succeed.** Error class:
```
ValueError: Invalid parameter: clamped_newton_step: non-finite gradient
```
at `mgcv_rust._fitter._single_fit` (`_fitter.py:649`). Same `clamped_newton_step`
call site as Cluster A but a different branch — the gradient itself goes
non-finite before the eigendecomposition is attempted.

**0.16.1 behavior on the same seeds** — a related but differently-worded error:
```
ValueError: Optimization failed: fit_pirls_fastreml: non-finite candidate at
            outer iter 17 after observed-info AND Fisher retries — likely
            indefinite X'WX from extreme outlier rows that pivoted-Chol's rank
            truncation cannot resolve. Try restricting `df` away from the
            heavy-tail boundary, or fall back to method='REML'.
```
That error path hints at `method='REML'` as a workaround. We have **not**
verified yet whether `REML` (instead of our GamFitter default `fREML`) actually
makes the nb seeds fit — that's part of what the investigation below should
test.

**Fit signature (test_gam_confidence_intervals):**
- `GamFitter(family='nb', link='log', predictors=['x1'])`  ← our wrapper sets `method='fREML', k_default=4`
- X = `x1 - x1.mean()` (post-`estimators.Gam` shift), single column, 1000 rows, uniform[~−50, 50]
- y = multivariate sin/cos/poly construction + N(0, 2) noise, then shifted by `y.min()` to be non-negative
  - For `nb` family, target is supposed to be count-valued — but `generate_data()` produces real-valued shifted y. That mismatch may be the underlying issue: an `nb`-family fit on a non-count, heavy-tail continuous target.

**Reproduce one case:**
```python
import pandas as pd, json
from r_fitting.r_model import GamFitter
import numpy as np

case_dir = "em_nb_cases/case00_seed123_family_nb"
df = pd.read_parquet(f"{case_dir}/train.parquet")
p  = json.loads(open(f"{case_dir}/params.json").read())

X = df[["x1"]].to_numpy()
y = df["y"].to_numpy()
X = X - np.nanmean(X, axis=0)  # the shift estimators.Gam.fit applies

GamFitter(predictors=["x1"], family=p["family"], link=p["link"]).fit(X, y)
# → ValueError: ... non-finite gradient
```

---

## What an investigation should determine

For Cluster B specifically (the only one still failing on 0.16.2):

1. **Does `method='REML'` fix the nb cases?** The 0.16.1 error message explicitly
   suggested this; the 0.16.2 error doesn't, but the codepath may be the same.
2. **Is `nb` family appropriate for this y?** The fixture's `y` is not count-valued.
   If R's `mgcv` succeeded only by lenience, the test fixture itself may be wrong.
3. **Does `k_default=3` (vs 4) fix it?** Smaller spline basis = less chance of
   IRLS divergence.
4. **R baseline:** does `mgcv::bam(family=nb(), link='log', method='fREML')` on
   these same inputs succeed, and if so, what does its `summary()` look like?

`rpy2` is installed in `/tmp/mgcv_swap_venv` for the comparison.
