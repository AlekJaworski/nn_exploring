"""Real-data parity tests against the TrueFootage neighbourhoods handoff.

Reference outputs from R 4.6.0 + mgcv 1.9.4 + qgam 2.0.0 on real
`/sale_price_prediction` data (~5,000-6,400 rows × 6 folds). Captured
2026-05-14 per
`~/ObsidianVault/Projects/mgcv_rust/mgcv_rust - parity handoff 2026-05-14.md`.

The handoff lives at `~/vibe_coding/mgcv_rust_parity_handoff/` (gitignored — not
checked into the repo because the data is real customer data). Tests skip
gracefully when the dir is absent so CI doesn't fail without it; local dev with
the handoff present catches real-world regressions that the synthetic battery
misses (e.g. the R8 pivoted-Cholesky perf regression on production-shape XX).

Two reference fits:

1. **Mean GAM** — `scat, fREML, discrete=TRUE, weights, 5 smooths + 3
   parametric indicators`. Reference: `expected/mean_gam_<fold>.parquet`
   column `pred_prod`.
2. **qgam q=0.95** — `REML, k=5, bs='cr', 4 smooths, unweighted`. Reference:
   `expected/qgam_q95_entire_dataset.parquet` column `qgam_q95_pred`.

Tolerances are initially set to current measured values (last refreshed
2026-05-14 on master `2c544c1`). Tighten as fixes land.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")  # parquet backend; skip if missing

import mgcv_rust
from mgcv_rust._quantile import fit_quantile

HANDOFF = Path.home() / "vibe_coding" / "mgcv_rust_parity_handoff"
META_PATH = HANDOFF / "metadata.json"

SMOOTHS = [
    "current_list_price",
    "price_change_pct_from_original",
    "cum_dom_before_current_regime",
    "days_in_current_price_regime",
    "monthly_index",
]
PARAMETRIC = [
    "at_least_1_price_drop",
    "at_least_2_price_drops",
    "at_least_3_price_drops",
]
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]

# Current measured tolerances. Update as fixes (R9, well-conditioned fast path)
# land. Listed per fold because the handoff covers all 5 CV folds plus the
# entire-dataset fit; tolerance varies with fold composition.
MEAN_GAM_TOLERANCES = {
    # max_abs (per-row), p95_abs, rmse — measured 2026-05-14 master 2c544c1
    "split_0":         {"max_abs": 0.08, "p95_abs": 0.02, "rmse": 0.01},
    "split_1":         {"max_abs": 0.08, "p95_abs": 0.02, "rmse": 0.01},
    "split_2":         {"max_abs": 0.10, "p95_abs": 0.02, "rmse": 0.01},
    "split_3":         {"max_abs": 0.10, "p95_abs": 0.02, "rmse": 0.01},
    "split_4":         {"max_abs": 0.07, "p95_abs": 0.02, "rmse": 0.01},
    "entire_dataset":  {"max_abs": 0.08, "p95_abs": 0.02, "rmse": 0.01},
}

# qgam tolerances per config. Listed config is the operating point we expect to
# use in production swaps. Tighter than mean GAM because the qgam REML path is
# unaffected by the R8 fREML changes.
QGAM_TOLERANCES = {
    "default_heuristic": {"max_abs": 0.07, "subject_delta_abs": 0.05},
    "calibrate_pin":     {"max_abs": 0.05, "subject_delta_abs": 0.04},
    "calibrate_cal_kl":  {"max_abs": 0.06, "subject_delta_abs": 0.05},
}


def _handoff_available() -> bool:
    return META_PATH.exists()


pytestmark = pytest.mark.skipif(
    not _handoff_available(),
    reason=(
        "neighbourhoods parity handoff not present at "
        f"{HANDOFF} — see "
        "`~/ObsidianVault/Projects/mgcv_rust/mgcv_rust - parity handoff 2026-05-14.md`. "
        "Run `python generate.py` in that dir to materialise the references."
    ),
)


def _metadata() -> dict:
    return json.loads(META_PATH.read_text())


def _diff_metrics(rust_pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = rust_pred - ref
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "p95_abs": float(np.percentile(np.abs(diff), 95)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
    }


@pytest.fixture(scope="module")
def meta() -> dict:
    return _metadata()


@pytest.mark.parametrize("fold", list(MEAN_GAM_TOLERANCES.keys()))
def test_mean_gam_real_data_parity(fold: str, meta: dict) -> None:
    """`Gam(method='fREML', discrete=True, family='t-dist')` vs mgcv `pred_prod`."""
    df = pd.read_parquet(HANDOFF / "inputs" / f"{fold}_train.parquet")
    expected = pd.read_parquet(HANDOFF / "expected" / f"mean_gam_{fold}.parquet")
    k_map = meta["fixtures"][fold]["k_mean_gam"]
    tol = MEAN_GAM_TOLERANCES[fold]

    predictors = SMOOTHS + PARAMETRIC
    X = df[predictors]
    y = df["sale_to_list_price_ratio"].to_numpy()
    w = df["weight"].to_numpy()

    gam = mgcv_rust.Gam(
        predictors=predictors,
        target="sale_to_list_price_ratio",
        family="t-dist",
        method="fREML",
        discrete=True,
        term_k_mapping=k_map,
        predictor_basis_map={
            **{c: "cr" for c in SMOOTHS},
            **{c: "parametric" for c in PARAMETRIC},
        },
    )
    gam.fit(X, y, sample_weight=w)
    pred = np.asarray(gam.predict(X))
    ref = expected["pred_prod"].to_numpy()
    m = _diff_metrics(pred, ref)

    msg = (
        f"\n  fold={fold}  n={len(df)}"
        f"\n  max_abs = {m['max_abs']:.4f}  (tol {tol['max_abs']:.4f})"
        f"\n  p95_abs = {m['p95_abs']:.4f}  (tol {tol['p95_abs']:.4f})"
        f"\n  rmse    = {m['rmse']:.4f}  (tol {tol['rmse']:.4f})"
        f"\n  bias    = {m['bias']:+.4f}"
    )
    assert m["max_abs"] <= tol["max_abs"], msg
    assert m["p95_abs"] <= tol["p95_abs"], msg
    assert m["rmse"] <= tol["rmse"], msg


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("default_heuristic", {"calibrate": False}),
        ("calibrate_pin", {"calibrate": True, "loss": "pin"}),
        # cal_kl is a known regression candidate (slower AND worse parity than
        # `pin` at tau=0.95) — see customer feedback 0.16. Test still runs to
        # detect further regressions.
        ("calibrate_cal_kl", {"calibrate": True, "loss": "cal_kl"}),
    ],
)
def test_qgam_q95_real_data_parity(label: str, kwargs: dict, meta: dict) -> None:
    """`fit_quantile(tau=0.95)` vs qgam reference on entire_dataset."""
    df = pd.read_parquet(HANDOFF / "inputs" / "entire_dataset_train.parquet")
    expected_mean = pd.read_parquet(HANDOFF / "expected" / "mean_gam_entire_dataset.parquet")
    qgam_expected = pd.read_parquet(HANDOFF / "expected" / "qgam_q95_entire_dataset.parquet")
    tol = QGAM_TOLERANCES[label]

    resid = expected_mean["y"].to_numpy() - expected_mean["pred_prod"].to_numpy()
    X = df[QGAM_SMOOTHS].to_numpy(dtype=float)

    gam_q, _sigma, _info = fit_quantile(
        X, resid, tau=0.95, k=[5, 5, 5, 5], bs="cr", method="REML", **kwargs
    )
    pred = np.asarray(gam_q.predict(X))
    ref = qgam_expected["qgam_q95_pred"].to_numpy()
    m = _diff_metrics(pred, ref)

    # Subject-row check (median features) — closest proxy to predict_subject_scalar
    subject_X = X.mean(axis=0, keepdims=True)
    rust_subject = float(gam_q.predict(subject_X)[0])
    qgam_subject = meta["fixtures"]["qgam_q95"]["subject_q95_pred"]
    subject_delta = rust_subject - qgam_subject

    msg = (
        f"\n  config={label}"
        f"\n  row max_abs = {m['max_abs']:.4f}  (tol {tol['max_abs']:.4f})"
        f"\n  rust subject q95 = {rust_subject:+.4f}"
        f"\n  qgam subject q95 = {qgam_subject:+.4f}"
        f"\n  subject Δ        = {subject_delta:+.4f}  (tol ±{tol['subject_delta_abs']:.4f})"
    )
    assert m["max_abs"] <= tol["max_abs"], msg
    assert abs(subject_delta) <= tol["subject_delta_abs"], msg


def test_mean_gam_subject_scalar_parity(meta: dict) -> None:
    """Subject-row prediction across all 6 folds is the production-relevant call.

    Customer feedback 0.16.0 measured -0.005 to -0.013 (rust slightly below mgcv).
    The subject row is the captured KBB property's actual feature values
    (in `subject_predictions.json::<fold>.subject_features`), not the training
    median.
    """
    subject = json.loads((HANDOFF / "subject_predictions.json").read_text())

    deltas = []
    for fold in MEAN_GAM_TOLERANCES.keys():
        df = pd.read_parquet(HANDOFF / "inputs" / f"{fold}_train.parquet")
        k_map = meta["fixtures"][fold]["k_mean_gam"]
        predictors = SMOOTHS + PARAMETRIC

        gam = mgcv_rust.Gam(
            predictors=predictors,
            target="sale_to_list_price_ratio",
            family="t-dist",
            method="fREML",
            discrete=True,
            term_k_mapping=k_map,
            predictor_basis_map={
                **{c: "cr" for c in SMOOTHS},
                **{c: "parametric" for c in PARAMETRIC},
            },
        )
        gam.fit(
            df[predictors],
            df["sale_to_list_price_ratio"].to_numpy(),
            sample_weight=df["weight"].to_numpy(),
        )
        feats = subject[fold]["subject_features"]
        subject_row = pd.DataFrame([{c: feats[c] for c in predictors}])
        rust_subject = float(gam.predict(subject_row)[0])
        mgcv_subject = subject[fold]["mean_gam_prod"]
        delta = rust_subject - mgcv_subject
        deltas.append((fold, rust_subject, mgcv_subject, delta))

    # Per-fold cap: rust must be within 0.02 of mgcv on the subject scalar
    # (current measured: -0.005 to -0.013 per customer feedback 0.16)
    failures = [(f, r, m, d) for (f, r, m, d) in deltas if abs(d) > 0.02]
    if failures:
        msg = "\n".join(
            f"  {f}: rust={r:.5f}  mgcv={m:.5f}  Δ={d:+.5f}"
            for (f, r, m, d) in deltas
        )
        pytest.fail(f"subject-scalar parity exceeded ±0.02 on:\n{msg}")
