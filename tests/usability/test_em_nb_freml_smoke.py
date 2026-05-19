"""Regression smoke for the captured em_nb fREML failure cluster.

Captured 2026-05-17→18 (see `data/mgcv_rust_lapack_failures/README.md`):
20 nb(log) seeds (123..142) where production code called
`GamFitter(family='nb', link='log', predictors=['x1']).fit(...)` —
which dispatches to `method='fREML', k_default=4, min_k=3` — and 7 of
them blew up inside the `clamped_newton_step` path with
`ValueError: Invalid parameter: clamped_newton_step: non-finite gradient`.

Root cause: log-link `Family::inverse_link` clamps `eta_safe = eta.min(20)`
but `Family::d_inverse_link` did not, so an over-shooting IRLS iterate
gave `mu = exp(20)` (finite, clamped) while `dμ/dη = exp(eta) = inf`,
producing `w = (dμ/dη)²/V(μ) = nan/inf` working weights. The non-finite
weights propagated through `X'WX`, `X'Wz`, and the fREML gradient.

Fix (src/pirls.rs `Family::d_inverse_link`): clamp `eta` symmetrically
with `inverse_link` for the log-link families. This test pins the fix
by running fREML at k=4 on all 20 captured seeds and asserting each
fit completes without raising and produces finite predictions.

Parity vs mgcv is covered separately under `tests/parity/` via the
`1d_em_nb_seed*_n1000_k4_cr` fixtures (those run REML since mgcv's
`gam` coerces fREML→REML; the Rust fREML path here is exercised for
*successful completion*, not byte-identical parity, because mgcv's
`bam` fREML profiles θ where our wrapper holds it fixed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

import mgcv_rust

REPO_ROOT = Path(__file__).resolve().parents[2]
EM_NB_DIR = REPO_ROOT / "data" / "mgcv_rust_lapack_failures" / "em_nb_cases"

SEEDS = list(range(123, 143))


def _load_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = seed - 123
    case_dir = EM_NB_DIR / f"case{idx:02d}_seed{seed}_family_nb"
    df = pd.read_parquet(case_dir / "train.parquet")
    x = df[["x1"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    # Mirror estimators.Gam.fit() pre-shift on x1.
    x = x - x.mean(axis=0)
    return x, y


@pytest.mark.skipif(not EM_NB_DIR.exists(), reason="em_nb_cases data not checked out")
@pytest.mark.parametrize("seed", SEEDS)
def test_em_nb_freml_k4_no_crash(seed: int) -> None:
    """Production-equivalent fit completes and returns finite, in-range predictions.

    Covers two failure modes that both used to bite this case at k=4:
      1. Non-finite gradient at fit time (original crash, closed by the
         [`crate::link`] conjugation fix).
      2. Degenerate β converging on the η > eta_max saturation plateau
         (closed by mgcv-style β init at link(mean(y)) + deviance-based
         β-step blending in `fit_pirls_fastreml`).
    """
    x, y = _load_case(seed)
    gam = mgcv_rust.Gam(
        predictors=["x1"],
        family="nb",
        link="log",
        method="fREML",
        k_default=4,
        min_k=3,
    )
    gam.fit(x, y)
    pred = np.asarray(gam.predict(x), dtype=float)
    assert pred.shape == y.shape, (
        f"seed={seed}: prediction shape {pred.shape} != y shape {y.shape}"
    )
    assert np.all(np.isfinite(pred)), (
        f"seed={seed}: predictions contain non-finite values "
        f"(min={pred.min()}, max={pred.max()})"
    )
    assert pred.min() > 0.0, (
        f"seed={seed}: prediction has non-positive value {pred.min()}"
    )
    y_max = float(y.max())
    assert pred.max() < 10.0 * y_max, (
        f"seed={seed}: prediction max {pred.max()} far exceeds y_max {y_max}"
    )
