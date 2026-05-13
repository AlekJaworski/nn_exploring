"""End-to-end tests for sample weights (mgcv's `weights=` arg).

Three layers of coverage:

1. **Unit-weights equivalence** — `Gam(...).fit(X, y)` and
   `Gam(...).fit(X, y, sample_weight=np.ones(n))` produce coefficients
   matching to machine precision. Catches accidental scale shifts in
   the weighted code path.

2. **Non-uniform weights move the fit** — upweighting one half of the
   data 10× shifts coefficients by an order of magnitude more than the
   tolerance allows for the equivalence test. Catches the case where
   weights are silently ignored.

3. **mgcv parity** — fit `gam(y ~ s(x), weights=w, method="REML")` in R
   via rpy2, then fit the same problem in Python and assert
   coefficients agree to rtol=1e-6. Catches numerical incorrectness in
   our weighted normal equations (e.g. wrong factor of W in the deviance
   or in X'Wy).

The parity case is skipped when rpy2 / mgcv aren't importable so the
ergonomics suite still runs on minimal CI configurations; it's run
locally and in the parity battery.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgcv_rust import Gam


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _make_data(seed: int = 2026, n: int = 400):
    """Single-predictor cubic problem with smooth signal + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.0, 3.0, n)
    y_true = 0.5 * x + 0.3 * x**2 - 0.05 * x**3
    y = y_true + rng.normal(0, 0.2, n)
    return x, y


def _coef(gam: Gam) -> np.ndarray:
    """Get coefficients from a fitted Gam, regardless of wrapper version."""
    return np.asarray(gam._native.get_coefficients(), dtype=float)


# ---------------------------------------------------------------------- #
# Tests                                                                  #
# ---------------------------------------------------------------------- #


def test_uniform_weights_equivalent_to_unweighted():
    """`weights=ones(n)` must reproduce the unweighted fit to machine precision.

    A non-trivially-wrong implementation (e.g. one that adds a spurious
    sqrt(w) factor on one side of the normal equations) would shift
    coefficients by O(1) here.
    """
    x, y = _make_data(seed=11, n=300)
    X = x.reshape(-1, 1)
    n = X.shape[0]

    gam_unw = Gam(predictors=("x0",), k_default=10).fit(X, y)
    gam_wgt = Gam(predictors=("x0",), k_default=10).fit(X, y, sample_weight=np.ones(n))

    beta_unw = _coef(gam_unw)
    beta_wgt = _coef(gam_wgt)
    assert beta_unw.shape == beta_wgt.shape
    np.testing.assert_allclose(beta_wgt, beta_unw, rtol=1e-10, atol=1e-12)


def test_nonuniform_weights_differ():
    """Upweighting half the data 10× must measurably move the fit.

    Catches the failure mode where the weights argument is accepted but
    silently dropped on the way into the Rust core. We require the
    L∞ coefficient diff to exceed the equivalence-test tolerance by
    several orders of magnitude.
    """
    x, y = _make_data(seed=12, n=400)
    X = x.reshape(-1, 1)
    n = X.shape[0]

    # Upweight the second half by 10×. With such asymmetric weighting
    # the smoother shifts noticeably toward the upweighted half.
    w = np.ones(n)
    w[n // 2:] = 10.0

    gam_unw = Gam(predictors=("x0",), k_default=10).fit(X, y)
    gam_wgt = Gam(predictors=("x0",), k_default=10).fit(X, y, sample_weight=w)

    beta_unw = _coef(gam_unw)
    beta_wgt = _coef(gam_wgt)
    max_diff = float(np.max(np.abs(beta_wgt - beta_unw)))
    # Equivalence test holds at rtol=1e-10; a real weight effect should be
    # at least 1e-3 in coefficient magnitude. Anything in between would
    # indicate weights getting partial (probably wrong) effect.
    assert max_diff > 1e-3, (
        f"non-uniform weights produced identical coefficients (max diff "
        f"{max_diff:.3e}); weights are likely being silently dropped"
    )


def test_weights_validation_rejects_negative_and_wrong_length():
    """Catch bad inputs early. Mirrors mgcv's behaviour of failing on
    non-positive weights and on mismatched lengths."""
    x, y = _make_data(seed=13, n=100)
    X = x.reshape(-1, 1)

    # Wrong length
    with pytest.raises(ValueError):
        Gam(predictors=("x0",), k_default=8).fit(X, y, sample_weight=np.ones(50))

    # Negative weights
    bad_w = np.ones(X.shape[0])
    bad_w[0] = -1.0
    with pytest.raises(Exception):  # ValueError-equivalent from Rust side
        Gam(predictors=("x0",), k_default=8).fit(X, y, sample_weight=bad_w)


def test_predictions_unchanged_by_unit_weights():
    """End-to-end check: predictions on new data must match between
    unweighted and unit-weighted fits."""
    x, y = _make_data(seed=14, n=250)
    X = x.reshape(-1, 1)
    X_new = np.linspace(-2.5, 2.5, 50).reshape(-1, 1)

    gam_unw = Gam(predictors=("x0",), k_default=10).fit(X, y)
    gam_wgt = Gam(predictors=("x0",), k_default=10).fit(
        X, y, sample_weight=np.ones(X.shape[0])
    )

    p_unw = np.asarray(gam_unw.predict(X_new), dtype=float)
    p_wgt = np.asarray(gam_wgt.predict(X_new), dtype=float)
    np.testing.assert_allclose(p_wgt, p_unw, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------- #
# Parity vs mgcv (rpy2-gated)                                            #
# ---------------------------------------------------------------------- #


try:
    import rpy2.robjects as _ro
    from rpy2.rinterface_lib.callbacks import logger as _rpy2_logger
    from rpy2.robjects import default_converter as _default_converter
    from rpy2.robjects import pandas2ri as _pandas2ri
    from rpy2.robjects.conversion import localconverter as _localconverter
    from rpy2.robjects.packages import importr
    import logging as _logging

    _rpy2_logger.setLevel(_logging.ERROR)
    _mgcv = importr("mgcv")
    _stats = importr("stats")
    _ro.r('options(contrasts = c("contr.sum", "contr.poly"))')
    _ro.r("options(warn = -1)")
    _RPY2_AVAILABLE = True
except Exception:
    _RPY2_AVAILABLE = False


@pytest.mark.skipif(not _RPY2_AVAILABLE, reason="rpy2 / mgcv not available")
def test_weights_parity_against_mgcv():
    """Fit `gam(y ~ s(x, bs='cr', k=10), weights=w, method='REML')` in R,
    fit the same problem in mgcv_rust, and assert coefficients agree to
    rtol=1e-6.

    Tolerance rationale: unweighted Gaussian parity holds to <1e-10 (the
    fixture battery checks this). Weighted Gaussian adds one extra
    matrix-vector multiplication (X'Wy and the diag(W) into X'WX) so we
    expect the same machine-precision agreement; we set rtol=1e-6 as a
    generous bar so noise from BLAS ordering / ridge doesn't trip the
    test on weird platforms.
    """
    import pandas as pd

    rng = np.random.default_rng(2026)
    n = 300
    x = np.linspace(-3.0, 3.0, n)
    y = 0.5 * x + 0.3 * x**2 - 0.05 * x**3 + rng.normal(0, 0.2, n)
    # Non-trivial weight pattern: smooth function of x so weighted vs
    # unweighted fits diverge meaningfully.
    w = 1.0 + np.abs(x)

    df_r = pd.DataFrame({"x0": x, "y": y, "w": w})

    formula = _ro.Formula("y ~ s(x0, bs='cr', k=10)")
    with _localconverter(_pandas2ri.converter + _default_converter):
        r_train = _pandas2ri.py2rpy(df_r)
    fit = _mgcv.gam(
        formula,
        data=r_train,
        weights=_ro.FloatVector(w.tolist()),
        method="REML",
        family=_ro.r('gaussian(link="identity")'),
    )
    beta_mgcv = np.asarray(_stats.coef(fit), dtype=float)
    lambda_mgcv = float(np.asarray(fit.rx2("sp"))[0])

    # mgcv_rust side. bs='cr' to match.
    gam = Gam(predictors=("x0",), k_default=10).fit(
        x.reshape(-1, 1), y, sample_weight=w
    )
    beta_ours = _coef(gam)
    lambda_ours = float(np.asarray(gam._native.get_all_lambdas())[0])

    # Beta length must match (one intercept + k-1 centred basis cols).
    assert beta_ours.shape == beta_mgcv.shape, (
        f"coef length mismatch: ours {beta_ours.shape} vs mgcv {beta_mgcv.shape}"
    )

    # Parity assertion. We're a bit loose on the absolute tolerance
    # because the intercept-relative offsets can be ~10 while individual
    # smooth basis coefficients are ~1e-2.
    np.testing.assert_allclose(beta_ours, beta_mgcv, rtol=1e-6, atol=1e-6)

    # Lambda should agree closely too (within the REML optimizer's
    # converge tol; battery shows rtol ~5e-2 for cr-spline cases).
    assert abs(lambda_ours - lambda_mgcv) / max(abs(lambda_mgcv), 1e-12) < 5e-2, (
        f"lambda mismatch: ours {lambda_ours:.4e} vs mgcv {lambda_mgcv:.4e}"
    )
