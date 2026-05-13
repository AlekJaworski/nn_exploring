"""0.12.1: ergonomic input coercion on the low-level `GAM` facade.

The previous `mgcv_rust.GAM` was the raw Rust binding, which rejected:
- 1-D ``x`` arrays (``np.linspace(0, 1, n)`` shape ``(n,)``)
- non-``float64`` dtypes
- scalar ``k`` (``k=10`` instead of ``[10]``)

The new ``GAM`` is a composition facade that coerces these inputs
before delegating to the native binding. All other methods forward
transparently via ``__getattr__``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mgcv_rust import GAM


# ---------------------------------------------------------------------- #
# The notebook-style call that used to TypeError                          #
# ---------------------------------------------------------------------- #


def test_notebook_style_call_works():
    """gam.fit(xs, ys, k=10) with 1-D xs and scalar k — the case the user hit."""
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 1, 200)
    ys = np.sin(2 * np.pi * xs) + 0.1 * rng.normal(0, 1, 200)
    gam = GAM()
    gam.fit(xs, ys, k=10)
    preds = gam.predict(xs)
    assert preds.shape == (200,)
    assert np.isfinite(preds).all()


def test_int_dtype_coerced():
    xs = np.arange(50)
    ys = (xs * 2 + 1).astype(int)
    gam = GAM()
    gam.fit(xs, ys, k=8)
    preds = gam.predict(xs)
    np.testing.assert_allclose(preds, ys.astype(float), atol=1e-6)


def test_non_contiguous_x_coerced():
    """Strided slices are converted to contiguous before the FFI hop."""
    rng = np.random.default_rng(1)
    full = rng.uniform(0, 1, (200, 4))
    xs = full[::2, ::2]  # non-contiguous view
    ys = xs[:, 0] + 0.1 * rng.normal(0, 1, xs.shape[0])
    gam = GAM()
    gam.fit(xs, ys, k=6)
    preds = gam.predict(xs)
    assert preds.shape == (xs.shape[0],)


# ---------------------------------------------------------------------- #
# k coercion                                                             #
# ---------------------------------------------------------------------- #


def test_scalar_k_broadcast_to_per_column():
    rng = np.random.default_rng(2)
    n = 200
    X = rng.uniform(0, 1, (n, 2))
    y = X[:, 0] + X[:, 1]
    gam = GAM()
    gam.fit(X, y, k=8)  # broadcasts to [8, 8]
    assert gam.predict(X).shape == (n,)


def test_list_k_passes_through():
    rng = np.random.default_rng(3)
    n = 200
    X = rng.uniform(0, 1, (n, 2))
    y = X[:, 0] + X[:, 1]
    gam = GAM()
    gam.fit(X, y, k=[6, 10])
    assert gam.predict(X).shape == (n,)


def test_singleton_k_list_broadcasts():
    rng = np.random.default_rng(4)
    n = 200
    X = rng.uniform(0, 1, (n, 3))
    y = X.sum(axis=1)
    gam = GAM()
    gam.fit(X, y, k=[7])  # single-element list broadcast across 3 cols
    assert gam.predict(X).shape == (n,)


def test_wrong_length_k_raises():
    X = np.zeros((50, 3))
    y = np.zeros(50)
    gam = GAM()
    with pytest.raises(ValueError, match="length"):
        gam.fit(X, y, k=[4, 5])  # 2 != 3 columns


def test_non_iterable_k_raises():
    X = np.zeros((50, 1))
    y = np.zeros(50)
    gam = GAM()
    with pytest.raises(TypeError, match="int or an iterable"):
        gam.fit(X, y, k=1.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# x shape validation                                                     #
# ---------------------------------------------------------------------- #


def test_3d_x_raises():
    X = np.zeros((10, 5, 2))
    y = np.zeros(10)
    gam = GAM()
    with pytest.raises(ValueError, match="1-D or 2-D"):
        gam.fit(X, y, k=4)


# ---------------------------------------------------------------------- #
# predict / evaluate_lpmatrix accept 1-D too                              #
# ---------------------------------------------------------------------- #


def test_predict_accepts_1d():
    xs = np.linspace(0, 1, 100)
    ys = np.sin(2 * np.pi * xs)
    gam = GAM()
    gam.fit(xs, ys, k=6)
    new_xs = np.linspace(0.1, 0.9, 20)
    assert gam.predict(new_xs).shape == (20,)


def test_evaluate_lpmatrix_accepts_1d():
    xs = np.linspace(0, 1, 100)
    ys = np.sin(2 * np.pi * xs)
    gam = GAM()
    gam.fit(xs, ys, k=6)
    lp = gam.evaluate_lpmatrix(xs)
    assert lp.shape[0] == 100


# ---------------------------------------------------------------------- #
# Forwarding via __getattr__                                              #
# ---------------------------------------------------------------------- #


def test_native_getters_forwarded():
    """Non-overridden methods (getters, family params) reach the native binding."""
    xs = np.linspace(0, 1, 100)
    ys = np.sin(2 * np.pi * xs)
    gam = GAM()
    gam.fit(xs, ys, k=6)
    # These are native methods we didn't override.
    coef = gam.get_coefficients()
    lam = gam.get_lambda()
    fam = gam.get_family()
    assert coef.shape == (6,)  # intercept + (k-1) basis fns
    assert isinstance(lam, float)
    assert fam == "gaussian"


def test_repr_smoke():
    gam = GAM()
    r = repr(gam)
    assert "gaussian" in r
    assert "identity" in r


# ---------------------------------------------------------------------- #
# Back-compat: the strict form still works                                #
# ---------------------------------------------------------------------- #


def test_strict_2d_float64_list_still_works():
    """The pre-0.12.1 strict form must still produce identical results."""
    rng = np.random.default_rng(5)
    xs = rng.uniform(0, 1, 200).reshape(-1, 1).astype(np.float64)
    ys = np.sin(2 * np.pi * xs.ravel()).astype(np.float64)
    gam = GAM()
    gam.fit(xs, ys, k=[10], method="REML")
    a = gam.predict(xs)

    # Loose form: 1-D xs, scalar k.
    xs_loose = xs.ravel()
    gam2 = GAM()
    gam2.fit(xs_loose, ys, k=10)
    b = gam2.predict(xs_loose)

    np.testing.assert_allclose(a, b, rtol=1e-12)
