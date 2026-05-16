"""Ergonomic facade over the native Rust :class:`GAM` binding.

The low-level Rust class (``_NativeGAM``) takes its arguments very
literally: ``x`` must be a 2-D ``float64`` :class:`numpy.ndarray`, ``y``
must be 1-D ``float64``, and ``k`` must be a ``list[int]``. That makes
sense at the FFI boundary but trips notebook-style usage like
``gam.fit(xs, ys, k=10)`` with ``xs = np.linspace(...)`` (a 1-D array).

This module wraps the native class in :class:`GAM`, a thin
composition facade that coerces ``x``/``y``/``k`` to the expected
shapes and dtypes before delegating to the binding. All other native
methods (getters, family params, etc.) are forwarded transparently
via ``__getattr__``.

Most users should still prefer :class:`mgcv_rust.Gam` — the high-level
wrapper with DataFrame support, subset views, predict_ci, etc. The
low-level :class:`GAM` is for callers who want direct control over
the Rust core; this facade just removes the dtype-and-shape papercut.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import numpy as np

from .mgcv_rust import GAM as _NativeGAM


def _ensure_2d_float64(x: Any) -> np.ndarray:
    """Coerce ``x`` to a 2-D ``float64`` ``ndarray``.

    - 1-D inputs are reshaped to ``(n, 1)`` — the common single-predictor
      notebook case.
    - Non-``float64`` dtypes are cast to ``float64``.
    - Non-contiguous inputs are copied to a contiguous layout (the
      Rust binding requires this).
    """
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"x must be 1-D or 2-D; got ndim={arr.ndim} (shape={arr.shape})"
        )
    return arr


def _ensure_1d_float64(y: Any) -> np.ndarray:
    """Coerce ``y`` to a 1-D ``float64`` ``ndarray``."""
    arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _ensure_k_list(k: Any, expected_len: int) -> list[int]:
    """Coerce ``k`` to a ``list[int]`` of length ``expected_len``.

    Scalars are broadcast: ``k=10`` on a 3-column ``x`` becomes
    ``[10, 10, 10]``. Lists / tuples / ndarrays are accepted as-is
    (cast to ``int``). Length must either equal ``expected_len`` or be
    1 (broadcast).
    """
    if isinstance(k, (int, np.integer)):
        return [int(k)] * expected_len
    try:
        ks = [int(v) for v in k]
    except TypeError as exc:
        raise TypeError(
            f"k must be an int or an iterable of ints; got {type(k).__name__}"
        ) from exc
    if len(ks) == 1 and expected_len != 1:
        return ks * expected_len
    if len(ks) != expected_len:
        raise ValueError(
            f"k has length {len(ks)} but x has {expected_len} columns; "
            "pass either one int per column or a single int to broadcast"
        )
    return ks


class GAM:
    """Ergonomic facade over the native Rust GAM binding.

    Drop-in compatible with the previous ``mgcv_rust.GAM`` (which was
    the native binding directly) — same constructor args, same methods.
    The difference is that ``fit`` / ``predict`` / ``evaluate_lpmatrix``
    now accept 1-D arrays, integer ``k``, and non-float dtypes; they're
    coerced before being handed to the Rust core.

    All other methods are forwarded transparently to the underlying
    native instance via ``__getattr__``.
    """

    __slots__ = ("_native",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "_native", _NativeGAM(*args, **kwargs))

    # --- Methods with input coercion ----------------------------------- #

    def fit(
        self,
        x: Any,
        y: Any,
        k: Union[int, Iterable[int]],
        method: str = "REML",
        bs: Optional[str] = None,
        max_iter: Optional[int] = None,
        use_edf: Optional[bool] = None,
        algorithm: Optional[str] = None,
        pc_values: Any = None,
        bs_list: Any = None,
        weights: Any = None,
    ) -> Any:
        x_arr = _ensure_2d_float64(x)
        y_arr = _ensure_1d_float64(y)
        ks = _ensure_k_list(k, x_arr.shape[1])
        # Per-row prior weights (mgcv's `weights=`). Coerce to a
        # contiguous float64 1-D array so the Rust binding sees the
        # exact layout it expects. None ⇒ unweighted fit.
        w_arr = (
            np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
            if weights is not None
            else None
        )
        if algorithm is not None:
            return self._native.fit_auto_optimized(
                x_arr, y_arr, ks, method,
                bs=bs, max_iter=max_iter, use_edf=use_edf,
                algorithm=algorithm, pc_values=pc_values, bs_list=bs_list,
                weights=w_arr,
            )
        return self._native.fit(
            x_arr, y_arr, ks, method,
            bs=bs, max_iter=max_iter, use_edf=use_edf,
            pc_values=pc_values, bs_list=bs_list, weights=w_arr,
        )

    def fit_auto(
        self,
        x: Any,
        y: Any,
        k: Union[int, Iterable[int]],
        method: str = "REML",
        bs: Optional[str] = None,
        max_iter: Optional[int] = None,
    ) -> Any:
        x_arr = _ensure_2d_float64(x)
        y_arr = _ensure_1d_float64(y)
        ks = _ensure_k_list(k, x_arr.shape[1])
        return self._native.fit_auto(x_arr, y_arr, ks, method, bs=bs, max_iter=max_iter)

    def fit_manual(
        self,
        x: Any,
        y: Any,
        method: str = "REML",
        max_iter: Optional[int] = None,
    ) -> Any:
        x_arr = _ensure_2d_float64(x)
        y_arr = _ensure_1d_float64(y)
        return self._native.fit_manual(x_arr, y_arr, method, max_iter=max_iter)

    def fit_formula(
        self,
        x: Any,
        y: Any,
        formula: str,
        method: str = "REML",
        max_iter: Optional[int] = None,
    ) -> Any:
        x_arr = _ensure_2d_float64(x)
        y_arr = _ensure_1d_float64(y)
        return self._native.fit_formula(x_arr, y_arr, formula, method, max_iter=max_iter)

    def predict(self, x: Any) -> np.ndarray:
        x_arr = _ensure_2d_float64(x)
        return self._native.predict(x_arr)

    def evaluate_lpmatrix(self, x: Any) -> np.ndarray:
        x_arr = _ensure_2d_float64(x)
        return self._native.evaluate_lpmatrix(x_arr)

    def calibrate_quantile_intercept(self, y: Any) -> float:
        """Shift fitted quantile intercept so training coverage matches tau."""
        y_arr = _ensure_1d_float64(y)
        return float(self._native.calibrate_quantile_intercept(y_arr))

    def fit_quantile_fixed_sp(
        self,
        x: Any,
        y: Any,
        k: Union[int, Iterable[int]],
        sp: Any,
        tau: float,
        sigma: float,
        co: float,
        bs: Optional[str] = None,
        bs_list: Any = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> dict:
        """Run ELF PIRLS at fixed smoothing parameters (no REML outer loop).

        Used for staged parity contracts: verifies PIRLS convergence at
        R-derived (co, sigma, sp) before touching lambda optimization.
        """
        x_arr = _ensure_2d_float64(x)
        y_arr = _ensure_1d_float64(y)
        ks = _ensure_k_list(k, x_arr.shape[1])
        sp_arr = list(float(v) for v in sp)
        return self._native.fit_quantile_fixed_sp(
            x_arr, y_arr, ks, sp_arr, float(tau), float(sigma), float(co),
            bs=bs, bs_list=bs_list,
            max_iter=max_iter, tol=tol,
        )

    # --- Forwarding ---------------------------------------------------- #

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, so every
        # non-overridden attribute / method on _NativeGAM is reachable.
        return getattr(self._native, name)

    def __repr__(self) -> str:
        return f"GAM(family={self._native.get_family()!r}, link={self._native.get_link()!r})"
