"""`GamPredictor` — frozen, inference-only view of a fitted :class:`Gam`.

Use this in deployment / serving paths where you want:

- **Strict input validation** — column drift in production silently
  shifts predictions; the predictor refuses any X that doesn't match
  the fitted ``feature_names_in_``.
- **Round-trip verification** — :meth:`check_against` asserts the
  predictor reproduces a reference :class:`Gam`'s output exactly on a
  sample, catching serialization or mismatch bugs.
- **A frozen API surface** — no ``fit``, no constructor knobs to twiddle;
  same ``predict`` / ``predict_ci`` / ``predict_diff`` / ``__getitem__``
  as :class:`Gam`.

Construction:

    >>> gam = Gam(family="gaussian").fit(X_train, y_train)
    >>> predictor = GamPredictor(gam)
    >>> predictor.predict(X_serve)
    >>> predictor.check_against(gam, X_train[:50])  # raises if it diverges

Subset views work as on :class:`Gam`:

    >>> sub = predictor[["x0"]]
    >>> sub.predict(X)

Closes two TrueTracts bug classes structurally:

- ``fa6d420`` (column index drift on predict): every ``predict`` call
  goes through ``_to_numpy_with_columns(X, feature_names_in_)`` which
  reorders / errors on missing columns.
- ``f74bad9`` (x-grid clamping forces ``range_multiplier=1000``): the
  predictor delegates to :meth:`Gam.predict`, which recomputes the
  basis at the requested X via ``evaluate_lpmatrix``. There is no
  pre-computed grid and therefore nothing to clamp against.
"""

from __future__ import annotations

from typing import Any, Iterable, Union

import numpy as np

from ._fitter import ArrayLike, Gam, TermContributions


class GamPredictor:
    """Frozen, inference-only view of a fitted :class:`Gam`.

    Wraps a fitted :class:`Gam` via composition and exposes only the
    inference-time API. Mutating the predictor is disallowed
    (``__slots__`` + no ``fit``) — once built, the bound :class:`Gam`'s
    coefficients, vcov, and feature schema are the contract.
    """

    __slots__ = ("_gam",)

    def __init__(self, gam: Gam) -> None:
        if not isinstance(gam, Gam):
            raise TypeError(
                f"GamPredictor expects a Gam, got {type(gam).__name__}"
            )
        if gam._native is None:
            raise RuntimeError(
                "GamPredictor requires a fitted Gam — call .fit() first."
            )
        # Use object.__setattr__ to bypass __slots__ semantics during construction.
        object.__setattr__(self, "_gam", gam)

    # ------------------------- sklearn-style attrs ------------------------- #

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self._gam.feature_names_in_

    @property
    def n_features_in_(self) -> int:
        return self._gam.n_features_in_

    @property
    def coef_(self) -> np.ndarray:
        return self._gam.coef_

    @property
    def intercept_(self) -> float:
        return self._gam.intercept_

    @property
    def intercept_response_(self) -> float:
        return self._gam.intercept_response_

    @property
    def lambda_(self) -> np.ndarray:
        return self._gam.lambda_

    @property
    def vcov_(self) -> np.ndarray:
        return self._gam.vcov_

    @property
    def k_(self) -> np.ndarray:
        return self._gam.k_

    @property
    def bs_(self) -> np.ndarray:
        return self._gam.bs_

    @property
    def edf_(self) -> np.ndarray:
        return self._gam.edf_

    @property
    def family(self) -> str:
        return self._gam.family

    @property
    def link(self) -> str:
        return self._gam.link

    # ------------------------------ Predict ------------------------------- #

    def predict(
        self,
        X: ArrayLike,
        scale: str = "response",
        type: Any = None,
    ) -> Union[np.ndarray, TermContributions]:
        """Delegates to :meth:`Gam.predict`.

        Column drift is caught by :func:`_to_numpy_with_columns` inside
        :meth:`Gam.predict`, which projects the input to
        ``feature_names_in_`` order and raises if any expected column is
        missing.
        """
        return self._gam.predict(X, scale=scale, type=type)

    def predict_ci(
        self,
        X: ArrayLike,
        alpha: Any = None,
        n_samples: int = 1000,
        predictor: Any = None,
        seed: int = 42,
        *,
        level: float = 0.95,
        scale: str = "response",
    ) -> tuple:
        """Delegates to :meth:`Gam.predict_ci`."""
        return self._gam.predict_ci(
            X,
            alpha=alpha,
            n_samples=n_samples,
            predictor=predictor,
            seed=seed,
            level=level,
            scale=scale,
        )

    def predict_diff(
        self,
        from_X: ArrayLike,
        to_X: ArrayLike,
        level: Any = None,
        broadcast: str = "none",
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Delegates to :meth:`Gam.predict_diff`."""
        return self._gam.predict_diff(
            from_X,
            to_X,
            level=level,
            broadcast=broadcast,
            n_samples=n_samples,
            seed=seed,
        )

    # ----------------------------- Subset view ---------------------------- #

    def __getitem__(self, predictors: Union[str, Iterable[str]]) -> "GamPredictor":
        """Return a :class:`GamPredictor` over the subset of smooths.

        Same semantics as :meth:`Gam.__getitem__`: pass a single name or
        an iterable of names; ``"__constant__"`` selects the intercept.
        Unknown names raise ``KeyError``.
        """
        sub_gam = self._gam[predictors]
        return GamPredictor(sub_gam)

    # --------------------------- Round-trip check ------------------------- #

    def check_against(
        self,
        gam: Gam,
        X_sample: ArrayLike,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        """Assert that this predictor matches ``gam.predict`` on ``X_sample``.

        Use this at deployment time to catch:
        - The predictor was built from a different fit than expected.
        - The bound :class:`Gam`'s state has drifted between
          serialization and load.

        Args:
            gam: a :class:`Gam` whose output should match this predictor's.
            X_sample: a small batch of input rows.
            rtol / atol: passed to :func:`numpy.allclose`.

        Raises:
            AssertionError: if any prediction diverges beyond
                ``(rtol, atol)``. The error message includes the max
                absolute and relative gap so the call site can decide
                whether to fail-closed or warn.
        """
        ours = np.asarray(self.predict(X_sample), dtype=float)
        theirs = np.asarray(gam.predict(X_sample), dtype=float)
        if not np.allclose(ours, theirs, rtol=rtol, atol=atol):
            abs_err = float(np.max(np.abs(ours - theirs)))
            rel_err = float(np.max(np.abs((ours - theirs) / np.where(theirs == 0, 1, theirs))))
            raise AssertionError(
                f"GamPredictor predictions diverge from Gam: max abs err "
                f"{abs_err:.3e}, max rel err {rel_err:.3e} "
                f"(rtol={rtol}, atol={atol}). The predictor may have been "
                "built from a different fit, or the bound Gam's state has "
                "drifted."
            )

    # ------------------------------- Repr --------------------------------- #

    def __repr__(self) -> str:
        return (
            f"GamPredictor(family={self.family!r}, link={self.link!r}, "
            f"features={list(self.feature_names_in_)})"
        )
