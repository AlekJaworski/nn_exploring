"""GAMFitter — ergonomics layer over the Rust `GAM` core.

Designed as a drop-in replacement for `r_fitting.GamFitter` in the
neighbourhoods repo: same constructor signature, same `fit(X, y)` /
`predict(X)` / `predict_ci` / `get_posterior_samples` / `serialize`
methods, plus pandas / polars / numpy input handling.

Status legend in this file:
- ✅  fully implemented and tested.
- 🚧  signature in place but raises NotImplementedError until the
      underlying Rust accessor lands.
- 📋  not yet wired (subset indexing, auto-k tuning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

# Import the compiled Rust core. We import it directly (not as
# `mgcv_rust.mgcv_rust`) so this file works whether the package is
# installed in mixed-layout mode or as a flat compiled module.
from .mgcv_rust import GAM as _NativeGAM

# Optional dependencies — pandas is the typical input format in the
# neighbourhoods code. Polars also shows up there. We accept either.
try:  # pragma: no cover - import-time
    import pandas as _pd  # type: ignore
except ImportError:  # pragma: no cover - import-time
    _pd = None

try:  # pragma: no cover - import-time
    import polars as _pl  # type: ignore
except ImportError:  # pragma: no cover - import-time
    _pl = None


ArrayLike = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _to_numpy_with_columns(
    X: ArrayLike, expected_cols: Optional[Sequence[str]] = None
) -> tuple[np.ndarray, list[str]]:
    """Convert a numpy / pandas / polars input to a 2D numpy array, returning
    column names alongside.

    If `expected_cols` is provided, the output is reordered/projected to
    match (and a clear error is raised on missing columns).
    """
    if _pd is not None and isinstance(X, _pd.DataFrame):
        cols = list(X.columns)
        if expected_cols is not None:
            missing = [c for c in expected_cols if c not in cols]
            if missing:
                raise ValueError(
                    f"DataFrame is missing expected columns: {missing}. "
                    f"Got: {cols}"
                )
            X = X[list(expected_cols)]
            cols = list(expected_cols)
        arr = X.to_numpy()
        return np.asarray(arr, dtype=float), cols

    if _pl is not None and isinstance(X, _pl.DataFrame):
        cols = list(X.columns)
        if expected_cols is not None:
            missing = [c for c in expected_cols if c not in cols]
            if missing:
                raise ValueError(
                    f"polars DataFrame is missing expected columns: {missing}. "
                    f"Got: {cols}"
                )
            X = X.select(list(expected_cols))
            cols = list(expected_cols)
        arr = X.to_numpy()
        return np.asarray(arr, dtype=float), cols

    # numpy array (or ndarray-compatible)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if expected_cols is None:
        cols = [f"x{i}" for i in range(arr.shape[1])]
    else:
        if len(expected_cols) != arr.shape[1]:
            raise ValueError(
                f"X has {arr.shape[1]} columns but expected {len(expected_cols)} "
                f"({list(expected_cols)})"
            )
        cols = list(expected_cols)
    return arr, cols


def _to_1d_numpy(y: Any) -> np.ndarray:
    if _pd is not None and isinstance(y, _pd.Series):
        return np.asarray(y.values, dtype=float)
    if _pd is not None and isinstance(y, _pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y must be 1-D, got DataFrame with {y.shape[1]} cols")
        return np.asarray(y.iloc[:, 0].values, dtype=float)
    if _pl is not None and isinstance(y, _pl.Series):
        return np.asarray(y.to_numpy(), dtype=float)
    if _pl is not None and isinstance(y, _pl.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"y must be 1-D, got DataFrame with {y.shape[1]} cols")
        return np.asarray(y.to_numpy().ravel(), dtype=float)
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


# ---------------------------------------------------------------------- #
# GAMFitter — the user-facing class                                      #
# ---------------------------------------------------------------------- #


class GAMFitter:
    """Ergonomic GAM wrapper. Drop-in for `r_fitting.GamFitter`.

    Parameters mirror the neighbourhoods `GamFitter`. Unsupported
    parameters are accepted but ignored with a clear note in the
    relevant method (rather than the constructor) so calling code can
    migrate without per-arg edits.

    Args:
        predictors: ordered list of predictor names. If `None`, defaults
            to `x0, x1, ...` derived from the input shape at fit time.
        target: name of the response column (used only by `serialize`).
        k_default: default basis dimension `k` for any predictor not
            listed in `term_k_mapping`.
        term_k_mapping: per-predictor `k` overrides, e.g.
            `{"days_ago": 25, "quality": 12}`.
        term_pc_mapping: per-predictor placeholder constant `pc` — for
            each `(predictor, value)` the smooth is shifted so that
            `f(value) = 0`. Used for missing-value sentinels and for
            constraints like "concessions = 0 → no adjustment". 🚧
            Currently passes through but the Rust core does NOT yet
            apply pc; tracked as Ergo followup.
        family: GLM family. One of
            `"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`.
        link: link function. Defaults to the canonical link per family.
            For gamma we additionally support `"log"`. Other links and
            `family="t-dist"` are not yet wired.
        method: smoothing-parameter selection method. Accepted values
            `"REML"`, `"fREML"`. Both currently route to mgcv-exact
            REML; the byte-for-byte fREML path is a followup.
        consider_categorical: if True, columns with two or fewer unique
            values are dropped from the smooth specification. 📋 Not
            yet implemented; pass-through for API compatibility.
        min_k / edf_cutoff / knots_increase_ratio /
        min_points_to_save / max_points_to_save: auto-k tuning knobs
            used by `_determine_optimal_k`. 📋 Auto-k is a followup.
    """

    # -------------------------- Constructor -------------------------- #

    def __init__(
        self,
        predictors: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        min_k: int = 3,
        k_default: int = 4,
        edf_cutoff: int = 2,
        knots_increase_ratio: float = 1.5,
        min_points_to_save: int = 100,
        max_points_to_save: int = 1000,
        method: str = "REML",
        family: str = "gaussian",
        link: str = "identity",
        term_k_mapping: Optional[dict[str, int]] = None,
        term_pc_mapping: Optional[dict[str, float]] = None,
        consider_categorical: bool = False,
        **kwargs: Any,
    ) -> None:
        self.predictors: Optional[list[str]] = list(predictors) if predictors else None
        self.target: str = target or "y"
        self.min_k = min_k
        self.k_default = k_default
        self.edf_cutoff = edf_cutoff
        self.knots_increase_ratio = knots_increase_ratio
        self.min_points_to_save = min_points_to_save
        self.max_points_to_save = max_points_to_save
        self.method = method
        self.family = family
        self.link = link
        self.term_k_mapping: dict[str, int] = dict(term_k_mapping or {})
        self.term_pc_mapping: dict[str, float] = dict(term_pc_mapping or {})
        self.consider_categorical = consider_categorical

        if self.term_pc_mapping:
            # 🚧 Rust core does not yet apply pc-anchoring; flag it so
            # callers know predictions won't be pinned at the sentinel.
            # Still accepted to keep the API drop-in.
            self._pc_unsupported = list(self.term_pc_mapping.keys())
        else:
            self._pc_unsupported = []

        # Filled at fit time:
        self._native: Optional[_NativeGAM] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.pc_map: dict[str, float] = {}
        self.prediction_range: Optional[dict[str, dict[str, float]]] = None
        self._effective_predictors: Optional[list[str]] = None

    # -------------------------- Fit / Predict ------------------------- #

    def fit(self, X: ArrayLike, y: Any) -> "GAMFitter":
        """Fit the GAM. Mirrors `r_fitting.GamFitter.fit`.

        Accepts numpy / pandas / polars inputs. If a DataFrame is given,
        column names are matched against `self.predictors` (if set)
        before the fit.
        """
        X_arr, cols = _to_numpy_with_columns(X, self.predictors)
        y_arr = _to_1d_numpy(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} elements"
            )

        if self.predictors is None:
            self.predictors = cols
        self._effective_predictors = list(cols)

        # Per-predictor k vector aligned with the (effective) predictor
        # order. For predictors with very few unique values we cap k —
        # mgcv would error on `k > nunique`, but we pre-cap to behave
        # gracefully (matches r_fitting.GamFitter's defensive trim).
        ks: list[int] = []
        for name in self._effective_predictors:
            requested = self.term_k_mapping.get(name, self.k_default)
            n_unique = int(np.unique(X_arr[:, self._effective_predictors.index(name)]).size)
            ks.append(max(self.min_k, min(requested, max(n_unique - 1, self.min_k))))

        # Capture per-predictor data range — used by `serialize` to grow
        # the prediction grid.
        self.prediction_range = {
            p: {"min": float(np.nanmin(X_arr[:, i])), "max": float(np.nanmax(X_arr[:, i]))}
            for i, p in enumerate(self._effective_predictors)
        }
        self.X = X_arr
        self.y = y_arr
        self.pc_map = dict(self.term_pc_mapping)  # surface to consumers

        # Build the native GAM. Family + link route through the
        # canonical-link path for gaussian/binomial/poisson and accept
        # `link="log"` for gamma. Anything else raises clearly.
        if self.family == "gamma" and self.link == "log":
            self._native = _NativeGAM(family="gamma", link="log")
        elif self.link in (None, "", "identity") and self.family == "gaussian":
            self._native = _NativeGAM()
        else:
            self._native = _NativeGAM(self.family, link=self.link)

        # mgcv's bs="cr" is the production default — match
        # neighbourhoods. (Term-level bs override is a followup.)
        self._native.fit(
            X_arr,
            y_arr,
            k=ks,
            method="REML",  # fREML is still routed through REML internally
            bs="cr",
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict at given X. Mirrors `r_fitting.GamFitter.predict`."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet — call .fit() first.")
        X_arr, _ = _to_numpy_with_columns(X, self._effective_predictors)
        return np.asarray(self._native.predict(X_arr), dtype=float)

    # -------------------------- Diagnostics --------------------------- #

    def get_lambdas(self) -> np.ndarray:
        """Smoothing parameters λ, one per smooth term, in predictor order."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._native.get_all_lambdas(), dtype=float)

    def get_coefficients(self) -> np.ndarray:
        """Fitted β coefficients (full vector — intercept first, then
        per-smooth blocks in predictor order)."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._native.get_coefficients(), dtype=float)

    def get_design_matrix(self) -> np.ndarray:
        """Design matrix at the training x — `[1 | smooth_1 | smooth_2 | ...]`."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._native.get_design_matrix(), dtype=float)

    # ------------------------- Followup stubs ------------------------- #

    def predict_ci(
        self, X: ArrayLike, predictor: Optional[str] = None, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confidence interval for predictions. 🚧 Requires `vcov` from
        the Rust core (Ergo-3 followup)."""
        raise NotImplementedError(
            "predict_ci needs vcov access from the Rust core (Ergo-3 followup). "
            "Track in `mgcv_rust - Next Steps.md`."
        )

    def get_posterior_samples(
        self, X: ArrayLike, predictor: Optional[str] = None, n_samples: int = 1000
    ) -> np.ndarray:
        """Posterior samples of the predicted curve. 🚧 Same blocker as
        `predict_ci`."""
        raise NotImplementedError(
            "get_posterior_samples needs vcov + lpmatrix-at-x from the Rust core."
        )

    def get_edf_df(self) -> Any:
        """Per-smooth EDF table for auto-k tuning. 🚧 Needs Rust EDF
        accessor (Ergo-3 followup)."""
        raise NotImplementedError(
            "get_edf_df needs per-smooth EDF from the Rust core."
        )

    def serialize(self, prediction_range: Optional[dict] = None) -> dict[str, Any]:
        """Serialize to the schema consumed by `GamPredictor` in the
        neighbourhoods repo. 🚧 Stub — returns coefficients only until
        vcov + lpmatrix-at-x land."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "predictors": self._effective_predictors,
            "family": self.family,
            "link": self.link,
            "pc_map": self.pc_map,
            "coefficients": self.get_coefficients(),
            # The full schema also needs lp_matrix, cov_matrix,
            # predictors_info — Ergo-4 followup.
            "_incomplete": True,
        }

    def __getitem__(self, predictors: Iterable[str]) -> "GAMFitter":
        """Marginal-effect view: `gam[["x0", "x2"]].predict(X)` returns
        predictions using only those smooth contributions. 📋 Auto-k
        followup."""
        raise NotImplementedError(
            "Marginal subset indexing is an Ergo followup."
        )
