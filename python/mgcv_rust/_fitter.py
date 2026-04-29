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

    def get_vcov(self) -> np.ndarray:
        """Posterior covariance matrix of β̂. Same shape as the design
        matrix's column count (intercept + per-smooth blocks). For
        Gaussian / Gamma this is `σ² · (X'WX + λS)⁻¹` with σ² profiled
        from the deviance; for binomial / poisson `σ² = 1`."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.asarray(self._native.get_vcov(), dtype=float)

    def evaluate_lpmatrix(self, X: ArrayLike) -> np.ndarray:
        """Design matrix `[1 | smooth_1 | ...]` at the given X. mgcv's
        `predict.gam(..., type="lpmatrix")`."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_arr, _ = _to_numpy_with_columns(X, self._effective_predictors)
        return np.asarray(self._native.evaluate_lpmatrix(X_arr), dtype=float)

    def get_posterior_samples(
        self,
        X: ArrayLike,
        predictor: Optional[str] = None,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample `n_samples` posterior curves of the linear predictor at
        the given X. Returns an `(n_x, n_samples)` array.

        Mirrors `r_fitting.GamFitter.get_posterior_samples` (which seeds
        with 42 by default). The `predictor` argument is currently
        unused — included for API compatibility; the lpmatrix is built
        from the FULL set of smooths and the resulting samples are the
        joint posterior over all of them.

        Sampling is from a multivariate normal centered at `coef` with
        covariance `vcov` — the standard Bayesian posterior for the
        REML/Laplace approximation (Wood 2011)."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        coef = self.get_coefficients()
        vcov = self.get_vcov()
        lp = self.evaluate_lpmatrix(X)
        rng = np.random.default_rng(seed)
        # Use `multivariate_normal` for parity with r_fitting.GamFitter
        # which calls np.random.multivariate_normal seeded with 42.
        coef_samples = rng.multivariate_normal(coef, vcov, n_samples)
        # lp: (n_x, p), coef_samples: (n_samples, p) — return (n_x, n_samples)
        return lp @ coef_samples.T

    def predict_ci(
        self,
        X: ArrayLike,
        alpha: float = 0.05,
        n_samples: int = 1000,
        predictor: Optional[str] = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise confidence interval for predictions at the given X.

        Returns `(lower, upper)` as 1-D arrays. Computed via posterior
        sampling (matches `r_fitting.GamFitter.predict_ci`): draws
        `n_samples` β-vectors from the multivariate normal posterior,
        evaluates the linear predictor at X, takes the alpha/2 and
        1-alpha/2 quantiles, then applies the inverse link.

        For non-canonical / non-identity links, the returned interval is
        on the response scale.
        """
        post = self.get_posterior_samples(
            X, predictor=predictor, n_samples=n_samples, seed=seed
        )
        lo, hi = np.quantile(post, [alpha / 2, 1 - alpha / 2], axis=1)
        # Apply inverse link on the response scale.
        link = (self.link or "identity").lower()
        if link in ("identity", ""):
            return lo, hi
        if link == "log":
            return np.exp(lo), np.exp(hi)
        if link == "logit":
            return 1.0 / (1.0 + np.exp(-lo)), 1.0 / (1.0 + np.exp(-hi))
        if link == "inverse":
            # Avoid division by zero — clamp to tiny epsilon (matches
            # what the Rust inverse_link does for Gamma).
            lo_safe = np.where(np.abs(lo) < 1e-10, np.sign(lo) * 1e-10 + 1e-15, lo)
            hi_safe = np.where(np.abs(hi) < 1e-10, np.sign(hi) * 1e-10 + 1e-15, hi)
            return 1.0 / lo_safe, 1.0 / hi_safe
        raise NotImplementedError(f"predict_ci does not yet support link={link!r}")

    def get_edf_df(self) -> Any:
        """Per-smooth EDF table for auto-k tuning. 🚧 Needs Rust EDF
        accessor (Ergo-3 followup)."""
        raise NotImplementedError(
            "get_edf_df needs per-smooth EDF from the Rust core."
        )

    def serialize(
        self,
        prediction_range: Optional[dict[str, dict[str, float]]] = None,
        n_points: int = 1000,
        range_multiplier: float = 1000.0,
    ) -> dict[str, Any]:
        """Serialize to the schema consumed by `GamPredictor` in the
        neighbourhoods repo. Mirrors `r_fitting.GamFitter.serialize`:
        builds a dense prediction grid spanning each predictor's range
        plus a far-extrapolation tail, evaluates the lpmatrix on that
        grid, and packages it with the coefficients and covariance.

        The downstream `GamPredictor` interpolates against this grid at
        prediction time, so the serialized dict is the only thing the
        consumer needs.
        """
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet.")
        if prediction_range is None:
            prediction_range = self.prediction_range or {}
        if not prediction_range:
            raise RuntimeError(
                "No prediction_range available — was the model fitted? "
                "Pass `prediction_range` explicitly to override."
            )

        # Build the prediction grid: per-predictor [low_extrap, ...n_points
        # linspaced over data range..., high_extrap]. The two tails sit far
        # outside the data range so downstream consumers can saturate
        # extrapolation predictions to those endpoints.
        predictors = self._effective_predictors or []
        data_for_df: dict[str, np.ndarray] = {}
        for p in predictors:
            rng = prediction_range[p]
            value_range = rng["max"] - rng["min"]
            lowest_val = rng["min"] - range_multiplier * value_range
            highest_val = rng["max"] + range_multiplier * value_range
            data_for_df[p] = np.concatenate(
                [
                    np.array([lowest_val]),
                    np.linspace(rng["min"], rng["max"], n_points),
                    np.array([highest_val]),
                ]
            )
        # Stack columns in predictor order.
        X_grid = np.column_stack([data_for_df[p] for p in predictors])
        lp_matrix = self.evaluate_lpmatrix(X_grid)

        # Per-term column index ranges (matches mgcv's `extract_term_indices`).
        # The native call returns a list of (name, first, last) tuples
        # with INCLUSIVE ends and 0-based indices against the full
        # design (intercept at column 0). The Rust core uses internal
        # names `x0, x1, ...` — map them back to the user-supplied
        # predictor names so consumers see the names they passed in.
        term_indices_raw = self._native.get_term_indices()  # type: ignore[union-attr]
        if len(term_indices_raw) != len(predictors):
            raise RuntimeError(
                f"Term count mismatch: native returned {len(term_indices_raw)} "
                f"smooths but wrapper has {len(predictors)} predictors"
            )
        predictors_info: dict[str, dict[str, int]] = {
            "constant": {"first_index": 0, "last_index": 0}
        }
        for user_name, (_native_name, first, last) in zip(
            predictors, term_indices_raw, strict=False
        ):
            predictors_info[user_name] = {"first_index": first, "last_index": last}

        coefficients = self.get_coefficients()
        cov_matrix = self.get_vcov()

        # `lp_feature_values`: prepend a constant 1 column to the grid so
        # downstream code sees `[1 | predictor_1 | predictor_2 | ...]`,
        # matching the lpmatrix's intercept layout.
        lp_feature_values = np.concatenate(
            [np.ones((X_grid.shape[0], 1)), X_grid], axis=1
        )

        return {
            "predictors_info": predictors_info,
            "lp_matrix": lp_matrix,
            "coefficients": coefficients,
            "cov_matrix": cov_matrix,
            "lp_feature_values": lp_feature_values,
            "predictors": list(predictors),
            "family": self.family,
            "link": self.link,
            "pc_map": dict(self.pc_map),
        }

    def __getitem__(self, predictors: Iterable[str]) -> "GAMFitter":
        """Marginal-effect view: `gam[["x0", "x2"]].predict(X)` returns
        predictions using only those smooth contributions. 📋 Auto-k
        followup."""
        raise NotImplementedError(
            "Marginal subset indexing is an Ergo followup."
        )
