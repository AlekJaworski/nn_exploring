"""`Gam` — ergonomics layer over the Rust core.

User-facing class: :class:`Gam`. Was historically named ``GAMFitter`` to be a
drop-in for ``r_fitting.GamFitter`` in the neighbourhoods repo; ``GAMFitter``
is now a deprecated alias that forwards to :class:`Gam`. Same constructor
signature, same ``fit(X, y)`` / ``predict(X)`` / ``predict_ci`` /
``get_posterior_samples`` / ``serialize`` methods, plus pandas / polars /
numpy input handling.

Status legend in this file:
- ✅  fully implemented and tested.
- 🚧  signature in place but raises NotImplementedError until the
      underlying Rust accessor lands.
- 📋  not yet wired (subset indexing).
"""

from __future__ import annotations

import copy
import math
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
# Gam — the user-facing class                                            #
# ---------------------------------------------------------------------- #


class Gam:
    """Ergonomic GAM wrapper. Drop-in for `r_fitting.GamFitter`.

    Was historically named ``GAMFitter``; the old name is preserved as a
    deprecated alias (see :class:`GAMFitter` at the bottom of this module).
    New code should use :class:`Gam`.

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
        predictor_basis_map: per-predictor basis-type overrides, e.g.
            ``{"cluster": "re", "days": "cr"}``. Supported values: ``"cr"``,
            ``"bs"``, ``"re"`` (random effects). Overrides the global
            ``bs="cr"`` default for the listed predictors.
        term_pc_mapping: per-predictor placeholder constant `pc` — for
            each `(predictor, value)` the smooth is shifted so that
            `f(value) = 0`. Used for missing-value sentinels and for
            constraints like "concessions = 0 → no adjustment". 🚧
            Currently passes through but the Rust core does NOT yet
            apply pc; tracked as Ergo followup.
        family: GLM family. One of
            `"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`, `"t-dist"`.
            ``"t-dist"`` is the scaled t-distribution (mgcv's ``scat`` family)
            with identity link and heavier-tailed residuals.
        link: link function. Defaults to the canonical link per family.
            For gamma we additionally support `"log"`.
        df: degrees of freedom for ``family="t-dist"``. If given (must be in
            [2, 100]), df is held fixed during fitting. If omitted or ``None``,
            df is profiled jointly with σ² each iteration.
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
        df: Optional[float] = None,
        tweedie_p: Optional[float] = None,
        negbin_theta: Optional[float] = None,
        term_k_mapping: Optional[dict[str, int]] = None,
        term_pc_mapping: Optional[dict[str, float]] = None,
        predictor_basis_map: Optional[dict[str, str]] = None,
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
        self.df = df  # degrees of freedom for t-dist family (None = profile)
        self.tweedie_p = tweedie_p  # power parameter for Tweedie family (None → default 1.5)
        self.negbin_theta = negbin_theta  # dispersion for NegBin family (None = profile via nb())
        self.method = method
        self.family = family
        self.link = link
        # Validate df for t-dist family at construction time (not just at fit)
        if df is not None and family == "t-dist":
            if df < 2.0:
                raise ValueError(f"t-dist df must be >= 2.0, got {df}")
            if df > 100.0:
                raise ValueError(f"t-dist df must be <= 100.0, got {df}. Use df ∈ [2, 100].")
        self.term_k_mapping: dict[str, int] = dict(term_k_mapping or {})
        self.term_pc_mapping: dict[str, float] = dict(term_pc_mapping or {})
        self.predictor_basis_map: dict[str, str] = dict(predictor_basis_map or {})
        self.consider_categorical = consider_categorical

        # Filled at fit time:
        self._native: Optional[_NativeGAM] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.pc_map: dict[str, float] = {}
        self.prediction_range: Optional[dict[str, dict[str, float]]] = None
        self._effective_predictors: Optional[list[str]] = None
        # Stores final per-term k after fit (including auto-k results):
        self._term_k: dict[str, int] = {}
        # Stores iteration count when auto-k loop ran:
        self._auto_k_iterations: int = 0

    # -------------------------- Fit / Predict ------------------------- #

    def fit(self, X: ArrayLike, y: Any) -> "Gam":
        """Fit the GAM. Mirrors `r_fitting.GamFitter.fit`.

        Accepts numpy / pandas / polars inputs. If a DataFrame is given,
        column names are matched against `self.predictors` (if set)
        before the fit.

        When `term_k_mapping` covers all predictors, a single fit is
        run (existing behavior). Otherwise, an iterative auto-k loop
        grows k for under-saturated terms until convergence.
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

        # Capture per-predictor data range — used by `serialize` to grow
        # the prediction grid.
        self.prediction_range = {
            p: {"min": float(np.nanmin(X_arr[:, i])), "max": float(np.nanmax(X_arr[:, i]))}
            for i, p in enumerate(self._effective_predictors)
        }
        self.X = X_arr
        self.y = y_arr
        self.pc_map = dict(self.term_pc_mapping)  # surface to consumers

        # Random-effect terms don't need k tuning (k = #levels, set by Rust).
        # Treat them as "covered" so we skip the auto-k loop for them.
        re_terms = {
            name for name in (self._effective_predictors or [])
            if self.predictor_basis_map.get(name) == "re"
        }
        all_covered = all(
            name in self.term_k_mapping or name in re_terms
            for name in self._effective_predictors
        )

        if all_covered:
            # Fixed-k path: single fit, no iteration.
            ks = self._resolve_ks(X_arr, self._effective_predictors, self.term_k_mapping)
            self._term_k = dict(zip(self._effective_predictors, ks))
            self._auto_k_iterations = 0
            self._single_fit(X_arr, y_arr, ks)
        else:
            # Auto-k path: iteratively refit growing k for uncovered terms.
            self._auto_fit_k(X_arr, y_arr)

        return self

    def _resolve_ks(
        self,
        X_arr: np.ndarray,
        predictors: list[str],
        k_mapping: dict[str, int],
    ) -> list[int]:
        """Compute the capped k vector for a given k_mapping.

        For each predictor uses k_mapping if present, else self.k_default.
        Caps at n_unique(x_j) - 1 per predictor column.

        Random-effect terms (``predictor_basis_map[name] == "re"``) use a
        placeholder k=1 — the Rust side ignores it and sets k = #unique
        levels from the training data.
        """
        ks: list[int] = []
        for name in predictors:
            if self.predictor_basis_map.get(name) == "re":
                # k is ignored for random effects; pass placeholder 1.
                ks.append(1)
                continue
            requested = k_mapping.get(name, self.k_default)
            col_idx = predictors.index(name)
            n_unique = int(np.unique(X_arr[:, col_idx]).size)
            cap = max(n_unique - 1, self.min_k)
            ks.append(max(self.min_k, min(requested, cap)))
        return ks

    def _make_native(self) -> _NativeGAM:
        """Construct a fresh native GAM for this fitter's family/link."""
        if self.family == "gamma" and self.link == "log":
            return _NativeGAM(family="gamma", link="log")
        if self.link in (None, "", "identity") and self.family == "gaussian":
            return _NativeGAM()
        if self.family == "t-dist":
            # Pass optional df kwarg — Rust side accepts df=None for profiling
            return _NativeGAM(family="t-dist", df=self.df)
        if self.family == "tweedie":
            # Pass optional p kwarg — Rust side defaults to 1.5 for fixed-p mode
            return _NativeGAM(family="tweedie", link="log", p=self.tweedie_p)
        if self.family in ("negbin", "negative.binomial"):
            # Fixed-θ NegBin: pass theta (defaults to 2.0 if not provided)
            return _NativeGAM(family="negbin", link="log", theta=self.negbin_theta or 2.0)
        if self.family == "nb":
            # Profile-θ NegBin: omit theta so the Rust side starts at θ=2.0
            return _NativeGAM(family="nb", link="log")
        return _NativeGAM(self.family, link=self.link)

    def _build_pc_values(self, predictors: list[str]) -> list[float | None] | None:
        """Build a positional list of pc values for the given predictors.

        Returns None if no pc anchoring is requested, otherwise a list of
        length len(predictors) where entry i is the pc anchor for predictor i
        (or None if that predictor has no pc constraint).
        """
        if not self.term_pc_mapping:
            return None
        pcs = [self.term_pc_mapping.get(name, None) for name in predictors]
        return pcs if any(p is not None for p in pcs) else None

    def _build_bs_list(self, predictors: list[str]) -> list[str | None] | None:
        """Build a positional list of per-term basis types.

        Returns None if no per-predictor overrides are requested, otherwise
        a list of length len(predictors) where entry i is the basis type
        string for predictor i (or None to use the global default).
        """
        if not self.predictor_basis_map:
            return None
        bs_vals = [self.predictor_basis_map.get(name, None) for name in predictors]
        return bs_vals if any(b is not None for b in bs_vals) else None

    def _single_fit(self, X_arr: np.ndarray, y_arr: np.ndarray, ks: list[int]) -> None:
        """Run one native fit with the given k vector (mgcv bs='cr', REML)."""
        self._native = self._make_native()
        pc_values = self._build_pc_values(self._effective_predictors or [])
        bs_list = self._build_bs_list(self._effective_predictors or [])
        self._native.fit(
            X_arr, y_arr, k=ks, method="REML", bs="cr", pc_values=pc_values, bs_list=bs_list
        )

    def _auto_fit_k(self, X_arr: np.ndarray, y_arr: np.ndarray) -> None:
        """Iteratively refit, growing k for terms whose EDF is near saturation.

        Fixed-k terms (those in term_k_mapping) are frozen. Uncovered terms
        start at k_default=4 and grow by ceil(k * knots_increase_ratio) each
        iteration where (k - 1) - edf < edf_cutoff.

        Stop when: no term grew, every term hit its cap, or 10 iterations.
        """
        predictors = self._effective_predictors

        # Pre-compute per-term caps from unique-value counts.
        caps: dict[str, int] = {}
        for i, name in enumerate(predictors):
            n_unique = int(np.unique(X_arr[:, i]).size)
            caps[name] = max(n_unique - 1, self.min_k)

        # Initial k for each term.
        current_k: dict[str, int] = {}
        for name in predictors:
            if name in self.term_k_mapping:
                requested = self.term_k_mapping[name]
                current_k[name] = max(self.min_k, min(requested, caps[name]))
            else:
                current_k[name] = max(self.min_k, min(self.k_default, caps[name]))

        for iteration in range(10):
            ks = [current_k[name] for name in predictors]
            self._single_fit(X_arr, y_arr, ks)

            edf_map = dict(self._native.get_edf_per_smooth())

            grew = False
            all_capped = True
            for name in predictors:
                if name in self.term_k_mapping:
                    # Fixed: never grow.
                    continue
                k_j = current_k[name]
                edf_j = edf_map.get(name, 0.0)
                # Grow when headroom (k-1) - edf < edf_cutoff (mirrors mgcv k.check).
                if (k_j - 1) - edf_j < self.edf_cutoff:
                    new_k = math.ceil(k_j * self.knots_increase_ratio)
                    capped_k = min(new_k, caps[name])
                    if capped_k > k_j:
                        current_k[name] = capped_k
                        grew = True
                if current_k[name] < caps[name]:
                    all_capped = False

            self._auto_k_iterations = iteration + 1
            if not grew or all_capped:
                break

        self._term_k = dict(current_k)

    # ---------------------- Subset-mask helper ----------------------- #

    def _apply_subset_mask(self, lp: np.ndarray) -> np.ndarray:
        """Zero lpmatrix columns that belong to non-selected smooths.

        Only called when `_subset_mask` is set (i.e. this instance is a
        subset view created by `__getitem__`).  The lpmatrix has shape
        `(n, p)` with column 0 as the intercept and the remaining columns
        split into per-smooth blocks according to `get_term_indices()`.

        Rules:
        - Column 0 (intercept): kept if ``"__constant__" in _subset_mask``,
          zeroed otherwise.
        - Smooth block ``[first, last]`` (inclusive, 1-based in the full
          design): kept if the smooth's user-facing name is in
          ``_subset_mask``, zeroed otherwise.

        Returns a copy with un-selected columns set to zero.
        """
        mask = self._subset_mask  # type: ignore[attr-defined]
        predictors = self._effective_predictors or []

        # get_term_indices() returns [(native_name, first, last), ...] in
        # predictor order with inclusive, 0-based indices.
        term_indices_raw = self._native.get_term_indices()  # type: ignore[union-attr]

        lp = lp.copy()

        # Intercept (column 0).
        if "__constant__" not in mask:
            lp[:, 0] = 0.0

        # Per-smooth blocks — map native order → user predictor names.
        for user_name, (_native_name, first, last) in zip(
            predictors, term_indices_raw, strict=False
        ):
            if user_name not in mask:
                lp[:, first : last + 1] = 0.0

        return lp

    # ----------------------------------------------------------------- #

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict at given X. Mirrors `r_fitting.GamFitter.predict`."""
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet — call .fit() first.")
        X_arr, _ = _to_numpy_with_columns(X, self._effective_predictors)

        subset_mask = getattr(self, "_subset_mask", None)
        if subset_mask is not None:
            lp = np.asarray(self._native.evaluate_lpmatrix(X_arr), dtype=float)
            lp = self._apply_subset_mask(lp)
            coef = np.asarray(self._native.get_coefficients(), dtype=float)
            return lp @ coef

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

        subset_mask = getattr(self, "_subset_mask", None)
        if subset_mask is not None:
            lp = self._apply_subset_mask(lp)

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
        """Per-smooth EDF table. Returns a pandas DataFrame with columns
        [predictor, k, edf, ratio] where ratio = edf / (k - 1).

        Mirrors `r_fitting.GamFitter.get_edf_df()`. pandas is lazy-imported
        so it remains an optional dependency.
        """
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet — call .fit() first.")
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "get_edf_df() requires pandas. Install it with: pip install pandas"
            ) from exc
        edf = dict(self._native.get_edf_per_smooth())
        rows = []
        for name, k in self._term_k.items():
            e = edf.get(name, float("nan"))
            rows.append({"predictor": name, "k": k, "edf": e, "ratio": e / max(k - 1, 1)})
        return pd.DataFrame(rows)

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

    def __getitem__(self, predictors: Union[str, Iterable[str]]) -> "Gam":
        """Return a subset view for marginal predictions.

        ``gam[["x0", "x2"]].predict(X)`` returns predictions using only
        the x0 and x2 smooth contributions (all other columns of the
        lpmatrix are zeroed).  The special token ``"__constant__"``
        selects the intercept column.

        Args:
            predictors: a single smooth name, or an iterable of smooth
                names.  Each name must match a predictor known to this
                fitted model, or be ``"__constant__"``.

        Returns:
            A shallow copy of this :class:`Gam` with ``_subset_mask``
            set.  The underlying Rust model is shared; only prediction
            behaviour changes.

        Raises:
            RuntimeError: if the model has not been fitted yet.
            KeyError: if any requested name is not a known smooth and is
                not ``"__constant__"``.
        """
        if self._native is None:
            raise RuntimeError("Model has not been fitted yet — call .fit() first.")

        if isinstance(predictors, str):
            requested: list[str] = [predictors]
        else:
            requested = list(predictors)

        known = set(self._effective_predictors or []) | {"__constant__"}
        for name in requested:
            if name not in known:
                raise KeyError(name)

        view = copy.copy(self)
        view._subset_mask = set(requested)
        return view


# ---------------------------------------------------------------------- #
# Discoverable sentinel for the intercept column in __getitem__          #
# ---------------------------------------------------------------------- #
# Lives on `Gam` so users write `gam[Gam.INTERCEPT]` instead of the magic
# string `"__constant__"`. The value is unchanged for back-compat — code
# that already passes `"__constant__"` keeps working.
Gam.INTERCEPT = "__constant__"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- #
# Deprecated alias                                                       #
# ---------------------------------------------------------------------- #


class GAMFitter(Gam):
    """Deprecated alias for :class:`Gam`. Will be removed in a future release.

    Emits :class:`DeprecationWarning` on instantiation. Use :class:`Gam`
    instead — same constructor, same methods.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn(
            "GAMFitter is deprecated; use mgcv_rust.Gam instead. "
            "GAMFitter will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
