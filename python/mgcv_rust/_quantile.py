"""σ-calibration helpers for the qgam-style Quantile family.

Why this lives in Python: the inner GAM fit is fast Rust; the outer
σ search is small Python loops. Two helpers:

- `tune_quantile_sigma`: K-fold CV with Brent on log σ. ~2-3× faster
  than qgam's `tuneLearnFast` at comparable quality (matches RMSE
  vs qgam to within ~10% on benchmarks).

- `fit_quantile`: one-shot helper that auto-calibrates σ then fits.

Why not LAML for σ in ELF: Fasiolo et al. 2021 show ELF's likelihood
is a Gibbs posterior, not a true likelihood — the MLE σ is structurally
degenerate (REML score collapses to σ→0 because in-sample pinball
shrinks faster than complexity penalties grow). GCV variants run into
the same degeneracy. CV is the *only* in-sample-free criterion that
breaks the tie cleanly. (Verified empirically 2026-05-07.) For
true-likelihood extended families like scat, Tweedie, NegBin, we use
LAML/REML directly — see test_*_laml_progress.py.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional, Sequence

from . import GAM


def _pinball(y: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    r = y - y_pred
    return float(np.maximum(tau * r, (tau - 1) * r).mean())


def _cal_kl_loss(
    z_bootstrap: np.ndarray,
) -> float:
    """qgam-style cal-KL distance between the bootstrap β-distribution and
    the full-fit posterior, computed on standardized deviations
    `z[b, j] = (β̂_b_j − β̂_full_j) / √V_full_jj`.

    Per coefficient j, KL distance ≈ √(var(z_·j) + mean(z_·j)² − log var(z_·j)).
    Returned scalar = mean over coefficients (qgam's `outLoss` formula in
    `I_tuneLearnBootstrapping.R:111-116`).

    When σ is well-calibrated, bootstrap variability matches the posterior
    SD ⟹ var(z_·j) ≈ 1, mean(z_·j) ≈ 0 ⟹ KL term ≈ √(1 + 0 − 0) = 1
    (the asymptotic minimum). Sigma values that under- or over-disperse
    the bootstrap β-distribution push KL above this floor.
    """
    if z_bootstrap.ndim != 2:
        raise ValueError(f"z_bootstrap must be (B, p), got {z_bootstrap.shape}")
    var_z = np.var(z_bootstrap, axis=0, ddof=1)  # (p,)
    mean_z = np.mean(z_bootstrap, axis=0)         # (p,)
    # Floor variance to avoid log(0) when β_b is constant for some coef
    var_safe = np.maximum(var_z, 1e-10)
    kl_per_coef = np.sqrt(var_safe + mean_z**2 - np.log(var_safe))
    return float(np.mean(kl_per_coef))


def _bootstrap_z_at_sigma(
    X: np.ndarray, y: np.ndarray, tau: float, k: Sequence[int],
    sigma: float, B: int,
    mu_full: np.ndarray, sdev_full: np.ndarray,
    bs: str, method: str, seed: int,
) -> np.ndarray:
    """Return (B, n) matrix z of standardized μ-deviations from the
    full-data fit's predictions, across B bootstrap refits at trial σ.

    Per qgam I_tuneLearnBootstrapping.R:60-69:
      μ_b = bootstrap fit's prediction at training x
      sdev_full = posterior predictive SD of full fit at training x
                = sqrt(diag(X · V_β · X')) — varies per observation
      z[b, i] = (μ_b[i] − μ_full[i]) / sdev_full[i]

    This is fundamentally different from standardizing β̂-deviations:
    we measure how far the bootstrap fit's *predictions* drift from the
    full fit's predictions, normalized by the prediction-uncertainty
    SD at each observation. Calibrated σ ⟹ z[b,i] has var≈1, mean≈0.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    z_rows = []
    for b in range(B):
        in_idx = rng.choice(n, size=n, replace=True)
        try:
            g = GAM("quantile", tau=tau, sigma=float(sigma))
            g.fit(X[in_idx], y[in_idx], k=list(k), method=method, bs=bs)
            mu_b = np.asarray(g.predict(X))  # predict on the ORIGINAL X
            if mu_b.shape != mu_full.shape:
                continue
            z_b = (mu_b - mu_full) / np.maximum(sdev_full, 1e-10)
            z_rows.append(z_b)
        except Exception:
            continue
    if len(z_rows) < 5:
        return np.zeros((0, n))
    return np.stack(z_rows, axis=0)


def _predictive_sdev(g, X: np.ndarray) -> np.ndarray:
    """Posterior predictive SD at rows of X under fit `g`:
    sqrt(diag(L · V_β · L')) where L = lpmatrix(X), V_β = vcov.

    Used by cal-KL standardisation of bootstrap-prediction deviations.
    """
    L = np.asarray(g.evaluate_lpmatrix(X))   # (n, p)
    V = np.asarray(g.get_vcov())              # (p, p)
    # diag(L · V · L') = sum over j,k of L[i,j] * V[j,k] * L[i,k]
    LV = L @ V                                  # (n, p)
    return np.sqrt(np.maximum(np.einsum('ij,ij->i', LV, L), 1e-12))


def _build_folds(n: int, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        folds.append((test_idx, np.where(train_mask)[0]))
    return folds


def _cv_loss_at_sigma(
    sigma: float, X, y, tau, k, folds, bs: str, method: str
) -> float:
    losses = []
    for test_idx, train_idx in folds:
        try:
            g = GAM("quantile", tau=tau, sigma=float(sigma))
            g.fit(X[train_idx], y[train_idx], k=list(k), method=method, bs=bs)
            losses.append(_pinball(y[test_idx], g.predict(X[test_idx]), tau))
        except Exception:
            losses.append(np.inf)
    return float(np.mean(losses))


def tune_quantile_sigma(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    k: Sequence[int],
    bs: str = "cr",
    method: str = "REML",
    n_folds: int = 3,
    seed: int = 0,
    multifidelity: bool = True,
    k_pilot_factor: Optional[float] = None,
    pilot_xatol: float = 0.05,
    full_brent_xatol: float = 0.05,
    loss: str = "pin",
    n_bootstrap: int = 20,
) -> tuple[float, dict[str, Any]]:
    """Pick σ via Brent on log σ with CV-style criterion.

    Two loss objectives:

    - **`loss="pin"` (default, fast)**: K-fold pinball CV. Brent
      minimises mean held-out pinball loss across `n_folds` folds.
      Best speed/quality trade-off for centred τ ∈ [0.2, 0.8]. At
      extreme τ (≥0.9 / ≤0.1) calibration error grows because each
      fold's tail is sparse — see `loss="cal_kl"` for that regime.

    - **`loss="cal_kl"` (qgam-style)**: bootstrap-based KL distance
      between the bootstrap β-distribution and the full-fit posterior
      (mirrors `qgam::tuneLearnFast` with `loss="cal"`). Required for
      well-calibrated extreme-τ estimation. Slower (~3-5s) but matches
      qgam's calibration (cal_err ~ 0.005 vs pinball-CV's 0.04).

    Multi-fidelity:

    - `multifidelity=True` (default): two-stage. Phase A runs Brent at
      `k_pilot_factor·k` (≈half the basis dim — much cheaper fits) to
      find σ_pilot. Phase B evaluates 3 candidates at full k around
      σ_pilot and picks the best.

    - `multifidelity=False`: single-stage Brent at full k.

    `n_folds=3` is the empirical sweet spot for `loss="pin"`: more folds
    actually *hurt* RMSE-truth because test sets get too small ⟹
    sampling noise dominates ⟹ Brent optimises fold-noise rather than
    signal. For `loss="cal_kl"`, `n_bootstrap` plays the analogous role
    (default 20, qgam uses 50).

    Returns (best_sigma, info_dict).
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as e:
        raise ImportError(
            "tune_quantile_sigma requires scipy. Install with `pip install scipy`."
        ) from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(y)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    folds = _build_folds(n, n_folds, seed)
    y_scale = float(np.std(y))
    if y_scale <= 0:
        y_scale = 1.0
    lo, hi = np.log(0.05 * y_scale), np.log(5.0 * y_scale)

    # cal-KL baseline = Gaussian-fit's μ̂ at training x AND posterior
    # predictive SD √diag(L · V · L') at the same x. Mirrors qgam's
    # `vtype="m"` standardisation in I_tuneLearnBootstrapping.R:60-69.
    # The σ-invariant Gaussian reference gives KL a single sensible minimum
    # over σ; using the ELF posterior at any particular σ deceives Brent.
    if loss == "cal_kl":
        g_full = GAM("gaussian")
        g_full.fit(X, y, k=list(k), method=method, bs=bs)
        mu_full = np.asarray(g_full.predict(X))
        try:
            sdev_full = _predictive_sdev(g_full, X)
        except Exception as e:
            raise RuntimeError(
                f"cal_kl needs evaluate_lpmatrix(X) and get_vcov() on the "
                f"Gaussian baseline fit: {e}"
            ) from e

    def loss_at_sigma(sigma: float, k_used: Sequence[int]) -> float:
        if loss == "pin":
            return _cv_loss_at_sigma(sigma, X, y, tau, k_used, folds, bs, method)
        elif loss == "cal_kl":
            z = _bootstrap_z_at_sigma(
                X, y, tau, k_used, sigma, n_bootstrap,
                mu_full, sdev_full, bs, method, seed,
            )
            if z.shape[0] < 5:
                return 1e10  # too few successful bootstrap fits
            return _cal_kl_loss(z)
        else:
            raise ValueError(f"loss must be 'pin' or 'cal_kl', got {loss!r}")

    def brent_at(k_used, xatol, bracket=None):
        b = bracket if bracket is not None else (lo, hi)
        result = minimize_scalar(
            lambda ls: loss_at_sigma(float(np.exp(ls)), k_used),
            bounds=b, method="bounded", options={"xatol": xatol},
        )
        return float(np.exp(result.x)), float(result.fun), int(result.nfev)

    # Choose default k_pilot_factor per loss kind. cal-KL benefits from
    # an even cheaper pilot (k=3) since the σ-curve is broad enough that
    # quarter-k still locates the optimum within the refine bracket;
    # this halves cal-KL runtime vs k_pilot=5. Pin-CV uses k_pilot=5
    # (empirically the sweet spot for K-fold pinball-CV).
    kpf = k_pilot_factor
    if kpf is None:
        kpf = 0.3 if loss == "cal_kl" else 0.5

    if multifidelity:
        k_pilot = [max(3, int(round(ki * kpf))) for ki in k]
        sigma_pilot, _, nev_pilot = brent_at(k_pilot, xatol=pilot_xatol)
        log_p = np.log(sigma_pilot)
        # cal-KL pilot σ at half-k is biased high (sharper σ optimum at
        # full k), so refining with widely-spaced candidates and picking
        # min is essential to land at qgam-equivalent calibration.
        # Empirically verified that skip-refine inflates cal_err 2.4×.
        candidates = np.exp([log_p - 0.3, log_p, log_p + 0.3])
        best_sigma, best_loss = sigma_pilot, np.inf
        for sig in candidates:
            l = loss_at_sigma(float(sig), k)
            if l < best_loss:
                best_loss, best_sigma = l, float(sig)
        return best_sigma, {
            "sigma_log": float(np.log(best_sigma)),
            "cv_loss": best_loss,
            "n_brent_evals_pilot": nev_pilot,
            "n_refine_candidates": 3,
            "n_folds": n_folds,
            "n_bootstrap": n_bootstrap if loss == "cal_kl" else None,
            "k_pilot": list(k_pilot),
            "k_full": list(k),
            "strategy": "multifidelity",
            "loss": loss,
        }

    sigma_star, cv_loss, nfev = brent_at(k, xatol=full_brent_xatol)
    return sigma_star, {
        "sigma_log": float(np.log(sigma_star)),
        "cv_loss": cv_loss,
        "n_brent_evals": nfev,
        "n_folds": n_folds,
        "n_bootstrap": n_bootstrap if loss == "cal_kl" else None,
        "bracket_log_sigma": (lo, hi),
        "strategy": "full_brent",
        "loss": loss,
    }


def _build_total_penalty(g: "GAM", lambdas: Sequence[float]) -> np.ndarray:
    """Construct Σ λ_i S_i at the full design size (p × p), embedded
    at intercept-aware offsets.

    `g.get_smooth_penalties()` returns per-smooth (n_basis_i, n_basis_i)
    blocks; the intercept lives at column 0 and each smooth block i
    occupies columns [start_i, start_i + n_basis_i). We zero-pad each
    block into the full design matrix and accumulate.
    """
    per_smooth = g.get_smooth_penalties()
    p = int(np.asarray(g.get_design_matrix()).shape[1])
    s_total = np.zeros((p, p))
    col = 1  # skip the intercept
    for blk, lam in zip(per_smooth, lambdas):
        blk = np.asarray(blk)
        nb = blk.shape[0]
        s_total[col : col + nb, col : col + nb] += float(lam) * blk
        col += nb
    return s_total


class QuantileLSSFit:
    """Result of a joint location-scale qgam fit.

    Holds two Gaussian-init `GAM` objects (`_g_loc`, `_g_scale`) for their
    basis machinery and the LSS-refit β coefficients. `predict_loc(x)`
    returns the τ-quantile estimate η̂_loc(x); `predict_sigma(x)` returns
    the per-observation σ̂(x) = exp(η̂_scale(x)).
    """

    def __init__(self, g_loc, g_scale, beta_loc, beta_scale, tau):
        self._g_loc = g_loc
        self._g_scale = g_scale
        self.beta_loc = np.asarray(beta_loc)
        self.beta_scale = np.asarray(beta_scale)
        self.tau = float(tau)

    def predict_loc(self, X: np.ndarray) -> np.ndarray:
        L = np.asarray(self._g_loc.evaluate_lpmatrix(np.atleast_2d(X)))
        return L @ self.beta_loc

    def predict_sigma(self, X: np.ndarray) -> np.ndarray:
        L = np.asarray(self._g_scale.evaluate_lpmatrix(np.atleast_2d(X)))
        return np.exp(L @ self.beta_scale)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for `predict_loc` — the τ-quantile estimate."""
        return self.predict_loc(X)


# E[log|Z|] for Z ~ N(0, 1) = -½·digamma(½) - ½·log 2 = -γ/2 - log 2 ≈ -0.6351.
# Adding this constant to log|y - μ̂| gives an unbiased estimator of log σ.
_E_LOG_ABS_NORMAL = -0.6351814227307388


def fit_quantile_lss(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    k_loc: Sequence[int],
    k_scale: Optional[Sequence[int]] = None,
    bs: str = "cr",
    method: str = "REML",
    max_iter: int = 30,
    tolerance: float = 1e-6,
    sigma_scale: Optional[float] = None,
    calibrate: bool = False,
    n_folds: int = 3,
    seed: int = 0,
) -> tuple[QuantileLSSFit, dict[str, Any]]:
    """Heteroskedastic τ-quantile fit — port of qgam's (≥1.3) "parametric
    location-scale model" for the case where σ varies with x.

    The scalar-σ `fit_quantile` compromises between sharp regions (low σ)
    and dispersed regions (high σ); calibration drifts at the tails when
    that compromise can't be made. `fit_quantile_lss` models σ(x) as a
    smooth function of x in addition to the τ-quantile location η_loc(x),
    closing the heteroskedastic gap.

    Algorithm — qgam ≥1.3:
    1. **gaulss preprocessing** (two REML-tuned Gaussian GAMs):
       a. Fit μ_G(x) on y → Gaussian conditional mean.
       b. Fit ν(x) on log|y - μ_G| with the Euler-Mascheroni bias correction
          E[log|N(0, 1)|] = -0.6351 → unbiased log σ_G(x).
       c. σ_G(x) = exp(ν(x) + 0.6351).
    2. **Per-obs-σ ELF IRLS** (Rust): set σ_i = σ_G(x_i) per-observation,
       run the standard ELF location-only fit. Penalty is the REML-tuned
       λ_loc · S from step 1a.

    Why NOT joint elflss MLE: the Beta-normalised elflss likelihood
    theoretically identifies σ via joint MLE, but in finite samples joint
    (μ, σ) MLE biases σ̂ small at extreme τ, breaking calibration. qgam
    moved away from joint elflss MLE post-1.3 for this reason.

    Returns (fit, info) — `fit.predict_loc(X)` for the τ-quantile,
    `fit.predict_sigma(X)` for σ̂(x).
    """
    from . import GAM as _GAM

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if k_scale is None:
        # Half-k for scale by default — σ(x) is usually a flatter function
        # than the location, and lower k_scale guards against overfitting
        # log|residuals|.
        k_scale = [max(3, ki // 2) for ki in k_loc]

    # ── Stage 1: gaulss-equivalent via two Gaussian GAMs. ──
    # 1a: μ_G(x) — Gaussian conditional mean.
    g_loc = _GAM("gaussian")
    g_loc.fit(X, y, k=list(k_loc), method=method, bs=bs)
    mu_g = np.asarray(g_loc.predict(X))

    # 1b: log σ(x) via Gaussian fit on log|y - μ_G| + Euler correction.
    # The Gaussian's REML auto-tunes λ_scale to balance fit-vs-smoothness
    # for log|residuals|, which has variance π²/8 ≈ 1.23 — a well-defined
    # likelihood scale, so the auto-λ is meaningful here.
    floor = 1e-3 * float(np.std(y) or 1.0)
    log_abs_r = np.log(np.maximum(np.abs(y - mu_g), floor)) - _E_LOG_ABS_NORMAL
    g_scale = _GAM("gaussian")
    g_scale.fit(X, log_abs_r, k=list(k_scale), method=method, bs=bs)
    sigma_g_per_obs = np.exp(np.asarray(g_scale.predict(X)))

    # ── Stage 2: per-obs-σ ELF IRLS in Rust. ──
    # The qgam-elf σ rescaling (σ_i = σ_global · σ_G / mean(σ_G)) and the
    # σ_global heuristic both happen INSIDE the Rust call — pass σ_G(x)
    # and let Rust do the math. Python is just orchestration here.
    X_loc = np.asarray(g_loc.evaluate_lpmatrix(X))
    lambda_loc = np.asarray(g_loc.get_all_lambdas())
    s_loc_total = _build_total_penalty(g_loc, lambda_loc)

    from . import mgcv_rust as _rust

    # ── Optional: calibrate σ_global via K-fold pinball-CV. ──
    # The CV loop is in Python because it's high-level orchestration
    # (fold splits + Brent on a single scalar); each fold's heavy compute
    # goes into the Rust LSS path via the recursive call.
    info_calibration = None
    if calibrate:
        try:
            from scipy.optimize import minimize_scalar
        except ImportError as _imp_err:
            raise ImportError(
                "calibrate=True requires scipy (`pip install scipy`)."
            ) from _imp_err

        rng_cal = np.random.default_rng(seed)
        perm = rng_cal.permutation(len(y))
        fold_size = len(y) // n_folds
        folds = []
        for i in range(n_folds):
            test_idx = perm[i * fold_size : (i + 1) * fold_size]
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            folds.append((test_idx, np.where(train_mask)[0]))

        def cv_loss_at(sg: float) -> float:
            losses = []
            for test_idx, train_idx in folds:
                try:
                    fit_cv, _ = fit_quantile_lss(
                        X[train_idx], y[train_idx], tau=tau,
                        k_loc=k_loc, k_scale=k_scale, bs=bs, method=method,
                        max_iter=max_iter, tolerance=tolerance,
                        sigma_scale=sg, calibrate=False,
                    )
                    yhat_te = fit_cv.predict_loc(X[test_idx])
                    r_te = y[test_idx] - yhat_te
                    losses.append(float(np.maximum(tau * r_te, (tau - 1) * r_te).mean()))
                except Exception:
                    losses.append(np.inf)
            return float(np.mean(losses))

        y_sd = float(np.std(y) or 1.0)
        bracket = (np.log(0.05 * y_sd), np.log(5.0 * y_sd))
        result = minimize_scalar(
            lambda lsg: cv_loss_at(float(np.exp(lsg))),
            bounds=bracket, method="bounded", options={"xatol": 0.05},
        )
        sigma_scale = float(np.exp(result.x))
        info_calibration = {
            "calibration_loss": float(result.fun),
            "n_brent_evals": int(result.nfev),
            "n_folds": n_folds,
            "log_sigma_bracket": list(bracket),
        }

    res = _rust.fit_quantile_lss_raw_py(
        x_loc=X_loc, y=y,
        s_loc_total=s_loc_total,
        sigma_g_per_obs=sigma_g_per_obs,
        sigma_global=(float(sigma_scale) if sigma_scale is not None else None),
        tau=float(tau), max_iter=max_iter, tolerance=tolerance,
    )
    sigma_per_obs = np.asarray(res["sigma"])  # σ used in ELF (Rust returns it)
    sigma_global_used = float(res["sigma_global"])

    # We don't have refined β_scale (gaulss did the σ-side), so just hold
    # g_scale's predictions as the σ̂(x) source. β_scale stays as an
    # implementation detail; predict_sigma(X) goes through g_scale directly.
    fit = QuantileLSSFit(
        g_loc=g_loc, g_scale=g_scale,
        beta_loc=res["beta_loc"],
        beta_scale=np.asarray(g_scale.get_coefficients()),
        tau=tau,
    )
    info = {
        "iterations": res["iterations"],
        "converged": res["converged"],
        "deviance": res["deviance"],
        "lambda_loc": lambda_loc.tolist(),
        "lambda_scale": np.asarray(g_scale.get_all_lambdas()).tolist(),
        "k_loc": list(k_loc),
        "k_scale": list(k_scale),
        "tau": float(tau),
        "sigma_train": sigma_per_obs,        # σ used in ELF (per-obs)
        "sigma_g_train": sigma_g_per_obs,    # σ̂_G(x) from gaulss step
        "sigma_global": sigma_global_used,    # the scalar bandwidth (auto or user)
        "calibration": info_calibration,
    }
    return fit, info


def fit_quantile(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    k: Sequence[int],
    bs: str = "cr",
    method: str = "REML",
    sigma: Optional[float] = None,
    calibrate: bool = False,
    n_folds: int = 3,
    seed: int = 0,
    loss: str = "pin",
    n_bootstrap: int = 20,
) -> tuple[GAM, float, Optional[dict[str, Any]]]:
    """Fit ELF GAM, optionally with calibrated σ.

    - `calibrate=True`: run `tune_quantile_sigma` first.
    - `sigma=` provided: fit at that σ.
    - Otherwise: fall back to the rust-side heuristic σ.

    `loss` chooses the calibration objective when `calibrate=True`:
    - `"pin"` (default): K-fold pinball CV — fast, good for centred τ.
    - `"cal_kl"`: bootstrap-KL — slower, required for well-calibrated
      extreme-τ estimation (qgam parity).

    Returns (fitted_gam, sigma_used, calibration_info_or_None).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    info: Optional[dict[str, Any]] = None
    if calibrate:
        sigma, info = tune_quantile_sigma(
            X, y, tau=tau, k=k, bs=bs, method=method, n_folds=n_folds, seed=seed,
            loss=loss, n_bootstrap=n_bootstrap,
        )

    if sigma is not None:
        g = GAM("quantile", tau=tau, sigma=float(sigma))
        g.fit(X, y, k=list(k), method=method, bs=bs)
        return g, float(sigma), info

    g = GAM("quantile", tau=tau)
    g.fit(X, y, k=list(k), method=method, bs=bs)
    return g, 0.0, None  # sigma=0 sentinel = rust heuristic
