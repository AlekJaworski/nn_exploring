"""Extract qgam fixed-state Fellner-Schall update terms (ladder step 7).

This is a diagnostic contract, not the qgam optimizer implementation. qgam/mgcv
uses LAML/Newton for smoothing-parameter optimization, while Rust's current ELF
path uses Fellner-Schall. To make the remaining lambda/sp gap debuggable, this
script freezes the R fixed-sp ELF fit state and computes the FS-style terms:

    rank_i, tr(A^-1 S_i), beta' S_i beta, numerator, log step, lambda_new

The same matrices are then passed through Rust's `fellner_schall_step_terms`
hook, so any disagreement is a bug in the shared update algebra rather than a
basis/penalty-coordinate issue.

Writes to: test_data/qgam_fs_step_terms_contract.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from mgcv_rust import fellner_schall_step_terms

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
REFERENCE_DIR = FIXTURE_DIR / "mgcv_rust_parity"
OUT = ROOT / "test_data" / "qgam_fs_step_terms_contract.json"

TARGET = "sale_to_list_price_ratio"
QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]
TAU = 0.95
K = 5
LSIG = -4.9598429613200636
ERR = 0.038489416415544007
VAR_HAT = 0.0020164420863363727
FINAL_SP_R = [
    1004340.6643248051,
    607622.6891643470,
    421.87326626306725,
    737690.6701503786,
]


def _co_from_err(err: float, var_hat: float) -> float:
    return err * math.sqrt(2 * math.pi * var_hat) / (2 * math.log(2))


def _full_penalty(block: np.ndarray, first_para_1based: int, p: int) -> np.ndarray:
    first = int(first_para_1based) - 1
    k = int(block.shape[0])
    out = np.zeros((p, p), dtype=float)
    out[first : first + k, first : first + k] = np.asarray(block, dtype=float)
    return out


def _fs_terms_python(
    penalty_blocks: list[np.ndarray],
    ranks: np.ndarray,
    lambdas: np.ndarray,
    a_inv: np.ndarray,
    beta: np.ndarray,
    phi: float = 1.0,
    log_step_clamp: float = 3.0,
    lambda_bounds: tuple[float, float] = (1e-9, 1e7),
) -> list[dict[str, float]]:
    terms: list[dict[str, float]] = []
    tiny = 1e-10
    for block, rank, lam in zip(penalty_blocks, ranks, lambdas):
        trace_a_inv_s = float(np.sum(a_inv * block.T))
        beta_s_beta = float(beta @ block @ beta)
        beta_s_beta_safe = max(beta_s_beta, tiny)
        numerator_raw = float(rank / max(lam, 1e-20) - trace_a_inv_s)
        numerator = max(numerator_raw, tiny)
        log_ratio_raw = float(np.log(phi * numerator / beta_s_beta_safe))
        log_ratio = float(np.clip(log_ratio_raw, -log_step_clamp, log_step_clamp))
        lambda_new = float(
            np.exp(
                np.clip(
                    np.log(lam) + log_ratio,
                    np.log(lambda_bounds[0]),
                    np.log(lambda_bounds[1]),
                )
            )
        )
        terms.append(
            {
                "lambda_old": float(lam),
                "lambda_new": lambda_new,
                "rank": float(rank),
                "trace_a_inv_s": trace_a_inv_s,
                "beta_s_beta": beta_s_beta,
                "numerator_raw": numerator_raw,
                "numerator": float(numerator),
                "log_ratio_raw": log_ratio_raw,
                "log_ratio": log_ratio,
            }
        )
    return terms


def _max_abs_term_delta(a: list[dict[str, float]], b: list[dict[str, float]]) -> float:
    keys = [
        "lambda_old",
        "lambda_new",
        "rank",
        "trace_a_inv_s",
        "beta_s_beta",
        "numerator_raw",
        "numerator",
        "log_ratio_raw",
        "log_ratio",
    ]
    return float(max(abs(float(x[k]) - float(y[k])) for x, y in zip(a, b) for k in keys))


def main() -> None:
    importr("mgcv")
    importr("qgam")

    df = pd.read_parquet(FIXTURE_DIR / "entire_dataset_train.parquet")
    mean_ref = pd.read_parquet(REFERENCE_DIR / "mean_gam_entire_dataset.parquet")
    df = df.copy()
    df["resid"] = df[TARGET].to_numpy() - mean_ref["pred_prod"].to_numpy()

    co = _co_from_err(ERR, VAR_HAT)
    formula_str = "resid ~ " + " + ".join(f"s({c}, k={K}, bs='cr')" for c in QGAM_SMOOTHS)

    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
    ro.globalenv["r_df"] = r_df

    r_code = f"""
    library(qgam)
    library(mgcv)
    fam <- elf(qu={TAU}, co={co:.17g}, theta={LSIG:.17g}, link="identity")
    fit <- gam(
        as.formula("{formula_str}"),
        family=fam,
        sp=c({",".join(str(v) for v in FINAL_SP_R)}),
        data=r_df,
        method="REML"
    )
    X <- predict(fit, type="lpmatrix")
    beta <- as.numeric(coef(fit))
    mu <- as.numeric(fitted(fit))
    dd <- fam$Dd(r_df$resid, mu, {LSIG:.17g}, rep(1, length(mu)), level=0)
    w <- as.numeric(dd$Dmu2) / 2
    smooths <- lapply(fit$smooth, function(sm) list(
        S=sm$S[[1]],
        first=sm$first.para,
        last=sm$last.para,
        rank=sm$rank[1]
    ))
    list(X=X, beta=beta, mu=mu, w=w, smooths=smooths)
    """
    result_r = ro.r(r_code)

    x_r = np.asarray(result_r.rx2("X"), dtype=float)
    beta_r = np.asarray(result_r.rx2("beta"), dtype=float)
    w_r = np.asarray(result_r.rx2("w"), dtype=float)
    smooths = result_r.rx2("smooths")

    p = x_r.shape[1]
    penalty_blocks: list[np.ndarray] = []
    ranks: list[float] = []
    for sm in smooths:
        block = np.asarray(sm.rx2("S"), dtype=float)
        first = int(sm.rx2("first")[0])
        penalty_blocks.append(_full_penalty(block, first, p))
        ranks.append(float(sm.rx2("rank")[0]))

    lambdas = np.asarray(FINAL_SP_R, dtype=float)
    ranks_arr = np.asarray(ranks, dtype=float)
    s_total = np.zeros((p, p), dtype=float)
    for lam, block in zip(lambdas, penalty_blocks):
        s_total += lam * block
    a = x_r.T @ (w_r[:, None] * x_r) + s_total
    a_inv = np.linalg.inv(a)

    terms_py = _fs_terms_python(penalty_blocks, ranks_arr, lambdas, a_inv, beta_r)
    terms_rust = fellner_schall_step_terms(penalty_blocks, ranks_arr, lambdas, a_inv, beta_r)
    terms_rust_json = [{k: float(v) for k, v in row.items()} for row in terms_rust]
    max_delta = _max_abs_term_delta(terms_py, terms_rust_json)

    payload = {
        "source": "qgam fixed-state Fellner-Schall term diagnostic (ladder step 7)",
        "formula": formula_str,
        "n": int(len(df)),
        "tau": TAU,
        "lsig": LSIG,
        "sigma": math.exp(LSIG),
        "co": co,
        "final_sp": FINAL_SP_R,
        "r_state": {
            "coef_head": beta_r[: min(8, len(beta_r))].tolist(),
            "weight_summary": np.quantile(w_r, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
            "penalty_ranks": ranks,
        },
        "terms_python": terms_py,
        "terms_rust": terms_rust_json,
        "max_abs_python_vs_rust": max_delta,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"max_abs_python_vs_rust = {max_delta:.3e}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
