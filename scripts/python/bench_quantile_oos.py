"""OOS benchmark for qgam-style quantile fits.

Compares Rust quantile variants on real sale-price residual fixtures and
synthetic cases with known conditional quantiles. Public preset variants
(`fast_oos`, `quality_oos`) are included as aliases for their explicit option
forms so preset behavior can be benchmarked directly. qgam is included where an
existing OOS contract is available; live R/qgam calls are intentionally out of
scope so this remains repeatable in CI/dev shells.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mgcv_rust._quantile import fit_quantile, fit_quantile_lss


ROOT = Path(__file__).resolve().parents[2]
SALE_FIXTURE_DIR = ROOT / "data" / "sale_price_fixtures"
PARITY_DIR = SALE_FIXTURE_DIR / "mgcv_rust_parity"
QGAM_HOLDOUT_CONTRACT = ROOT / "test_data" / "qgam_holdout_pinball_contract.json"
QGAM_REAL_CONTRACTS = ROOT / "test_data" / "qgam_oos_real_contracts.json"

QGAM_SMOOTHS = [
    "days_in_current_price_regime",
    "cum_dom_before_current_regime",
    "price_change_pct_from_original",
    "monthly_index",
]


@dataclass
class Case:
    case_id: str
    case_kind: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    tau: float
    k: list[int]
    bs: str = "cr"
    true_quantile_test: np.ndarray | None = None
    qgam_pred_test: np.ndarray | None = None
    qgam_pinball: float | None = None
    qgam_coverage: float | None = None
    qgam_fit_time_s: float | None = None
    source_files: list[str] | None = None


def _pinball(y: np.ndarray, pred: np.ndarray, tau: float) -> float:
    r = y - pred
    return float(np.mean(np.maximum(tau * r, (tau - 1.0) * r)))


def _metrics(case: Case, pred: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=float)
    out: dict[str, Any] = {
        "pinball": _pinball(case.y_test, pred, case.tau),
        "coverage": float(np.mean(case.y_test < pred)),
        "coverage_error": float(np.mean(case.y_test < pred) - case.tau),
        "pred_finite": bool(np.all(np.isfinite(pred))),
    }
    out["abs_coverage_error"] = abs(out["coverage_error"])

    if case.true_quantile_test is not None:
        diff = pred - case.true_quantile_test
        out.update(
            {
                "rmse_true_quantile": float(np.sqrt(np.mean(diff * diff))),
                "mae_true_quantile": float(np.mean(np.abs(diff))),
                "max_abs_true_quantile": float(np.max(np.abs(diff))),
            }
        )
    if case.qgam_pred_test is not None:
        diff = pred - case.qgam_pred_test
        out.update(
            {
                "qgam_pinball": case.qgam_pinball,
                "qgam_fit_time_s": case.qgam_fit_time_s,
                "pinball_ratio_vs_qgam": out["pinball"] / case.qgam_pinball
                if case.qgam_pinball and case.qgam_pinball > 0.0
                else None,
                "qgam_coverage": case.qgam_coverage,
                "coverage_delta_vs_qgam": out["coverage"] - case.qgam_coverage
                if case.qgam_coverage is not None
                else None,
                "pred_rmse_vs_qgam": float(np.sqrt(np.mean(diff * diff))),
                "pred_mean_delta_vs_qgam": float(np.mean(diff)),
            }
        )
        if np.std(pred) > 0 and np.std(case.qgam_pred_test) > 0:
            out["pred_corr_vs_qgam"] = float(np.corrcoef(pred, case.qgam_pred_test)[0, 1])
    return out


def _base_row(case: Case, variant: str, seed: int) -> dict[str, Any]:
    return {
        "benchmark_version": "quantile_oos_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id,
        "case_kind": case.case_kind,
        "variant": variant,
        "tau": case.tau,
        "n_train": int(case.X_train.shape[0]),
        "n_test": int(case.X_test.shape[0]),
        "d": int(case.X_train.shape[1]),
        "k": case.k,
        "bs": case.bs,
        "seed": seed,
        "source_files": case.source_files or [],
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 20:
            return {
                "shape": list(value.shape),
                "mean": float(np.nanmean(value)),
                "std": float(np.nanstd(value)),
                "min": float(np.nanmin(value)),
                "max": float(np.nanmax(value)),
            }
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _fit_heuristic(case: Case, X: np.ndarray, y: np.ndarray):
    fit, sigma_used, info = fit_quantile(X, y, tau=case.tau, k=case.k, bs=case.bs)
    return fit, sigma_used, info


def _apply_intercept_shift(fit: Any, shift: float) -> float:
    fitted = np.asarray(fit.get_fitted_values(), dtype=float)
    pseudo_y = fitted + float(shift)
    return float(fit.calibrate_quantile_intercept(pseudo_y))


def _residual_quantile_shift(y: np.ndarray, pred: np.ndarray, tau: float) -> float:
    residuals = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.quantile(residuals, tau, method="lower"))


def _fit_heuristic_holdout_covcal(case: Case, seed: int) -> tuple[Any, float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = case.X_train.shape[0]
    idx = rng.permutation(n)
    n_fit = max(1, int(0.8 * n))
    fit_idx = np.sort(idx[:n_fit])
    cal_idx = np.sort(idx[n_fit:])

    cal_fit, _, _ = _fit_heuristic(case, case.X_train[fit_idx], case.y_train[fit_idx])
    cal_pred = np.asarray(cal_fit.predict(case.X_train[cal_idx]), dtype=float)
    shift = _residual_quantile_shift(case.y_train[cal_idx], cal_pred, case.tau)

    final_fit, sigma_used, info = _fit_heuristic(case, case.X_train, case.y_train)
    applied_shift = _apply_intercept_shift(final_fit, shift)
    return final_fit, sigma_used, {
        "coverage_calibration": "holdout",
        "n_extra_fits": 1,
        "n_fit": int(len(fit_idx)),
        "n_cal": int(len(cal_idx)),
        "shift": shift,
        "applied_shift": applied_shift,
        "base_info": info,
    }


def _fit_heuristic_kfold_covcal(case: Case, seed: int, n_folds: int) -> tuple[Any, float, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = case.X_train.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    oof_pred = np.empty(n, dtype=float)

    for fold in folds:
        fold = np.asarray(fold, dtype=int)
        train_idx = np.setdiff1d(np.arange(n), fold, assume_unique=False)
        fold_fit, _, _ = _fit_heuristic(case, case.X_train[train_idx], case.y_train[train_idx])
        oof_pred[fold] = np.asarray(fold_fit.predict(case.X_train[fold]), dtype=float)

    shift = _residual_quantile_shift(case.y_train, oof_pred, case.tau)
    final_fit, sigma_used, info = _fit_heuristic(case, case.X_train, case.y_train)
    applied_shift = _apply_intercept_shift(final_fit, shift)
    return final_fit, sigma_used, {
        "coverage_calibration": f"kfold_{n_folds}",
        "n_extra_fits": n_folds,
        "n_folds": n_folds,
        "shift": shift,
        "applied_shift": applied_shift,
        "base_info": info,
    }


def _run_variant(case: Case, variant: str, seed: int) -> dict[str, Any]:
    row = _base_row(case, variant, seed)
    try:
        t0 = time.perf_counter()
        info: dict[str, Any] | None = None
        sigma_used: float | None = None

        if variant == "heuristic":
            fit, sigma_used, info = fit_quantile(
                case.X_train, case.y_train, tau=case.tau, k=case.k, bs=case.bs
            )
            pred_fn = fit.predict
        elif variant == "heuristic_covcal":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                coverage_calibrate=True,
            )
            pred_fn = fit.predict
        elif variant == "fast_oos":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                preset="fast_oos",
            )
            pred_fn = fit.predict
        elif variant == "heuristic_holdout_covcal":
            fit, sigma_used, info = _fit_heuristic_holdout_covcal(case, seed)
            pred_fn = fit.predict
        elif variant == "heuristic_kfold3_covcal":
            fit, sigma_used, info = _fit_heuristic_kfold_covcal(case, seed, 3)
            pred_fn = fit.predict
        elif variant == "heuristic_kfold5_covcal":
            fit, sigma_used, info = _fit_heuristic_kfold_covcal(case, seed, 5)
            pred_fn = fit.predict
        elif variant == "pin_cv":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                calibrate=True,
                loss="pin",
                n_folds=3,
                seed=seed,
            )
            pred_fn = fit.predict
        elif variant == "pin_cv_covcal":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                calibrate=True,
                loss="pin",
                n_folds=3,
                seed=seed,
                coverage_calibrate=True,
            )
            pred_fn = fit.predict
        elif variant == "quality_oos":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                preset="quality_oos",
                n_folds=3,
                seed=seed,
            )
            pred_fn = fit.predict
        elif variant == "cal_kl_small":
            fit, sigma_used, info = fit_quantile(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k=case.k,
                bs=case.bs,
                calibrate=True,
                loss="cal_kl",
                n_bootstrap=5,
                seed=seed,
            )
            pred_fn = fit.predict
        elif variant == "lss_retune":
            fit, info = fit_quantile_lss(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k_loc=case.k,
                k_scale=case.k,
                bs=case.bs,
                retune_lambda=True,
                seed=seed,
            )
            pred_fn = fit.predict
        elif variant == "lss_no_retune":
            fit, info = fit_quantile_lss(
                case.X_train,
                case.y_train,
                tau=case.tau,
                k_loc=case.k,
                k_scale=case.k,
                bs=case.bs,
                retune_lambda=False,
                seed=seed,
            )
            pred_fn = fit.predict
        else:
            raise ValueError(f"unknown variant: {variant}")

        fit_time_ms = (time.perf_counter() - t0) * 1000.0
        tp = time.perf_counter()
        pred = np.asarray(pred_fn(case.X_test), dtype=float)
        predict_time_ms = (time.perf_counter() - tp) * 1000.0

        row.update(_metrics(case, pred))
        row.update(
            {
                "fit_time_ms": fit_time_ms,
                "predict_time_ms": predict_time_ms,
                "total_time_ms": fit_time_ms + predict_time_ms,
                "sigma_used": sigma_used,
                "calibration": _jsonable(info),
                "status": "ok",
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - benchmark row should capture failures.
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return row


def _sale_price_contract_case() -> Case:
    pd = _import_pandas()
    contract = json.loads(QGAM_HOLDOUT_CONTRACT.read_text())
    train_path = SALE_FIXTURE_DIR / "entire_dataset_train.parquet"
    mean_path = PARITY_DIR / "mean_gam_entire_dataset.parquet"
    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = mean_ref["y"].to_numpy(dtype=float) - mean_ref["pred_prod"].to_numpy(dtype=float)
    X = df[QGAM_SMOOTHS].to_numpy(dtype=float)
    split = int(contract["split_index"])
    return Case(
        case_id="sale_price_q95_contract_80_20",
        case_kind="real",
        X_train=X[:split],
        y_train=resid[:split],
        X_test=X[split:],
        y_test=resid[split:],
        tau=float(contract["tau"]),
        k=[5, 5, 5, 5],
        qgam_pred_test=np.asarray(contract["qgam_pred_test"], dtype=float),
        qgam_pinball=float(contract["qgam_oos_pinball"]),
        qgam_coverage=float(contract["qgam_oos_coverage"]),
        qgam_fit_time_s=float(contract["fit_time_s"]),
        source_files=[str(train_path), str(mean_path), str(QGAM_HOLDOUT_CONTRACT)],
    )


def _load_sale_price_arrays(source_name: str, mean_name: str) -> tuple[np.ndarray, np.ndarray]:
    pd = _import_pandas()
    train_path = SALE_FIXTURE_DIR / source_name
    mean_path = PARITY_DIR / mean_name
    df = pd.read_parquet(train_path)
    mean_ref = pd.read_parquet(mean_path)
    resid = mean_ref["y"].to_numpy(dtype=float) - mean_ref["pred_prod"].to_numpy(dtype=float)
    return df[QGAM_SMOOTHS].to_numpy(dtype=float), resid


def _case_from_qgam_contract(contract: dict[str, Any]) -> Case:
    source_files = list(contract.get("source_files") or [])
    source_name = source_files[0] if source_files else "entire_dataset_train.parquet"
    mean_name = source_files[1] if len(source_files) > 1 else "mgcv_rust_parity/mean_gam_entire_dataset.parquet"
    if mean_name.startswith("mgcv_rust_parity/"):
        mean_name = mean_name.split("/", 1)[1]
    X, resid = _load_sale_price_arrays(source_name, mean_name)
    train_idx = np.asarray(contract["train_idx_1based"], dtype=int) - 1
    test_idx = np.asarray(contract["test_idx_1based"], dtype=int) - 1
    return Case(
        case_id=str(contract["case_id"]),
        case_kind="real",
        X_train=X[train_idx],
        y_train=resid[train_idx],
        X_test=X[test_idx],
        y_test=resid[test_idx],
        tau=float(contract["tau"]),
        k=[5, 5, 5, 5],
        qgam_pred_test=np.asarray(contract["qgam_pred_test"], dtype=float),
        qgam_pinball=float(contract["qgam_oos_pinball"]),
        qgam_coverage=float(contract["qgam_oos_coverage"]),
        qgam_fit_time_s=float(contract["fit_time_s"]),
        source_files=[str(SALE_FIXTURE_DIR / source_name), str(PARITY_DIR / mean_name), str(QGAM_REAL_CONTRACTS)],
    )


def _qgam_real_contract_cases(path: Path, quick: bool) -> list[Case]:
    payload = json.loads(path.read_text())
    contracts = list(payload.get("contracts", []))
    if quick:
        contracts = [c for c in contracts if c.get("case_id") == "sale_price_q95_contract_80_20"]
    return [_case_from_qgam_contract(c) for c in contracts]


def _sale_price_split_cases(limit: int | None) -> list[Case]:
    pd = _import_pandas()
    cases: list[Case] = []
    for i in range(5):
        train_path = SALE_FIXTURE_DIR / f"split_{i}_train.parquet"
        mean_path = PARITY_DIR / f"mean_gam_split_{i}.parquet"
        if not train_path.exists() or not mean_path.exists():
            continue
        df = pd.read_parquet(train_path)
        mean_ref = pd.read_parquet(mean_path)
        resid = mean_ref["y"].to_numpy(dtype=float) - mean_ref["pred_prod"].to_numpy(dtype=float)
        X = df[QGAM_SMOOTHS].to_numpy(dtype=float)
        n = X.shape[0] if limit is None else min(limit, X.shape[0])
        split = max(50, int(n * 0.8))
        cases.append(
            Case(
                case_id=f"sale_price_split_{i}_q95_internal_80_20",
                case_kind="real_no_qgam_ref",
                X_train=X[:split],
                y_train=resid[:split],
                X_test=X[split:n],
                y_test=resid[split:n],
                tau=0.95,
                k=[5, 5, 5, 5],
                source_files=[str(train_path), str(mean_path)],
            )
        )
    return cases


def _synthetic_cases(quick: bool, seed: int) -> list[Case]:
    stats = _import_scipy_stats()
    rng = np.random.default_rng(seed)
    n_train = 600 if quick else 1000
    n_test = 600 if quick else 1000
    cases: list[Case] = []

    def add_case(
        case_id: str,
        tau: float,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        q_test: np.ndarray,
    ) -> None:
        cases.append(
            Case(
                case_id=f"{case_id}_tau_{tau:.2f}",
                case_kind="synthetic",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                tau=tau,
                k=[10] * X_train.shape[1],
                true_quantile_test=q_test,
            )
        )

    for tau in ([0.5, 0.9] if quick else [0.1, 0.5, 0.9, 0.95]):
        X_all = rng.uniform(0.0, 1.0, (n_train + n_test, 2))
        f = np.sin(2 * np.pi * X_all[:, 0]) + 0.5 * np.cos(2 * np.pi * X_all[:, 1])
        y = f + 0.3 * rng.standard_normal(n_train + n_test)
        q = f[n_train:] + 0.3 * stats.norm.ppf(tau)
        add_case(
            "syn_gaussian_additive",
            tau,
            X_all[:n_train],
            y[:n_train],
            X_all[n_train:],
            y[n_train:],
            q,
        )

    for tau in ([0.9] if quick else [0.5, 0.9]):
        X_all = rng.uniform(0.0, 1.0, (n_train + n_test, 2))
        f = np.sin(2 * np.pi * X_all[:, 0]) + X_all[:, 1]
        y = f + 0.25 * rng.standard_t(df=3, size=n_train + n_test)
        q = f[n_train:] + 0.25 * stats.t.ppf(tau, df=3)
        add_case("syn_t_heavytail", tau, X_all[:n_train], y[:n_train], X_all[n_train:], y[n_train:], q)

    for tau in ([0.9] if quick else [0.5, 0.75, 0.9]):
        X_all = rng.uniform(-1.0, 1.0, (n_train + n_test, 2))
        f = np.sin(2.0 * X_all[:, 0]) + 0.5 * X_all[:, 1]
        sigma = 0.1 + 0.4 * np.abs(X_all[:, 0])
        y = f + sigma * rng.standard_normal(n_train + n_test)
        q = f[n_train:] + sigma[n_train:] * stats.norm.ppf(tau)
        add_case(
            "syn_heteroskedastic_normal",
            tau,
            X_all[:n_train],
            y[:n_train],
            X_all[n_train:],
            y[n_train:],
            q,
        )

    return cases


def _import_pandas():
    import pandas as pd

    return pd


def _import_scipy_stats():
    from scipy import stats

    return stats


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    failures = [r for r in rows if r.get("status") != "ok"]
    best: dict[str, dict[str, Any]] = {}
    for row in ok:
        key = f"{row['case_id']}|tau={row['tau']}"
        old = best.get(key)
        if old is None or row.get("pinball", math.inf) < old.get("pinball", math.inf):
            best[key] = {
                "variant": row["variant"],
                "pinball": row.get("pinball"),
                "coverage": row.get("coverage"),
                "fit_time_ms": row.get("fit_time_ms"),
                "pinball_ratio_vs_qgam": row.get("pinball_ratio_vs_qgam"),
                "rmse_true_quantile": row.get("rmse_true_quantile"),
            }
    payload = {
        "benchmark_version": "quantile_oos_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_failures": len(failures),
        "best_by_case_tau_pinball": best,
        "failures": failures,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _selected_cases(args: argparse.Namespace) -> list[Case]:
    cases: list[Case] = []
    if args.real:
        qgam_contracts = Path(args.qgam_contracts)
        if qgam_contracts.exists():
            cases.extend(_qgam_real_contract_cases(qgam_contracts, args.quick))
        else:
            cases.append(_sale_price_contract_case())
        if not args.quick and not qgam_contracts.exists():
            cases.extend(_sale_price_split_cases(args.real_limit))
    if args.synthetic:
        cases.extend(_synthetic_cases(args.quick, args.seed))
    return cases


def _variants(args: argparse.Namespace) -> list[str]:
    if args.variants:
        return [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = ["heuristic", "heuristic_covcal", "pin_cv", "pin_cv_covcal"]
    if not args.quick:
        variants.append("cal_kl_small")
    if args.include_lss:
        variants.extend(["lss_no_retune", "lss_retune"])
    return variants


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/opencode/quantile_oos_rows.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("/tmp/opencode/quantile_oos_summary.json"))
    parser.add_argument("--quick", action="store_true", help="smaller synthetic cases and fewer variants")
    parser.add_argument("--real", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lss", action="store_true")
    parser.add_argument("--variants", help="comma-separated variant names")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-limit", type=int, default=None)
    parser.add_argument("--qgam-contracts", type=Path, default=QGAM_REAL_CONTRACTS)
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    variants = _variants(args)
    with args.output.open("w") as fh:
        for case in _selected_cases(args):
            for variant in variants:
                row = _run_variant(case, variant, args.seed)
                rows.append(row)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                status = row.get("status")
                pin = row.get("pinball")
                ratio = row.get("pinball_ratio_vs_qgam")
                rmse = row.get("rmse_true_quantile")
                t_ms = row.get("fit_time_ms")
                print(
                    f"{status:5s} {case.case_id:42s} {variant:18s} "
                    f"tau={case.tau:.2f} pin={pin} ratio={ratio} rmse={rmse} fit_ms={t_ms}"
                )

    _write_summary(rows, args.summary)
    print(f"rows: {args.output}")
    print(f"summary: {args.summary}")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
