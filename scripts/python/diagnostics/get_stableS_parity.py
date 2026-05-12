"""Validate Rust `gam_reparam_core_py` against the mgcv-C oracle.

Diffs the Rust port (src/reparam.rs::gam_reparam_core, exposed via
`mgcv_rust.gam_reparam_core_py`) against the ctypes oracle wrapping
mgcv's C `get_stableS` (scripts/python/diagnostics/get_stableS_oracle.py).
Reports max abs/rel diff on each of (S, Qs, rS, det, det1, det2).

Run:
    source .venv/bin/activate
    python scripts/python/diagnostics/get_stableS_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mgcv_rust  # noqa: E402
from get_stableS_oracle import gam_reparam_oracle  # noqa: E402


def _pack_rs(rs: list[np.ndarray]) -> tuple[np.ndarray, list[int], int]:
    """Column-major flatten the rs list to match the Rust binding's contract."""
    q = rs[0].shape[0]
    ncols = [r.shape[1] for r in rs]
    flat = np.concatenate([np.asarray(r, dtype=np.float64, order="F").ravel("F") for r in rs])
    return flat, ncols, q


def diff_pair(name: str, a, b, atol: float = 0.0):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return f"{name}: SHAPE MISMATCH {a.shape} vs {b.shape}"
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_diff = np.where(denom > 1e-15, abs_diff / denom, 0.0)
    return (
        f"{name:8s} shape={str(a.shape):16s} "
        f"max|abs|={abs_diff.max():.3e} max|rel|={rel_diff.max():.3e}"
    )


def case_synthetic():
    rng = np.random.default_rng(42)
    q, M = 10, 2

    rs = []
    for _ in range(M):
        a = rng.standard_normal((q, q - 1))
        rs.append(a)
    log_sp = np.log(np.array([1.5, 0.7]))
    deriv = 2

    oracle = gam_reparam_oracle(rs, log_sp, deriv=deriv, fixed_penalty=False)
    flat, ncols, qq = _pack_rs(rs)
    rust = mgcv_rust.gam_reparam_core_py(flat, ncols, qq, log_sp, deriv, False)

    # Re-pack Rust's rs in same order
    print(f"--- case_synthetic (q={q}, M={M}, deriv={deriv}) ---")
    print(diff_pair("S", oracle["S"], rust["S"]))
    print(diff_pair("Qs", oracle["Qs"], rust["Qs"]))
    for i, (a, b) in enumerate(zip(oracle["rS"], rust["rs"])):
        print(diff_pair(f"rs[{i}]", a, b))
    print(f"det      oracle={oracle['det']:.10f}  rust={rust['det']:.10f}  abs={abs(oracle['det']-rust['det']):.3e}")
    print(diff_pair("det1", oracle["det1"], rust["det1"]))
    print(diff_pair("det2", oracle["det2"], rust["det2"]))


def case_invgauss_n800():
    """Pull the actual penalty roots from the InvGauss n=800 fit and diff."""
    import json
    fix_path = Path(__file__).resolve().parents[3] / "tests" / "parity" / "fixtures" / "2d_invgauss_log_n800_k10_cr.json"
    if not fix_path.exists():
        print(f"--- case_invgauss_n800: skipped (no fixture at {fix_path}) ---")
        return
    fix = json.load(open(fix_path))
    inp = fix["inputs"]
    gam = mgcv_rust.GAM("inverse.gaussian")
    gam.fit(
        np.asarray(inp["x_train"], dtype=float),
        np.asarray(inp["y_train"], dtype=float),
        k=list(inp["k"]),
        method=inp["method"],
        bs=inp["bs"][0],
    )
    # Pull per-smooth penalty matrices and design matrix.
    penalties = gam.get_smooth_penalties()  # list of (k_i x k_i) matrices
    lambdas_mgcv = gam.get_all_lambdas()    # length M, optimizer-scale; convert to mgcv-scale
    sf = gam.get_penalty_scale_factors()    # length M, optimizer→mgcv-scale factor

    # Build q × q embedded square roots from each k_i × k_i penalty.
    design = np.asarray(gam.get_design_matrix())
    q = design.shape[1]
    term_idx = gam.get_term_indices()  # list of (name, first_col, last_col)
    rs_list = []
    for (name, lo, hi), pen in zip(term_idx, penalties):
        k = pen.shape[0]
        assert (hi - lo + 1) == k, f"shape mismatch {name}: {hi - lo + 1} vs {k}"
        # Square root via eigh (drop near-zero eigenvalues)
        evals, evecs = np.linalg.eigh(pen)
        rank = int(np.sum(evals > evals.max() * 1e-12))
        # rs_block_k = evecs[:, -rank:] · diag(sqrt(evals[-rank:]))
        rs_k = evecs[:, -rank:] * np.sqrt(evals[-rank:])
        # Embed into q × rank with zeros outside [lo, hi]
        rs_full = np.zeros((q, rank), order="F")
        rs_full[lo : hi + 1, :] = rs_k
        rs_list.append(rs_full)

    # mgcv-scale λ = optimizer-scale λ × scale_factor
    sp_mgcv = np.array(lambdas_mgcv) * np.array(sf)
    log_sp = np.log(sp_mgcv)
    deriv = 2

    oracle = gam_reparam_oracle(rs_list, log_sp, deriv=deriv, fixed_penalty=False)
    flat, ncols, qq = _pack_rs(rs_list)
    rust = mgcv_rust.gam_reparam_core_py(flat, ncols, qq, log_sp, deriv, False)

    print(f"--- case_invgauss_n800 (q={q}, M={len(rs_list)}, λ_mgcv={sp_mgcv}) ---")
    print(diff_pair("S", oracle["S"], rust["S"]))
    print(diff_pair("Qs", oracle["Qs"], rust["Qs"]))
    for i, (a, b) in enumerate(zip(oracle["rS"], rust["rs"])):
        print(diff_pair(f"rs[{i}]", a, b))
    print(f"det      oracle={oracle['det']:.10f}  rust={rust['det']:.10f}  abs={abs(oracle['det']-rust['det']):.3e}")
    print(diff_pair("det1", oracle["det1"], rust["det1"]))
    print(diff_pair("det2", oracle["det2"], rust["det2"]))


if __name__ == "__main__":
    case_synthetic()
    print()
    case_invgauss_n800()
