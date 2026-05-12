"""Diff Rust vs mgcv outer-Newton trajectories on the InvGauss-log n=800 fixture.

Reads:
  /tmp/invgauss_rust_outer_trace.txt   (produced by MGCV_RUST_TRACE_OUTER=1)
  /tmp/invgauss_mgcv_outer_trace.txt   (produced by /tmp/run_mgcv_trace.R)

Writes:
  /tmp/invgauss_n800_trajectory_diff.json
  Pretty side-by-side table to stdout.

The two optimizers iterate over different parameter spaces:
  - mgcv: lsp = (log λ_1, log λ_2, log φ)  -- joint with the log-scale,
    so its REML score numerically differs from Rust's (which works on
    (log λ_1, log λ_2) only and conditions on the inner-PIRLS-implied φ).
  - Rust: lsp = (log λ_1, log λ_2)

We line them up by iter index and report the smooth-only λ's to compare
trajectories. mgcv's third lsp entry is parked alongside.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

RUST_PATH = Path("/tmp/invgauss_rust_outer_trace.txt")
MGCV_PATH = Path("/tmp/invgauss_mgcv_outer_trace.txt")
OUT_PATH = Path("/tmp/invgauss_n800_trajectory_diff.json")


def parse_float_list(s: str) -> list[float]:
    """Parse '[1.0, 2.0]' or '1.0,2.0' or 'a,b' formats."""
    s = s.strip()
    s = s.strip("[]")
    return [float(x) for x in s.split(",") if x.strip()]


def parse_rust_trace(path: Path) -> list[dict]:
    text = path.read_text()
    iters: dict[int, dict] = {}
    # Pattern A: iter header
    pat_head = re.compile(
        r"\[OUTER iter=(\d+)\] log_sp=\[([^\]]*)\] sp=\[([^\]]*)\] REML=([\-0-9.eE+]+) "
        r"grad_inf=([\-0-9.eE+]+) grad=\[([^\]]*)\]"
    )
    pat_dir = re.compile(
        r"\[OUTER iter=(\d+)\]\s+dir=\[([^\]]*)\] step_norm=([\-0-9.eE+]+) "
        r"grad_dot_step=([\-0-9.eE+]+) max_half=(\d+)"
    )
    pat_done = re.compile(
        r"\[OUTER iter=(\d+)\]\s+linesearch_done step_size=([\-0-9.eE+]+) "
        r"best_scale=([\-0-9.eE+]+) halvings=(\d+) best_REML=([\-0-9.eE+]+) "
        r"accepted=(\w+) reason=(\w+)"
    )

    for m in pat_head.finditer(text):
        i = int(m.group(1))
        iters.setdefault(i, {})["iter"] = i
        iters[i]["log_sp"] = parse_float_list(m.group(2))
        iters[i]["sp"] = parse_float_list(m.group(3))
        iters[i]["reml"] = float(m.group(4))
        iters[i]["grad_inf"] = float(m.group(5))
        iters[i]["grad"] = parse_float_list(m.group(6))

    for m in pat_dir.finditer(text):
        i = int(m.group(1))
        iters.setdefault(i, {})["dir"] = parse_float_list(m.group(2))
        iters[i]["step_norm"] = float(m.group(3))
        iters[i]["grad_dot_step"] = float(m.group(4))
        iters[i]["max_half"] = int(m.group(5))

    for m in pat_done.finditer(text):
        i = int(m.group(1))
        d = iters.setdefault(i, {})
        d["step_size_taken"] = float(m.group(2))
        d["best_step_scale"] = float(m.group(3))
        d["halvings"] = int(m.group(4))
        d["best_reml"] = float(m.group(5))
        d["accepted"] = m.group(6) == "true"
        d["reason"] = m.group(7)

    return [iters[k] for k in sorted(iters)]


def parse_mgcv_trace(path: Path) -> list[dict]:
    text = path.read_text()
    iters: dict[int, dict] = {}

    pat1 = re.compile(
        r"\[MGCV iter=(\d+)\] lsp=([\-0-9.eE+,]+) score=([\-0-9.eE+]+) "
        r"grad=([\-0-9.eE+,]+) grad_inf=([\-0-9.eE+]+)"
    )
    pat2 = re.compile(
        r"\[MGCV iter=(\d+)\]\s+diag\(hess\)=([\-0-9.eE+,]+) pdef=(\w+) ii=(\-?\d+) "
        r"Nstep=([\-0-9.eE+,]+) sd_unused=(\w+)"
    )
    pat3 = re.compile(
        r"\[MGCV iter=(\d+)\]\s+score1=([\-0-9.eE+]+) score_change=([\-0-9.eE+]+) "
        r"qerror=([\-0-9.eE+]+) pred_change=([\-0-9.eE+]+) Sstep=([\-0-9.eE+,]+)"
    )

    for m in pat1.finditer(text):
        i = int(m.group(1))
        d = iters.setdefault(i, {"iter": i})
        d["lsp"] = parse_float_list(m.group(2))
        d["score"] = float(m.group(3))
        d["grad"] = parse_float_list(m.group(4))
        d["grad_inf"] = float(m.group(5))

    for m in pat2.finditer(text):
        i = int(m.group(1))
        d = iters.setdefault(i, {"iter": i})
        d["diag_hess"] = parse_float_list(m.group(2))
        d["pdef"] = m.group(3) == "TRUE"
        d["ii_halvings"] = int(m.group(4))
        d["Nstep"] = parse_float_list(m.group(5))
        d["sd_unused"] = m.group(6) == "TRUE"

    for m in pat3.finditer(text):
        i = int(m.group(1))
        d = iters.setdefault(i, {"iter": i})
        d["score1_trial"] = float(m.group(2))
        d["score_change"] = float(m.group(3))
        d["qerror"] = float(m.group(4))
        d["pred_change"] = float(m.group(5))
        d["Sstep"] = parse_float_list(m.group(6))

    return [iters[k] for k in sorted(iters)]


def fmt(v, w=12, prec=6):
    if v is None:
        return f"{'-':>{w}}"
    if isinstance(v, list):
        return "[" + ",".join(f"{x:+.4f}" for x in v) + "]"
    if isinstance(v, bool):
        return f"{str(v):>{w}}"
    if isinstance(v, int):
        return f"{v:>{w}d}"
    if isinstance(v, float):
        if abs(v) >= 1e5 or (v != 0.0 and abs(v) < 1e-3):
            return f"{v:>{w}.3e}"
        return f"{v:>{w}.{prec}f}"
    return f"{str(v):>{w}}"


def main():
    rust = parse_rust_trace(RUST_PATH)
    mgcv = parse_mgcv_trace(MGCV_PATH)

    n_iters = max(len(rust), len(mgcv))

    print("=" * 130)
    print("Per-iter Rust vs mgcv outer-Newton trajectory (InvGauss-log n=800)")
    print("=" * 130)
    print()
    print(f"{'iter':>4} | {'RUST log_sp':>26} {'RUST REML':>14} {'g_inf':>10} {'dir':>26} {'step':>10} {'halv':>5} {'acc':>5} {'reason':>20}")
    print("-" * 130)
    for r in rust:
        i = r["iter"]
        log_sp = fmt(r.get("log_sp"))
        reml = r.get("reml", float("nan"))
        ginf = r.get("grad_inf", float("nan"))
        d = fmt(r.get("dir"))
        ss = r.get("step_size_taken", float("nan"))
        hv = r.get("halvings", -1)
        ac = r.get("accepted", False)
        rs = r.get("reason", "")
        print(f"{i:>4} | {log_sp:>26} {reml:>14.4f} {ginf:>10.3e} {d:>26} {ss:>10.3e} {hv:>5d} {str(ac):>5} {rs:>20}")

    print()
    print(f"{'iter':>4} | {'MGCV lsp (incl. lphi)':>40} {'MGCV REML':>14} {'g_inf':>10} {'Nstep':>40} {'qerror':>10} {'pdef':>6} {'pred_dREML':>13} {'true_dREML':>13}")
    print("-" * 130)
    for m in mgcv:
        i = m["iter"]
        lsp = fmt(m.get("lsp"))
        scr = m.get("score", float("nan"))
        ginf = m.get("grad_inf", float("nan"))
        ns = fmt(m.get("Nstep"))
        qe = m.get("qerror", float("nan"))
        pdef = m.get("pdef", False)
        pc = m.get("pred_change", float("nan"))
        sc = m.get("score_change", float("nan"))
        print(f"{i:>4} | {lsp:>40} {scr:>14.4f} {ginf:>10.3e} {ns:>40} {qe:>10.3e} {str(pdef):>6} {pc:>13.4e} {sc:>13.4e}")

    print()
    print("=" * 130)
    print("Key observations (Rust vs mgcv at each iter):")
    print("=" * 130)
    for i in range(1, n_iters + 1):
        r = next((x for x in rust if x["iter"] == i), None)
        m = next((x for x in mgcv if x["iter"] == i), None)
        if r is None or m is None:
            continue
        # Trajectory dist in (log_sp_1, log_sp_2) -- compare first 2 components
        rlsp = r.get("log_sp", [math.nan, math.nan])
        mlsp = m.get("lsp", [math.nan, math.nan, math.nan])[:2]
        delta = [rlsp[j] - mlsp[j] if j < len(rlsp) and j < len(mlsp) else math.nan for j in range(2)]
        print(
            f"  iter={i}: Δ(log_sp) Rust-mgcv = [{delta[0]:+.4f}, {delta[1]:+.4f}]  "
            f"| Rust REML={r.get('reml'):.4f} | mgcv REML={m.get('score'):.4f} "
            f"| Rust halv={r.get('halvings','?')} mgcv halv={m.get('ii_halvings','?')} "
            f"| Rust accepted={r.get('accepted','?')}"
        )

    payload = {
        "rust": rust,
        "mgcv": mgcv,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
