"""Phase A scout for N-1, N-2 (Gamma+log xfails).

A1: at our converged ρ, compare rust σ̂² = D/(n-trA) against mgcv's $sig2.
A2: at our converged ρ, the *profiled* REML's gradient (numerical FD of
    `evaluate_reml_mgcv_formula`) minus the *fixed-σ²* IFT gradient
    (`evaluate_reml_gradient_ift`) is the residual σ²-chain term
    (∂REML/∂σ²) · (∂σ̂²/∂ρ_k). If this matches the residual gradient norm
    Newton hits at convergence, joint-σ² is the missing piece.

Run from repo root with the venv activated:
    source venv/bin/activate
    python scripts/python/diagnostics/joint_sigma_check.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures"

CASES = {
    "N-1": "2d_gamma_log_n200_k10_cr",
    "N-2": "4d_gamma_log_n2000_k8_cr",
}


# ---------------------------------------------------------------------------
# mgcv side — re-fit and pull sig2 + sp directly from the gam object.
# ---------------------------------------------------------------------------
def fit_mgcv(fixture: dict) -> dict:
    inputs = fixture["inputs"]
    x = np.asarray(inputs["x_train"], dtype=float)
    y = np.asarray(inputs["y_train"], dtype=float)
    k = inputs["k"]
    bs = inputs["bs"]
    family = inputs["family"]
    link = inputs["link"]
    method = inputs["method"]

    n, d = x.shape

    smooth_terms = " + ".join(
        f's(x{i}, k={k[i]}, bs="{bs[i]}")' for i in range(d)
    )
    formula = f"y ~ {smooth_terms}"

    df_cols = ", ".join(f"x{i}=x[,{i+1}]" for i in range(d))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        x_path = tmpdir / "x.csv"
        y_path = tmpdir / "y.csv"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(y_path, y, delimiter=",")

        rscript = textwrap.dedent(f"""
            suppressMessages(library(mgcv))
            x <- as.matrix(read.csv("{x_path}", header=FALSE))
            y <- read.csv("{y_path}", header=FALSE)[,1]
            df <- data.frame({df_cols}, y=y)
            fam <- {family}(link="{link}")
            g <- gam({formula}, data=df, family=fam, method="{method}")
            cat("SIG2:", g$sig2, "\\n")
            cat("SCALE_EST:", g$scale.estimated, "\\n")
            cat("SP:", paste(g$sp, collapse=","), "\\n")
            cat("DEV:", g$deviance, "\\n")
            cat("REML:", g$gcv.ubre, "\\n")
            pred <- predict(g, type="response")
            cat("PRED_HEAD:", paste(head(pred, 5), collapse=","), "\\n")
            cat("PRED_MEAN:", mean(pred), "\\n")
            saveRDS(list(sig2=g$sig2, sp=g$sp, dev=g$deviance,
                         reml=g$gcv.ubre, pred=as.numeric(pred)),
                    file="{tmpdir / 'out.rds'}")
            # Dump preds as csv too
            write.table(pred, "{tmpdir / 'pred.csv'}",
                        sep=",", row.names=FALSE, col.names=FALSE)
        """)
        rfile = tmpdir / "fit.R"
        rfile.write_text(rscript)
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(rfile)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            print("--- R stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print("--- R stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            print("--- script ---", file=sys.stderr)
            print(rscript, file=sys.stderr)
            raise RuntimeError("Rscript failed; see above")

        out = {}
        for line in proc.stdout.splitlines():
            if line.startswith("SIG2:"):
                out["sig2"] = float(line.split()[1])
            elif line.startswith("SCALE_EST:"):
                out["scale_est"] = line.split()[1]
            elif line.startswith("SP:"):
                out["sp"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            elif line.startswith("DEV:"):
                out["dev"] = float(line.split()[1])
            elif line.startswith("REML:"):
                out["reml"] = float(line.split()[1])
        out["pred"] = np.loadtxt(tmpdir / "pred.csv", delimiter=",")
    return out


# ---------------------------------------------------------------------------
# rust side — fit, pull σ̂² + trA from MGCV_EXACT_DEBUG, FD the profile score.
# ---------------------------------------------------------------------------
DEBUG_RE = re.compile(
    r"sigma2=(?P<sig2>[-0-9.eE+]+).*?"
    r"trA=(?P<trA>[-0-9.eE+]+)\s+n-trA=(?P<n_minus_tra>[-0-9.eE+]+).*?"
    r"REML\s+(?P<reml>[-0-9.eE+]+)",
    re.DOTALL,
)


def parse_debug(stderr: str) -> dict | None:
    m = DEBUG_RE.search(stderr)
    if not m:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}


def rust_eval_with_debug(model, y, lambdas):
    """Call evaluate_reml_mgcv_formula with debug, parse out σ̂², trA, REML."""
    import io
    from contextlib import redirect_stderr

    # The debug print uses eprintln (Rust stderr); we need to capture stderr
    # at the OS level. Use os.dup/dup2 to swap fd 2.
    import os as _os
    r, w = _os.pipe()
    saved = _os.dup(2)
    _os.dup2(w, 2)
    _os.close(w)
    os.environ["MGCV_EXACT_DEBUG"] = "1"
    try:
        reml = model.evaluate_reml_mgcv_formula(np.asarray(y, dtype=float),
                                                list(lambdas))
    finally:
        os.environ.pop("MGCV_EXACT_DEBUG", None)
        _os.dup2(saved, 2)
        _os.close(saved)
        # Drain pipe
        chunks = []
        while True:
            try:
                chunk = _os.read(r, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < 65536:
                break
        _os.close(r)
    stderr = b"".join(chunks).decode("utf-8", errors="replace")
    info = parse_debug(stderr)
    if info is None:
        raise RuntimeError(f"failed to parse MGCV_EXACT_DEBUG output:\n{stderr[:1000]}")
    info["reml_returned"] = reml
    return info


def fit_rust(fixture: dict) -> dict:
    from mgcv_rust import GAMFitter

    inputs = fixture["inputs"]
    x = np.asarray(inputs["x_train"], dtype=float)
    y = np.asarray(inputs["y_train"], dtype=float)
    k = inputs["k"]
    bs = inputs["bs"]

    fam = inputs["family"]
    link = inputs["link"]

    family_arg = "gamma" if fam == "Gamma" else fam.lower()
    predictors = [f"x{i}" for i in range(x.shape[1])]
    term_k_mapping = {p: int(k[i]) for i, p in enumerate(predictors)}
    predictor_basis_map = {p: str(bs[i]) for i, p in enumerate(predictors)}

    fitter = GAMFitter(
        predictors=predictors,
        family=family_arg,
        link=link,
        method="REML",
        term_k_mapping=term_k_mapping,
        predictor_basis_map=predictor_basis_map,
    )
    fitter.fit(x, y)

    pred = fitter.predict(x)
    lambdas = list(np.asarray(fitter.get_lambdas()).tolist())

    # σ̂², trA, REML at our converged ρ
    info = rust_eval_with_debug(fitter._native, y, lambdas)

    # Numerical FD of the profiled REML score (evaluate_reml_mgcv_formula
    # internally profiles σ̂²) — gives the TOTAL derivative dREML_profile/dρ_k
    # = ∂REML/∂ρ_k|σ² + (∂REML/∂σ²)·∂σ̂²/∂ρ_k. Our outer Newton drives the
    # ∂REML/∂ρ_k|σ² term to zero (it uses the IFT fixed-σ² gradient for
    # descent), so at convergence the FD profile gradient ≈ the chain term.
    eps = 1e-4
    fd_grad = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        # ρ_i = log λ_i; perturb in log-space, then exponentiate
        log_lam = np.log(lambdas)
        log_lam_plus = log_lam.copy()
        log_lam_plus[i] += eps
        log_lam_minus = log_lam.copy()
        log_lam_minus[i] -= eps
        lam_plus = list(np.exp(log_lam_plus))
        lam_minus = list(np.exp(log_lam_minus))
        r_plus = fitter._native.evaluate_reml_mgcv_formula(
            np.asarray(y, dtype=float), lam_plus
        )
        r_minus = fitter._native.evaluate_reml_mgcv_formula(
            np.asarray(y, dtype=float), lam_minus
        )
        fd_grad[i] = (r_plus - r_minus) / (2.0 * eps)

    # At convergence the IFT fixed-σ² gradient is ~0, so the FD profile
    # gradient IS the chain term (∂REML/∂σ²)·(∂σ̂²/∂ρ_k).
    chain_term = fd_grad

    # Also FD σ̂² wrt each ρ_k for explicit chain-term magnitude
    fd_sig2 = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        log_lam = np.log(lambdas)
        log_lam_plus = log_lam.copy()
        log_lam_plus[i] += eps
        log_lam_minus = log_lam.copy()
        log_lam_minus[i] -= eps
        info_plus = rust_eval_with_debug(fitter._native, y,
                                         list(np.exp(log_lam_plus)))
        info_minus = rust_eval_with_debug(fitter._native, y,
                                          list(np.exp(log_lam_minus)))
        fd_sig2[i] = (info_plus["sig2"] - info_minus["sig2"]) / (2.0 * eps)

    # The σ̂² printed by MGCV_EXACT_DEBUG comes from the working-RSS path
    # (evaluate_reml_mgcv_formula passes y_original=None — see lib.rs:1155).
    # That's NOT the σ̂² mgcv reports. To compare apples-to-apples, recompute
    # σ̂² from the true GLM deviance using rust's predictions (= μ̂):
    #   Gamma deviance: D = 2 · Σ ((y - μ)/μ - log(y/μ))
    y_arr = np.asarray(y, dtype=float)
    mu = np.asarray(pred, dtype=float)
    # Avoid log(0) on tiny predictions
    safe = np.maximum(mu, 1e-300)
    glm_dev = 2.0 * float(np.sum((y_arr - safe) / safe - np.log(y_arr / safe)))
    n = y_arr.shape[0]
    sigma2_glm = glm_dev / (n - info["trA"])

    return {
        "lambdas": lambdas,
        "pred": pred,
        "sig2_working_rss": info["sig2"],
        "sig2_glm": sigma2_glm,
        "glm_dev": glm_dev,
        "trA": info["trA"],
        "n_minus_tra": info["n_minus_tra"],
        "reml": info["reml_returned"],
        "fd_grad_profile": fd_grad,
        "chain_term": chain_term,
        "fd_sig2": fd_sig2,
    }


def main():
    sys.path.insert(0, str(REPO / "python"))
    summary = {}

    for tag, name in CASES.items():
        path = FIXTURES_DIR / f"{name}.json"
        fixture = json.loads(path.read_text())
        n = fixture["inputs"]["n"]

        print(f"\n{'='*70}\n{tag} = {name}  (n={n})\n{'='*70}")

        print(f"[mgcv] re-fitting {name}...")
        m = fit_mgcv(fixture)
        print(f"  mgcv sig2     = {m['sig2']:.6e}")
        print(f"  mgcv sp       = {m['sp']}")
        print(f"  mgcv dev      = {m['dev']:.6e}")
        print(f"  mgcv pred[:5] = {m['pred'][:5]}")

        print(f"[rust] fitting {name}...")
        r = fit_rust(fixture)
        print(f"  rust σ̂² (GLM dev / n-trA)  = {r['sig2_glm']:.6e}")
        print(f"  rust σ̂² (working-RSS form) = {r['sig2_working_rss']:.6e}  (diag-entry artefact)")
        print(f"  rust GLM deviance          = {r['glm_dev']:.6e}")
        print(f"  rust trA       = {r['trA']:.4f}, n-trA = {r['n_minus_tra']:.4f}")
        print(f"  rust λ (our)   = {r['lambdas']}")
        print(f"  rust pred[:5]  = {r['pred'][:5]}")

        # A1 verdict — Δlog φ on the GLM-deviance σ̂² (the one mgcv reports)
        log_sig2_rust = np.log(r["sig2_glm"])
        log_sig2_mgcv = np.log(m["sig2"])
        delta = log_sig2_rust - log_sig2_mgcv
        print(f"\n  [A1] log σ̂²_rust(GLM) - log σ²_mgcv = {delta:+.4e}")
        print(f"        rust σ̂²_GLM / mgcv σ² ratio    = {r['sig2_glm']/m['sig2']:.6f}")
        print(f"        rust GLM dev vs mgcv dev: {r['glm_dev']:.4f} vs {m['dev']:.4f}")

        # A2 — at converged ρ the IFT fixed-σ² gradient ≈ 0 (Newton minimised
        # it); FD of the profile REML therefore IS the chain term we'd need
        # to add to drive the *true* profile gradient to zero.
        print(f"  [A2] FD profile gradient (= chain term at convergence):")
        print(f"          {r['fd_grad_profile']}")
        print(f"        ‖chain term‖∞              : {np.max(np.abs(r['chain_term'])):.4e}")
        print(f"        FD ∂σ̂²/∂ρ                  : {r['fd_sig2']}")

        # Train absdiff vs mgcv
        absdiff = np.max(np.abs(np.asarray(r["pred"]) - np.asarray(m["pred"])))
        print(f"\n  train max |rust_pred - mgcv_pred|: {absdiff:.4e}")

        summary[tag] = {
            "case": name,
            "delta_log_sigma2": delta,
            "rust_sig2_glm": r["sig2_glm"],
            "rust_sig2_working_rss": r["sig2_working_rss"],
            "rust_glm_dev": r["glm_dev"],
            "mgcv_sig2": m["sig2"],
            "mgcv_dev": m["dev"],
            "fd_profile_grad_inf": float(np.max(np.abs(r["fd_grad_profile"]))),
            "chain_inf": float(np.max(np.abs(r["chain_term"]))),
            "train_absdiff": float(absdiff),
        }

    print(f"\n\n{'='*70}\nVERDICT\n{'='*70}")
    for tag, s in summary.items():
        print(f"\n{tag} ({s['case']}):")
        print(f"  Δlog σ²(GLM)        = {s['delta_log_sigma2']:+.4e}")
        print(f"  rust σ̂² (GLM)/mgcv  = {s['rust_sig2_glm']/s['mgcv_sig2']:.6f}")
        print(f"  rust GLM dev / mgcv = {s['rust_glm_dev']/s['mgcv_dev']:.6f}")
        print(f"  chain term ∞        = {s['chain_inf']:.4e}")
        print(f"  train absdiff       = {s['train_absdiff']:.4e}")

    print("\n[A1 interpretation]")
    print("  |Δlog σ²| < 1e-4 → our σ̂² ≈ mgcv's converged dispersion.")
    print("  |Δlog σ²| > 1e-2 → joint-σ² treatment likely needed (Path B).")
    print("\n[A2 interpretation]")
    print("  ‖chain term‖∞ ≈ ‖IFT grad‖∞ at converged ρ → chain term is the")
    print("    residual gradient our Newton can't drive to zero. Either")
    print("    Path B (joint σ²) or Path C (add chain term analytically).")

    out_path = REPO / "scripts" / "python" / "diagnostics" / "joint_sigma_check_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
