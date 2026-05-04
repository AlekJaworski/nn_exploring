"""N-3 saturation diagnostic for 2d_gamma_inverse_n1000_k10_cr.

Stage 1: Diagnose why Rust converges to a worse REML point than mgcv.
  - Compare converged λ, log-λ ratios, REML scores
  - Compare gradient ∞-norms at convergence (Rust vs mgcv)
  - Check Hessian diagonal vs gradient: |H_ii| < |g_i|·100 → saturation
  - Check if Armijo early-exit is culprit vs genuinely different stationary point

Stage 2: If Armijo-damping confirmed, loosen c1 from 0.01 to 1e-4.

Run from repo root with the venv activated:
    source venv/bin/activate
    python scripts/python/diagnostics/n3_saturation_check.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures"
CASE = "2d_gamma_inverse_n1000_k10_cr"


def fit_mgcv(fixture: dict) -> dict:
    """Re-fit with fresh mgcv::gam, capture lambda, grad, n_iter, score, predictions."""
    inputs = fixture["inputs"]
    x = np.asarray(inputs["x_train"], dtype=float)
    y = np.asarray(inputs["y_train"], dtype=float)
    k = inputs["k"]
    bs = inputs["bs"]
    family = inputs["family"]
    link = inputs["link"]
    method = inputs["method"]

    n, d = x.shape
    smooth_terms = " + ".join(f's(x{i}, k={k[i]}, bs="{bs[i]}")' for i in range(d))
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
            cat("SP:", paste(g$sp, collapse=","), "\\n")
            cat("REML:", g$gcv.ubre, "\\n")
            cat("N_ITER:", g$outer.info$iter, "\\n")
            cat("FINAL_GRAD:", paste(g$outer.info$grad, collapse=","), "\\n")
            pred <- predict(g, type="response")
            write.table(pred, "{tmpdir / 'pred.csv'}", sep=",", row.names=FALSE, col.names=FALSE)
            # Gradient and Hessian at convergence (deriv=2 evaluation)
            # Extract from outer.info if available
            if (!is.null(g$outer.info)) {{
                cat("OUTER_INFO_CONV:", g$outer.info$conv, "\\n")
                if (!is.null(g$outer.info$hess)) {{
                    h <- g$outer.info$hess
                    cat("HESS_DIAG:", paste(diag(h), collapse=","), "\\n")
                }}
            }}
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
            raise RuntimeError("Rscript failed")

        out: dict = {}
        for line in proc.stdout.splitlines():
            if line.startswith("SP:"):
                out["sp"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            elif line.startswith("REML:"):
                out["reml"] = float(line.split()[1])
            elif line.startswith("N_ITER:"):
                out["n_iter"] = int(line.split()[1])
            elif line.startswith("FINAL_GRAD:"):
                out["final_grad"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            elif line.startswith("OUTER_INFO_CONV:"):
                out["conv"] = line.split(":", 1)[1].strip()
            elif line.startswith("HESS_DIAG:"):
                out["hess_diag"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
        out["pred"] = np.loadtxt(tmpdir / "pred.csv", delimiter=",")
    return out


def fit_rust(fixture: dict, verbose_profile: bool = False) -> dict:
    """Fit with our Rust pipeline via GAMFitter, capture lambda, grad, REML."""
    sys.path.insert(0, str(REPO / "python"))
    from mgcv_rust import GAMFitter

    inputs = fixture["inputs"]
    x = np.asarray(inputs["x_train"], dtype=float)
    y = np.asarray(inputs["y_train"], dtype=float)
    k = inputs["k"]
    bs = inputs["bs"]
    d = x.shape[1]

    predictors = [f"x{i}" for i in range(d)]
    term_k_mapping = {p: int(k[i]) for i, p in enumerate(predictors)}
    predictor_basis_map = {p: str(bs[i]) for i, p in enumerate(predictors)}

    env_backup = {}
    if verbose_profile:
        env_backup["MGCV_PROFILE"] = os.environ.get("MGCV_PROFILE", "")
        os.environ["MGCV_PROFILE"] = "1"

    fitter = GAMFitter(
        predictors=predictors,
        family="gamma",
        link="inverse",
        method="REML",
        term_k_mapping=term_k_mapping,
        predictor_basis_map=predictor_basis_map,
    )
    fitter.fit(x, y)

    if verbose_profile:
        for k2, v in env_backup.items():
            if v:
                os.environ[k2] = v
            else:
                os.environ.pop(k2, None)

    pred = fitter.predict(x)
    lambdas = list(np.asarray(fitter.get_lambdas()).tolist())

    # Gradient and Hessian at our converged λ via mgcv-exact IFT formula
    # Use FD if direct access unavailable
    native = fitter._native
    eps = 1e-4
    log_lam = np.log(lambdas)
    m = len(lambdas)

    # Compute REML at our converged λ
    reml_ours = native.evaluate_reml_mgcv_formula(np.asarray(y, dtype=float), lambdas)

    # FD gradient in log-λ space
    fd_grad = np.zeros(m)
    for i in range(m):
        ll_p = log_lam.copy(); ll_p[i] += eps
        ll_m = log_lam.copy(); ll_m[i] -= eps
        r_p = native.evaluate_reml_mgcv_formula(y.astype(float), list(np.exp(ll_p)))
        r_m = native.evaluate_reml_mgcv_formula(y.astype(float), list(np.exp(ll_m)))
        fd_grad[i] = (r_p - r_m) / (2.0 * eps)

    # FD Hessian diagonal
    fd_hess_diag = np.zeros(m)
    for i in range(m):
        ll_p = log_lam.copy(); ll_p[i] += eps
        ll_m = log_lam.copy(); ll_m[i] -= eps
        r_p = native.evaluate_reml_mgcv_formula(y.astype(float), list(np.exp(ll_p)))
        r_m = native.evaluate_reml_mgcv_formula(y.astype(float), list(np.exp(ll_m)))
        fd_hess_diag[i] = (r_p - 2.0 * reml_ours + r_m) / (eps ** 2)

    return {
        "lambdas": lambdas,
        "pred": pred,
        "reml": reml_ours,
        "fd_grad": fd_grad,
        "fd_hess_diag": fd_hess_diag,
    }


def eval_reml_at_sp(fixture: dict, sp: list[float]) -> float:
    """Evaluate our Rust REML formula at mgcv's converged sp."""
    sys.path.insert(0, str(REPO / "python"))
    from mgcv_rust import GAMFitter

    inputs = fixture["inputs"]
    x = np.asarray(inputs["x_train"], dtype=float)
    y = np.asarray(inputs["y_train"], dtype=float)
    k = inputs["k"]
    bs = inputs["bs"]
    d = x.shape[1]

    predictors = [f"x{i}" for i in range(d)]
    term_k_mapping = {p: int(k[i]) for i, p in enumerate(predictors)}
    predictor_basis_map = {p: str(bs[i]) for i, p in enumerate(predictors)}

    # Fit with mgcv's λ as initial (but then override with init lambda = sp)
    # We just need the native handle for evaluate_reml_mgcv_formula.
    # Fit normally first, then call evaluate_reml at mgcv sp.
    fitter = GAMFitter(
        predictors=predictors,
        family="gamma",
        link="inverse",
        method="REML",
        term_k_mapping=term_k_mapping,
        predictor_basis_map=predictor_basis_map,
    )
    fitter.fit(x, y)
    return fitter._native.evaluate_reml_mgcv_formula(y.astype(float), sp)


def main():
    path = FIXTURES_DIR / f"{CASE}.json"
    fixture = json.loads(path.read_text())

    print(f"\n{'='*70}")
    print(f"N-3 saturation diagnostic: {CASE}")
    print(f"{'='*70}\n")

    # --- Fixture reference ---
    mo = fixture["mgcv_output"]
    fixture_lambda = mo.get("lambda", [])
    fixture_n_iter = mo.get("n_iter")
    fixture_final_grad = mo.get("final_grad", [])
    fixture_score_history = mo.get("score_history", [])
    print(f"[Fixture] mgcv λ         : {fixture_lambda}")
    print(f"[Fixture] mgcv n_iter    : {fixture_n_iter}")
    print(f"[Fixture] mgcv final_grad: {fixture_final_grad}")
    if fixture_score_history:
        print(f"[Fixture] mgcv score hist: {fixture_score_history}")
        print(f"[Fixture] mgcv final REML: {fixture_score_history[-1]:.6f}")

    # --- Re-fit mgcv (fresh run) ---
    print(f"\n[mgcv] re-fitting {CASE}...")
    m = fit_mgcv(fixture)
    print(f"  mgcv converged sp      : {m['sp']}")
    print(f"  mgcv converged log-sp  : {np.log(m['sp'])}")
    print(f"  mgcv REML              : {m['reml']:.6f}")
    print(f"  mgcv n_iter            : {m['n_iter']}")
    print(f"  mgcv outer.info$grad   : {m['final_grad']}")
    mgcv_grad_inf = float(np.max(np.abs(m['final_grad']))) if m['final_grad'] else float('nan')
    print(f"  mgcv grad ∞-norm       : {mgcv_grad_inf:.4e}")
    if 'hess_diag' in m:
        print(f"  mgcv H_diag            : {m['hess_diag']}")
    if 'conv' in m:
        print(f"  mgcv conv              : {m['conv']}")

    # --- Rust fit ---
    print(f"\n[rust] fitting {CASE}...")
    r = fit_rust(fixture)
    print(f"  rust converged λ       : {r['lambdas']}")
    print(f"  rust converged log-λ   : {list(np.log(r['lambdas']))}")
    print(f"  rust REML at our λ     : {r['reml']:.6f}")
    rust_grad_inf = float(np.max(np.abs(r['fd_grad'])))
    print(f"  rust grad ∞-norm (FD)  : {rust_grad_inf:.4e}")
    print(f"  rust FD gradient       : {r['fd_grad']}")
    print(f"  rust FD Hess diag      : {r['fd_hess_diag']}")

    # --- Log-λ ratio ---
    print(f"\n--- Log-λ comparison ---")
    rust_log_lam = np.log(r['lambdas'])
    mgcv_log_lam = np.log(m['sp'])
    for i in range(len(r['lambdas'])):
        ratio = rust_log_lam[i] - mgcv_log_lam[i]
        print(f"  smooth {i}: rust log-λ={rust_log_lam[i]:.4f}, mgcv log-λ={mgcv_log_lam[i]:.4f}, diff={ratio:+.4f}")

    # --- REML score comparison ---
    print(f"\n--- REML score comparison ---")
    reml_at_mgcv_sp = eval_reml_at_sp(fixture, m['sp'])
    print(f"  Our REML formula at rust λ : {r['reml']:.6f}")
    print(f"  Our REML formula at mgcv λ : {reml_at_mgcv_sp:.6f}")
    score_gap = r['reml'] - reml_at_mgcv_sp
    print(f"  Score gap (rust - mgcv)    : {score_gap:+.6f}")
    if score_gap > 0.01:
        print(f"  => Rust is at a WORSE point (score +{score_gap:.3f} above mgcv)")
    else:
        print(f"  => Score gap small (<0.01), optimum is similar")

    # --- Saturation check: |H_ii| < |g_i|·100 ---
    print(f"\n--- Saturation check: |H_ii| < |g_i|·100 ---")
    for i, (g, h) in enumerate(zip(r['fd_grad'], r['fd_hess_diag'])):
        saturating = abs(h) < abs(g) * 100
        print(f"  smooth {i}: |g|={abs(g):.3e}, |H_ii|={abs(h):.3e}, ratio={abs(h)/max(abs(g),1e-30):.1f}, saturating={saturating}")

    # --- Armijo diagnostic ---
    print(f"\n--- Armijo early-exit diagnosis ---")
    print(f"  mgcv grad ∞-norm at convergence : {mgcv_grad_inf:.4e}")
    print(f"  rust grad ∞-norm at convergence : {rust_grad_inf:.4e}")
    if rust_grad_inf > 10 * mgcv_grad_inf and score_gap > 0.01:
        verdict = "ARMIJO_EARLY_EXIT"
        print(f"  => VERDICT: {verdict}")
        print(f"     Rust gradient is {rust_grad_inf/max(mgcv_grad_inf,1e-30):.1f}× larger than mgcv's.")
        print(f"     Score is {score_gap:.3f} above mgcv's optimal.")
        print(f"     Armijo c1=0.01 damps Newton step once score is near-flat,")
        print(f"     triggering score-change exit before gradient converges.")
    elif rust_grad_inf < 5 * mgcv_grad_inf and score_gap < 0.01:
        verdict = "DIFFERENT_STATIONARY_POINT"
        print(f"  => VERDICT: {verdict}")
        print(f"     Both rust and mgcv have small gradients but at different λ.")
        print(f"     Multiple local optima in flat REML landscape.")
    else:
        verdict = "UNCLEAR"
        print(f"  => VERDICT: {verdict}")
        print(f"     Score gap={score_gap:.4f}, grad ratio={rust_grad_inf/max(mgcv_grad_inf,1e-30):.2f}×")

    # --- Train absdiff ---
    fixture_pred = np.asarray(fixture["mgcv_output"]["predictions_train"], dtype=float)
    rust_pred = np.asarray(r["pred"], dtype=float)
    mgcv_pred = np.asarray(m["pred"], dtype=float)
    absdiff_vs_fixture = float(np.max(np.abs(rust_pred - fixture_pred)))
    absdiff_vs_fresh = float(np.max(np.abs(rust_pred - mgcv_pred)))
    print(f"\n--- Prediction comparison ---")
    print(f"  rust vs fixture max|diff| : {absdiff_vs_fixture:.4e}")
    print(f"  rust vs fresh mgcv max|diff|: {absdiff_vs_fresh:.4e}")
    print(f"  (rtol threshold = 1e-3)")

    return verdict


if __name__ == "__main__":
    main()
