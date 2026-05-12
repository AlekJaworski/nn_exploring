"""InvGauss-log n=800 sub-step diff: localize where Rust REML diverges from mgcv.

Goal: at two fixed λ points
  A = mgcv's converged sp  = (192.16284812521303, 504.4534182343682)
  B = Rust's converged sp  = (191.54, 482.97)   # filled from a live Rust fit
compute REML score, gradient (in log-λ), and Hessian (in log-λ) on
*both* sides — mgcv via mgcv:::gam.fit3(deriv=2), Rust via the
mgcv-exact criterion + IFT grad/Hess.

Coordinate-system note (src/lib.rs:1411-1424):
  evaluate_reml_mgcv_formula / evaluate_reml_gradient_ift / evaluate_reml_hessian_ift
  all internally do `penalty.scaled_add_to(&mut a, *lambda)` directly,
  so they take *raw* (= mgcv-coord) λ values that multiply S_j with no
  scale_factor. So we can pass mgcv's sp directly. To convert a Rust
  fit's reported λ (optimizer-scaled coords) back to mgcv coords for
  comparison, multiply by get_penalty_scale_factors().

Output table covers REML, gradient (∂/∂log λ_j), and Hessian (∂²/∂log λ_j∂log λ_k).
Result is saved to /tmp/invgauss_n800_substep_diff.json.

Run:
    source .venv/bin/activate
    python scripts/python/diagnostics/invgauss_n800_substep_diff.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures"
CASE = "2d_invgauss_log_n800_k10_cr"

# Point A is mgcv's converged sp from the fixture.
POINT_A = np.array([192.16284812521303, 504.4534182343682])


# --------------------------------------------------------------------------- #
# R side — mgcv:::gam.fit3 at fixed sp with deriv=2.                          #
# --------------------------------------------------------------------------- #

R_EVAL_SCRIPT = r"""
suppressMessages(library(mgcv))
x <- as.matrix(read.csv("__XPATH__", header=FALSE))
y <- read.csv("__YPATH__", header=FALSE)[,1]
df <- data.frame(x0=x[,1], x1=x[,2], y=y)
fam <- inverse.gaussian(link="log")

# First fit unconstrained: gives us the converged scale (and a sanity check sp).
g_full <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
              data=df, family=fam, method="REML")
cat("MGCV_FIT_SP:", paste(g_full$sp, collapse=","), "\n")
cat("MGCV_FIT_REML:", g_full$gcv.ubre, "\n")
cat("MGCV_FIT_SCALE:", g_full$scale, "\n")
cat("MGCV_FIT_GRAD:", paste(g_full$outer.info$grad, collapse=","), "\n")
cat("MGCV_FIT_HESS:", paste(as.vector(g_full$outer.info$hess), collapse=","), "\n")

# REML-profiled scale (= argmin of REML w.r.t. log(scale) at fixed sp). At the
# outer-converged point this lives in g$reml.scale and DIFFERS from g$scale
# (which is the Pearson scale.est). gam.fit3's grad w.r.t. log(scale) is only
# zero when we plug in reml.scale, so we extract it via a fixed-sp gam refit.
scale_at <- function(sp_t) {
  gp <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
            data=df, family=fam, method="REML", sp=sp_t)
  return(gp$reml.scale)
}

# Build G via fit=FALSE then replicate the pre-outer setup mgcv does
# inside estimate.gam (mgcv.r:1654-1672).
G <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
         data=df, family=fam, method="REML", fit=FALSE)
G$family <- mgcv:::fix.family(G$family)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)
G$UrS <- list()
if (length(G$S) > 0) {
  for (i in 1:length(G$S)) G$UrS[[i]] <- t(Ssp$Y) %*% G$rS[[i]]
}
G$family <- mgcv:::fix.family.ls(
              mgcv:::fix.family.var(
                mgcv:::fix.family.link(G$family)))
G$null.coef <- rep(0, ncol(G$X))

points <- list(A=c(__SP_A_1__, __SP_A_2__),
               B=c(__SP_B_1__, __SP_B_2__))
for (lab in names(points)) {
  sp_t <- points[[lab]]
  # Scale is unknown → mgcv expects sp to include log(scale) as last entry.
  # Use the profiled scale at this sp (refit with sp fixed → scale picks).
  scale_loc <- scale_at(sp_t)
  lsp <- c(log(sp_t), log(scale_loc))
  b <- mgcv:::gam.fit3(
    x=G$X, y=G$y, sp=lsp, Eb=G$Eb, UrS=G$UrS,
    offset=G$offset, U1=G$U1, Mp=G$Mp, family=G$family,
    weights=G$w, deriv=2, control=gam.control(),
    gamma=1, scale=-1, printWarn=FALSE,
    scoreType="REML", null.coef=G$null.coef, Sl=G$Sl
  )
  cat("POINT:", lab, "\n")
  cat("SP:", paste(sp_t, collapse=","), "\n")
  cat("SCALE_USED:", scale_loc, "\n")
  cat("REML:", b$REML, "\n")
  cat("REML1:", paste(b$REML1, collapse=","), "\n")
  cat("REML2:", paste(as.vector(b$REML2), collapse=","), "\n")
  cat("SCALE_EST:", b$scale.est, "\n")
  cat("END\n")
}
"""


def mgcv_eval_at_sps(x: np.ndarray, y: np.ndarray, sp_A: np.ndarray, sp_B: np.ndarray) -> dict:
    """Call mgcv:::gam.fit3(deriv=2) at sp_A and sp_B, return REML/grad/hess."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        xp = tmp / "x.csv"
        yp = tmp / "y.csv"
        np.savetxt(xp, x, delimiter=",")
        np.savetxt(yp, y, delimiter=",")

        rscript = (
            R_EVAL_SCRIPT.replace("__XPATH__", str(xp))
                         .replace("__YPATH__", str(yp))
                         .replace("__SP_A_1__", f"{sp_A[0]:.17g}")
                         .replace("__SP_A_2__", f"{sp_A[1]:.17g}")
                         .replace("__SP_B_1__", f"{sp_B[0]:.17g}")
                         .replace("__SP_B_2__", f"{sp_B[1]:.17g}")
        )
        rfile = tmp / "eval.R"
        rfile.write_text(rscript)
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(rfile)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write("---- R stdout ----\n" + proc.stdout +
                             "\n---- R stderr ----\n" + proc.stderr + "\n")
            raise RuntimeError("Rscript failed")

    results = {"full_fit": {}}
    cur = None
    for line in proc.stdout.splitlines():
        if line.startswith("MGCV_FIT_SP:"):
            results["full_fit"]["sp"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
        elif line.startswith("MGCV_FIT_REML:"):
            results["full_fit"]["reml"] = float(line.split()[1])
        elif line.startswith("MGCV_FIT_SCALE:"):
            results["full_fit"]["scale"] = float(line.split()[1])
        elif line.startswith("MGCV_FIT_GRAD:"):
            results["full_fit"]["grad"] = [
                float(v) for v in line.split(":", 1)[1].strip().split(",")
            ]
        elif line.startswith("MGCV_FIT_HESS:"):
            flat = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            m = int(np.sqrt(len(flat)))
            results["full_fit"]["hess"] = np.asarray(flat).reshape(m, m, order="F").tolist()
        elif line.startswith("POINT:"):
            cur = {"label": line.split(":", 1)[1].strip()}
        elif line.startswith("SP:"):
            cur["sp"] = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
        elif line.startswith("SCALE_USED:"):
            cur["scale_used"] = float(line.split()[1])
        elif line.startswith("REML:"):
            cur["reml"] = float(line.split()[1])
        elif line.startswith("REML1:"):
            full = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            # mgcv with scale unknown returns grad of length (m + 1):
            # last entry is d/dlog(scale). Lambda block is first m.
            cur["grad_full"] = full
            cur["grad"] = full[:-1] if len(full) > 1 else full
        elif line.startswith("REML2:"):
            flat = [float(v) for v in line.split(":", 1)[1].strip().split(",")]
            d = int(np.sqrt(len(flat)))
            H = np.asarray(flat).reshape(d, d, order="F")
            cur["hess_full"] = H.tolist()
            cur["hess"] = H[:-1, :-1].tolist() if d > 1 else H.tolist()
        elif line.startswith("SCALE_EST:"):
            cur["scale_est"] = float(line.split()[1])
        elif line.startswith("END"):
            results[cur["label"]] = cur
            cur = None
    return results


# --------------------------------------------------------------------------- #
# Rust side — fit then probe REML / grad / Hess at fixed λ in mgcv coords.    #
# --------------------------------------------------------------------------- #

def rust_fit_and_probe(x: np.ndarray, y: np.ndarray, sp_A: np.ndarray) -> dict:
    """Fit with GAMFitter; return converged λ (in mgcv coords) + REML / grad / Hess
       evaluated at A (mgcv coords, fixed) and at Rust's converged λ (B)."""
    sys.path.insert(0, str(REPO / "python"))
    from mgcv_rust import GAMFitter  # noqa: WPS433

    d = x.shape[1]
    predictors = [f"x{i}" for i in range(d)]
    fitter = GAMFitter(
        predictors=predictors,
        family="inverse.gaussian",
        link="log",
        method="REML",
        term_k_mapping={p: 10 for p in predictors},
        predictor_basis_map={p: "cr" for p in predictors},
    )
    fitter.fit(x, y)
    native = fitter._native

    # Rust's converged λ are in optimizer-scaled coords. Multiply by the
    # per-smooth scale_factor to convert to mgcv-coord (raw) λ that the
    # IFT entry points consume directly.
    rust_lam_scaled = np.asarray(native.get_all_lambdas(), dtype=float)
    scale_factors = np.asarray(native.get_penalty_scale_factors(), dtype=float)
    rust_lam_mgcv = rust_lam_scaled * scale_factors
    sp_B = rust_lam_mgcv.copy()

    y64 = y.astype(float)
    y_orig = y64  # InvGauss → pass the original y for the GLM-deviance form
    out = {
        "sp_B": sp_B.tolist(),
        "sp_B_scaled_coords": rust_lam_scaled.tolist(),
        "scale_factors": scale_factors.tolist(),
        "points": {},
    }

    # IFT grad/Hess work in raw-λ coords. The score returned by
    # evaluate_reml_mgcv_formula is in the same units. We additionally
    # transform grad/Hess from ∂/∂λ to ∂/∂log(λ) using the standard
    # chain rule:
    #   ∂f/∂log λ_j      = λ_j · ∂f/∂λ_j
    #   ∂²f/∂log λ_j λ_k = λ_j λ_k ∂²f/∂λ_j∂λ_k + δ_{jk} λ_j · ∂f/∂λ_j
    #
    # NOTE: the Rust IFT functions ALREADY return derivatives w.r.t.
    # log(λ) in mgcv's convention (gdi.c uses ρ = log λ). So we report
    # both b$REML1 and Rust grad as ∂/∂ρ and they should match without
    # any chain-rule wrapper.  (compute_reml_grad_in_logsp in
    # reml.rs:2236+ multiplies internally by λ.)  If this assumption is
    # wrong, both endpoint and ratio diagnostics below will reveal it.
    for label, sp in (("A", sp_A), ("B", sp_B)):
        lam = list(map(float, sp))
        reml = native.evaluate_reml_mgcv_formula(y64, lam)
        grad = np.asarray(
            native.evaluate_reml_gradient_ift(y64, lam, y_original=y_orig),
            dtype=float,
        )
        hess = np.asarray(
            native.evaluate_reml_hessian_ift(y64, lam, y_original=y_orig),
            dtype=float,
        )
        out["points"][label] = {
            "sp": lam,
            "reml": float(reml),
            "grad": grad.tolist(),
            "hess": hess.tolist(),
        }
    return out


# --------------------------------------------------------------------------- #
# Pretty printing.                                                            #
# --------------------------------------------------------------------------- #

def fmt_vec(v, fmt="{:+.6e}"):
    return "[" + ", ".join(fmt.format(x) for x in v) + "]"


def fmt_mat(M, fmt="{:+.4e}"):
    return "\n".join("  " + " ".join(fmt.format(x) for x in row) for row in M)


def main():
    path = FIXTURES_DIR / f"{CASE}.json"
    fixture = json.loads(path.read_text())
    x = np.asarray(fixture["inputs"]["x_train"], dtype=float)
    y = np.asarray(fixture["inputs"]["y_train"], dtype=float)

    print(f"\n{'=' * 76}")
    print(f"InvGauss-log n=800 sub-step diff — {CASE}")
    print(f"{'=' * 76}\n")

    # --- Rust fit → also returns sp_B and REML/grad/Hess at A and B ---
    print("[rust] fitting + probing…")
    rust = rust_fit_and_probe(x, y, POINT_A)
    sp_B = np.asarray(rust["sp_B"])
    print(f"  Rust converged λ (mgcv coords):   {sp_B}")
    print(f"  Rust converged λ (scaled coords): {rust['sp_B_scaled_coords']}")
    print(f"  Per-smooth scale_factor:          {rust['scale_factors']}")

    # --- R side: mgcv:::gam.fit3 at both points ---
    print("\n[mgcv] evaluating at A and B…")
    mgcv = mgcv_eval_at_sps(x, y, POINT_A, sp_B)

    # --- Side-by-side table ---
    print(f"\n{'-' * 76}")
    print(f"POINT A  = mgcv converged sp = {POINT_A.tolist()}")
    print(f"POINT B  = Rust converged sp = {sp_B.tolist()}")
    print(f"{'-' * 76}\n")

    out = {
        "case": CASE,
        "point_A": POINT_A.tolist(),
        "point_B": sp_B.tolist(),
        "scale_factors": rust["scale_factors"],
        "mgcv": mgcv,
        "rust": rust["points"],
    }

    if "full_fit" in mgcv:
        ff = mgcv["full_fit"]
        print(f"\n[mgcv full-fit sanity check] sp = {ff.get('sp')}   "
              f"REML(gcv.ubre) = {ff.get('reml')}   scale = {ff.get('scale')}")
        print(f"  outer.info$grad (3D incl. log-scale): {ff.get('grad')}")
        print()

    for label in ("A", "B"):
        m = mgcv[label]
        r = rust["points"][label]
        m_reml = m["reml"]
        r_reml = r["reml"]
        m_grad = np.asarray(m["grad"])
        r_grad = np.asarray(r["grad"])
        m_hess = np.asarray(m["hess"])
        r_hess = np.asarray(r["hess"])
        scale_used = m.get("scale_used")
        scale_str = f"{scale_used:.6f}" if scale_used is not None else "n/a"
        print(f"=== Point {label}  sp = {m['sp']}   (scale_used = {scale_str}) ===")
        print(f"  REML(gam.fit3) mgcv = {m_reml:+.6f}   rust = {r_reml:+.6f}   "
              f"Δ(rust-mgcv) = {r_reml - m_reml:+.6e}")
        print(f"  Full mgcv grad incl. log-scale: {fmt_vec(m['grad_full'])}")
        print(f"  Grad d/dlog λ  (λ-block only):")
        print(f"    mgcv: {fmt_vec(m_grad)}")
        print(f"    rust: {fmt_vec(r_grad)}")
        print(f"    Δ:    {fmt_vec(r_grad - m_grad)}")
        print(f"  Hess d²/dlog λ (λ-block only):")
        print(f"    mgcv:\n{fmt_mat(m_hess)}")
        print(f"    rust:\n{fmt_mat(r_hess)}")
        print(f"    Δ (rust − mgcv):\n{fmt_mat(r_hess - m_hess)}")
        print()

    # --- Optimum-of-Rust-criterion question ---
    print(f"{'-' * 76}")
    r_A = rust["points"]["A"]["reml"]
    r_B = rust["points"]["B"]["reml"]
    delta = r_B - r_A
    print(f"Rust REML(A) = {r_A:+.6f}   Rust REML(B) = {r_B:+.6f}   Δ = {delta:+.6e}")
    if delta < 0:
        print("=> Rust REML is LOWER at B (Rust's own λ) than at A (mgcv's λ).")
        print("   Rust IS finding a better optimum of its own criterion ⇒")
        print("   the Rust REML formula DISAGREES with mgcv's (Port 1 / Port 2).")
    else:
        print("=> Rust REML is LOWER at A (mgcv's λ) than at B (Rust's λ).")
        print("   Rust's optimizer is FAILING to find its own minimum ⇒")
        print("   trajectory issue (step-blending / Phase B).")

    # --- Dominant Hess piece ---
    print(f"\n{'-' * 76}")
    print("Hessian disagreement breakdown at A:")
    Hm = np.asarray(mgcv["A"]["hess"])
    Hr = np.asarray(rust["points"]["A"]["hess"])
    dH = Hr - Hm
    print(f"  diag(Δ):          {fmt_vec(np.diag(dH))}")
    print(f"  offdiag Δ[0,1]:   {dH[0,1]:+.4e}")
    print(f"  ‖Δ‖_F:            {np.linalg.norm(dH):.4e}")
    print(f"  ‖diag(Δ)‖:        {np.linalg.norm(np.diag(dH)):.4e}")
    print(f"  ‖offdiag(Δ)‖_F:   "
          f"{np.linalg.norm(dH - np.diag(np.diag(dH))):.4e}")

    out["analysis"] = {
        "rust_reml_A": r_A,
        "rust_reml_B": r_B,
        "rust_REML_B_minus_A": delta,
        "hess_delta_A": dH.tolist(),
        "hess_delta_A_offdiag_norm": float(
            np.linalg.norm(dH - np.diag(np.diag(dH)))
        ),
        "hess_delta_A_diag_norm": float(np.linalg.norm(np.diag(dH))),
    }

    json_path = Path("/tmp/invgauss_n800_substep_diff.json")
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
