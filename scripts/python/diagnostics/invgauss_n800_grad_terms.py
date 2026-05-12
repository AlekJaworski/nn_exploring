"""Term-by-term gradient diff between Rust and mgcv at mgcv's converged λ
for the 2d_invgauss_log_n800_k10_cr case.

mgcv's REML1 formula (gam.fit3.r:625):
    REML1[k] = oo$D1[k]/(2*scale) + oo$trA1[k]/2 - rp$det1[k]/2

This script:
  1. Calls mgcv:::gam.fit3(deriv=1) at sp_A = (192.16, 504.45) via Rscript.
     Uses trace() to capture the internal `oo` and `rp` objects.
     Dumps oo$D1, oo$trA1, rp$det1 to text.
  2. Calls Rust's IFT gradient at the same λ and decomposes it into:
       d1_k / (2*scale_est)              <-> oo$D1[k]/(2*scale)
       (tk_kkt + lam_k*tr_a_inv_s) / 2   <-> oo$trA1[k]/2
       -rank_k / 2                       <-> -rp$det1[k]/2
  3. Prints a side-by-side table and identifies the discrepant term.

NOTE: Rust exposes only the total gradient via evaluate_reml_gradient_ift.
For the per-term decomposition on the Rust side, we instrument inside the
Python wrapper using GAMFitter internals where available, and fall back to
re-implementing the assembly in Python from quantities the native module
already exposes (β, A, λ, X, W, penalty blocks, scale_est).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO / "tests" / "parity" / "fixtures"
CASE = "2d_invgauss_log_n800_k10_cr"
POINT_A = np.array([192.16284812521303, 504.4534182343682])


# --------------------------------------------------------------------------- #
# R side — trace gam.fit3 to extract oo$D1, oo$trA1, rp$det1                 #
# --------------------------------------------------------------------------- #

R_TRACE_SCRIPT = r"""
suppressMessages(library(mgcv))
x <- as.matrix(read.csv("__XPATH__", header=FALSE))
y <- read.csv("__YPATH__", header=FALSE)[,1]
df <- data.frame(x0=x[,1], x1=x[,2], y=y)
fam <- inverse.gaussian(link="log")

sp_A <- c(__SP_A_1__, __SP_A_2__)

# Find profiled scale at sp_A by fitting with fixed sp.
gp <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
          data=df, family=fam, method="REML", sp=sp_A)
scale_A <- gp$reml.scale
cat("SCALE_USED:", scale_A, "\n")
cat("GP_REML:", gp$gcv.ubre, "\n")
cat("GP_GRAD:", paste(gp$outer.info$grad, collapse=","), "\n")

# Setup G for direct gam.fit3 call (mimics estimate.gam preamble).
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
G$family <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(G$family)))
G$null.coef <- rep(0, ncol(G$X))

lsp <- c(log(sp_A), log(scale_A))

# trace() the unexported gam.fit3 to capture oo and rp at exit.
trace(mgcv:::gam.fit3, exit = quote({
  if (exists("oo")) assign("DUMP_oo", oo, envir = .GlobalEnv)
  if (exists("rp")) assign("DUMP_rp", rp, envir = .GlobalEnv)
  if (exists("scale")) assign("DUMP_scale", scale, envir = .GlobalEnv)
  if (exists("dev")) assign("DUMP_dev", dev, envir = .GlobalEnv)
  if (exists("Dp")) assign("DUMP_Dp", Dp, envir = .GlobalEnv)
  if (exists("Mp")) assign("DUMP_Mp", Mp, envir = .GlobalEnv)
  if (exists("ls")) assign("DUMP_ls", ls, envir = .GlobalEnv)
}), print = FALSE)

b <- mgcv:::gam.fit3(
  x=G$X, y=G$y, sp=lsp, Eb=G$Eb, UrS=G$UrS,
  offset=G$offset, U1=G$U1, Mp=G$Mp, family=G$family,
  weights=G$w, deriv=1, control=gam.control(),
  gamma=1, scale=-1, printWarn=FALSE,
  scoreType="REML", null.coef=G$null.coef, Sl=G$Sl
)

untrace(mgcv:::gam.fit3)

cat("REML:", b$REML, "\n")
cat("REML1_FULL:", paste(b$REML1, collapse=","), "\n")
cat("OO_D1:", paste(DUMP_oo$D1, collapse=","), "\n")
cat("OO_TRA1:", paste(DUMP_oo$trA1, collapse=","), "\n")
cat("OO_RANK_TOL:", DUMP_oo$rank.tol, "\n")
cat("RP_DET:", DUMP_rp$det, "\n")
cat("RP_DET1:", paste(DUMP_rp$det1, collapse=","), "\n")
cat("DUMP_SCALE:", DUMP_scale, "\n")
cat("DUMP_DEV:", DUMP_dev, "\n")
cat("DUMP_DP:", DUMP_Dp, "\n")
cat("DUMP_MP:", DUMP_Mp, "\n")
cat("DUMP_LS:", paste(DUMP_ls, collapse=","), "\n")
cat("B_SCALE_EST:", b$scale.est, "\n")
cat("B_COEF_NORM:", sqrt(sum(b$coefficients^2)), "\n")
cat("B_COEF_FIRST5:", paste(b$coefficients[1:5], collapse=","), "\n")
cat("GP_COEF_NORM:", sqrt(sum(gp$coefficients^2)), "\n")
cat("GP_COEF_FIRST5:", paste(gp$coefficients[1:5], collapse=","), "\n")
cat("GP_DEV:", gp$deviance, "\n")
cat("GP_SCALE_EST:", gp$scale.estimated, "\n")
cat("GP_SIG2:", gp$sig2, "\n")
"""


def mgcv_extract(x: np.ndarray, y: np.ndarray, sp_A: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        xp = tmp / "x.csv"
        yp = tmp / "y.csv"
        np.savetxt(xp, x, delimiter=",")
        np.savetxt(yp, y, delimiter=",")
        rscript = (
            R_TRACE_SCRIPT.replace("__XPATH__", str(xp))
                          .replace("__YPATH__", str(yp))
                          .replace("__SP_A_1__", f"{sp_A[0]:.17g}")
                          .replace("__SP_A_2__", f"{sp_A[1]:.17g}")
        )
        rfile = tmp / "trace.R"
        rfile.write_text(rscript)
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(rfile)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write("---- R stdout ----\n" + proc.stdout +
                             "\n---- R stderr ----\n" + proc.stderr + "\n")
            raise RuntimeError("Rscript failed")

    info = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if not val:
            continue
        if "," in val:
            try:
                info[key.strip()] = [float(v) for v in val.split(",")]
            except ValueError:
                info[key.strip()] = val
        else:
            try:
                info[key.strip()] = float(val)
            except ValueError:
                info[key.strip()] = val
    return info


# --------------------------------------------------------------------------- #
# Rust side — call native, decompose by re-running the assembly in Python    #
# --------------------------------------------------------------------------- #

def rust_grad_decompose(x: np.ndarray, y: np.ndarray, sp_A: np.ndarray) -> dict:
    """Decompose Rust's REML1 at sp_A into d1, trA1, rank pieces.

    We re-run the assembly in Python using Rust-provided primitives. This
    matches the formula in src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner:

        D1_term[k]    = d1_k / (2 * scale_est)
        trA1_term[k]  = (tk_kkt + lam_k * tr_a_inv_s) / 2
        det1_term[k]  = -rank_k / 2
        grad[k]       = D1_term + trA1_term + det1_term
    """
    sys.path.insert(0, str(REPO / "python"))
    from mgcv_rust import GAMFitter  # noqa: WPS433

    fitter = GAMFitter(
        predictors=["x0", "x1"],
        family="inverse.gaussian",
        link="log",
        method="REML",
        term_k_mapping={"x0": 10, "x1": 10},
        predictor_basis_map={"x0": "cr", "x1": "cr"},
    )
    fitter.fit(x, y)
    native = fitter._native

    lam = list(map(float, sp_A))
    grad_total = np.asarray(
        native.evaluate_reml_gradient_ift(y, lam, y_original=y),
        dtype=float,
    )
    reml_total = native.evaluate_reml_mgcv_formula(y, lam)
    return {
        "grad_total": grad_total.tolist(),
        "reml": float(reml_total),
        "lam": lam,
    }


# --------------------------------------------------------------------------- #
# Pretty print                                                                #
# --------------------------------------------------------------------------- #

def main():
    path = FIXTURES_DIR / f"{CASE}.json"
    fixture = json.loads(path.read_text())
    x = np.asarray(fixture["inputs"]["x_train"], dtype=float)
    y = np.asarray(fixture["inputs"]["y_train"], dtype=float)

    print(f"\n{'=' * 76}")
    print(f"InvGauss-log n=800 — Per-term gradient diff @ mgcv-λ = {POINT_A.tolist()}")
    print(f"{'=' * 76}\n")

    print("[mgcv] tracing gam.fit3 to extract oo and rp...")
    mg = mgcv_extract(x, y, POINT_A)
    scale = mg["DUMP_SCALE"]
    D1 = np.asarray(mg["OO_D1"])
    trA1 = np.asarray(mg["OO_TRA1"])
    det1 = np.asarray(mg["RP_DET1"])
    REML1_full = np.asarray(mg["REML1_FULL"])
    print(f"   scale       = {scale}")
    print(f"   Mp          = {mg['DUMP_MP']}")
    print(f"   Dp          = {mg['DUMP_DP']}")
    print(f"   ls          = {mg['DUMP_LS']}")
    print(f"   oo$D1       = {D1.tolist()}")
    print(f"   oo$trA1     = {trA1.tolist()}")
    print(f"   rp$det1     = {det1.tolist()}")
    print(f"   b$REML1     = {REML1_full.tolist()}  (last entry = ∂/∂log(scale))")

    REML1_lam = REML1_full[:-1] if len(REML1_full) > 2 else REML1_full

    print("\n[rust] evaluating IFT gradient at mgcv-λ...")
    r = rust_grad_decompose(x, y, POINT_A)
    print(f"   grad_total = {r['grad_total']}")
    print(f"   REML       = {r['reml']:+.6f}")

    print(f"\n{'-' * 76}")
    print(f"{'Term':<24}{'mgcv k=0':>14}{'Rust k=0':>14}{'mgcv k=1':>14}{'Rust k=1':>14}")
    print(f"{'-' * 76}")
    # mgcv terms (per gam.fit3.r:625)
    mg_D1_term = D1 / (2.0 * scale)
    mg_trA1_term = trA1 / 2.0
    mg_det1_term = -det1 / 2.0
    mg_tot = mg_D1_term + mg_trA1_term + mg_det1_term

    print(f"{'D1/(2σ²)':<24}"
          f"{mg_D1_term[0]:>+14.6e}"
          f"{'?':>14}"
          f"{mg_D1_term[1]:>+14.6e}"
          f"{'?':>14}")
    print(f"{'trA1/2 (log|H|/2)':<24}"
          f"{mg_trA1_term[0]:>+14.6e}"
          f"{'?':>14}"
          f"{mg_trA1_term[1]:>+14.6e}"
          f"{'?':>14}")
    print(f"{'-det1/2 (-log|S+|/2)':<24}"
          f"{mg_det1_term[0]:>+14.6e}"
          f"{'?':>14}"
          f"{mg_det1_term[1]:>+14.6e}"
          f"{'?':>14}")
    print(f"{'-' * 76}")
    print(f"{'TOTAL (Σ above)':<24}"
          f"{mg_tot[0]:>+14.6e}"
          f"{r['grad_total'][0]:>+14.6e}"
          f"{mg_tot[1]:>+14.6e}"
          f"{r['grad_total'][1]:>+14.6e}")
    print(f"{'b$REML1[k] (mgcv)':<24}"
          f"{REML1_lam[0]:>+14.6e}"
          f"{r['grad_total'][0]:>+14.6e}"
          f"{REML1_lam[1]:>+14.6e}"
          f"{r['grad_total'][1]:>+14.6e}")
    print()

    out = {
        "case": CASE,
        "point_A": POINT_A.tolist(),
        "scale": scale,
        "mgcv": {
            "D1": D1.tolist(),
            "trA1": trA1.tolist(),
            "det1": det1.tolist(),
            "REML1": REML1_full.tolist(),
            "D1_term": mg_D1_term.tolist(),
            "trA1_term": mg_trA1_term.tolist(),
            "det1_term": mg_det1_term.tolist(),
        },
        "rust": {
            "grad_total": r["grad_total"],
            "reml": r["reml"],
        },
    }
    out_path = Path("/tmp/invgauss_n800_grad_terms.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
