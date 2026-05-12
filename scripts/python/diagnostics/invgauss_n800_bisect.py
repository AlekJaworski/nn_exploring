"""Systematic bisection: where does Rust diverge from mgcv on InvGauss-n800?

At freshly-fit β at mgcv-converged sp = (192.16, 504.45) on the
2d_invgauss_log_n800_k10_cr fixture, walk through every quantity in mgcv's
REML formula and find the first one that disagrees.

The strategy is "compute it in Python from Rust primitives". Rust exposes
X (design), β (PIRLS-converged), W_pirls (Fisher fallback). We rebuild
W_newton, A_pirls, A_newton, b1, η₁, a1, lev_uw, λ·tr(A⁻¹·S_k), tk_kkt
in Python. We extract the analogous quantities from mgcv via trace() of
gam.fit3. Then we diff piece-by-piece.

mgcv REML formula (gam.fit3.r:621):
    REML = Dp/(2σ²) − ls[1] + log|H|/2 − log|S+|/2 − Mp/2·log(2πσ²)
    REML1[k] = oo$D1[k]/(2σ²) + oo$trA1[k]/2 − rp$det1[k]/2
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/alex/vibe_coding/nn_exploring")
FIXTURE = REPO / "tests" / "parity" / "fixtures" / "2d_invgauss_log_n800_k10_cr.json"
POINT_A = [192.16284812521303, 504.4534182343682]

sys.path.insert(0, str(REPO / "python"))
from mgcv_rust import GAMFitter  # noqa: E402


# ---- mgcv side: trace() everything we need at sp = point A ---------------- #

R_DUMP = r"""
suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))
fixture <- fromJSON("__FIXTURE__")
x <- as.matrix(fixture$inputs$x_train)
y <- as.numeric(fixture$inputs$y_train)
df <- data.frame(x0=x[,1], x1=x[,2], y=y)
fam <- inverse.gaussian(link="log")
sp_A <- c(__SP1__, __SP2__)

# Fit fixed-sp to get reml.scale and the converged β
gp <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
          data=df, family=fam, method="REML", sp=sp_A)
scale_A <- gp$reml.scale

# Setup G for gam.fit3 with same preamble as estimate.gam
G <- gam(y ~ s(x0, k=10, bs="cr") + s(x1, k=10, bs="cr"),
         data=df, family=fam, method="REML", fit=FALSE)
G$family <- mgcv:::fix.family(G$family)
G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)
Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))
G$Eb <- Ssp$E
G$U1 <- cbind(Ssp$Y, Ssp$Z)
G$Mp <- ncol(Ssp$Z)
G$UrS <- list(); if (length(G$S) > 0) for (i in 1:length(G$S)) G$UrS[[i]] <- t(Ssp$Y) %*% G$rS[[i]]
G$family <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(G$family)))
G$null.coef <- rep(0, ncol(G$X))
lsp <- c(log(sp_A), log(scale_A))

trace(mgcv:::gam.fit3, exit = quote({
  assign("DUMP_oo", oo, envir=.GlobalEnv)
  assign("DUMP_rp", rp, envir=.GlobalEnv)
  assign("DUMP_scale", scale, envir=.GlobalEnv)
  assign("DUMP_dev", dev, envir=.GlobalEnv)
  assign("DUMP_Dp", Dp, envir=.GlobalEnv)
  assign("DUMP_Mp", Mp, envir=.GlobalEnv)
  assign("DUMP_ls", ls, envir=.GlobalEnv)
  if (exists("w")) assign("DUMP_w", w, envir=.GlobalEnv)
  if (exists("z")) assign("DUMP_z", z, envir=.GlobalEnv)
  if (exists("mu")) assign("DUMP_mu", mu, envir=.GlobalEnv)
  if (exists("eta")) assign("DUMP_eta", eta, envir=.GlobalEnv)
  if (exists("coef")) assign("DUMP_coef", coef, envir=.GlobalEnv)
}), print=FALSE)

b <- mgcv:::gam.fit3(x=G$X, y=G$y, sp=lsp, Eb=G$Eb, UrS=G$UrS,
                     offset=G$offset, U1=G$U1, Mp=G$Mp, family=G$family,
                     weights=G$w, deriv=1, control=gam.control(),
                     gamma=1, scale=-1, printWarn=FALSE,
                     scoreType="REML", null.coef=G$null.coef, Sl=G$Sl)
untrace(mgcv:::gam.fit3)

# Save key quantities to JSON for Python consumption
out <- list(
  scale = DUMP_scale,
  Mp = DUMP_Mp,
  ls = DUMP_ls,
  dev = DUMP_dev,
  Dp = DUMP_Dp,
  oo_D1 = DUMP_oo$D1,
  oo_trA1 = DUMP_oo$trA1,
  oo_rank_tol = DUMP_oo$rank.tol,
  rp_det = DUMP_rp$det,
  rp_det1 = DUMP_rp$det1,
  REML = b$REML,
  REML1 = b$REML1,
  beta = b$coefficients,
  # also dump gam G$X for an exact basis comparison
  gp_coef = gp$coefficients,
  gp_dev = gp$deviance,
  gp_sig2 = gp$sig2,
  gp_w = gp$weights,
  gp_z = gp$y - gp$linear.predictors,  # working residual proxy
  gp_eta = as.numeric(gp$linear.predictors),
  gp_mu = as.numeric(gp$fitted.values),
  gp_working_weights = gp$prior.weights * (1 / gp$family$variance(gp$fitted.values)) * (gp$family$mu.eta(gp$linear.predictors))^2,
  # X and S blocks
  G_X = G$X,
  G_S = G$S,
  G_off = G$off,
  G_w = G$w
)
writeLines(toJSON(out, digits=NA, auto_unbox=FALSE), "__OUTPATH__")
"""


def mgcv_dump() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_json = tmp / "mgcv.json"
        rscript = (
            R_DUMP.replace("__FIXTURE__", str(FIXTURE))
            .replace("__SP1__", f"{POINT_A[0]:.17g}")
            .replace("__SP2__", f"{POINT_A[1]:.17g}")
            .replace("__OUTPATH__", str(out_json))
        )
        rfile = tmp / "dump.R"
        rfile.write_text(rscript)
        proc = subprocess.run(
            ["Rscript", "--vanilla", str(rfile)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write("R stdout:\n" + proc.stdout + "\nR stderr:\n" + proc.stderr + "\n")
            raise RuntimeError("R script failed")
        return json.loads(out_json.read_text())


# ---- Rust side: extract design + fit at sp_A, return primitives ----------- #

def rust_dump():
    fixture = json.loads(FIXTURE.read_text())
    x_in = np.asarray(fixture["inputs"]["x_train"], dtype=float)
    y_in = np.asarray(fixture["inputs"]["y_train"], dtype=float)
    fitter = GAMFitter(
        predictors=["x0", "x1"], family="inverse.gaussian", link="log", method="REML",
        term_k_mapping={"x0": 10, "x1": 10},
        predictor_basis_map={"x0": "cr", "x1": "cr"},
    )
    fitter.fit(x_in, y_in)
    native = fitter._native
    X = np.asarray(native.get_design_matrix(), dtype=float)
    fresh = native.evaluate_reml_at_sp_freshly_fit(y_in, POINT_A)
    smooth_penalties = [np.asarray(p) for p in native.get_smooth_penalties()]
    return {
        "X": X,
        "y_raw": y_in,
        "fresh": fresh,
        "smooth_penalties": smooth_penalties,
    }


# ---- Python: rebuild W_newton, A, b1, η₁, a1, lev_uw, tk_kkt ------------- #

def newton_weights_invgauss_log(y, mu):
    """For InvGauss + log:
       wf = 1/μ
       g₂n = -1/μ
       v1n = 3/μ
       α_raw = 1 + (y - μ)·(v1n + g₂n) = 1 + (y - μ)·2/μ = 2y/μ - 1
       W_newton = wf · α_raw = (1/μ)·(2y/μ - 1) = 2y/μ² - 1/μ
    """
    return 2.0 * y / (mu ** 2) - 1.0 / mu


def fmt(arr, n=5, prec=6):
    if np.ndim(arr) == 0:
        return f"{arr:.{prec}f}"
    a = np.asarray(arr).ravel()
    if len(a) <= 2 * n:
        return "[" + ", ".join(f"{v:+.{prec}e}" for v in a) + "]"
    return "[" + ", ".join(f"{v:+.{prec}e}" for v in a[:n]) + " ... " + ", ".join(f"{v:+.{prec}e}" for v in a[-n:]) + "]"


def main():
    print("=" * 80)
    print("BISECTION: InvGauss+log n=800 at mgcv-λ — find first divergence")
    print("=" * 80)

    print("\n[1/2] Dumping mgcv internals via trace()...")
    mg = mgcv_dump()
    print("[2/2] Dumping Rust primitives + fresh PIRLS fit at sp_A...")
    ru = rust_dump()

    # mgcv arrays
    G_X = np.asarray(mg["G_X"], dtype=float)
    G_S = [np.asarray(S, dtype=float) for S in mg["G_S"]]
    G_off = np.asarray(mg["G_off"], dtype=int)
    beta_mg = np.asarray(mg["beta"], dtype=float)
    gp_coef = np.asarray(mg["gp_coef"], dtype=float)
    gp_eta = np.asarray(mg["gp_eta"], dtype=float)
    gp_mu = np.asarray(mg["gp_mu"], dtype=float)
    gp_w = np.asarray(mg["gp_working_weights"], dtype=float)
    oo_D1 = np.asarray(mg["oo_D1"], dtype=float)
    oo_trA1 = np.asarray(mg["oo_trA1"], dtype=float)
    rp_det1 = np.asarray(mg["rp_det1"], dtype=float)
    scale_mg = float(np.asarray(mg["scale"]).item())
    Mp_mg = int(np.asarray(mg["Mp"]).item())
    dev_mg = float(np.asarray(mg["dev"]).item())
    REML_mg = float(np.asarray(mg["REML"]).item())
    rank_tol_mg = float(np.asarray(mg["oo_rank_tol"]).item())

    # Rust arrays
    X_ru = ru["X"]
    y_raw = ru["y_raw"]
    fresh = ru["fresh"]
    beta_ru = np.asarray(fresh["beta"], dtype=float)
    w_pirls = np.asarray(fresh["weights"], dtype=float)
    z_pirls = np.asarray(fresh["working_response"], dtype=float)
    dev_ru = float(fresh["deviance"])
    reml_ru = float(fresh["reml"])
    grad_ru = np.asarray(fresh["grad"], dtype=float)
    s_blocks_ru = ru["smooth_penalties"]

    print()
    print("--- 0. Design / β / dev / scale -----------------------------------")
    print(f"  Rust X shape: {X_ru.shape}    mgcv G$X shape: {G_X.shape}")
    print(f"  Rust ‖β‖:  {np.linalg.norm(beta_ru):.6f}")
    print(f"  mgcv ‖β‖:  {np.linalg.norm(beta_mg):.6f}")
    print(f"  Rust dev:  {dev_ru:.6f}")
    print(f"  mgcv dev:  {dev_mg:.6f}")
    print(f"  Rust REML: {reml_ru:.6f}")
    print(f"  mgcv REML: {REML_mg:.6f}    Δ = {reml_ru - REML_mg:+.6f}")
    print(f"  Rust scale_est used internally: see below")
    print(f"  mgcv scale: {scale_mg:.6f}")
    print(f"  mgcv Mp = {Mp_mg}")

    # Basis-compare: X vs G_X
    # Bases may differ by rotation. Compare column spans first.
    print(f"\n  X mismatch (Rust X − mgcv G$X) — element 0,0: {X_ru[0, 0]} vs {G_X[0, 0]}")
    # Rebuild Rust's η = X·β  vs mgcv's gp$linear.predictor
    eta_ru = X_ru @ beta_ru
    print(f"  Rust η[:3]:  {eta_ru[:3]}")
    print(f"  mgcv η[:3]:  {gp_eta[:3]}")
    print(f"  ‖η_ru − η_mg‖∞: {np.max(np.abs(eta_ru - gp_eta)):.6e}")

    print()
    print("--- 1. Working weights at converged η -----------------------------")
    mu_ru = np.exp(eta_ru)
    w_newton_py = newton_weights_invgauss_log(y_raw, mu_ru)
    print(f"  Rust  W_pirls (Fisher-fallback): min={np.min(w_pirls):.4e} max={np.max(w_pirls):.4e} #neg={int(np.sum(w_pirls<0))}")
    print(f"  Rust  W_newton (Python rebuild): min={np.min(w_newton_py):.4e} max={np.max(w_newton_py):.4e} #neg={int(np.sum(w_newton_py<0))}")
    print(f"  mgcv  gp$working.weights:        min={np.min(gp_w):.4e} max={np.max(gp_w):.4e} #neg={int(np.sum(gp_w<0))}")
    print(f"  W_newton ≈ mgcv working weights?  ‖Δ‖∞ = {np.max(np.abs(w_newton_py - gp_w)):.4e}")
    print(f"  W_pirls vs mgcv:                  ‖Δ‖∞ = {np.max(np.abs(w_pirls - gp_w)):.4e}")

    # The bisection: if mgcv working weights have negs but PIRLS doesn't, that's
    # the divergence root.

    print()
    print("--- 2. A = X'WX + ΣλS  →  log|H| ----------------------------------")
    n, p = X_ru.shape
    # Build S blocks at full p×p
    if len(s_blocks_ru) == 2:
        # Smooth offsets in Rust design: 1, 1+nb1
        nb_list = [pb.shape[0] for pb in s_blocks_ru]
        offs = [1, 1 + nb_list[0]]
    else:
        raise RuntimeError("expected 2 smooth penalties")

    def expand_S(p, off, block):
        full = np.zeros((p, p))
        full[off:off + block.shape[0], off:off + block.shape[1]] = block
        return full

    S0_full = expand_S(p, offs[0], s_blocks_ru[0])
    S1_full = expand_S(p, offs[1], s_blocks_ru[1])
    lams = POINT_A
    # Newton-based A
    XtWnX = (X_ru * w_newton_py[:, None]).T @ X_ru
    A_newton = XtWnX + lams[0] * S0_full + lams[1] * S1_full
    # PIRLS-based A
    XtWpX = (X_ru * w_pirls[:, None]).T @ X_ru
    A_pirls = XtWpX + lams[0] * S0_full + lams[1] * S1_full
    sign_n, logabs_n = np.linalg.slogdet(A_newton)
    sign_p, logabs_p = np.linalg.slogdet(A_pirls)
    print(f"  Rust log|det(A_pirls)|  = {sign_p:+.0f} × {logabs_p:.6f}")
    print(f"  Rust log|det(A_newton)| = {sign_n:+.0f} × {logabs_n:.6f}")
    print(f"  mgcv oo$rank.tol         = {rank_tol_mg:.6f}    (mgcv's log|H| for REML formula)")
    # Which matches?
    print(f"  match (A_newton)?  Δ = {logabs_n - rank_tol_mg:+.6e}")
    print(f"  match (A_pirls)?   Δ = {logabs_p - rank_tol_mg:+.6e}")

    print()
    print("--- 3. λ_k · tr(A⁻¹ · S_k)  →  appears in trA1[k] ----------------")
    Ainv_newton = np.linalg.inv(A_newton)
    Ainv_pirls = np.linalg.inv(A_pirls)
    tr_AinvS_newton = [lams[k] * np.trace(Ainv_newton @ S_full) for k, S_full in enumerate([S0_full, S1_full])]
    tr_AinvS_pirls = [lams[k] * np.trace(Ainv_pirls @ S_full) for k, S_full in enumerate([S0_full, S1_full])]
    print(f"  Rust  λ·tr(A_pirls⁻¹·S_k)   = {tr_AinvS_pirls}")
    print(f"  Rust  λ·tr(A_newton⁻¹·S_k)  = {tr_AinvS_newton}")
    # mgcv's λ·tr(P'S_k P) is hidden in oo$trA1 but we can compute it from
    # G's X/S directly (mgcv uses A_newton too).
    G_S_full = []
    for i, S in enumerate(G_S):
        full = np.zeros((G_X.shape[1], G_X.shape[1]))
        off = int(G_off[i] if hasattr(G_off, '__len__') else G_off) - 1
        full[off:off + S.shape[0], off:off + S.shape[1]] = S
        G_S_full.append(full)
    G_A_newton = (G_X * gp_w[:, None]).T @ G_X + lams[0] * G_S_full[0] + lams[1] * G_S_full[1]
    G_Ainv_newton = np.linalg.inv(G_A_newton)
    tr_AinvS_mgcv = [lams[k] * np.trace(G_Ainv_newton @ G_S_full[k]) for k in range(2)]
    print(f"  mgcv  λ·tr(A_newton⁻¹·S_k)  = {tr_AinvS_mgcv}")
    print(f"  match with Rust A_newton? Δ = {[tr_AinvS_newton[k] - tr_AinvS_mgcv[k] for k in range(2)]}")
    print(f"  match with Rust A_pirls?  Δ = {[tr_AinvS_pirls[k] - tr_AinvS_mgcv[k] for k in range(2)]}")

    print()
    print("--- 4. tk_kkt[k] = oo$trA1[k] - λ_k·tr(A⁻¹·S_k)  ------------------")
    tk_kkt_mgcv_eff = [oo_trA1[k] - tr_AinvS_mgcv[k] for k in range(2)]
    print(f"  mgcv  tk_kkt[k] (back-out):  {tk_kkt_mgcv_eff}")
    # Rebuild Rust's tk_kkt in Python from primitives
    # a1 for InvGauss+log Newton:
    #   v1n = 3/μ, g2n = -1/μ, v2n = 6/μ², g3n = 2/μ²
    #   c_resid = y - μ
    #   α_raw = 1 + c·(v1n + g2n) = 1 + c·2/μ = 2y/μ - 1
    #   α = α_raw if α_raw > 0 else 1
    #   xx = v2n - v1n² + g3n - g2n² = 6/μ² - 9/μ² + 2/μ² - 1/μ² = -2/μ²
    #   α1 = (-(v1n+g2n) + c·xx)/α = (-2/μ + (y-μ)·(-2/μ²))/α = (-2/μ - 2y/μ² + 2/μ)/α = -2y/μ²/α
    #   g1 = 1/(dμ/dη) = 1/μ
    #   a1 = w · (α1 - v1n - 2g2n)/g1 = w · (α1 - 3/μ + 2/μ)·μ = w·(α1 - 1/μ)·μ
    def rebuild_a1(w, y, mu):
        v1n = 3.0 / mu
        g2n = -1.0 / mu
        v2n = 6.0 / (mu ** 2)
        g3n = 2.0 / (mu ** 2)
        c = y - mu
        alpha_raw = 1.0 + c * (v1n + g2n)
        alpha = np.where(alpha_raw <= 0.0, 1.0, alpha_raw)
        xx = v2n - v1n ** 2 + g3n - g2n ** 2
        alpha1 = (-(v1n + g2n) + c * xx) / alpha
        # g1 = 1 / (dμ/dη) = 1/μ
        a1 = w * (alpha1 - v1n - 2.0 * g2n) / (1.0 / mu)
        return a1

    a1_pirls = rebuild_a1(w_pirls, y_raw, mu_ru)
    a1_newton = rebuild_a1(w_newton_py, y_raw, mu_ru)
    # mgcv α (no clamping done by us — mgcv uses raw α too?)
    a1_mgcv = rebuild_a1(gp_w, y_raw, gp_mu)

    # b1 and η₁: b1[:,k] = -λ_k · A⁻¹ · S_k · β,  η₁[:,k] = X · b1[:,k]
    def compute_eta1(X, Ainv, beta, lams, S_blocks):
        eta1 = np.zeros((X.shape[0], len(lams)))
        for k in range(len(lams)):
            b1_k = -lams[k] * Ainv @ (S_blocks[k] @ beta)
            eta1[:, k] = X @ b1_k
        return eta1

    eta1_pirls = compute_eta1(X_ru, Ainv_pirls, beta_ru, lams, [S0_full, S1_full])
    eta1_newton = compute_eta1(X_ru, Ainv_newton, beta_ru, lams, [S0_full, S1_full])
    eta1_mgcv = compute_eta1(G_X, G_Ainv_newton, gp_coef, lams, G_S_full)

    # lev_uw[i] = x_i' A⁻¹ x_i
    XAi_pirls = X_ru @ Ainv_pirls
    lev_uw_pirls = np.einsum("ij,ij->i", XAi_pirls, X_ru)
    XAi_newton = X_ru @ Ainv_newton
    lev_uw_newton = np.einsum("ij,ij->i", XAi_newton, X_ru)
    GXAi = G_X @ G_Ainv_newton
    lev_uw_mgcv = np.einsum("ij,ij->i", GXAi, G_X)

    # tk_kkt in three flavours: rust-pirls (current Rust formula),
    # rust-newton-py (what we think mgcv computes), and direct mgcv via back-out.
    for k in range(2):
        # Current Rust formula (Fisher-fallback W, with sign_w factor)
        tk_rust = np.sum(a1_pirls * eta1_pirls[:, k] * np.sign(w_pirls) * lev_uw_pirls)
        # gdi.c formula with Newton W and no sign_w
        tk_newton = np.sum(a1_newton * eta1_newton[:, k] * lev_uw_newton)
        # mgcv via independent rebuild
        tk_mgcv_rb = np.sum(a1_mgcv * eta1_mgcv[:, k] * lev_uw_mgcv)
        print(f"  k={k}:")
        print(f"    Rust formula (pirls W + sign_w):  {tk_rust:+.6e}")
        print(f"    Newton W, no sign_w:              {tk_newton:+.6e}")
        print(f"    mgcv rebuild (gp$w):              {tk_mgcv_rb:+.6e}")
        print(f"    mgcv back-out (oo$trA1−λtr):      {tk_kkt_mgcv_eff[k]:+.6e}")

    print()
    print("--- 5. Decompose total trA1[k] and gradient by step ---------------")
    REML1_mgcv = np.asarray(mg["REML1"])[:2]
    print(f"  mgcv REML1[k] (λ-block):    {REML1_mgcv.tolist()}")
    print(f"  Rust grad[k] (our formula): {grad_ru.tolist()}")

    print()
    print("=" * 80)
    print("CONCLUSION: which step first diverges? Look at sections 1, 2, 3, 4.")
    print("=" * 80)


if __name__ == "__main__":
    main()
