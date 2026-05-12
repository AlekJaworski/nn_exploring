"""Systematic bisection: where does Rust diverge from mgcv on 4d_binomial_logit_n2000?

At freshly-fit β at mgcv-converged sp on the 4d_binomial_logit_n2000_k8_cr fixture,
walk through every quantity in mgcv's REML formula and find the first one that
disagrees.

This is the canonical-link sibling of `invgauss_n800_bisect.py`. Because Binomial+logit
is canonical, Newton ≡ Fisher and W ≥ 0 everywhere — the InvGauss-era Newton-W /
sign(w) bugs cannot bite here. The remaining suspects from the predecessor note are:
  (a) stable similarity basis (gam.reparam) — `MGCV_REPARAM=1` flag
  (b) saturating-λ handling on the 4th smooth (λ ≈ 22118)
  (c) outer-Newton convergence criterion / step direction near the saturating coord.

mgcv REML formula (gam.fit3.r:621):
    REML = Dp/(2σ²) − ls[1] + log|H|/2 − log|S+|/2 − Mp/2·log(2πσ²)
    REML1[k] = oo$D1[k]/(2σ²) + oo$trA1[k]/2 − rp$det1[k]/2

For Binomial scale ≡ 1 (fixed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/alex/vibe_coding/nn_exploring")
FIXTURE = REPO / "tests" / "parity" / "fixtures" / "4d_binomial_logit_n2000_k8_cr.json"

# mgcv-converged λ from the fixture
POINT_A = [18.172327330552687, 3659.557823778686, 1051.0303173526806, 22117.877681649614]

sys.path.insert(0, str(REPO / "python"))
from mgcv_rust import GAMFitter  # noqa: E402


R_DUMP = r"""
suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))
fixture <- fromJSON("__FIXTURE__")
x <- as.matrix(fixture$inputs$x_train)
y <- as.numeric(fixture$inputs$y_train)
df <- data.frame(x0=x[,1], x1=x[,2], x2=x[,3], x3=x[,4], y=y)
fam <- binomial(link="logit")
sp_A <- c(__SP1__, __SP2__, __SP3__, __SP4__)

# Fit fixed-sp to get the converged β
gp <- gam(y ~ s(x0, k=8, bs="cr") + s(x1, k=8, bs="cr") +
              s(x2, k=8, bs="cr") + s(x3, k=8, bs="cr"),
          data=df, family=fam, method="REML", sp=sp_A)
scale_A <- 1.0  # Binomial: scale fixed at 1

# Setup G for gam.fit3 with same preamble as estimate.gam
G <- gam(y ~ s(x0, k=8, bs="cr") + s(x1, k=8, bs="cr") +
             s(x2, k=8, bs="cr") + s(x3, k=8, bs="cr"),
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
# Binomial: scale is fixed at 1, so no log-scale appended to lsp.
lsp <- log(sp_A)

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
                     gamma=1, scale=1, printWarn=FALSE,
                     scoreType="REML", null.coef=G$null.coef, Sl=G$Sl)
untrace(mgcv:::gam.fit3)

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
  gp_coef = gp$coefficients,
  gp_dev = gp$deviance,
  gp_sig2 = gp$sig2,
  gp_w = gp$weights,
  gp_eta = as.numeric(gp$linear.predictors),
  gp_mu = as.numeric(gp$fitted.values),
  gp_working_weights = gp$prior.weights * (1 / gp$family$variance(gp$fitted.values)) * (gp$family$mu.eta(gp$linear.predictors))^2,
  G_X = G$X,
  G_S = G$S,
  G_off = G$off,
  G_w = G$w,
  G_rank = G$rank
)
writeLines(toJSON(out, digits=NA, auto_unbox=FALSE), "__OUTPATH__")
"""


def mgcv_dump() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_json = tmp / "mgcv.json"
        rscript = R_DUMP.replace("__FIXTURE__", str(FIXTURE))
        for i, v in enumerate(POINT_A, start=1):
            rscript = rscript.replace(f"__SP{i}__", f"{v:.17g}")
        rscript = rscript.replace("__OUTPATH__", str(out_json))
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


def rust_dump():
    fixture = json.loads(FIXTURE.read_text())
    x_in = np.asarray(fixture["inputs"]["x_train"], dtype=float)
    y_in = np.asarray(fixture["inputs"]["y_train"], dtype=float)
    fitter = GAMFitter(
        predictors=["x0", "x1", "x2", "x3"],
        family="binomial", link="logit", method="REML",
        term_k_mapping={f"x{i}": 8 for i in range(4)},
        predictor_basis_map={f"x{i}": "cr" for i in range(4)},
    )
    fitter.fit(x_in, y_in)
    native = fitter._native
    X = np.asarray(native.get_design_matrix(), dtype=float)
    fresh = native.evaluate_reml_at_sp_freshly_fit(y_in, POINT_A)
    smooth_penalties = [np.asarray(p) for p in native.get_smooth_penalties()]
    rust_lams = list(native.get_all_lambdas())
    return {
        "X": X,
        "y_raw": y_in,
        "fresh": fresh,
        "smooth_penalties": smooth_penalties,
        "rust_lambdas": rust_lams,
    }


def main():
    print("=" * 80)
    print("BISECTION: 4d_binomial_logit n=2000 at mgcv-λ — find first divergence")
    print("=" * 80)
    print(f"  fixture: {FIXTURE.name}")
    print(f"  mgcv-converged λ: {POINT_A}")

    print("\n[1/2] Dumping mgcv internals via trace()...")
    mg = mgcv_dump()
    print("[2/2] Dumping Rust primitives + fresh PIRLS fit at sp_A...")
    ru = rust_dump()

    # mgcv arrays
    G_X = np.asarray(mg["G_X"], dtype=float)
    G_S = [np.asarray(S, dtype=float) for S in mg["G_S"]]
    G_off_raw = mg["G_off"]
    G_off = np.atleast_1d(np.asarray(G_off_raw, dtype=int))
    beta_mg = np.asarray(mg["beta"], dtype=float)
    gp_coef = np.asarray(mg["gp_coef"], dtype=float)
    gp_eta = np.asarray(mg["gp_eta"], dtype=float)
    gp_mu = np.asarray(mg["gp_mu"], dtype=float)
    gp_w = np.asarray(mg["gp_working_weights"], dtype=float)
    oo_D1 = np.asarray(mg["oo_D1"], dtype=float)
    oo_trA1 = np.asarray(mg["oo_trA1"], dtype=float)
    rp_det = float(np.asarray(mg["rp_det"]).item())
    rp_det1 = np.asarray(mg["rp_det1"], dtype=float)
    Mp_mg = int(np.asarray(mg["Mp"]).item())
    dev_mg = float(np.asarray(mg["dev"]).item())
    REML_mg = float(np.asarray(mg["REML"]).item())
    REML1_mg = np.asarray(mg["REML1"], dtype=float)
    rank_tol_mg = float(np.asarray(mg["oo_rank_tol"]).item())

    # Rust arrays
    X_ru = ru["X"]
    y_raw = ru["y_raw"]
    fresh = ru["fresh"]
    beta_ru = np.asarray(fresh["beta"], dtype=float)
    w_pirls = np.asarray(fresh["weights"], dtype=float)
    dev_ru = float(fresh["deviance"])
    reml_ru = float(fresh["reml"])
    grad_ru = np.asarray(fresh["grad"], dtype=float)
    s_blocks_ru = ru["smooth_penalties"]
    rust_lams = ru["rust_lambdas"]

    print()
    print("--- 0. Design / β / dev / scale -----------------------------------")
    print(f"  Rust X shape: {X_ru.shape}    mgcv G$X shape: {G_X.shape}")
    print(f"  Rust ‖β‖:  {np.linalg.norm(beta_ru):.6f}")
    print(f"  mgcv ‖β‖:  {np.linalg.norm(beta_mg):.6f}")
    print(f"  Rust dev:  {dev_ru:.6f}")
    print(f"  mgcv dev:  {dev_mg:.6f}")
    print(f"  Rust REML: {reml_ru:.6f}")
    print(f"  mgcv REML: {REML_mg:.6f}    Δ = {reml_ru - REML_mg:+.6e}")
    print(f"  mgcv Mp = {Mp_mg}")
    print(f"  Rust optimizer-converged λ: {rust_lams}")
    print(f"  mgcv             converged λ: {POINT_A}")
    print(f"  rel-λ Δ: {[abs(rust_lams[k] - POINT_A[k])/POINT_A[k] for k in range(len(POINT_A))]}")

    print(f"  ‖X_rust − G$X‖∞: {np.max(np.abs(X_ru - G_X)):.6e}")
    eta_ru = X_ru @ beta_ru
    print(f"  ‖η_ru − η_mg‖∞: {np.max(np.abs(eta_ru - gp_eta)):.6e}")
    print(f"  ‖β_ru − β_mg‖∞: {np.max(np.abs(beta_ru - gp_coef)):.6e}")

    print()
    print("--- 1. Working weights at converged η (canonical: W = μ(1−μ)) ----")
    mu_ru = 1.0 / (1.0 + np.exp(-eta_ru))
    w_canonical_py = mu_ru * (1.0 - mu_ru)
    print(f"  Rust  W_pirls (Fisher = Newton, canonical):  min={np.min(w_pirls):.4e} max={np.max(w_pirls):.4e}")
    print(f"  Python W = μ(1−μ) (canonical):               min={np.min(w_canonical_py):.4e} max={np.max(w_canonical_py):.4e}")
    print(f"  mgcv  gp$working.weights:                    min={np.min(gp_w):.4e} max={np.max(gp_w):.4e}")
    print(f"  W_pirls vs μ(1−μ):  ‖Δ‖∞ = {np.max(np.abs(w_pirls - w_canonical_py)):.4e}")
    print(f"  W_pirls vs mgcv:    ‖Δ‖∞ = {np.max(np.abs(w_pirls - gp_w)):.4e}")

    print()
    print("--- 2. A = X'WX + ΣλS  →  log|H| ----------------------------------")
    n, p = X_ru.shape
    # Build full S blocks at p×p (offsets from mgcv's G$off, 1-indexed)
    # We use mgcv's G_X / G_S to compute a reference; we use Rust X with Rust S blocks.
    # First: Rust S offsets — the smooth.rs places them at fixed positions in the design.
    # The fixture has 4 smooths each k=8 with cr basis; each contributes (k-1)=7 columns of penalty.
    nb_list = [pb.shape[0] for pb in s_blocks_ru]
    print(f"  Rust smooth penalty sizes: {nb_list}    sum={sum(nb_list)}    p={p}")
    # First column is intercept; smooths come after. Compute offsets.
    rust_offs = [1]
    for nb in nb_list[:-1]:
        rust_offs.append(rust_offs[-1] + nb)
    print(f"  Rust offsets in design: {rust_offs}")

    def expand_S(p, off, block):
        full = np.zeros((p, p))
        full[off:off + block.shape[0], off:off + block.shape[1]] = block
        return full

    S_full_ru = [expand_S(p, rust_offs[k], s_blocks_ru[k]) for k in range(len(s_blocks_ru))]
    lams = POINT_A
    XtWX = (X_ru * w_pirls[:, None]).T @ X_ru
    A_ru = XtWX + sum(lams[k] * S_full_ru[k] for k in range(len(lams)))
    sign_n, logabs_n = np.linalg.slogdet(A_ru)
    print(f"  Rust log|det(A)| (PIRLS W)  = {sign_n:+.0f} × {logabs_n:.6f}")
    # mgcv side via independent rebuild
    G_S_full = []
    for i, S in enumerate(G_S):
        full = np.zeros((G_X.shape[1], G_X.shape[1]))
        off = int(G_off[i]) - 1
        full[off:off + S.shape[0], off:off + S.shape[1]] = S
        G_S_full.append(full)
    G_A = (G_X * gp_w[:, None]).T @ G_X + sum(lams[k] * G_S_full[k] for k in range(len(lams)))
    sign_m, logabs_m = np.linalg.slogdet(G_A)
    print(f"  mgcv log|det(A)| (rebuild) = {sign_m:+.0f} × {logabs_m:.6f}    Δ = {logabs_n - logabs_m:+.6e}")
    print(f"  mgcv oo$rank.tol           = {rank_tol_mg:.6f}")

    print()
    print("--- 3. λ_k · tr(A⁻¹ · S_k) ---------------------------------------")
    Ainv_ru = np.linalg.inv(A_ru)
    G_Ainv = np.linalg.inv(G_A)
    tr_AinvS_ru = [lams[k] * np.trace(Ainv_ru @ S_full_ru[k]) for k in range(len(lams))]
    tr_AinvS_mg = [lams[k] * np.trace(G_Ainv @ G_S_full[k]) for k in range(len(lams))]
    for k in range(len(lams)):
        print(f"  k={k}: Rust = {tr_AinvS_ru[k]:+.6e}    mgcv = {tr_AinvS_mg[k]:+.6e}    Δ = {tr_AinvS_ru[k] - tr_AinvS_mg[k]:+.3e}")

    print()
    print("--- 4. tk_kkt[k] = oo$trA1[k] - λ_k·tr(A⁻¹·S_k) (canonical: a1 = 0) ---")
    # For canonical link Binomial+logit, the gdi.c Newton-extra term collapses:
    #   v1n = (1-2μ)/(μ(1-μ));  g2n = -(1-2μ)/(μ(1-μ));  v1n + g2n = 0
    # so α_raw = 1, α1 vanishes, and a1 ∝ (α1 - v1n - 2g2n) reduces to a1 = w·(-v1n - 2g2n)/g1
    # which equals -w·(v1n + 2g2n)·μ(1-μ) = -w·(-(1-2μ)/(μ(1-μ)))·μ(1-μ) = w·(1-2μ)... actually let's
    # just rebuild it directly. The key point is that for canonical link, the Hessian's
    # outer-derivative-of-W term is small but not strictly zero.
    def rebuild_a1_binomial(w, y, mu):
        v1n = (1.0 - 2.0 * mu) / (mu * (1.0 - mu))
        g2n = -(1.0 - 2.0 * mu) / (mu * (1.0 - mu))
        v2n = ((1.0 - 2.0 * mu) ** 2 + 2.0 * mu * (1.0 - mu)) / (mu * (1.0 - mu)) ** 2
        g3n = (1.0 - 2.0 * mu) ** 2 / (mu * (1.0 - mu)) ** 2 + 2.0  # placeholder
        c = y - mu
        # For canonical link, v1n + g2n = 0 so α_raw == 1
        alpha_raw = 1.0 + c * (v1n + g2n)
        alpha = np.where(alpha_raw <= 0.0, 1.0, alpha_raw)
        xx = v2n - v1n ** 2 + g3n - g2n ** 2
        alpha1 = (-(v1n + g2n) + c * xx) / alpha
        # g1 = 1 / (dμ/dη) = 1/(μ(1-μ))
        g1 = 1.0 / (mu * (1.0 - mu))
        a1 = w * (alpha1 - v1n - 2.0 * g2n) / g1
        return a1

    a1_ru = rebuild_a1_binomial(w_pirls, y_raw, mu_ru)
    a1_mg = rebuild_a1_binomial(gp_w, y_raw, gp_mu)
    print(f"  a1 (canonical link): max|a1_ru| = {np.max(np.abs(a1_ru)):.3e}    max|a1_mg| = {np.max(np.abs(a1_mg)):.3e}")
    print(f"  → For canonical, a1 should still be ~tiny but contribute to tk_kkt at saturating-λ")

    # η₁[:,k] = X·b1[:,k] = X·(-λ·A⁻¹·S_k·β)
    def compute_eta1(X, Ainv, beta, lams, S_blocks):
        eta1 = np.zeros((X.shape[0], len(lams)))
        for k in range(len(lams)):
            b1_k = -lams[k] * Ainv @ (S_blocks[k] @ beta)
            eta1[:, k] = X @ b1_k
        return eta1

    eta1_ru = compute_eta1(X_ru, Ainv_ru, beta_ru, lams, S_full_ru)
    eta1_mg = compute_eta1(G_X, G_Ainv, gp_coef, lams, G_S_full)

    # lev_uw[i] = x_i' A⁻¹ x_i
    XAi_ru = X_ru @ Ainv_ru
    lev_ru = np.einsum("ij,ij->i", XAi_ru, X_ru)
    GXAi = G_X @ G_Ainv
    lev_mg = np.einsum("ij,ij->i", GXAi, G_X)
    print(f"  lev_uw range: Rust [{np.min(lev_ru):.3e}, {np.max(lev_ru):.3e}]   mgcv [{np.min(lev_mg):.3e}, {np.max(lev_mg):.3e}]")

    tk_kkt_mg_eff = [oo_trA1[k] - tr_AinvS_mg[k] for k in range(len(lams))]
    print()
    print("  Per-smooth tk_kkt comparison:")
    for k in range(len(lams)):
        tk_newton = np.sum(a1_ru * eta1_ru[:, k] * lev_ru)
        tk_mg_rb = np.sum(a1_mg * eta1_mg[:, k] * lev_mg)
        print(f"  k={k}:")
        print(f"    Rust rebuild (canon W, no sign_w):  {tk_newton:+.6e}")
        print(f"    mgcv rebuild:                       {tk_mg_rb:+.6e}")
        print(f"    mgcv back-out (oo$trA1−λtr):        {tk_kkt_mg_eff[k]:+.6e}")
        print(f"    Δ rebuild-vs-back-out (mgcv):       {tk_mg_rb - tk_kkt_mg_eff[k]:+.3e}")

    print()
    print("--- 5. Total gradient (REML1) -------------------------------------")
    print(f"  mgcv oo$D1[k]:    {oo_D1.tolist()}")
    print(f"  mgcv oo$trA1[k]:  {oo_trA1.tolist()}")
    print(f"  mgcv rp$det1[k]:  {rp_det1.tolist()}")
    # REML1[k] = oo$D1[k]/(2σ²) + oo$trA1[k]/2 − rp$det1[k]/2  (binomial: σ² = 1)
    reml1_rebuild = [oo_D1[k] / 2.0 + oo_trA1[k] / 2.0 - rp_det1[k] / 2.0 for k in range(len(lams))]
    print(f"  REML1[k] rebuild: {reml1_rebuild}")
    print(f"  mgcv REML1[k]:    {REML1_mg.tolist()}")
    print()
    print(f"  Rust grad[k]:     {grad_ru.tolist()}")
    print(f"  mgcv REML1[k]:    {REML1_mg.tolist()}")
    print(f"  Δ per smooth:     {[grad_ru[k] - REML1_mg[k] for k in range(len(lams))]}")

    print()
    print("=" * 80)
    print("CONCLUSION:")
    print(f"  REML(Rust at mgcv-λ) - REML(mgcv at mgcv-λ) = {reml_ru - REML_mg:+.3e}")
    print(f"  ‖grad_rust − grad_mgcv‖∞ at mgcv-λ = {np.max(np.abs(grad_ru - REML1_mg)):.3e}")
    print(f"  ‖β_rust − β_mgcv‖∞ at mgcv-λ        = {np.max(np.abs(beta_ru - gp_coef)):.3e}")
    print("Look at section [0,1,2,3,5] for first systematic divergence.")
    print("=" * 80)


if __name__ == "__main__":
    main()
