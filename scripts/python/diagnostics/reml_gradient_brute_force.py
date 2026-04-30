"""Brute-force REML gradient reference for full validation of our IFT path.

Two cases (so we cover both Fisher and Newton branches):
  - Binomial + logit (canonical, Fisher PIRLS)
  - Gamma + log (non-canonical, Newton PIRLS)

For each case, computes the FULL ∂REML/∂ρ_k two independent ways:

  (A) Analytic per gam.fit3.r:622:
       REML1[k] = D1[k]/(2·scale) + trA1[k]/2 - rp$det1[k]/2
      where:
       D1[k]    = (∂D/∂β)' b1[:,k] + λ_k β'S_kβ + 2 (ΣλSβ)' b1[:,k]   (penalised dev derivative)
       trA1[k]  = ∂log|X'WX+S|/∂ρ_k = tr(Tk[:,k]·KK') + λ_k·tr(A⁻¹·S_k)
       det1[k]  = ∂log|S|+/∂ρ_k

  (B) Central finite difference on the REML score itself, refitting β̂(ρ) at each ρ±h.

If (A) ≈ (B), the analytic decomposition is correct. Then by validating
each piece (tk_kkt, lam_k·tr_a_inv_s, d1_k) against this reference in Rust,
we can pinpoint any mismatch.
"""

import sys
import numpy as np
import scipy.linalg as sla


# ----------------------------------------------------------------------------
# Family abstractions
# ----------------------------------------------------------------------------
class Family:
    canonical: bool

    def link(self, mu): ...
    def inverse_link(self, eta): ...
    def variance(self, mu): ...
    def saturated_loglik(self, y): ...
    def family_terms(self, mu):
        """Return mgcv-scaled derivatives (g1 raw, g2/g3 normalized; V0 raw, V1/V2 normalized)."""
        ...


class BinomialLogit(Family):
    canonical = True

    def inverse_link(self, eta):
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    def variance(self, mu):
        return mu * (1.0 - mu)

    def saturated_loglik(self, y):
        # dbinom(y, 1, y) where y∈{0,1} gives 0; entropy term otherwise.
        out = np.zeros_like(y)
        m = (y > 0) & (y < 1)
        out[m] = y[m] * np.log(y[m]) + (1 - y[m]) * np.log(1 - y[m])
        return out.sum()

    def family_terms(self, mu):
        # logit: g'=1/(μ(1-μ)), g''=(2μ-1)/(μ²(1-μ)²), g'''=2/μ³+2/(1-μ)³
        # V'=1-2μ, V''=-2
        g1 = 1.0 / (mu * (1.0 - mu))
        g2_raw = (2 * mu - 1) / (mu**2 * (1 - mu) ** 2)
        g3_raw = 2.0 / mu**3 + 2.0 / (1 - mu) ** 3
        V0 = mu * (1.0 - mu)
        V1_raw = 1.0 - 2 * mu
        V2_raw = -2.0
        # Normalised
        g2 = g2_raw / g1
        g3 = g3_raw / g1
        V1 = V1_raw / V0
        V2 = V2_raw / V0
        dmu_deta = 1.0 / g1
        return g1, g2, g3, V0, V1, V2, dmu_deta

    def deviance(self, y, mu):
        # Binomial deviance: 2·Σ [y log(y/μ) + (1-y) log((1-y)/(1-μ))]
        eps = 1e-300
        a = np.where(y > 0, y * np.log(y.clip(eps) / mu.clip(eps)), 0.0)
        b = np.where(y < 1, (1 - y) * np.log((1 - y).clip(eps) / (1 - mu).clip(eps)), 0.0)
        return 2.0 * (a + b).sum()


class GammaLog(Family):
    canonical = False

    def inverse_link(self, eta):
        return np.exp(np.clip(eta, -30, 30))

    def variance(self, mu):
        return mu**2

    def saturated_loglik(self, y):
        # We don't need this for the gradient comparison since we'll use the
        # working-deviance form. Returning 0 keeps the score formula cleanly
        # additive — but we'll reuse the Gamma+log script's reference values.
        return 0.0

    def family_terms(self, mu):
        g1 = 1.0 / mu
        g2 = -1.0 / mu
        g3 = 2.0 / mu**2
        V0 = mu**2
        V1 = 2.0 / mu
        V2 = 2.0 / mu**2
        dmu_deta = mu
        return g1, g2, g3, V0, V1, V2, dmu_deta

    def deviance(self, y, mu):
        # Gamma deviance: 2·Σ [-log(y/μ) + (y-μ)/μ]
        return 2.0 * (-np.log(np.clip(y / mu, 1e-300, None)) + (y - mu) / mu).sum()


# ----------------------------------------------------------------------------
# PIRLS (Fisher for canonical, Newton for non-canonical) by Newton on penalised
# deviance (the Hessian uses the appropriate weight type).
# ----------------------------------------------------------------------------
def fit_inner_pirls(family, X, y, S_list, log_lambda, beta0=None, tol=1e-12, max_iter=300):
    n, p = X.shape
    M = len(S_list)
    lam = np.exp(log_lambda)
    S_lambda = sum(l * S for l, S in zip(lam, S_list))

    if beta0 is None:
        # Cold start: regress link(y_safe) on X.
        if isinstance(family, BinomialLogit):
            ys = np.clip(y, 0.05, 0.95)
            z = np.log(ys / (1 - ys))
        else:  # GammaLog
            z = np.log(np.maximum(y, 1e-3))
        beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(p) + S_lambda, X.T @ z)
    else:
        beta = beta0.copy()

    def pen_dev(beta_):
        eta_ = X @ beta_
        mu_ = family.inverse_link(eta_)
        return family.deviance(y, mu_) + beta_ @ S_lambda @ beta_

    def step_terms(beta_):
        eta_ = X @ beta_
        mu_ = family.inverse_link(eta_)
        g1, g2, g3, V0, V1, V2, dmu_deta = family.family_terms(mu_)
        if family.canonical:
            alpha_ = np.ones_like(mu_)
        else:
            alpha_ = 1.0 + (y - mu_) * (V1 + g2)
        # Newton weight (full): α·(dμ/dη)²/V₀; for canonical α=1 → Fisher.
        w_ = alpha_ * dmu_deta**2 / V0
        # Penalised deviance gradient.
        # ∂D_unpen/∂β = -2 X' (y-μ) / (V·g')  -- mgcv convention.
        grad_unpen = -2.0 * X.T @ ((y - mu_) * dmu_deta / V0)
        grad_pen = grad_unpen + 2.0 * S_lambda @ beta_
        H = 2.0 * X.T @ (w_[:, None] * X) + 2.0 * S_lambda
        return mu_, eta_, alpha_, w_, grad_pen, H

    for it in range(max_iter):
        mu, eta, alpha, w, grad, H = step_terms(beta)
        gn = np.linalg.norm(grad)
        if gn < tol:
            break
        try:
            step = sla.solve(H, grad, assume_a="sym")
        except sla.LinAlgError:
            step = sla.solve(H + 1e-6 * np.eye(p), grad, assume_a="sym")
        # Backtracking on penalised deviance.
        f0 = pen_dev(beta)
        t = 1.0
        for _ in range(60):
            cand = beta - t * step
            f1 = pen_dev(cand)
            if np.isfinite(f1) and f1 <= f0 - 1e-4 * t * (grad @ step):
                break
            t *= 0.5
        beta = beta - t * step

    mu, eta, alpha, w, grad, H = step_terms(beta)
    return beta, mu, eta, alpha, w, S_lambda, grad, H, lam


# ----------------------------------------------------------------------------
# REML score (mgcv-form) and analytic gradient
# ----------------------------------------------------------------------------
def reml_score(family, X, y, S_list, beta, mu, w, S_lambda, lam, scale_known=None):
    """REML at converged β̂(ρ).

    REML = D/(2σ²) - ls + (1/2)log|H| - (1/2)log|S|+ - (Mp/2)log(2π·σ²)
    For known scale (binomial): σ² = 1, no Mp/2 log term in mgcv (gam.fit3.r:622).
    For Gamma: σ² estimated from D/(n - tr(A)) — but here we test gradient at fixed β,
    so we use the Pearson-style σ² estimate from mgcv: phi = D/(n - Mp).
    """
    n, p = X.shape
    A = X.T @ (w[:, None] * X) + S_lambda
    sign, logdet_H = np.linalg.slogdet(A)
    # log|S|+ = sum log of nonzero eigvals of S_lambda
    eigS = np.linalg.eigvalsh(S_lambda)
    eigS_pos = eigS[eigS > 1e-12 * (eigS.max() if eigS.size else 1.0)]
    logdet_S_plus = np.log(eigS_pos).sum()
    Mp = p - len(eigS_pos)  # nullspace dim

    D_unpen = family.deviance(y, mu)
    bSb = beta @ S_lambda @ beta
    D_pen = D_unpen + bSb
    ls = family.saturated_loglik(y)

    if scale_known is not None:
        scale = scale_known
        Mp_term = 0.0  # binomial/poisson: no σ² log term in REML formula per mgcv
    else:
        scale = D_pen / max(n - Mp, 1.0)
        Mp_term = (Mp / 2.0) * np.log(2.0 * np.pi * scale)

    reml = D_pen / (2.0 * scale) - ls + 0.5 * logdet_H - 0.5 * logdet_S_plus - Mp_term
    return reml, scale, logdet_H, logdet_S_plus


def analytic_gradient(family, X, y, S_list, beta, mu, alpha, w, S_lambda, lam, scale_known=None):
    """Compute ∂REML/∂ρ_k via the IFT-based analytic formula.
    Returns dict with all components for inspection."""
    n, p = X.shape
    M = len(S_list)
    A = X.T @ (w[:, None] * X) + S_lambda
    Ainv = sla.inv(A)
    g1, g2n, g3n, V0, v1n, v2n, dmu_deta = family.family_terms(mu)

    # b1 = -λ_k A⁻¹ S_k β
    b1 = np.zeros((p, M))
    for k in range(M):
        b1[:, k] = -lam[k] * Ainv @ (S_list[k] @ beta)
    eta1 = X @ b1

    # leverage h[i] = w[i] · x_i' A⁻¹ x_i
    lev_uw = np.einsum("ij,jk,ik->i", X, Ainv, X)
    h = w * lev_uw

    # a1 = dw/dη
    if family.canonical:
        a1 = -w * (v1n + 2.0 * g2n) / g1
    else:
        xx = v2n - v1n**2 + g3n - g2n**2
        alpha1 = (-(v1n + g2n) + (y - mu) * xx) / alpha
        a1 = w * (alpha1 - v1n - 2.0 * g2n) / g1

    # tk_kkt[k] = Σᵢ a1[i]·η1[i,k]·sign(w)·lev_uw[i] = Σᵢ Tk[i,k]·h[i]
    tk_kkt = np.zeros(M)
    for k in range(M):
        tk_kkt[k] = np.sum(a1 * eta1[:, k] * np.sign(w) * lev_uw)

    # tr(A⁻¹·S_k)
    tr_ainv_sk = np.array([np.trace(Ainv @ S_list[k]) for k in range(M)])

    # Penalised deviance gradient: dev_grad·b1 + λ_k β'S_kβ + 2(ΣλSβ)'b1
    dev_grad = -2.0 * X.T @ ((y - mu) * dmu_deta / V0)  # ∂D_unpen/∂β
    sum_lambda_S_beta = sum(lam[j] * S_list[j] @ beta for j in range(M))
    bsb_per_k = np.array([beta @ S_list[k] @ beta for k in range(M)])

    d1 = np.zeros(M)
    for k in range(M):
        dev_dot_b1 = dev_grad @ b1[:, k]
        sls_dot_b1 = sum_lambda_S_beta @ b1[:, k]
        d1[k] = dev_dot_b1 + lam[k] * bsb_per_k[k] + 2.0 * sls_dot_b1

    # rp$det1: ∂log|S|+/∂ρ_k. For penalty only: |S_lambda(ρ)| = ∏ eigvals.
    # Use FD on log|S|+ (cheap and stable).
    h_fd = 1e-5
    rp_det1 = np.zeros(M)
    for k in range(M):
        Sp = sum((lam[j] * (np.exp(h_fd) if j == k else 1.0)) * S_list[j] for j in range(M))
        Sm = sum((lam[j] * (np.exp(-h_fd) if j == k else 1.0)) * S_list[j] for j in range(M))
        ep = np.linalg.eigvalsh(Sp); ep = ep[ep > 1e-12 * ep.max()]
        em = np.linalg.eigvalsh(Sm); em = em[em > 1e-12 * em.max()]
        rp_det1[k] = (np.log(ep).sum() - np.log(em).sum()) / (2 * h_fd)

    # scale handling
    Mp = p - len(np.linalg.eigvalsh(S_lambda)[np.linalg.eigvalsh(S_lambda) > 1e-12 * np.linalg.eigvalsh(S_lambda).max()])
    if scale_known is not None:
        scale = scale_known
    else:
        scale = (family.deviance(y, mu) + beta @ S_lambda @ beta) / max(n - Mp, 1.0)

    # rank_k for the penalty K — # of positive eigvals of S_k.
    rank_k = np.zeros(M, dtype=int)
    for k in range(M):
        ev = np.linalg.eigvalsh(S_list[k])
        rank_k[k] = int((ev > 1e-12 * ev.max()).sum())

    # Final: REML1 = D1/(2 scale) + (tk_kkt + λ_k·tr(Ainv S_k))/2 - rp_det1/2
    grad = np.zeros(M)
    for k in range(M):
        grad[k] = d1[k] / (2.0 * scale) + (tk_kkt[k] + lam[k] * tr_ainv_sk[k]) / 2.0 - rp_det1[k] / 2.0

    return {
        "grad": grad,
        "d1": d1,
        "tk_kkt": tk_kkt,
        "lam_tr_ainv_sk": lam * tr_ainv_sk,
        "rp_det1": rp_det1,
        "rank_k": rank_k,
        "scale": scale,
        "a1": a1,
        "h": h,
        "lev_uw": lev_uw,
        "Ainv": Ainv,
        "b1": b1,
        "eta1": eta1,
    }


# ----------------------------------------------------------------------------
# FD-on-score reference
# ----------------------------------------------------------------------------
def fd_gradient(family, X, y, S_list, log_lambda, beta_warm, scale_known=None, h_fd=5e-5):
    M = len(S_list)
    grad = np.zeros(M)
    for k in range(M):
        rp = log_lambda.copy(); rp[k] += h_fd
        rm = log_lambda.copy(); rm[k] -= h_fd
        bp, mp, ep, ap, wp, S_p, *_ = fit_inner_pirls(family, X, y, S_list, rp, beta0=beta_warm, tol=1e-13)
        bm, mm, em, am, wm, S_m, *_ = fit_inner_pirls(family, X, y, S_list, rm, beta0=beta_warm, tol=1e-13)
        sp_score, _, _, _ = reml_score(family, X, y, S_list, bp, mp, wp, S_p, np.exp(rp), scale_known=scale_known)
        sm_score, _, _, _ = reml_score(family, X, y, S_list, bm, mm, wm, S_m, np.exp(rm), scale_known=scale_known)
        grad[k] = (sp_score - sm_score) / (2 * h_fd)
    return grad


# ----------------------------------------------------------------------------
# Synthetic problem
# ----------------------------------------------------------------------------
def make_problem(family_cls, seed=0, n=30, p=8):
    rng = np.random.default_rng(seed)
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    B1 = np.column_stack([x1, x1**2, x1**3, np.sin(2.0 * x1)])
    B2 = np.column_stack([x2, x2**2, np.cos(1.5 * x2)])
    intercept = np.ones((n, 1))
    X = np.column_stack([intercept, B1, B2])

    def diff2(d):
        D = np.zeros((d - 1, d))
        for i in range(d - 1):
            D[i, i] = 1.0; D[i, i + 1] = -1.0
        return D.T @ D
    S1 = np.zeros((p, p)); S1[1:5, 1:5] = diff2(4)
    S2 = np.zeros((p, p)); S2[5:8, 5:8] = diff2(3)

    if family_cls is BinomialLogit:
        beta_true = np.array([0.2, 0.4, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
        eta_true = X @ beta_true
        mu_true = 1.0 / (1.0 + np.exp(-eta_true))
        y = (rng.uniform(size=n) < mu_true).astype(float)
    else:  # GammaLog
        beta_true = np.array([0.6, 0.4, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
        eta_true = X @ beta_true
        mu_true = np.exp(eta_true)
        y = rng.gamma(shape=4.0, scale=mu_true / 4.0)

    return X, y, [S1, S2], beta_true


# ----------------------------------------------------------------------------
def run_case(family_cls, scale_known, label):
    print(f"\n{'='*78}\nCase: {label}\n{'='*78}")
    family = family_cls()
    X, y, S_list, _ = make_problem(family_cls)
    n, p = X.shape
    M = len(S_list)
    log_lambda = np.array([0.0, 1.0])

    beta, mu, eta, alpha, w, S_lambda, grad_pirls, H, lam = fit_inner_pirls(
        family, X, y, S_list, log_lambda
    )
    print(f"PIRLS converged: ||grad||_inf = {np.max(np.abs(grad_pirls)):.3e}")
    print(f"min(w)={w.min():.3e}, max(w)={w.max():.3e}, neg_w? {bool((w<0).any())}")

    out = analytic_gradient(family, X, y, S_list, beta, mu, alpha, w, S_lambda, lam, scale_known=scale_known)
    fd_grad = fd_gradient(family, X, y, S_list, log_lambda, beta, scale_known=scale_known, h_fd=5e-5)

    print(f"\n--- Per-component analytic decomposition ---")
    for k in range(M):
        print(f"k={k}:")
        print(f"  d1[k]                = {out['d1'][k]:+.10e}")
        print(f"  tk_kkt[k]            = {out['tk_kkt'][k]:+.10e}")
        print(f"  lam[k]·tr(Ainv·S_k)  = {out['lam_tr_ainv_sk'][k]:+.10e}")
        print(f"  rp$det1[k]           = {out['rp_det1'][k]:+.10e}")
        print(f"  rank_k               = {out['rank_k'][k]}")
        print(f"  scale                = {out['scale']:.10e}")

    print(f"\n--- Final REML gradient: analytic vs FD ---")
    for k in range(M):
        diff = out["grad"][k] - fd_grad[k]
        rel = abs(diff) / max(abs(fd_grad[k]), 1e-12)
        print(f"k={k}: analytic={out['grad'][k]:+.10e}, FD={fd_grad[k]:+.10e}, |diff|={abs(diff):.3e}, rel={rel:.3e}")

    print(f"\nbeta_hat = {beta.tolist()}")
    return out, fd_grad, beta, w, mu, alpha


def main():
    np.set_printoptions(precision=8, suppress=False, linewidth=140)
    # Binomial+logit (canonical, scale=1)
    out_bin, fd_bin, beta_bin, w_bin, mu_bin, alpha_bin = run_case(
        BinomialLogit, scale_known=1.0, label="Binomial+logit (canonical, Fisher PIRLS, σ²=1)"
    )
    # Gamma+log (non-canonical, scale estimated from D/(n-Mp))
    out_gam, fd_gam, beta_gam, w_gam, mu_gam, alpha_gam = run_case(
        GammaLog, scale_known=None, label="Gamma+log (non-canonical, Newton PIRLS, σ² profiled)"
    )


if __name__ == "__main__":
    main()
