"""Brute-force reference for the `tr(Tk[:,k] · K K')` term that mgcv adds
to `det1[k]` at gdi.c:857 (full-Newton, non-canonical link).

Self-contained reference for validating a Rust implementation. Builds a
tiny Gamma+log GLM (n=30, p=8, M=2, both penalties rank-deficient by 1),
fits the inner PIRLS by full-Newton on penalised deviance to ||grad||<1e-10,
then checks:

  (a) leverage h[i] = diag(K K')[i] from two routes:
        - direct:  h[i] = w[i] * x[i]' * (X'WX + S)^-1 * x[i]
        - aug-QR:  K = top n rows of Q in qr([sqrt(W) X; E]) where E'E = S
  (b) Tk[i,k] from analytic formulas for Gamma+log
  (c) the gradient term  det1_addition[k] = sum_i Tk[i,k] * h[i]
  (d) cross-check via central finite difference on log|A(rho)|.

Notation and formulas follow gdi.c (mgcv); see lines 2548 (alpha1) and 2556
(a1 = dw/d eta).
"""

import numpy as np
import scipy.linalg as sla


# ----------------------------------------------------------------------------
# Family: Gamma with log link
# ----------------------------------------------------------------------------
# mu = exp(eta), so eta = log(mu)
# g'(mu) = 1/mu, g''(mu) = -1/mu^2, g'''(mu) = 2/mu^3
# Variance: V(mu) = mu^2, V'(mu) = 2 mu, V''(mu) = 2
# d mu / d eta = mu (since eta = log mu)


def family_terms(mu):
    """Returns the *mgcv-scaled* derivatives:
       g1 = g'(mu)              (raw)
       g2 = g''(mu)/g'(mu)
       g3 = g'''(mu)/g'(mu)
       V0 = V(mu)               (raw)
       V1 = V'(mu)/V(mu)
       V2 = V''(mu)/V(mu)
       dmu_deta = 1/g1
    For Gamma+log:
       g'(mu) = 1/mu, g''(mu) = -1/mu^2, g'''(mu) = 2/mu^3
       V = mu^2, V' = 2 mu, V'' = 2
    Hence g2 = -1/mu, g3 = 2/mu^2; V1 = 2/mu, V2 = 2/mu^2.
    """
    g1 = 1.0 / mu
    g2 = -1.0 / mu                # = (-1/mu^2) / (1/mu)
    g3 = 2.0 / mu ** 2            # = (2/mu^3) / (1/mu)
    V0 = mu ** 2
    V1 = 2.0 / mu                 # = (2 mu) / (mu^2)
    V2 = 2.0 / mu ** 2            # = 2 / (mu^2)
    dmu_deta = mu                 # log link
    return g1, g2, g3, V0, V1, V2, dmu_deta


# ----------------------------------------------------------------------------
# Synthetic problem
# ----------------------------------------------------------------------------
def make_problem(seed=0, n=30, p=8):
    rng = np.random.default_rng(seed)
    # Smooth-like design: two cubic-ish bases concatenated, plus an intercept.
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)

    # Block 1: 4 basis cols on x1; Block 2: 3 basis cols on x2; plus intercept.
    B1 = np.column_stack([x1, x1 ** 2, x1 ** 3, np.sin(2.0 * x1)])  # p1=4
    B2 = np.column_stack([x2, x2 ** 2, np.cos(1.5 * x2)])           # p2=3
    intercept = np.ones((n, 1))
    X = np.column_stack([intercept, B1, B2])
    assert X.shape == (n, p)

    # Penalties: each on its own block, second-difference-like (rank deficient).
    # S1 acts on cols 1..4, S2 on cols 5..7. Build small p1xp1 and p2xp2
    # rank-(p-1) penalty matrices, embed.
    def diff2_penalty(d):
        D = np.zeros((d - 1, d))
        for i in range(d - 1):
            D[i, i] = 1.0
            D[i, i + 1] = -1.0
        return D.T @ D

    S1_block = diff2_penalty(4)  # 4x4, rank 3
    S2_block = diff2_penalty(3)  # 3x3, rank 2
    S1 = np.zeros((p, p))
    S1[1:5, 1:5] = S1_block
    S2 = np.zeros((p, p))
    S2[5:8, 5:8] = S2_block

    # Ground truth coefficients and response.
    beta_true = np.array([0.6, 0.4, -0.3, 0.2, 0.1, -0.4, 0.3, -0.2])
    eta_true = X @ beta_true
    mu_true = np.exp(eta_true)
    # Gamma noise with shape phi^-1 = 4 (sigma^2 = 0.25).
    shape = 4.0
    y = rng.gamma(shape=shape, scale=mu_true / shape)

    return X, y, [S1, S2], beta_true


# ----------------------------------------------------------------------------
# Penalised deviance, gradient, full-Newton Hessian
# ----------------------------------------------------------------------------
def penalised_deviance_terms(X, y, beta, S_lambda):
    """Return mu, eta, deviance gradient, full-Newton Hessian operator
    H = X' diag(w_newton) X + S (full Newton Hessian of penalised deviance/2).

    mgcv conventions:
        alpha = 1 + (y - mu) * (V1 + g2)           where V1=V'/V, g2=g''/g'
        w_newton = alpha * (dmu/d eta)^2 / V0      where V0 = V(mu) raw
    For Gamma+log this gives  alpha = w = y/mu.
    """
    eta = X @ beta
    mu = np.exp(eta)

    g1, g2, g3, V0, V1, V2, dmu_deta = family_terms(mu)

    alpha = 1.0 + (y - mu) * (V1 + g2)
    w_newton = alpha * dmu_deta ** 2 / V0

    # Gradient of penalised deviance/2:
    #   d (-loglik) / d beta = -X' (y - mu) * dmu_deta / V0
    # For Gamma+log this is -X' (y-mu)/mu; matches what penalised IRLS solves.
    grad_unpen = -X.T @ ((y - mu) * dmu_deta / V0)
    grad = grad_unpen + S_lambda @ beta

    H = X.T @ (w_newton[:, None] * X) + S_lambda
    return mu, eta, alpha, w_newton, grad, H


def fit_inner_pirls(X, y, S_list, log_lambda, beta0=None, tol=1e-10, max_iter=200):
    """Newton on penalised deviance / 2 until ||grad|| < tol.
    Uses Newton (not Fisher) weights for the Hessian, with backtracking on
    penalised deviance."""
    n, p = X.shape
    lam = np.exp(log_lambda)
    S_lambda = sum(l * S for l, S in zip(lam, S_list))

    if beta0 is None:
        # Cold start: regress log(y) on X with light ridge.
        z = np.log(np.maximum(y, 1e-6))
        beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(p) + S_lambda, X.T @ z)
    else:
        beta = beta0.copy()

    def pen_dev(beta):
        eta = X @ beta
        mu = np.exp(eta)
        # Gamma deviance, dropping additive constants in y.
        d = 2.0 * np.sum(-np.log(np.maximum(y / mu, 1e-300)) + (y - mu) / mu)
        return d + beta @ S_lambda @ beta

    for it in range(max_iter):
        mu, eta, alpha, w, grad, H = penalised_deviance_terms(X, y, beta, S_lambda)
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        # Newton step.  H may not be PD off-MLE (alpha can be negative), so add
        # mild Levenberg damping if eigenvalue is non-positive.
        try:
            step = sla.solve(H, grad, assume_a="sym")
        except sla.LinAlgError:
            step = sla.solve(H + 1e-6 * np.eye(p), grad, assume_a="sym")

        # Backtracking on penalised deviance (objective = -2 loglik + beta'Sbeta)
        f0 = pen_dev(beta)
        t = 1.0
        for _ in range(50):
            beta_new = beta - t * step
            f1 = pen_dev(beta_new)
            if np.isfinite(f1) and f1 <= f0 - 1e-4 * t * (grad @ step):
                break
            t *= 0.5
        beta = beta - t * step

    mu, eta, alpha, w, grad, H = penalised_deviance_terms(X, y, beta, S_lambda)
    return beta, mu, eta, alpha, w, S_lambda, grad, H, lam


# ----------------------------------------------------------------------------
# Tk and KK' diagonal
# ----------------------------------------------------------------------------
def leverage_direct(X, w, S_lambda):
    """h[i] = w[i] * x[i]' * (X'WX + S)^-1 * x[i].
    Allows w<0 (Newton)."""
    A = X.T @ (w[:, None] * X) + S_lambda
    Ainv = sla.inv(A)
    h = np.einsum("ij,jk,ik->i", X, Ainv, X) * w
    return h, A, Ainv


def leverage_qr_aug(X, w, S_lambda):
    """Aug-QR route: K = top n rows of Q, M = [sqrt(|w|) X ; E], E'E = S.
    With negative w, sqrt(|w|) is used and we carry the sign via the formula.
    For the leverage of (X'WX + S), we use sign(w)*|w|: rows scaled by sqrt|w|
    and then `diag(K K')` reproduces |w| * x'A^{-1} x.  We multiply by sign(w)
    to recover w * x'A^{-1} x."""
    n, p = X.shape
    sgn = np.sign(w)
    sw = np.sqrt(np.abs(w))
    # Cholesky of S (it's PSD; use eigendecomposition for safety since S is rank-deficient).
    eigvals, eigvecs = np.linalg.eigh(S_lambda)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    E = (eigvecs * np.sqrt(eigvals_clipped)).T  # (p, p), some rows zero
    # Drop zero rows for a thinner aug
    keep = eigvals_clipped > 1e-14 * (eigvals_clipped.max() if eigvals_clipped.size else 1.0)
    E = E[keep]
    M = np.vstack([sw[:, None] * X, E])
    Q, R = np.linalg.qr(M, mode="reduced")
    K = Q[:n, :]                  # (n, p)
    h_abs = np.einsum("ij,ij->i", K, K)  # = |w| * x' (X'WX + S)^-1 x
    h = sgn * h_abs
    return h, K, R


def compute_alpha1_mgcv(mu, y, alpha):
    """alpha1 per gdi.c:2548 (mgcv-scaled). This is *NOT* d alpha/d eta;
    it equals  (d log alpha)/(d mu) = (d alpha / d mu) / alpha.

       xx = V2 - V1^2 + g3 - g2^2
       alpha1 = (-(V1 + g2) + (y-mu)*xx) / alpha

    Derivation: alpha = 1 + (y-mu)(V1 + g2)  (with mgcv-scaled V1, g2). So
    d alpha / d mu = -(V1+g2) + (y-mu) * d(V1+g2)/dmu.
    d(V'/V)/dmu = (V'' V - V'^2)/V^2 = V2 - V1^2 (with mgcv scaling).
    d(g''/g')/dmu = (g'''/g' - (g''/g')^2) = g3 - g2^2.
    So d alpha / d mu = -(V1+g2) + (y-mu)(V2-V1^2 + g3 - g2^2). Divide by alpha.
    """
    g1, g2, g3, V0, V1, V2, dmu_deta = family_terms(mu)
    xx = V2 - V1 ** 2 + g3 - g2 ** 2
    alpha1 = (-(V1 + g2) + (y - mu) * xx) / alpha
    return alpha1


def compute_a1_mgcv(mu, y, alpha, w):
    """a1 = dw/d eta per gdi.c:2556 (mgcv-scaled):
       a1 = w * (alpha1 - V1 - 2 g2) / g1
    where alpha1 is from compute_alpha1_mgcv (i.e. d log alpha/d mu).
    """
    g1, g2, g3, V0, V1, V2, dmu_deta = family_terms(mu)
    alpha1 = compute_alpha1_mgcv(mu, y, alpha)
    a1 = w * (alpha1 - V1 - 2.0 * g2) / g1
    return a1, alpha1


def main():
    np.set_printoptions(precision=8, suppress=False, linewidth=140)
    X, y, S_list, beta_true = make_problem(seed=0)
    n, p = X.shape
    M = len(S_list)
    log_lambda = np.array([0.0, 1.0])

    print("=" * 78)
    print(f"Problem: n={n}, p={p}, M={M}, rho={log_lambda.tolist()}")
    print("=" * 78)

    beta, mu, eta, alpha, w, S_lambda, grad, H, lam = fit_inner_pirls(
        X, y, S_list, log_lambda
    )

    print(f"\nInner PIRLS done.  ||grad||_inf = {np.max(np.abs(grad)):.3e}, "
          f"||grad||_2 = {np.linalg.norm(grad):.3e}")
    print(f"lambda = {lam}")
    print(f"\nbeta_hat   = {beta}")
    print(f"\nmu (first 6) = {mu[:6]}")
    print(f"alpha (first 6) = {alpha[:6]}")
    print(f"w_newton (first 6) = {w[:6]}")
    print(f"min(w)  = {w.min():.6e}, max(w) = {w.max():.6e}, "
          f"any negative? {bool((w < 0).any())}")

    # ------------------------------------------------------------------
    # (a) Leverage h[i] = diag(K K')[i] two ways
    # ------------------------------------------------------------------
    h_direct, A, Ainv = leverage_direct(X, w, S_lambda)
    h_qr, K, R = leverage_qr_aug(X, w, S_lambda)
    print(f"\n--- Leverage check (h = w * x' A^-1 x) ---")
    print(f"max |h_direct - h_qr| = {np.max(np.abs(h_direct - h_qr)):.3e}")
    print(f"sum(h_direct) = {h_direct.sum():.10f}  (= EDF + p_unpen)")
    print(f"h_direct[:6] = {h_direct[:6]}")

    # ------------------------------------------------------------------
    # IFT: b1[:,k] = -lam_k * A^-1 * S_k * beta
    # ------------------------------------------------------------------
    b1 = np.zeros((p, M))
    for k in range(M):
        b1[:, k] = -lam[k] * (Ainv @ (S_list[k] @ beta))
    print(f"\n--- b1 (dbeta/drho) via IFT ---")
    for k in range(M):
        print(f"b1[:,{k}] = {b1[:, k]}")

    # Cross check b1 by central finite difference
    h_fd = 1e-5
    b1_fd = np.zeros_like(b1)
    for k in range(M):
        rho_p = log_lambda.copy(); rho_p[k] += h_fd
        rho_m = log_lambda.copy(); rho_m[k] -= h_fd
        bp, *_ = fit_inner_pirls(X, y, S_list, rho_p, beta0=beta, tol=1e-12)
        bm, *_ = fit_inner_pirls(X, y, S_list, rho_m, beta0=beta, tol=1e-12)
        b1_fd[:, k] = (bp - bm) / (2 * h_fd)
    print(f"\nb1 IFT vs central-FD max-abs:")
    for k in range(M):
        diff = np.max(np.abs(b1[:, k] - b1_fd[:, k]))
        print(f"  k={k}: max|IFT - FD| = {diff:.3e}")

    # ------------------------------------------------------------------
    # (b) Tk[i,k] = a1[i,k] * (X b1)[i,k] / |w[i]|
    #     (mgcv divides by |w|; allows w<0 under Newton.)
    # ------------------------------------------------------------------
    # alpha1 and a1 are per-observation, NOT per-rho — they're d log alpha/d mu
    # and dw/d eta respectively.  The rho-dependence enters via Xb1.
    a1, alpha1 = compute_a1_mgcv(mu, y, alpha, w)

    Tk = np.zeros((n, M))
    for k in range(M):
        Xb1_k = X @ b1[:, k]
        Tk[:, k] = a1 * Xb1_k / np.abs(w)

    print(f"\n--- alpha1 = d log alpha / d mu (first 6 obs) ---")
    print(alpha1[:6])
    print(f"\n--- a1 = dw/d eta (first 6 obs) ---")
    print(a1[:6])
    # Sanity: for Gamma+log, alpha = w = y/mu, so d log alpha/dmu = -1/mu, and
    # dw/d eta = -y/mu directly.
    print(f"     expected (Gamma+log): -1/mu[:6] = {-1.0/mu[:6]}")
    print(f"     expected (Gamma+log): -y/mu[:6] = {-y[:6]/mu[:6]}")
    print(f"\n--- Tk[:,k] (first 4 obs) ---")
    print(Tk[:4])

    # Cross-check Tk via direct central-FD on w(rho) per observation:
    #   Tk[i,k] should equal (dw_i/drho_k) / |w_i|  (with sign convention
    #   sign(w_i) absorbed into Tk because Tk[i] = a1[i]*Xb1[i]/|w[i]|
    #   and a1 = dw/d eta carries the right sign for w>0).
    # We verify (dw_i/drho_k)_FD / |w_i|  ≈  Tk[i, k].
    h_fd_w = 1e-5
    Tk_fd = np.zeros((n, M))
    for k in range(M):
        rp = log_lambda.copy(); rp[k] += h_fd_w
        rm = log_lambda.copy(); rm[k] -= h_fd_w
        bp, mp, ep, ap, wp, *_ = fit_inner_pirls(X, y, S_list, rp,
                                                  beta0=beta, tol=1e-14)
        bm, mm, em, am, wm, *_ = fit_inner_pirls(X, y, S_list, rm,
                                                  beta0=beta, tol=1e-14)
        Tk_fd[:, k] = (wp - wm) / (2 * h_fd_w) / np.abs(w)
    print(f"\n--- Tk vs FD-on-w / |w| (max abs diff) ---")
    for k in range(M):
        diff = np.max(np.abs(Tk[:, k] - Tk_fd[:, k]))
        print(f"  k={k}: max|Tk_analytic - Tk_fd| = {diff:.3e}")

    # ------------------------------------------------------------------
    # (c) det1_addition[k] = sum_i Tk[i,k] * h[i]
    # ------------------------------------------------------------------
    det1_addition = np.array([np.sum(Tk[:, k] * h_direct) for k in range(M)])
    print(f"\n--- tr(Tk[:,k] · K K') = sum_i Tk[i,k] * h[i] ---")
    for k in range(M):
        print(f"k={k}:  det1_addition[k] = {det1_addition[k]:+.12e}")

    # ------------------------------------------------------------------
    # (d) Cross-check via central FD on log|A(rho)|
    # ------------------------------------------------------------------
    # mgcv's det1[k] = d/drho_k log|X'WX + S|
    #              = lam_k * tr(A^-1 S_k)  +  tr(Tk[:,k] * KK')   (Newton)
    # Compare both sides via central FD.
    def logdetA(rho):
        # Use very tight PIRLS tol so beta(rho) is resolved well below the FD
        # truncation error.
        beta_r, mu_r, eta_r, alpha_r, w_r, S_r, grad_r, H_r, lam_r = \
            fit_inner_pirls(X, y, S_list, rho, beta0=beta, tol=1e-14, max_iter=400)
        Ar = X.T @ (w_r[:, None] * X) + S_r
        sign, ld = np.linalg.slogdet(Ar)
        return ld
    # Sweep central-FD over several step sizes so the user can see which is
    # truncation- vs roundoff-dominated; also do Richardson extrapolation.
    def central_fd(k, h):
        rp = log_lambda.copy(); rp[k] += h
        rm = log_lambda.copy(); rm[k] -= h
        return (logdetA(rp) - logdetA(rm)) / (2 * h)

    print("\n--- FD step sweep on d/drho_k log|A| ---")
    fd_steps = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    fd_table = np.zeros((len(fd_steps), M))
    for i, h in enumerate(fd_steps):
        for k in range(M):
            fd_table[i, k] = central_fd(k, h)
        print(f"  h={h:.0e}: " + ", ".join(
            f"k={k}: {fd_table[i, k]:+.12e}" for k in range(M)
        ))

    # For central FD with f-evals at machine precision ~1e-16, the optimal step
    # for d/drho log|A(rho)| ~ O(1) is h* ~ eps_M^(1/3) ~ 5e-6 to 5e-5 (cube
    # root of machine epsilon). At larger h we see truncation, smaller h we see
    # roundoff.  Pick h=5e-5 as the reference.
    fd_step_ref = 5e-5
    fd_step_per_k = np.full(M, fd_step_ref)
    det1_fd = np.array([central_fd(k, fd_step_ref) for k in range(M)])

    # Analytic: lam_k * tr(A^-1 S_k) + det1_addition[k]
    det1_analytic = np.zeros(M)
    for k in range(M):
        tr_AinvSk = np.trace(Ainv @ S_list[k])
        det1_analytic[k] = lam[k] * tr_AinvSk + det1_addition[k]

    print(f"\n--- Cross-check d/drho_k log|A| ---")
    for k in range(M):
        tr_AinvSk = np.trace(Ainv @ S_list[k])
        print(f"k={k}:  lam[k]*tr(A^-1 S_k) = {lam[k]*tr_AinvSk:+.12e}")
        print(f"       Tk*KK' addition     = {det1_addition[k]:+.12e}")
        print(f"       analytic total      = {det1_analytic[k]:+.12e}")
        print(f"       best central-FD (h={fd_step_per_k[k]:g}) = {det1_fd[k]:+.12e}")
        d = abs(det1_analytic[k] - det1_fd[k])
        rel = d / max(abs(det1_fd[k]), 1e-300)
        print(f"       |analytic - FD|    = {d:.3e}    rel = {rel:.3e}")

    # ------------------------------------------------------------------
    # Final summary line for grep-friendly regression-test extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("REFERENCE VALUES (for Rust regression test):")
    print("=" * 78)
    for k in range(M):
        print(f"  tr(Tk[:,{k}] * KK')  = {det1_addition[k]:+.15e}")
        print(f"  lam[{k}]*tr(A^-1 S_{k}) = {lam[k]*np.trace(Ainv @ S_list[k]):+.15e}")
        print(f"  det1_total[{k}]      = {det1_analytic[k]:+.15e}  (analytic)")
        print(f"  det1_total[{k}]      = {det1_fd[k]:+.15e}  (best central FD, h={fd_step_per_k[k]:g})")
    print(f"\n  beta_hat = {repr(beta.tolist())}")
    print(f"  log_lambda = {log_lambda.tolist()}")


if __name__ == "__main__":
    main()
