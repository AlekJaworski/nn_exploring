#!/usr/bin/env python3
"""
Pure Python reference implementation of the CORRECTED Hessian formula.

This can be used to test the Newton optimizer even without Rust compilation.
Uses the exact same formula as the Rust implementation.
"""

import numpy as np
from scipy.linalg import solve, qr

def estimate_rank(S, threshold=1e-10):
    """Estimate rank of penalty matrix using SVD."""
    if S.size == 0:
        return 0
    u, s, vh = np.linalg.svd(S)
    max_sv = s[0] if len(s) > 0 else 0.0
    tol = threshold * max(max_sv, 1.0)
    return np.sum(s > tol)

def reml_hessian_corrected(y, X, w, lambdas, penalties):
    """
    Compute REML Hessian using the CORRECTED IFT-based formula.

    This matches the Rust implementation in src/reml.rs::reml_hessian_multi_qr

    Returns: H[i,j] = ∂²REML/∂ρⱼ∂ρᵢ (m x m matrix)
    """
    n, p = X.shape
    m = len(lambdas)

    # Step 1: Build A = X'WX + Σλᵢ·Sᵢ
    W = np.diag(w)
    XtWX = X.T @ W @ X
    A = XtWX.copy()
    for lam, S in zip(lambdas, penalties):
        A += lam * S

    # Add ridge for stability (same as Rust)
    ridge = 1e-7 * (1.0 + np.sqrt(m))
    A += ridge * np.diag(np.abs(np.diag(A)).clip(min=1.0))

    # Compute A⁻¹
    A_inv = np.linalg.inv(A)

    # Step 2: Compute coefficients β
    beta = A_inv @ (X.T @ (w * y))

    # Step 3: Compute residuals, RSS, phi, P
    fitted = X @ beta
    residuals = y - fitted
    RSS = np.sum(w * residuals**2)

    ranks = [estimate_rank(S) for S in penalties]
    total_rank = sum(ranks)
    n_minus_r = n - total_rank
    phi = RSS / n_minus_r
    inv_phi = 1.0 / phi
    phi_sq = phi * phi
    phi_cb = phi * phi * phi

    # Compute P = RSS + Σⱼ λⱼ·β'·Sⱼ·β
    P = RSS
    for lam, S in zip(lambdas, penalties):
        P += lam * beta @ S @ beta

    print(f"[Hessian] n={n}, p={p}, m={m}, RSS={RSS:.3e}, phi={phi:.3e}, P={P:.3e}")

    # Step 4: Compute first derivatives (matching gradient)
    dbeta_drho = []
    drss_drho = []
    dphi_drho = []
    dp_drho = []

    for i in range(m):
        lam_i = lambdas[i]
        S_i = penalties[i]

        # ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
        dbeta_i = -A_inv @ (lam_i * S_i @ beta)
        dbeta_drho.append(dbeta_i)

        # ∂RSS/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ
        drss_i = -2.0 * residuals @ X @ dbeta_i
        drss_drho.append(drss_i)

        # ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        dphi_i = drss_i / n_minus_r
        dphi_drho.append(dphi_i)

        # ∂P/∂ρᵢ = ∂RSS/∂ρᵢ + λᵢ·β'·Sᵢ·β + 2·Σⱼ λⱼ·β'·Sⱼ·∂β/∂ρᵢ
        explicit_pen = lam_i * beta @ S_i @ beta

        implicit_pen = 0.0
        for j in range(m):
            S_j = penalties[j]
            lam_j = lambdas[j]
            implicit_pen += lam_j * (beta @ S_j @ dbeta_i + dbeta_i @ S_j @ beta)

        dp_i = drss_i + explicit_pen + implicit_pen
        dp_drho.append(dp_i)

    # Step 5: Compute Hessian
    H = np.zeros((m, m))

    for i in range(m):
        for j in range(i, m):  # Only upper triangle
            lam_i = lambdas[i]
            lam_j = lambdas[j]
            S_i = penalties[i]
            S_j = penalties[j]

            # ================================================================
            # TERM 1: ∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)] / 2
            # ================================================================
            # Part A: -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
            ainv_sj = A_inv @ S_j
            ainv_sj_ainv = ainv_sj @ A_inv
            si_ainv_sj_ainv = S_i @ ainv_sj_ainv
            trace1a = np.trace(si_ainv_sj_ainv)
            term1a = -lam_i * lam_j * trace1a

            # Part B: δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ)
            term1b = lam_i * np.trace(A_inv @ S_i) if i == j else 0.0

            term1 = (term1a + term1b) / 2.0

            # ================================================================
            # TERM 2: ∂²(P/φ)/∂ρⱼ∂ρᵢ / 2
            # ================================================================

            # Compute ∂²β/∂ρⱼ∂ρᵢ
            si_beta = S_i @ beta
            ainv_si_beta = A_inv @ si_beta
            lambda_i_ainv_si_beta = lam_i * ainv_si_beta
            sj_times_term = S_j @ lambda_i_ainv_si_beta
            part_a = A_inv @ sj_times_term * lam_j

            si_dbeta_j = S_i @ dbeta_drho[j]
            part_b = -lam_i * A_inv @ si_dbeta_j

            d2beta = part_a + part_b
            if i == j:
                d2beta -= dbeta_drho[i]

            # Compute ∂²RSS/∂ρⱼ∂ρᵢ
            x_dbeta_j = X @ dbeta_drho[j]
            x_dbeta_i = X @ dbeta_drho[i]
            d2rss_part1 = 2.0 * x_dbeta_j @ x_dbeta_i

            x_d2beta = X @ d2beta
            d2rss_part2 = -2.0 * residuals @ x_d2beta

            d2rss = d2rss_part1 + d2rss_part2

            # Compute ∂²φ/∂ρⱼ∂ρᵢ
            d2phi = d2rss / n_minus_r

            # Compute ∂²P/∂ρⱼ∂ρᵢ
            diag_explicit = lam_i * beta @ si_beta if i == j else 0.0
            dbeta_j_si_beta = dbeta_drho[j] @ si_beta
            explicit_cross = 2.0 * lam_i * dbeta_j_si_beta

            implicit_sum = 0.0
            for k in range(m):
                sk_beta = penalties[k] @ beta
                sk_dbeta_i = penalties[k] @ dbeta_drho[i]

                # δₖⱼ·λₖ·∂β'/∂ρᵢ·Sₖ·β
                term_k1 = lambdas[k] * dbeta_drho[i] @ sk_beta if k == j else 0.0

                # λₖ·∂²β'/∂ρⱼ∂ρᵢ·Sₖ·β
                sk_d2beta = d2beta @ sk_beta
                term_k2 = lambdas[k] * sk_d2beta

                # λₖ·∂β'/∂ρᵢ·Sₖ·∂β/∂ρⱼ
                dbeta_i_sk_dbeta_j = dbeta_drho[i] @ sk_dbeta_i
                term_k3 = lambdas[k] * dbeta_i_sk_dbeta_j

                implicit_sum += term_k1 + term_k2 + term_k3

            d2p = d2rss + diag_explicit + explicit_cross + 2.0 * implicit_sum

            # Assemble ∂²(P/φ)/∂ρⱼ∂ρᵢ
            term2a = inv_phi * d2p
            term2b = -(1.0 / phi_sq) * (dphi_drho[j] * dp_drho[i] + dp_drho[j] * dphi_drho[i])
            term2c = 2.0 * (P / phi_cb) * dphi_drho[j] * dphi_drho[i]
            term2d = -(P / phi_sq) * d2phi

            term2 = (term2a + term2b + term2c + term2d) / 2.0

            # ================================================================
            # TERM 3: ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
            # ================================================================
            term3a = n_minus_r * inv_phi * d2phi
            term3b = -n_minus_r * (1.0 / phi_sq) * dphi_drho[j] * dphi_drho[i]

            term3 = (term3a + term3b) / 2.0

            # ================================================================
            # TOTAL HESSIAN
            # ================================================================
            h_val = term1 + term2 + term3
            H[i, j] = h_val

            # Fill symmetric entry
            if i != j:
                H[j, i] = h_val

    print(f"[Hessian] Eigenvalues: {np.linalg.eigvalsh(H)}")
    print(f"[Hessian] Matrix:\n{H}")

    return H

# Make available as module
__all__ = ['reml_hessian_corrected', 'estimate_rank']
