#!/usr/bin/env python3
"""
Numerical validation of the corrected Hessian formula.

This script computes the Hessian numerically by finite differences
of the gradient, and can be used to validate the analytical formula.
"""

import numpy as np
from scipy.linalg import solve, qr
import sys

def estimate_rank(S, threshold=1e-10):
    """Estimate rank of penalty matrix using SVD."""
    if S.size == 0:
        return 0
    u, s, vh = np.linalg.svd(S)
    max_sv = s[0] if len(s) > 0 else 0.0
    tol = threshold * max(max_sv, 1.0)
    return np.sum(s > tol)

def reml_gradient_ift(y, X, w, lambdas, penalties):
    """
    Compute REML gradient using IFT-based formula (matching Rust implementation).

    ∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
    """
    n, p = X.shape
    m = len(lambdas)

    # Build A = X'WX + Σλᵢ·Sᵢ
    W = np.diag(w)
    sqrt_W = np.sqrt(W)
    XtWX = X.T @ W @ X
    A = XtWX.copy()
    for lam, S in zip(lambdas, penalties):
        A += lam * S

    # Add ridge for stability
    ridge = 1e-7 * (1.0 + np.sqrt(m))
    A += ridge * np.diag(np.abs(np.diag(A)).clip(min=1.0))

    # Solve for β
    A_inv = np.linalg.inv(A)
    beta = A_inv @ (X.T @ (w * y))

    # Compute residuals and RSS
    fitted = X @ beta
    residuals = y - fitted
    RSS = np.sum(w * residuals**2)

    # Compute ranks and phi
    ranks = [estimate_rank(S) for S in penalties]
    total_rank = sum(ranks)
    n_minus_r = n - total_rank
    phi = RSS / n_minus_r

    # Compute P = RSS + Σλⱼ·β'·Sⱼ·β
    P = RSS
    for lam, S in zip(lambdas, penalties):
        P += lam * beta @ S @ beta

    # Compute gradient for each parameter
    gradient = np.zeros(m)

    for i in range(m):
        lam_i = lambdas[i]
        S_i = penalties[i]
        rank_i = ranks[i]

        # Term 1: tr(A⁻¹·λᵢ·Sᵢ)
        trace_term = lam_i * np.trace(A_inv @ S_i)

        # Term 2: -rank(Sᵢ)
        rank_term = -rank_i

        # Compute ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
        dbeta_drho_i = -A_inv @ (lam_i * S_i @ beta)

        # Compute ∂RSS/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ
        dRSS_drho_i = -2.0 * residuals @ X @ dbeta_drho_i

        # Compute ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        dphi_drho_i = dRSS_drho_i / n_minus_r

        # Compute ∂P/∂ρᵢ
        explicit_pen = lam_i * beta @ S_i @ beta
        implicit_pen = 0.0
        for j in range(m):
            S_j = penalties[j]
            lam_j = lambdas[j]
            # 2·Σⱼ λⱼ·β'·Sⱼ·∂β/∂ρᵢ (using symmetry of S_j)
            implicit_pen += lam_j * (beta @ S_j @ dbeta_drho_i + dbeta_drho_i @ S_j @ beta)

        dP_drho_i = dRSS_drho_i + explicit_pen + implicit_pen

        # Compute ∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ
        dPphi_drho_i = dP_drho_i / phi - (P / phi**2) * dphi_drho_i

        # Compute (n-r)·(1/φ)·∂φ/∂ρᵢ
        phi_term = n_minus_r * dphi_drho_i / phi

        # Total gradient
        gradient[i] = (trace_term + rank_term + dPphi_drho_i + phi_term) / 2.0

    return gradient

def numerical_hessian(y, X, w, lambdas, penalties, eps=1e-5):
    """
    Compute Hessian numerically using finite differences of gradient.

    H[i,j] ≈ (∂g/∂ρⱼ)ᵢ ≈ [g(ρ + εeⱼ) - g(ρ - εeⱼ)]ᵢ / (2ε)
    """
    m = len(lambdas)
    H = np.zeros((m, m))

    # Base gradient at current point
    g0 = reml_gradient_ift(y, X, w, lambdas, penalties)

    for j in range(m):
        # Perturb ρⱼ forward
        lambdas_plus = lambdas.copy()
        lambdas_plus[j] = lambdas[j] * np.exp(eps)  # ρⱼ + ε => λⱼ' = λⱼ·exp(ε)
        g_plus = reml_gradient_ift(y, X, w, lambdas_plus, penalties)

        # Perturb ρⱼ backward
        lambdas_minus = lambdas.copy()
        lambdas_minus[j] = lambdas[j] * np.exp(-eps)  # ρⱼ - ε => λⱼ' = λⱼ·exp(-ε)
        g_minus = reml_gradient_ift(y, X, w, lambdas_minus, penalties)

        # Finite difference
        H[:, j] = (g_plus - g_minus) / (2 * eps)

    # Symmetrize (should be symmetric already, but numerical errors)
    H = (H + H.T) / 2

    return H

def main():
    """Test Hessian computation on a simple 2D problem."""
    print("=" * 70)
    print("Numerical Validation of Corrected Hessian Formula")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    n = 100
    p1 = 10
    p2 = 10
    p = p1 + p2

    # Design matrix for two smooths
    X1 = np.random.randn(n, p1)
    X2 = np.random.randn(n, p2)
    X = np.hstack([X1, X2])

    # Response
    y = np.random.randn(n)

    # Weights (all 1 for simplicity)
    w = np.ones(n)

    # Penalty matrices (simple: S = D'D where D is second difference)
    def second_diff_penalty(k):
        """Create second-difference penalty matrix."""
        D = np.zeros((k-2, k))
        for i in range(k-2):
            D[i, i:i+3] = [1, -2, 1]
        return D.T @ D

    S1 = np.zeros((p, p))
    S1[:p1, :p1] = second_diff_penalty(p1)

    S2 = np.zeros((p, p))
    S2[p1:, p1:] = second_diff_penalty(p2)

    penalties = [S1, S2]

    # Test at specific λ values
    lambdas = np.array([5.0, 3.0])

    print(f"\nTest configuration:")
    print(f"  n = {n}")
    print(f"  p = {p} (p1={p1}, p2={p2})")
    print(f"  m = {len(lambdas)}")
    print(f"  λ = {lambdas}")
    print(f"  rank(S1) = {estimate_rank(S1)}")
    print(f"  rank(S2) = {estimate_rank(S2)}")

    # Compute gradient at this point
    print(f"\nComputing gradient...")
    grad = reml_gradient_ift(y, X, w, lambdas, penalties)
    print(f"  Gradient: {grad}")
    print(f"  ||gradient|| = {np.linalg.norm(grad):.6e}")

    # Compute numerical Hessian
    print(f"\nComputing numerical Hessian (this may take a moment)...")
    H_numerical = numerical_hessian(y, X, w, lambdas, penalties, eps=1e-5)

    print(f"\nNumerical Hessian:")
    print(H_numerical)
    print(f"\nDiagonal: {np.diag(H_numerical)}")
    print(f"Off-diagonal: {H_numerical[0, 1]}")

    # Check if it's positive definite (for a minimum)
    eigenvalues = np.linalg.eigvals(H_numerical)
    print(f"\nEigenvalues of Hessian: {eigenvalues}")

    if np.all(eigenvalues > 0):
        print("✓ Hessian is positive definite (local minimum)")
    elif np.all(eigenvalues < 0):
        print("✗ Hessian is negative definite (local maximum)")
    else:
        print("⚠ Hessian is indefinite (saddle point)")

    # Check descent direction: g'·H⁻¹·g should be < 0 for minimization
    # Actually for Newton: Δρ = -H⁻¹·g, and g'·Δρ = -g'·H⁻¹·g should be < 0
    # This means g'·H⁻¹·g > 0 (which is guaranteed if H is positive definite)
    try:
        H_inv = np.linalg.inv(H_numerical)
        delta_rho = -H_inv @ grad
        descent_check = grad @ delta_rho

        print(f"\nDescent direction check:")
        print(f"  Δρ (Newton step): {delta_rho}")
        print(f"  g'·Δρ = {descent_check:.6e}")

        if descent_check < 0:
            print("  ✓ Valid descent direction (g'·Δρ < 0)")
        else:
            print("  ✗ NOT a descent direction (g'·Δρ > 0)")
            print("  This is the BUG that was mentioned!")
    except np.linalg.LinAlgError:
        print("  ✗ Hessian is singular, cannot compute Newton step")

    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)

    # Save results for comparison with Rust
    np.savez('hessian_validation_data.npz',
             X=X, y=y, w=w, lambdas=lambdas,
             S1=S1, S2=S2,
             gradient=grad,
             hessian_numerical=H_numerical)
    print("\nData saved to 'hessian_validation_data.npz' for Rust comparison")

if __name__ == '__main__':
    main()
