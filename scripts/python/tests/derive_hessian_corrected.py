"""
CORRECTED Hessian derivation accounting for λᵢ = exp(ρᵢ)

Key insight: ∂λᵢ/∂ρⱼ = λᵢ if i=j, else 0

This adds extra terms to the Hessian!
"""
import numpy as np
import pandas as pd

# Load test data
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
S1 = pd.read_csv('/tmp/bench_S1.csv').values
S2 = pd.read_csv('/tmp/bench_S2.csv').values

n, p = X.shape
print("="*80)
print("CORRECTED HESSIAN DERIVATION")
print("="*80)
print(f"Problem: n={n}, p={p}")
print()

# Test point
lambda1, lambda2 = 1.0, 2.0
rho1, rho2 = np.log(lambda1), np.log(lambda2)

# Build A and solve for beta
XtX = X.T @ X
A = XtX + lambda1 * S1 + lambda2 * S2
Ainv = np.linalg.inv(A)
beta = Ainv @ (X.T @ y)

# Residuals and RSS
fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)

# edf and phi
ainv_xtx = Ainv @ XtX
edf_total = np.trace(ainv_xtx)
phi = rss / (n - edf_total)
log_phi = np.log(phi)

print(f"At λ = [{lambda1}, {lambda2}]:")
print(f"  rss = {rss:.6f}")
print(f"  edf = {edf_total:.6f}")
print(f"  phi = {phi:.6f}")
print()

# Pre-compute
ainv_xtx_ainv = ainv_xtx @ Ainv

# ============================================================================
# GRADIENT (for reference)
# ============================================================================
def compute_gradient_components(S, lam):
    """Compute gradient components for one smooth"""
    trace = lam * np.trace(Ainv @ S)
    dbeta_drho = -Ainv @ (lam * S @ beta)
    drss_drho = -2 * residuals @ X @ dbeta_drho
    dedf_drho = -lam * np.trace(ainv_xtx_ainv @ S)
    grad = trace + dedf_drho * (-log_phi + 1) + drss_drho / phi
    return {
        'trace': trace,
        'dbeta_drho': dbeta_drho,
        'drss_drho': drss_drho,
        'dedf_drho': dedf_drho,
        'gradient': grad
    }

g1 = compute_gradient_components(S1, lambda1)
g2 = compute_gradient_components(S2, lambda2)

print("GRADIENT:")
print(f"∂REML/∂ρ₁ = {g1['gradient']:.6f}")
print(f"∂REML/∂ρ₂ = {g2['gradient']:.6f}")
print()

# ============================================================================
# HESSIAN WITH CORRECTED FORMULA
# ============================================================================
print("="*80)
print("HESSIAN DERIVATION (CORRECTED)")
print("="*80)
print()
print("KEY INSIGHT: Since λᵢ = exp(ρᵢ), we have ∂λᵢ/∂ρⱼ = λᵢ·δᵢⱼ")
print()
print("The gradient component from log|A| is:")
print("  ∂log|A|/∂ρᵢ = tr(A⁻¹·∂A/∂ρᵢ) = tr(A⁻¹·λᵢ·Sᵢ)")
print()
print("Taking the derivative with respect to ρⱼ:")
print("  ∂²log|A|/∂ρⱼ∂ρᵢ = ∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)]")
print("                     = tr(∂A⁻¹/∂ρⱼ·λᵢ·Sᵢ) + tr(A⁻¹·∂λᵢ/∂ρⱼ·Sᵢ)")
print("                     = -λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·λᵢ·Sᵢ) + δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ)")
print()
print("The δᵢⱼ term adds the GRADIENT itself to diagonal elements!")
print()

def hessian_analytical(Si, Sj, lambda_i, lambda_j, i, j, grad_i=None):
    """
    Complete Hessian computation

    H[i,j] = ∂²REML/∂ρⱼ∂ρᵢ

    Components:
    1. ∂²log|A|/∂ρⱼ∂ρᵢ
    2. ∂/∂ρⱼ[∂edf/∂ρᵢ·(-log(φ)+1)]
    3. ∂/∂ρⱼ[∂rss/∂ρᵢ/φ]
    """

    # ========================================================================
    # TERM 1: ∂²log|A|/∂ρⱼ∂ρᵢ
    # ========================================================================
    # Part A: -λⱼ·λᵢ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
    term1a = -lambda_i * lambda_j * np.trace(Ainv @ Sj @ Ainv @ Si)

    # Part B: δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ) = δᵢⱼ·(gradient component)
    term1b = 0.0
    if i == j:
        term1b = lambda_i * np.trace(Ainv @ Si)

    term1_total = term1a + term1b

    # ========================================================================
    # TERM 2: ∂/∂ρⱼ[∂edf/∂ρᵢ·(-log(φ)+1)]
    # ========================================================================
    # ∂edf/∂ρᵢ
    dedf_drho_i = -lambda_i * np.trace(ainv_xtx_ainv @ Si)

    # Part A: ∂²edf/∂ρⱼ∂ρᵢ·(-log(φ)+1)
    # ∂²edf/∂ρⱼ∂ρᵢ has TWO parts:
    #   (1) From ∂A⁻¹/∂ρⱼ: λᵢ·λⱼ·[tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ·A⁻¹·X'X) + tr(A⁻¹·Sᵢ·A⁻¹·Sⱼ·A⁻¹·X'X)]
    #   (2) From ∂λᵢ/∂ρⱼ: -δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ·A⁻¹·X'X)

    d2edf_part1 = lambda_i * lambda_j * (
        np.trace(Ainv @ Sj @ Ainv @ Si @ ainv_xtx) +
        np.trace(Ainv @ Si @ Ainv @ Sj @ ainv_xtx)
    )
    d2edf_part2 = 0.0
    if i == j:
        d2edf_part2 = -lambda_i * np.trace(Ainv @ Si @ ainv_xtx)

    d2edf = d2edf_part1 + d2edf_part2
    term2a = d2edf * (-log_phi + 1)

    # Part B: ∂edf/∂ρᵢ·∂/∂ρⱼ[-log(φ)]
    # ∂φ/∂ρⱼ = [∂rss/∂ρⱼ + φ·∂edf/∂ρⱼ] / (n-edf)
    dbeta_drho_j = -Ainv @ (lambda_j * Sj @ beta)
    drss_drho_j = -2 * residuals @ X @ dbeta_drho_j
    dedf_drho_j = -lambda_j * np.trace(ainv_xtx_ainv @ Sj)
    dphi_drho_j = (drss_drho_j + phi * dedf_drho_j) / (n - edf_total)
    dlogphi_drho_j = -dphi_drho_j / phi

    term2b = dedf_drho_i * dlogphi_drho_j

    term2_total = term2a + term2b

    # ========================================================================
    # TERM 3: ∂/∂ρⱼ[∂rss/∂ρᵢ/φ]
    # ========================================================================
    # ∂rss/∂ρᵢ
    dbeta_drho_i = -Ainv @ (lambda_i * Si @ beta)
    drss_drho_i = -2 * residuals @ X @ dbeta_drho_i

    # Part A: ∂²rss/∂ρⱼ∂ρᵢ/φ
    # ∂²rss/∂ρⱼ∂ρᵢ has THREE parts:
    #   (1) -2·(∂r/∂ρⱼ)'·X·∂β/∂ρᵢ = 2·(X·∂β/∂ρⱼ)'·X·∂β/∂ρᵢ
    #   (2) -2·r'·X·∂²β/∂ρⱼ∂ρᵢ (from ∂A⁻¹/∂ρⱼ and ∂β/∂ρⱼ)
    #   (3) -2·r'·X·δᵢⱼ·∂β/∂ρᵢ (from ∂λᵢ/∂ρⱼ)

    # Part (1)
    d2rss_part1 = 2 * (X @ dbeta_drho_j) @ X @ dbeta_drho_i

    # Part (2) - via IFT
    d2beta_part2 = Ainv @ (lambda_j * Sj) @ Ainv @ (lambda_i * Si @ beta) - Ainv @ (lambda_i * Si @ dbeta_drho_j)
    d2rss_part2 = -2 * residuals @ X @ d2beta_part2

    # Part (3) - diagonal correction
    d2rss_part3 = 0.0
    if i == j:
        d2rss_part3 = -2 * residuals @ X @ dbeta_drho_i

    d2rss = d2rss_part1 + d2rss_part2 + d2rss_part3
    term3a = d2rss / phi

    # Part B: -∂rss/∂ρᵢ·∂φ/∂ρⱼ/φ²
    term3b = -drss_drho_i * dphi_drho_j / phi**2

    term3_total = term3a + term3b

    # ========================================================================
    # TOTAL
    # ========================================================================
    total = term1_total + term2_total + term3_total

    return {
        'term1a': term1a,
        'term1b': term1b,
        'term1_total': term1_total,
        'term2a': term2a,
        'term2b': term2b,
        'term2_total': term2_total,
        'term3a': term3a,
        'term3b': term3b,
        'term3_total': term3_total,
        'total': total,
        'd2edf_part1': d2edf_part1,
        'd2edf_part2': d2edf_part2,
        'd2rss_part1': d2rss_part1,
        'd2rss_part2': d2rss_part2,
        'd2rss_part3': d2rss_part3
    }

# Compute all Hessian elements
print("Computing Hessian elements...")
print()

h11 = hessian_analytical(S1, S1, lambda1, lambda1, 0, 0, g1['gradient'])
h12 = hessian_analytical(S1, S2, lambda1, lambda2, 0, 1)
h21 = hessian_analytical(S2, S1, lambda2, lambda1, 1, 0)
h22 = hessian_analytical(S2, S2, lambda2, lambda2, 1, 1, g2['gradient'])

print("H[1,1] breakdown:")
print(f"  Term 1a (trace derivative): {h11['term1a']:.6f}")
print(f"  Term 1b (diagonal λ correction): {h11['term1b']:.6f}")
print(f"  Term 1 total: {h11['term1_total']:.6f}")
print(f"  Term 2a (∂²edf·factor): {h11['term2a']:.6f}")
print(f"    - d²edf part1: {h11['d2edf_part1']:.6f}")
print(f"    - d²edf part2: {h11['d2edf_part2']:.6f}")
print(f"  Term 2b (∂edf·∂logφ): {h11['term2b']:.6f}")
print(f"  Term 2 total: {h11['term2_total']:.6f}")
print(f"  Term 3a (∂²rss/φ): {h11['term3a']:.6f}")
print(f"    - d²rss part1: {h11['d2rss_part1']:.6f}")
print(f"    - d²rss part2: {h11['d2rss_part2']:.6f}")
print(f"    - d²rss part3: {h11['d2rss_part3']:.6f}")
print(f"  Term 3b (-∂rss·∂φ/φ²): {h11['term3b']:.6f}")
print(f"  Term 3 total: {h11['term3_total']:.6f}")
print(f"  TOTAL H[1,1]: {h11['total']:.6f}")
print()

print("H[1,2] breakdown:")
print(f"  Term 1 total: {h12['term1_total']:.6f}")
print(f"  Term 2 total: {h12['term2_total']:.6f}")
print(f"  Term 3 total: {h12['term3_total']:.6f}")
print(f"  TOTAL H[1,2]: {h12['total']:.6f}")
print()

H_analytical = np.array([
    [h11['total'], h12['total']],
    [h21['total'], h22['total']]
])

print("="*80)
print("ANALYTICAL HESSIAN:")
print("="*80)
print(H_analytical)
print()

# ============================================================================
# NUMERICAL VALIDATION
# ============================================================================
print("="*80)
print("NUMERICAL VALIDATION")
print("="*80)

def reml_criterion(rho):
    """REML criterion at log-scale smoothing parameters"""
    lambdas = np.exp(rho)
    lambda1, lambda2 = lambdas[0], lambdas[1]

    A = XtX + lambda1 * S1 + lambda2 * S2
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (X.T @ y)

    fitted = X @ beta
    residuals = y - fitted
    rss = np.sum(residuals**2)

    edf_total = np.trace(Ainv @ XtX)
    phi = rss / (n - edf_total)

    sign_A, logdet_A = np.linalg.slogdet(A)
    reml = (n - edf_total) * np.log(phi) + logdet_A

    return reml

# Numerical Hessian
rho = np.array([rho1, rho2])
eps = 1e-6

H_numerical = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        rho_pp = rho.copy()
        rho_pp[i] += eps
        rho_pp[j] += eps

        rho_pm = rho.copy()
        rho_pm[i] += eps
        rho_pm[j] -= eps

        rho_mp = rho.copy()
        rho_mp[i] -= eps
        rho_mp[j] += eps

        rho_mm = rho.copy()
        rho_mm[i] -= eps
        rho_mm[j] -= eps

        H_numerical[i, j] = (reml_criterion(rho_pp) - reml_criterion(rho_pm) -
                             reml_criterion(rho_mp) + reml_criterion(rho_mm)) / (4 * eps**2)

print("Numerical Hessian:")
print(H_numerical)
print()

print("Difference (Analytical - Numerical):")
diff = H_analytical - H_numerical
print(diff)
print()

print("Relative error:")
rel_error = np.abs(diff) / (np.abs(H_numerical) + 1e-10)
print(rel_error)
print()

print("Maximum absolute error:", np.max(np.abs(diff)))
print("Maximum relative error:", np.max(rel_error))

if np.max(np.abs(diff)) < 1e-3:
    print("\n✅ HESSIAN VALIDATED! Analytical matches numerical to high precision.")
else:
    print("\n⚠️  Large discrepancy detected. Review derivation.")
