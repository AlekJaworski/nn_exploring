"""
Derive and validate the Hessian matrix for REML criterion

The REML criterion is:
    REML(ρ) = (n - edf) * log(φ) + log|A|

where:
    ρᵢ = log(λᵢ)
    A = X'X + Σᵢ λᵢ·Sᵢ
    β = A⁻¹·X'y (implicit dependence on ρ)
    rss = ||y - X·β||²
    edf = tr(A⁻¹·X'X)
    φ = rss / (n - edf)

We need the Hessian: H[i,j] = ∂²REML/∂ρᵢ∂ρⱼ

The gradient is:
    ∂REML/∂ρᵢ = tr(A⁻¹·λᵢ·Sᵢ) + ∂edf/∂ρᵢ·(-log(φ)+1) + ∂rss/∂ρᵢ/φ

where (from IFT):
    ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
    ∂rss/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ
    ∂edf/∂ρᵢ = -tr(A⁻¹·λᵢ·Sᵢ·A⁻¹·X'X)

To get the Hessian, we differentiate the gradient components with respect to ρⱼ.
"""
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

# Load test data
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
S1 = pd.read_csv('/tmp/bench_S1.csv').values
S2 = pd.read_csv('/tmp/bench_S2.csv').values

n, p = X.shape
print("="*80)
print("DERIVING HESSIAN FOR REML CRITERION")
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
edf_total = np.trace(Ainv @ XtX)
phi = rss / (n - edf_total)
log_phi = np.log(phi)

print(f"At λ = [{lambda1}, {lambda2}]:")
print(f"  rss = {rss:.6f}")
print(f"  edf = {edf_total:.6f}")
print(f"  phi = {phi:.6f}")
print()

# Pre-compute useful quantities
ainv_xtx = Ainv @ XtX
ainv_xtx_ainv = ainv_xtx @ Ainv

# ============================================================================
# GRADIENT COMPONENTS (for reference)
# ============================================================================
print("GRADIENT COMPONENTS:")
print("-"*80)

def compute_gradient_components(S, lam):
    """Compute gradient components for one smooth"""
    # Trace term
    trace = lam * np.trace(Ainv @ S)

    # ∂β/∂ρ via IFT
    dbeta_drho = -Ainv @ (lam * S @ beta)

    # ∂rss/∂ρ
    drss_drho = -2 * residuals @ X @ dbeta_drho

    # ∂edf/∂ρ
    dedf_drho = -lam * np.trace(ainv_xtx_ainv @ S)

    # Total gradient
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

print(f"∂REML/∂ρ₁ = {g1['gradient']:.6f}")
print(f"∂REML/∂ρ₂ = {g2['gradient']:.6f}")
print()

# ============================================================================
# HESSIAN DERIVATION
# ============================================================================
print("="*80)
print("HESSIAN DERIVATION: ∂²REML/∂ρᵢ∂ρⱼ")
print("="*80)
print()

print("The gradient has form:")
print("  ∂REML/∂ρᵢ = tr(A⁻¹·λᵢ·Sᵢ) + ∂edf/∂ρᵢ·(-log(φ)+1) + ∂rss/∂ρᵢ/φ")
print()
print("Differentiating each term with respect to ρⱼ:")
print()

# ============================================================================
# Term 1: ∂/∂ρⱼ [tr(A⁻¹·λᵢ·Sᵢ)]
# ============================================================================
print("TERM 1: ∂/∂ρⱼ [tr(A⁻¹·λᵢ·Sᵢ)]")
print("-"*80)
print("Using ∂A⁻¹/∂ρⱼ = -A⁻¹·(∂A/∂ρⱼ)·A⁻¹ = -A⁻¹·λⱼ·Sⱼ·A⁻¹:")
print()
print("  ∂/∂ρⱼ [tr(A⁻¹·λᵢ·Sᵢ)] = tr(∂A⁻¹/∂ρⱼ·λᵢ·Sᵢ)")
print("                        = tr(-A⁻¹·λⱼ·Sⱼ·A⁻¹·λᵢ·Sᵢ)")
print("                        = -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)")
print()

# Compute for all pairs
def hessian_term1(Si, Sj, lambda_i, lambda_j):
    """∂/∂ρⱼ [tr(A⁻¹·λᵢ·Sᵢ)]"""
    return -lambda_i * lambda_j * np.trace(Ainv @ Sj @ Ainv @ Si)

h11_term1 = hessian_term1(S1, S1, lambda1, lambda1)
h12_term1 = hessian_term1(S1, S2, lambda1, lambda2)
h21_term1 = hessian_term1(S2, S1, lambda2, lambda1)
h22_term1 = hessian_term1(S2, S2, lambda2, lambda2)

print(f"H[1,1] term1 = {h11_term1:.6f}")
print(f"H[1,2] term1 = {h12_term1:.6f}")
print(f"H[2,1] term1 = {h21_term1:.6f}")
print(f"H[2,2] term1 = {h22_term1:.6f}")
print()

# ============================================================================
# Term 2: ∂/∂ρⱼ [∂edf/∂ρᵢ·(-log(φ)+1)]
# ============================================================================
print("TERM 2: ∂/∂ρⱼ [∂edf/∂ρᵢ·(-log(φ)+1)]")
print("-"*80)
print("This involves:")
print("  (a) ∂²edf/∂ρⱼ∂ρᵢ·(-log(φ)+1)")
print("  (b) ∂edf/∂ρᵢ·∂/∂ρⱼ[-log(φ)+1]")
print()

# Term 2a: ∂²edf/∂ρⱼ∂ρᵢ
print("Term 2a: ∂²edf/∂ρⱼ∂ρᵢ")
print("  ∂edf/∂ρᵢ = -tr(A⁻¹·λᵢ·Sᵢ·A⁻¹·X'X)")
print("  ∂²edf/∂ρⱼ∂ρᵢ = -tr(∂A⁻¹/∂ρⱼ·λᵢ·Sᵢ·A⁻¹·X'X + A⁻¹·λᵢ·Sᵢ·∂A⁻¹/∂ρⱼ·X'X)")
print("                 = tr(A⁻¹·λⱼ·Sⱼ·A⁻¹·λᵢ·Sᵢ·A⁻¹·X'X + A⁻¹·λᵢ·Sᵢ·A⁻¹·λⱼ·Sⱼ·A⁻¹·X'X)")
print("                 = λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ·A⁻¹·X'X + A⁻¹·Sᵢ·A⁻¹·Sⱼ·A⁻¹·X'X)")
print()

def d2edf_drho2(Si, Sj, lambda_i, lambda_j):
    """∂²edf/∂ρⱼ∂ρᵢ"""
    term1 = np.trace(Ainv @ Sj @ Ainv @ Si @ ainv_xtx)
    term2 = np.trace(Ainv @ Si @ Ainv @ Sj @ ainv_xtx)
    return lambda_i * lambda_j * (term1 + term2)

d2edf_11 = d2edf_drho2(S1, S1, lambda1, lambda1)
d2edf_12 = d2edf_drho2(S1, S2, lambda1, lambda2)
d2edf_21 = d2edf_drho2(S2, S1, lambda2, lambda1)
d2edf_22 = d2edf_drho2(S2, S2, lambda2, lambda2)

print(f"∂²edf/∂ρ₁∂ρ₁ = {d2edf_11:.6f}")
print(f"∂²edf/∂ρ₁∂ρ₂ = {d2edf_12:.6f}")
print(f"∂²edf/∂ρ₂∂ρ₁ = {d2edf_21:.6f}")
print(f"∂²edf/∂ρ₂∂ρ₂ = {d2edf_22:.6f}")
print()

# Term 2b: ∂edf/∂ρᵢ·∂/∂ρⱼ[-log(φ)+1]
print("Term 2b: ∂edf/∂ρᵢ·∂/∂ρⱼ[-log(φ)]")
print("  ∂/∂ρⱼ[-log(φ)] = -1/φ·∂φ/∂ρⱼ")
print("  φ = rss/(n-edf), so:")
print("  ∂φ/∂ρⱼ = [∂rss/∂ρⱼ·(n-edf) - rss·(-∂edf/∂ρⱼ)] / (n-edf)²")
print("         = [∂rss/∂ρⱼ + rss/(n-edf)·∂edf/∂ρⱼ] / (n-edf)")
print("         = [∂rss/∂ρⱼ + φ·∂edf/∂ρⱼ] / (n-edf)")
print()

def dphi_drho(S, lam):
    """∂φ/∂ρ"""
    dbeta_drho = -Ainv @ (lam * S @ beta)
    drss_drho = -2 * residuals @ X @ dbeta_drho
    dedf_drho = -lam * np.trace(ainv_xtx_ainv @ S)
    return (drss_drho + phi * dedf_drho) / (n - edf_total)

dphi_drho1 = dphi_drho(S1, lambda1)
dphi_drho2 = dphi_drho(S2, lambda2)

dlogphi_drho1 = -dphi_drho1 / phi
dlogphi_drho2 = -dphi_drho2 / phi

print(f"∂φ/∂ρ₁ = {dphi_drho1:.6e}")
print(f"∂φ/∂ρ₂ = {dphi_drho2:.6e}")
print(f"∂(-log(φ))/∂ρ₁ = {dlogphi_drho1:.6e}")
print(f"∂(-log(φ))/∂ρ₂ = {dlogphi_drho2:.6e}")
print()

# Total term 2
h11_term2a = d2edf_11 * (-log_phi + 1)
h12_term2a = d2edf_12 * (-log_phi + 1)
h21_term2a = d2edf_21 * (-log_phi + 1)
h22_term2a = d2edf_22 * (-log_phi + 1)

h11_term2b = g1['dedf_drho'] * dlogphi_drho1
h12_term2b = g1['dedf_drho'] * dlogphi_drho2
h21_term2b = g2['dedf_drho'] * dlogphi_drho1
h22_term2b = g2['dedf_drho'] * dlogphi_drho2

print("Term 2 contributions:")
print(f"H[1,1] term2a = {h11_term2a:.6f}, term2b = {h11_term2b:.6f}")
print(f"H[1,2] term2a = {h12_term2a:.6f}, term2b = {h12_term2b:.6f}")
print(f"H[2,1] term2a = {h21_term2a:.6f}, term2b = {h21_term2b:.6f}")
print(f"H[2,2] term2a = {h22_term2a:.6f}, term2b = {h22_term2b:.6f}")
print()

# ============================================================================
# Term 3: ∂/∂ρⱼ [∂rss/∂ρᵢ/φ]
# ============================================================================
print("TERM 3: ∂/∂ρⱼ [∂rss/∂ρᵢ/φ]")
print("-"*80)
print("  = ∂²rss/∂ρⱼ∂ρᵢ/φ - ∂rss/∂ρᵢ·∂φ/∂ρⱼ/φ²")
print()

print("Term 3a: ∂²rss/∂ρⱼ∂ρᵢ")
print("  ∂rss/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ")
print("  ∂²rss/∂ρⱼ∂ρᵢ = -2·(∂r/∂ρⱼ)'·X·∂β/∂ρᵢ - 2·r'·X·∂²β/∂ρⱼ∂ρᵢ")
print()
print("where:")
print("  ∂r/∂ρⱼ = -X·∂β/∂ρⱼ")
print("  ∂²β/∂ρⱼ∂ρᵢ = -∂A⁻¹/∂ρⱼ·λᵢ·Sᵢ·β - A⁻¹·λᵢ·Sᵢ·∂β/∂ρⱼ")
print("                = A⁻¹·λⱼ·Sⱼ·A⁻¹·λᵢ·Sᵢ·β - A⁻¹·λᵢ·Sᵢ·∂β/∂ρⱼ")
print()

def d2beta_drho2(Si, Sj, lambda_i, lambda_j):
    """∂²β/∂ρⱼ∂ρᵢ"""
    dbeta_drho_j = -Ainv @ (lambda_j * Sj @ beta)
    term1 = Ainv @ (lambda_j * Sj) @ Ainv @ (lambda_i * Si @ beta)
    term2 = -Ainv @ (lambda_i * Si @ dbeta_drho_j)
    return term1 + term2

def d2rss_drho2(Si, Sj, lambda_i, lambda_j):
    """∂²rss/∂ρⱼ∂ρᵢ"""
    dbeta_drho_i = -Ainv @ (lambda_i * Si @ beta)
    dbeta_drho_j = -Ainv @ (lambda_j * Sj @ beta)
    d2beta = d2beta_drho2(Si, Sj, lambda_i, lambda_j)

    dr_drho_j = -X @ dbeta_drho_j
    term1 = -2 * dr_drho_j @ X @ dbeta_drho_i
    term2 = -2 * residuals @ X @ d2beta
    return term1 + term2

d2rss_11 = d2rss_drho2(S1, S1, lambda1, lambda1)
d2rss_12 = d2rss_drho2(S1, S2, lambda1, lambda2)
d2rss_21 = d2rss_drho2(S2, S1, lambda2, lambda1)
d2rss_22 = d2rss_drho2(S2, S2, lambda2, lambda2)

print(f"∂²rss/∂ρ₁∂ρ₁ = {d2rss_11:.6e}")
print(f"∂²rss/∂ρ₁∂ρ₂ = {d2rss_12:.6e}")
print(f"∂²rss/∂ρ₂∂ρ₁ = {d2rss_21:.6e}")
print(f"∂²rss/∂ρ₂∂ρ₂ = {d2rss_22:.6e}")
print()

h11_term3a = d2rss_11 / phi
h12_term3a = d2rss_12 / phi
h21_term3a = d2rss_21 / phi
h22_term3a = d2rss_22 / phi

h11_term3b = -g1['drss_drho'] * dphi_drho1 / phi**2
h12_term3b = -g1['drss_drho'] * dphi_drho2 / phi**2
h21_term3b = -g2['drss_drho'] * dphi_drho1 / phi**2
h22_term3b = -g2['drss_drho'] * dphi_drho2 / phi**2

print("Term 3 contributions:")
print(f"H[1,1] term3a = {h11_term3a:.6f}, term3b = {h11_term3b:.6f}")
print(f"H[1,2] term3a = {h12_term3a:.6f}, term3b = {h12_term3b:.6f}")
print(f"H[2,1] term3a = {h21_term3a:.6f}, term3b = {h21_term3b:.6f}")
print(f"H[2,2] term3a = {h22_term3a:.6f}, term3b = {h22_term3b:.6f}")
print()

# ============================================================================
# TOTAL HESSIAN
# ============================================================================
print("="*80)
print("TOTAL HESSIAN")
print("="*80)

H_analytical = np.array([
    [h11_term1 + h11_term2a + h11_term2b + h11_term3a + h11_term3b,
     h12_term1 + h12_term2a + h12_term2b + h12_term3a + h12_term3b],
    [h21_term1 + h21_term2a + h21_term2b + h21_term3a + h21_term3b,
     h22_term1 + h22_term2a + h22_term2b + h22_term3a + h22_term3b]
])

print("Analytical Hessian:")
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

def reml_gradient_numerical(rho, eps=1e-7):
    """Numerical gradient"""
    grad = np.zeros(len(rho))
    for i in range(len(rho)):
        rho_plus = rho.copy()
        rho_plus[i] += eps
        rho_minus = rho.copy()
        rho_minus[i] -= eps
        grad[i] = (reml_criterion(rho_plus) - reml_criterion(rho_minus)) / (2 * eps)
    return grad

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

if np.max(np.abs(diff)) < 1e-4:
    print("\n✅ HESSIAN VALIDATED! Analytical matches numerical to high precision.")
else:
    print("\n⚠️  Large discrepancy detected. Review derivation.")
