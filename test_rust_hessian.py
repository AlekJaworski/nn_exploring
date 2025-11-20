"""
Test the Rust Hessian implementation against Python analytical formula
"""
import numpy as np
import pandas as pd
import mgcv_rust

# Load test data
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
S1 = pd.read_csv('/tmp/bench_S1.csv').values
S2 = pd.read_csv('/tmp/bench_S2.csv').values

n, p = X.shape
w = np.ones(n)

print("="*80)
print("TESTING RUST HESSIAN IMPLEMENTATION")
print("="*80)
print(f"Problem: n={n}, p={p}")
print()

# Test at lambda = [1.0, 2.0]
lambdas = np.array([1.0, 2.0])
penalties = [S1, S2]

print(f"Testing at λ = {lambdas}")

# Compute Hessian using Rust
hessian_rust = mgcv_rust.reml_hessian_multi_qr_py(y, X, w, lambdas.tolist(), penalties)

print("\nRust Hessian:")
print(hessian_rust)
print()

# Now compute using Python analytical formula
XtX = X.T @ X

# Build A
A = XtX + lambdas[0] * S1 + lambdas[1] * S2
Ainv = np.linalg.inv(A)

# Solve for beta
beta = Ainv @ (X.T @ y)

# Residuals
fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)

# edf and phi
ainv_xtx = Ainv @ XtX
edf_total = np.trace(ainv_xtx)
phi = rss / (n - edf_total)
log_phi = np.log(phi)

# Pre-compute
ainv_xtx_ainv = ainv_xtx @ Ainv

# Compute gradient components (needed for Hessian)
dedf_drho = []
drss_drho = []
for i, (lam, S) in enumerate(zip(lambdas, penalties)):
    # ∂β/∂ρᵢ
    dbeta_drho_i = -Ainv @ (lam * S @ beta)
    # ∂rss/∂ρᵢ
    drss_drho.append(-2 * residuals @ X @ dbeta_drho_i)
    # ∂edf/∂ρᵢ
    dedf_drho.append(-lam * np.sum(ainv_xtx_ainv * S.T))

def compute_hessian_element_python(i, j):
    """Compute H[i,j] using Python analytical formula"""
    lambda_i = lambdas[i]
    lambda_j = lambdas[j]
    Si = penalties[i]
    Sj = penalties[j]

    # Term 1: ∂²log|A|/∂ρⱼ∂ρᵢ
    # Part A: -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
    term1a = -lambda_i * lambda_j * np.trace(Ainv @ Sj @ Ainv @ Si)

    # Part B: δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ) (diagonal correction)
    term1b = 0.0
    if i == j:
        term1b = lambda_i * np.trace(Ainv @ Si)

    term1_total = term1a + term1b

    # Term 2: ∂/∂ρⱼ[∂edf/∂ρᵢ·(-log(φ)+1)]
    # Part A: ∂²edf/∂ρⱼ∂ρᵢ·(-log(φ)+1)
    d2edf_part1 = lambda_i * lambda_j * (
        np.trace(Ainv @ Sj @ Ainv @ Si @ ainv_xtx) +
        np.trace(Ainv @ Si @ Ainv @ Sj @ ainv_xtx)
    )
    d2edf_part2 = 0.0
    if i == j:
        d2edf_part2 = -lambda_i * np.trace(Ainv @ Si @ ainv_xtx)

    d2edf = d2edf_part1 + d2edf_part2
    term2a = d2edf * (-log_phi + 1.0)

    # Part B: ∂edf/∂ρᵢ·∂(-log(φ))/∂ρⱼ
    dphi_drho_j = (drss_drho[j] + phi * dedf_drho[j]) / (n - edf_total)
    dlogphi_drho_j = -dphi_drho_j / phi
    term2b = dedf_drho[i] * dlogphi_drho_j

    term2_total = term2a + term2b

    # Term 3: ∂/∂ρⱼ[∂rss/∂ρᵢ/φ]
    # Compute ∂β/∂ρ for both i and j
    dbeta_drho_i = -Ainv @ (lambda_i * Si @ beta)
    dbeta_drho_j = -Ainv @ (lambda_j * Sj @ beta)

    # Part A: ∂²rss/∂ρⱼ∂ρᵢ/φ
    d2rss_part1 = 2 * (X @ dbeta_drho_j) @ (X @ dbeta_drho_i)

    # ∂²β/∂ρⱼ∂ρᵢ
    d2beta = (Ainv @ (lambda_j * Sj) @ Ainv @ (lambda_i * Si @ beta) -
              Ainv @ (lambda_i * Si @ dbeta_drho_j))
    if i == j:
        d2beta = d2beta - dbeta_drho_i

    d2rss_part2 = -2 * residuals @ X @ d2beta

    d2rss = d2rss_part1 + d2rss_part2
    term3a = d2rss / phi

    # Part B: -∂rss/∂ρᵢ·∂φ/∂ρⱼ/φ²
    term3b = -drss_drho[i] * dphi_drho_j / phi**2

    term3_total = term3a + term3b

    # Total
    return term1_total + term2_total + term3_total

# Compute Python Hessian
hessian_python = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        hessian_python[i, j] = compute_hessian_element_python(i, j)

print("Python Analytical Hessian:")
print(hessian_python)
print()

print("Difference (Rust - Python):")
diff = hessian_rust - hessian_python
print(diff)
print()

print("Relative error:")
rel_error = np.abs(diff) / (np.abs(hessian_python) + 1e-10)
print(rel_error)
print()

print(f"Maximum absolute error: {np.max(np.abs(diff)):.6e}")
print(f"Maximum relative error: {np.max(rel_error):.6e}")

if np.max(np.abs(diff)) < 1e-6:
    print("\n✅ RUST HESSIAN VALIDATED! Matches Python analytical formula to high precision.")
else:
    print(f"\n⚠️  Error detected: {np.max(np.abs(diff)):.6e}")
