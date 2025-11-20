"""
Compute the CORRECT gradient using Implicit Function Theorem
"""
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('/tmp/fresh_data.csv')
X = pd.read_csv('/tmp/fresh_X.csv').values
y = df['y'].values
penalties = []
for i in range(5):
    S = pd.read_csv(f'/tmp/fresh_S{i+1}.csv').values
    penalties.append(S)

n, p = X.shape
w = np.ones(n)

mgcv_lambda = np.array([0.2705535, 9038.71, 150.8265, 400.144, 13747035])

print("="*80)
print("CORRECT GRADIENT COMPUTATION WITH IMPLICIT FUNCTION THEOREM")
print("="*80)
print()

# Forward pass
XtWX = X.T @ X
A = XtWX.copy()
for i, lam in enumerate(mgcv_lambda):
    A += lam * penalties[i]

A_inv = np.linalg.inv(A)
beta = A_inv @ (X.T @ y)
residuals = y - X @ beta
RSS = np.sum(residuals**2)
total_rank = 40
phi = RSS / (n - total_rank)

print(f"phi = {phi:.10e}")
print(f"RSS = {RSS:.10e}")
print()

# Compute correct gradient for each component
correct_gradient = np.zeros(5)

for i in range(5):
    lambda_i = mgcv_lambda[i]
    S_i = penalties[i]
    rank_i = np.linalg.matrix_rank(S_i, tol=1e-7)

    print(f"\n--- Component {i+1} (λ = {lambda_i:.6e}) ---")

    # Term 1: ∂[log|A|]/∂ρᵢ = λᵢ·tr(A⁻¹·Sᵢ)
    trace_term = lambda_i * np.trace(A_inv @ S_i)
    print(f"  ∂[log|A|]/∂ρᵢ = {trace_term:.10e}")

    # Term 2: ∂[-Σrⱼ·log(λⱼ)]/∂ρᵢ = -rᵢ
    rank_term = -rank_i
    print(f"  ∂[-Σr·log(λ)]/∂ρᵢ = {rank_term:.10e}")

    # Term 3: ∂[(RSS + Σλⱼ·β'·Sⱼ·β)/φ]/∂ρᵢ
    # Using IFT: ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
    S_i_beta = S_i @ beta
    dbeta_drho = -A_inv @ (lambda_i * S_i_beta)

    # ∂RSS/∂ρᵢ = -2·residuals'·X·∂β/∂ρᵢ
    dRSS_drho = -2.0 * (residuals @ (X @ dbeta_drho))

    # ∂φ/∂ρᵢ = ∂RSS/∂ρᵢ / (n-r)
    dphi_drho = dRSS_drho / (n - total_rank)

    # ∂[Σλⱼ·β'·Sⱼ·β]/∂ρᵢ = λᵢ·β'·Sᵢ·β + 2·Σλⱼ·β'·Sⱼ·∂β/∂ρᵢ
    explicit_pen = lambda_i * (beta @ S_i_beta)

    implicit_pen = 0.0
    for j, lam_j in enumerate(mgcv_lambda):
        S_j_beta = penalties[j] @ beta
        S_j_dbeta = penalties[j] @ dbeta_drho
        implicit_pen += lam_j * (S_j_beta @ dbeta_drho + beta @ S_j_dbeta)

    dP_drho = dRSS_drho + explicit_pen + implicit_pen

    # P = RSS + Σλⱼ·β'·Sⱼ·β
    P = RSS
    for j, lam_j in enumerate(mgcv_lambda):
        S_j_beta = penalties[j] @ beta
        P += lam_j * (beta @ S_j_beta)

    # ∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ
    penalty_total_deriv = (dP_drho / phi) - (P / (phi**2)) * dphi_drho

    print(f"  ∂β/∂ρᵢ norm = {np.linalg.norm(dbeta_drho):.10e}")
    print(f"  ∂RSS/∂ρᵢ = {dRSS_drho:.10e}")
    print(f"  ∂φ/∂ρᵢ = {dphi_drho:.10e}")
    print(f"  Explicit pen term: {explicit_pen:.10e}")
    print(f"  Implicit pen term: {implicit_pen:.10e}")
    print(f"  ∂P/∂ρᵢ = {dP_drho:.10e}")
    print(f"  ∂(P/φ)/∂ρᵢ = {penalty_total_deriv:.10e}")

    # Term 4: ∂[(n-r)·log(2πφ)]/∂ρᵢ = (n-r)·(1/φ)·∂φ/∂ρᵢ
    log_phi_term = (n - total_rank) * (dphi_drho / phi)
    print(f"  ∂[(n-r)·log(2πφ)]/∂ρᵢ = {log_phi_term:.10e}")

    # Total gradient (divide by 2)
    correct_gradient[i] = (trace_term + rank_term + penalty_total_deriv + log_phi_term) / 2.0
    print(f"  TOTAL ∂REML/∂ρᵢ = {correct_gradient[i]:.10e}")

print()
print("="*80)
print("COMPARISON WITH NUMERICAL")
print("="*80)
print()

# Numerical gradient
def reml_python(log_lambda):
    lambdas = np.exp(log_lambda)
    A_temp = XtWX.copy()
    for j, lam in enumerate(lambdas):
        A_temp += lam * penalties[j]

    Ainv_temp = np.linalg.inv(A_temp)
    beta_temp = Ainv_temp @ (X.T @ y)
    rss_temp = np.sum((y - X @ beta_temp)**2)

    penalty_sum = 0.0
    for j, lam in enumerate(lambdas):
        s_beta = penalties[j] @ beta_temp
        penalty_sum += lam * (beta_temp @ s_beta)

    log_lambda_sum = 0.0
    for j, S in enumerate(penalties):
        rank_s = np.linalg.matrix_rank(S, tol=1e-7)
        if rank_s > 0 and lambdas[j] > 1e-10:
            log_lambda_sum += rank_s * np.log(lambdas[j])

    phi_temp = rss_temp / (n - total_rank)
    sign, logdet = np.linalg.slogdet(A_temp)

    reml = ((rss_temp + penalty_sum) / phi_temp
            + (n - total_rank) * np.log(2 * np.pi * phi_temp)
            + logdet
            - log_lambda_sum) / 2.0

    return reml

eps = 1e-6
reml_0 = reml_python(np.log(mgcv_lambda))
numerical_gradient = np.zeros(5)
for i in range(5):
    rho = np.log(mgcv_lambda)
    rho_plus = rho.copy()
    rho_plus[i] += eps
    reml_plus = reml_python(rho_plus)
    numerical_gradient[i] = (reml_plus - reml_0) / eps

print(f"Correct analytical: {correct_gradient}")
print(f"Numerical:          {numerical_gradient}")
print(f"Difference:         {correct_gradient - numerical_gradient}")
print(f"Relative error:     {np.linalg.norm(correct_gradient - numerical_gradient) / np.linalg.norm(numerical_gradient):.6e}")
