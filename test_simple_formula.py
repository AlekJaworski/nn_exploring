"""
Test the OLD simple gradient formula (with -rank term added) vs numerical
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

# Compute all intermediate quantities
XtWX = X.T @ X
A = XtWX.copy()
for i, lam in enumerate(mgcv_lambda):
    A += lam * penalties[i]

A_inv = np.linalg.inv(A)
beta = A_inv @ (X.T @ y)
fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)

# EDF and phi
total_rank = 40
phi = rss / (n - total_rank)

print("="*80)
print("OLD SIMPLE FORMULA (WITH -rank TERM)")
print("="*80)
print()

# Compute gradient using old simple formula
simple_gradient = np.zeros(5)

for i in range(5):
    lambda_i = mgcv_lambda[i]
    S_i = penalties[i]
    rank_i = np.linalg.matrix_rank(S_i, tol=1e-7)

    # Component 1: tr(A^{-1}·λᵢ·Sᵢ)
    trace = lambda_i * np.trace(A_inv @ S_i)

    # Component 2: -rank    # Component 3: λᵢ·β'·Sᵢ·β / φ
    S_beta = S_i @ beta
    penalty_term = lambda_i * (beta @ S_beta)

    # Old formula: (trace - rank + penalty_term/φ) / 2
    simple_gradient[i] = (trace - rank_i + penalty_term / phi) / 2.0

    print(f"Penalty {i+1}: trace={trace:.6f}, rank={rank_i}, penalty_term/φ={penalty_term/phi:.6f}")
    print(f"  gradient = ({trace:.6f} - {rank_i} + {penalty_term/phi:.6f}) / 2 = {simple_gradient[i]:.6f}")

print()
print("="*80)
print("NUMERICAL GRADIENT (CORRECT)")
print("="*80)

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

print(f"Numerical gradient: {numerical_gradient}")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print()
print(f"Simple formula:     {simple_gradient}")
print(f"Numerical:          {numerical_gradient}")
print(f"Difference:         {simple_gradient - numerical_gradient}")
print(f"Relative error:     {np.linalg.norm(simple_gradient - numerical_gradient) / np.linalg.norm(numerical_gradient):.6e}")
