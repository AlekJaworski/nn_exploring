"""
Find the missing implicit dependencies in gradient
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
print("FINDING IMPLICIT DEPENDENCIES")
print("="*80)
print()

# Forward pass
XtWX = X.T @ X
A = XtWX.copy()
for i, lam in enumerate(mgcv_lambda):
    A += lam * penalties[i]

A_inv = np.linalg.inv(A)
beta = A_inv @ (X.T @ y)
RSS = np.sum((y - X @ beta)**2)
total_rank = 40
phi = RSS / (n - total_rank)

print("The REML formula has these terms:")
print("  REML = [(RSS + Σλᵢ·β'·Sᵢ·β)/φ + (n-r)·log(2πφ) + log|A| - Σr·log(λ)] / 2")
print()
print("When we differentiate w.r.t. ρᵢ = log(λᵢ):")
print()

# The full derivative accounting for implicit dependencies
print("EXPLICIT dependencies (what our simple formula has):")
print("  ∂[log|A|]/∂ρᵢ = λᵢ·tr(A⁻¹·Sᵢ)  ✓")
print("  ∂[-Σr·log(λ)]/∂ρᵢ = -rᵢ  ✓")
print()

print("IMPLICIT dependencies (what we're missing):")
print("  1. β depends on ρ through A: β = A⁻¹·X'y")
print("  2. RSS depends on β: RSS = ||y - X·β||²")
print("  3. φ depends on RSS: φ = RSS / (n-r)")
print()

print("So the penalty term (RSS + Σλᵢ·β'·Sᵢ·β)/φ has hidden dependencies!")
print()

# Let's compute numerical derivative of just the penalty term
def penalty_part(log_lam):
    lam = np.exp(log_lam)
    A_temp = XtWX.copy()
    for j, l in enumerate(lam):
        A_temp += l * penalties[j]

    A_inv_temp = np.linalg.inv(A_temp)
    beta_temp = A_inv_temp @ (X.T @ y)
    rss_temp = np.sum((y - X @ beta_temp)**2)

    pen_sum = 0.0
    for j, l in enumerate(lam):
        s_beta = penalties[j] @ beta_temp
        pen_sum += l * (beta_temp @ s_beta)

    phi_temp = rss_temp / (n - total_rank)

    return (rss_temp + pen_sum) / phi_temp

log_lambda = np.log(mgcv_lambda)
eps = 1e-6

print("Numerical ∂[(RSS + pen)/φ]/∂ρ for each component:")
for i in range(5):
    log_lam_plus = log_lambda.copy()
    log_lam_plus[i] += eps
    d_penalty_part = (penalty_part(log_lam_plus) - penalty_part(log_lambda)) / eps
    print(f"  Component {i+1}: {d_penalty_part:.10e}")

print()

# Now compute what our simple formula gives for this term
print("Our simple formula for penalty term: λᵢ·β'·Sᵢ·β / φ")
for i in range(5):
    lambda_i = mgcv_lambda[i]
    S_i = penalties[i]
    S_beta = S_i @ beta
    simple_term = lambda_i * (beta @ S_beta) / phi
    print(f"  Component {i+1}: {simple_term:.10e}")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("If these don't match, our simple formula is WRONG for the penalty term!")
print("We need to account for ∂β/∂ρ and ∂φ/∂ρ using Implicit Function Theorem.")
