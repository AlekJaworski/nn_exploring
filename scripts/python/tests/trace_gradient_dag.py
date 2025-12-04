"""
Step-by-step DAG tracing of gradient computation to find the 30% error
"""
import numpy as np
import pandas as pd
import mgcv_rust

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
print("GRADIENT DAG TRACE - Finding the 30% Error")
print("="*80)
print()

# ============================================================================
# Step 1: Compute forward pass - all intermediates
# ============================================================================
print("STEP 1: Forward pass at λ")
print("-" * 80)

XtWX = X.T @ X
A = XtWX.copy()
for i, lam in enumerate(mgcv_lambda):
    A += lam * penalties[i]

A_inv = np.linalg.inv(A)
beta = A_inv @ (X.T @ y)
fitted = X @ beta
residuals = y - fitted
RSS = np.sum(residuals**2)

# Compute total rank
total_rank = sum([np.linalg.matrix_rank(S, tol=1e-7) for S in penalties])

# φ using total_rank
phi = RSS / (n - total_rank)

print(f"RSS = {RSS:.10e}")
print(f"total_rank = {total_rank}")
print(f"φ = RSS / (n - total_rank) = {phi:.10e}")
print()

# ============================================================================
# Step 2: Compute REML and verify it matches our formula
# ============================================================================
print("STEP 2: Compute REML")
print("-" * 80)

# Penalty sum
penalty_sum = 0.0
for i, lam in enumerate(mgcv_lambda):
    S_beta = penalties[i] @ beta
    penalty_sum += lam * (beta @ S_beta)

# Log lambda sum
log_lambda_sum = 0.0
for i, S in enumerate(penalties):
    rank_s = np.linalg.matrix_rank(S, tol=1e-7)
    if rank_s > 0 and mgcv_lambda[i] > 1e-10:
        log_lambda_sum += rank_s * np.log(mgcv_lambda[i])

# log|A|
sign, logdet_A = np.linalg.slogdet(A)

# REML formula
REML = ((RSS + penalty_sum) / phi
        + (n - total_rank) * np.log(2 * np.pi * phi)
        + logdet_A
        - log_lambda_sum) / 2.0

print(f"REML = {REML:.10e}")
print()

# ============================================================================
# Step 3: Numerical gradient by finite differences
# ============================================================================
print("STEP 3: Numerical gradient (ground truth)")
print("-" * 80)

def reml_function(log_lam):
    """REML as function of log(λ)"""
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

    log_lam_sum = 0.0
    for j, S in enumerate(penalties):
        rank_s = np.linalg.matrix_rank(S, tol=1e-7)
        if rank_s > 0 and lam[j] > 1e-10:
            log_lam_sum += rank_s * np.log(lam[j])

    phi_temp = rss_temp / (n - total_rank)
    sign, logdet = np.linalg.slogdet(A_temp)

    reml = ((rss_temp + pen_sum) / phi_temp
            + (n - total_rank) * np.log(2 * np.pi * phi_temp)
            + logdet
            - log_lam_sum) / 2.0

    return reml

eps = 1e-6
log_lambda = np.log(mgcv_lambda)
numerical_grad = np.zeros(5)

for i in range(5):
    log_lam_plus = log_lambda.copy()
    log_lam_plus[i] += eps
    reml_plus = reml_function(log_lam_plus)
    numerical_grad[i] = (reml_plus - REML) / eps

print(f"Numerical ∂REML/∂ρ: {numerical_grad}")
print(f"Norm: {np.linalg.norm(numerical_grad):.6e}")
print()

# ============================================================================
# Step 4: Analytical gradient using our formula - COMPONENT BY COMPONENT
# ============================================================================
print("STEP 4: Analytical gradient - component by component")
print("-" * 80)
print()

analytical_grad = np.zeros(5)

for i in range(5):
    lambda_i = mgcv_lambda[i]
    S_i = penalties[i]
    rank_i = np.linalg.matrix_rank(S_i, tol=1e-7)

    print(f"--- Component {i+1} (λ={lambda_i:.6e}) ---")

    # Term 1: λᵢ·tr(A^{-1}·Sᵢ)
    trace_term = lambda_i * np.trace(A_inv @ S_i)
    print(f"  Term 1 (λ·tr(A⁻¹·S)): {trace_term:.10e}")

    # Term 2: -rank
    rank_term = -rank_i
    print(f"  Term 2 (-rank): {rank_term:.10e}")

    # Term 3: λᵢ·β'·Sᵢ·β / φ
    S_beta = S_i @ beta
    beta_S_beta = beta @ S_beta
    penalty_term = lambda_i * beta_S_beta / phi
    print(f"  Term 3 (λ·β'·S·β/φ): {penalty_term:.10e}")

    # Sum
    sum_terms = trace_term + rank_term + penalty_term
    print(f"  Sum before /2: {sum_terms:.10e}")

    # Divide by 2
    gradient_i = sum_terms / 2.0
    analytical_grad[i] = gradient_i
    print(f"  Gradient /2: {gradient_i:.10e}")
    print()

print(f"Analytical gradient: {analytical_grad}")
print(f"Numerical gradient:  {numerical_grad}")
print(f"Difference:          {analytical_grad - numerical_grad}")
print()

# ============================================================================
# Step 5: Check if numerical gradient for EACH TERM separately
# ============================================================================
print("="*80)
print("STEP 5: Numerical derivatives of INDIVIDUAL TERMS")
print("="*80)
print()

# Let's compute numerical derivative of each term in the REML formula
# REML = [(RSS+pen)/φ + (n-r)·log(2πφ) + log|A| - Σr·log(λ)] / 2

print("Term-by-term numerical derivatives:")
print()

for i in range(5):
    print(f"--- Component {i+1} ---")

    # Numerical ∂[log|A|]/∂ρᵢ
    def logdet_func(log_lam):
        lam = np.exp(log_lam)
        A_temp = XtWX.copy()
        for j, l in enumerate(lam):
            A_temp += l * penalties[j]
        sign, logdet = np.linalg.slogdet(A_temp)
        return logdet

    log_lam_plus = log_lambda.copy()
    log_lam_plus[i] += eps
    dlogdet_drho = (logdet_func(log_lam_plus) - logdet_func(log_lambda)) / eps
    print(f"  ∂[log|A|]/∂ρ: {dlogdet_drho:.10e}")

    # Compare with analytical: λᵢ·tr(A^{-1}·Sᵢ)
    lambda_i = mgcv_lambda[i]
    S_i = penalties[i]
    analytical_dlogdet = lambda_i * np.trace(A_inv @ S_i)
    print(f"  Analytical λ·tr(A⁻¹·S): {analytical_dlogdet:.10e}")
    print(f"  Difference: {dlogdet_drho - analytical_dlogdet:.10e}")
    print()

    # Numerical ∂[-Σrⱼ·log(λⱼ)]/∂ρᵢ = -rᵢ
    def log_lambda_term(log_lam):
        lam = np.exp(log_lam)
        log_lam_sum = 0.0
        for j, S in enumerate(penalties):
            rank_s = np.linalg.matrix_rank(S, tol=1e-7)
            if rank_s > 0 and lam[j] > 1e-10:
                log_lam_sum += rank_s * np.log(lam[j])
        return -log_lam_sum

    dlog_lambda_term = (log_lambda_term(log_lam_plus) - log_lambda_term(log_lambda)) / eps
    rank_i = np.linalg.matrix_rank(S_i, tol=1e-7)
    print(f"  ∂[-Σr·log(λ)]/∂ρ: {dlog_lambda_term:.10e}")
    print(f"  Analytical -rank: {-rank_i:.10e}")
    print(f"  Difference: {dlog_lambda_term - (-rank_i):.10e}")
    print()

print("="*80)
print("DIAGNOSIS")
print("="*80)
print()
print(f"Analytical vs Numerical error: {np.linalg.norm(analytical_grad - numerical_grad) / np.linalg.norm(numerical_grad):.2%}")
print()
print("If term-by-term derivatives match, but total doesn't:")
print("  → Problem is in how terms combine (φ dependence, implicit derivatives)")
print("If individual terms don't match:")
print("  → Problem is in individual term formulas")
