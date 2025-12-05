"""
Methodically trace through REML computation DAG to find divergence from mgcv
"""
import numpy as np
import pandas as pd
import subprocess

print("="*80)
print("STEP-BY-STEP REML COMPUTATION TRACE")
print("="*80)
print()

# Load the fresh data (same as mgcv used)
df = pd.read_csv('/tmp/fresh_data.csv')
X = pd.read_csv('/tmp/fresh_X.csv').values
y = df['y'].values
penalties = []
for i in range(5):
    S = pd.read_csv(f'/tmp/fresh_S{i+1}.csv').values
    penalties.append(S)

n, p = X.shape
w = np.ones(n)

# mgcv's solution
mgcv_lambda = np.array([0.2705535, 9038.71, 150.8265, 400.144, 13747035])

print(f"Input dimensions: n={n}, p={p}, d={len(penalties)}")
print(f"Testing at mgcv's λ: {mgcv_lambda}")
print()

# ============================================================================
# STEP 1: Compute X'WX
# ============================================================================
print("="*80)
print("STEP 1: Compute X'WX")
print("="*80)

# Since W = I (all weights = 1), X'WX = X'X
XtWX = X.T @ X

print(f"X'WX shape: {XtWX.shape}")
print(f"X'WX[0,0]: {XtWX[0,0]:.10e}")
print(f"X'WX trace: {np.trace(XtWX):.10e}")
print(f"X'WX Frobenius norm: {np.linalg.norm(XtWX, 'fro'):.10e}")
print()

# ============================================================================
# STEP 2: Compute A = X'WX + Σλᵢ·Sᵢ
# ============================================================================
print("="*80)
print("STEP 2: Compute A = X'WX + Σλᵢ·Sᵢ")
print("="*80)

A = XtWX.copy()
for i, (lam, S) in enumerate(zip(mgcv_lambda, penalties)):
    A += lam * S
    print(f"After adding λ_{i+1}·S_{i+1} (λ={lam:.2e}):")
    print(f"  A[0,0] = {A[0,0]:.10e}")
    print(f"  A trace = {np.trace(A):.10e}")

print(f"\nFinal A[0,0]: {A[0,0]:.10e}")
print(f"Final A trace: {np.trace(A):.10e}")
print(f"Final A Frobenius norm: {np.linalg.norm(A, 'fro'):.10e}")
print()

# ============================================================================
# STEP 3: Solve for β: A·β = X'Wy
# ============================================================================
print("="*80)
print("STEP 3: Solve for β")
print("="*80)

XtWy = X.T @ y
print(f"X'Wy[0]: {XtWy[0]:.10e}")
print(f"||X'Wy||: {np.linalg.norm(XtWy):.10e}")

beta = np.linalg.solve(A, XtWy)
print(f"\nβ[0]: {beta[0]:.10e}")
print(f"β[1]: {beta[1]:.10e}")
print(f"||β||: {np.linalg.norm(beta):.10e}")
print()

# ============================================================================
# STEP 4: Compute fitted values and residuals
# ============================================================================
print("="*80)
print("STEP 4: Compute fitted values and residuals")
print("="*80)

fitted = X @ beta
residuals = y - fitted

print(f"fitted[0]: {fitted[0]:.10e}")
print(f"||fitted||: {np.linalg.norm(fitted):.10e}")
print(f"residuals[0]: {residuals[0]:.10e}")
print(f"||residuals||: {np.linalg.norm(residuals):.10e}")
print()

# ============================================================================
# STEP 5: Compute RSS
# ============================================================================
print("="*80)
print("STEP 5: Compute RSS = r'·W·r")
print("="*80)

RSS = np.sum(residuals**2)  # W = I
print(f"RSS = {RSS:.10e}")
print()

# ============================================================================
# STEP 6: Compute penalty term Σλᵢ·β'·Sᵢ·β
# ============================================================================
print("="*80)
print("STEP 6: Compute penalty term Σλᵢ·β'·Sᵢ·β")
print("="*80)

penalty_sum = 0.0
for i, (lam, S) in enumerate(zip(mgcv_lambda, penalties)):
    S_beta = S @ beta
    beta_S_beta = beta @ S_beta
    penalty_sum += lam * beta_S_beta
    print(f"λ_{i+1}·β'·S_{i+1}·β = {lam * beta_S_beta:.10e}")

print(f"\nTotal penalty sum: {penalty_sum:.10e}")
print(f"RSS + penalty: {RSS + penalty_sum:.10e}")
print()

# ============================================================================
# STEP 7: Compute EDF = tr(A^{-1}·X'WX)
# ============================================================================
print("="*80)
print("STEP 7: Compute EDF = tr(A^{-1}·X'WX)")
print("="*80)

A_inv = np.linalg.inv(A)
A_inv_XtWX = A_inv @ XtWX
edf = np.trace(A_inv_XtWX)

print(f"tr(A^{-1}·X'WX) = {edf:.10e}")
print(f"n - EDF = {n - edf:.10e}")
print()

# ============================================================================
# STEP 8: Compute rank of penalties
# ============================================================================
print("="*80)
print("STEP 8: Compute rank of penalties")
print("="*80)

total_rank = 0
log_lambda_sum = 0.0
for i, (lam, S) in enumerate(zip(mgcv_lambda, penalties)):
    rank_S = np.linalg.matrix_rank(S, tol=1e-7)
    total_rank += rank_S
    if lam > 1e-10:
        log_lambda_sum += rank_S * np.log(lam)
    print(f"rank(S_{i+1}) = {rank_S}, λ_{i+1} = {lam:.6e}, contribution = {rank_S * np.log(lam):.6e}")

print(f"\nTotal rank: {total_rank}")
print(f"Σrank(Sᵢ)·log(λᵢ): {log_lambda_sum:.10e}")
print()

# ============================================================================
# STEP 9: Compute φ (scale parameter)
# ============================================================================
print("="*80)
print("STEP 9: Compute φ")
print("="*80)

phi = RSS / (n - total_rank)
print(f"φ = RSS / (n - Σrank(Sᵢ))")
print(f"φ = {RSS:.10e} / {n - total_rank}")
print(f"φ = {phi:.10e}")
print(f"log(φ) = {np.log(phi):.10e}")
print(f"log(2π·φ) = {np.log(2 * np.pi * phi):.10e}")
print()

# ============================================================================
# STEP 10: Compute log|A|
# ============================================================================
print("="*80)
print("STEP 10: Compute log|A|")
print("="*80)

sign_A, logdet_A = np.linalg.slogdet(A)
print(f"log|A| = {logdet_A:.10e}")
print()

# ============================================================================
# STEP 11: Compute REML using our formula
# ============================================================================
print("="*80)
print("STEP 11: Compute REML (our formula)")
print("="*80)

# Our formula:
# REML = [(RSS + Σλᵢ·β'·Sᵢ·β)/φ + (n-Σrank(Sᵢ))·log(2πφ) + log|A| - Σrank(Sᵢ)·log(λᵢ)] / 2

RSS_penalty = RSS + penalty_sum
term1 = RSS_penalty / phi
term2 = (n - total_rank) * np.log(2 * np.pi * phi)
term3 = logdet_A
term4 = -log_lambda_sum

REML_ours = (term1 + term2 + term3 + term4) / 2.0

print(f"Term 1 (RSS+penalty)/φ: {term1:.10e}")
print(f"Term 2 (n-r)·log(2πφ): {term2:.10e}")
print(f"Term 3 log|A|: {term3:.10e}")
print(f"Term 4 -Σr·log(λ): {term4:.10e}")
print(f"\nSum: {term1 + term2 + term3 + term4:.10e}")
print(f"REML (ours) = {REML_ours:.10e}")
print()

# ============================================================================
# STEP 12: Ask mgcv what it computed at these same values
# ============================================================================
print("="*80)
print("STEP 12: Ask mgcv to compute REML at same λ")
print("="*80)

# Save current state for R to use
pd.DataFrame(beta).to_csv('/tmp/trace_beta.csv', index=False, header=False)

r_code = """
library(mgcv)

# Load the data
df <- read.csv('/tmp/fresh_data.csv')
X <- as.matrix(read.csv('/tmp/fresh_X.csv'))
y <- df$y
n <- nrow(X)
p <- ncol(X)

# Load penalties
penalties <- list()
for (i in 1:5) {
    penalties[[i]] <- as.matrix(read.csv(paste0('/tmp/fresh_S', i, '.csv')))
}

# Use mgcv's λ
lambda <- c(0.2705535, 9038.71, 150.8265, 400.144, 13747035)

# Compute A = X'X + Σλᵢ·Sᵢ
A <- t(X) %*% X
for (i in 1:5) {
    A <- A + lambda[i] * penalties[[i]]
}

# Solve for beta
beta <- solve(A, t(X) %*% y)

# RSS
fitted <- X %*% beta
residuals <- y - fitted
RSS <- sum(residuals^2)

# Penalty
penalty_sum <- 0
for (i in 1:5) {
    penalty_sum <- penalty_sum + lambda[i] * t(beta) %*% penalties[[i]] %*% beta
}

# EDF
A_inv <- solve(A)
XtX <- t(X) %*% X
edf <- sum(diag(A_inv %*% XtX))

# Rank
total_rank <- 0
log_lambda_sum <- 0
for (i in 1:5) {
    rank_S <- qr(penalties[[i]])$rank
    total_rank <- total_rank + rank_S
    log_lambda_sum <- log_lambda_sum + rank_S * log(lambda[i])
}

# phi
phi <- RSS / (n - total_rank)

# log|A|
logdet_A <- determinant(A, logarithm=TRUE)$modulus[1]

# Now compute what mgcv would report
# Try to extract REML from a gam fit with these exact parameters
fit <- gam(y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr') +
               s(x3, k=10, bs='cr') + s(x4, k=10, bs='cr') +
               s(x5, k=10, bs='cr'),
           data=df, method='REML', sp=lambda)

cat('\\nMy R computation:\\n')
cat('RSS:', RSS, '\\n')
cat('Penalty sum:', penalty_sum, '\\n')
cat('RSS + penalty:', RSS + penalty_sum, '\\n')
cat('EDF:', edf, '\\n')
cat('Total rank:', total_rank, '\\n')
cat('phi:', phi, '\\n')
cat('log|A|:', logdet_A, '\\n')
cat('log_lambda_sum:', log_lambda_sum, '\\n')

# Manual REML calculation
term1 <- (RSS + penalty_sum) / phi
term2 <- (n - total_rank) * log(2 * pi * phi)
term3 <- logdet_A
term4 <- -log_lambda_sum
REML_manual <- (term1 + term2 + term3 + term4) / 2

cat('\\nREML (manual formula):', REML_manual, '\\n')
cat('mgcv fit$gcv.ubre:', fit$gcv.ubre, '\\n')
cat('mgcv deviance:', deviance(fit), '\\n')

# Check beta match
beta_saved <- as.matrix(read.csv('/tmp/trace_beta.csv', header=FALSE))
cat('\\nBeta[1] (R):', beta[1], '\\n')
cat('Beta[1] (Python):', beta_saved[1], '\\n')
cat('Max |beta_R - beta_Python|:', max(abs(beta - beta_saved)), '\\n')
"""

result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    errors = [line for line in result.stderr.split('\n')
              if line and 'Loading' not in line and 'This is mgcv' not in line]
    if errors:
        print("R messages:", '\n'.join(errors[:5]))

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Our REML computation: {REML_ours:.6f}")
print(f"mgcv reported: 1238.813")
print(f"Difference: {1238.813 - REML_ours:.6f}")
