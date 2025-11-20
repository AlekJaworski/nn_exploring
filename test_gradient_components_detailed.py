"""
Detailed component-by-component comparison of REML gradient
"""
import numpy as np
import pandas as pd
import subprocess

# Load matrices
X = pd.read_csv('/tmp/X_matrix.csv').values
S1_full = pd.read_csv('/tmp/S1_full.csv').values
S2_full = pd.read_csv('/tmp/S2_full.csv').values
y = pd.read_csv('/tmp/trace_step_data.csv')['y'].values

lambda1, lambda2 = 2.0, 3.0

# Get detailed breakdown from mgcv
r_code = f"""
library(mgcv)
df <- read.csv('/tmp/trace_step_data.csv')
X <- as.matrix(read.csv('/tmp/X_matrix.csv'))
S1_full <- as.matrix(read.csv('/tmp/S1_full.csv'))
S2_full <- as.matrix(read.csv('/tmp/S2_full.csv'))

lambda <- c({lambda1}, {lambda2})
y <- df$y
n <- nrow(X)
p <- ncol(X)

# Compute A = X'X + λ1·S1 + λ2·S2
XtX <- t(X) %*% X
A <- XtX + lambda[1] * S1_full + lambda[2] * S2_full
Ainv <- solve(A)

# Compute beta
beta <- Ainv %*% (t(X) %*% y)

# Residuals
fitted <- X %*% beta
residuals <- y - fitted
rss <- sum(residuals^2)

# Trace terms
trace1 <- sum(diag(Ainv %*% (lambda[1] * S1_full)))
trace2 <- sum(diag(Ainv %*% (lambda[2] * S2_full)))

# Penalty terms
penalty1 <- lambda[1] * t(beta) %*% S1_full %*% beta
penalty2 <- lambda[2] * t(beta) %*% S2_full %*% beta

# Rank estimation - CORRECT way for CR splines
# For s(x, k=10, bs="cr"), the rank is k-2 NOT k-3
rank1 <- sum(eigen(S1_full, only.values=TRUE)$values > 1e-10 * max(eigen(S1_full, only.values=TRUE)$values))
rank2 <- sum(eigen(S2_full, only.values=TRUE)$values > 1e-10 * max(eigen(S2_full, only.values=TRUE)$values))

# edf (effective degrees of freedom)
edf1 <- trace1
edf2 <- trace2

# phi estimation - multiple possible formulas
# Method 1: Using total edf
phi_v1 <- rss / (n - edf1 - edf2 - 1)

# Method 2: Using sum of ranks
phi_v2 <- rss / (n - rank1 - rank2 - 1)

# Method 3: Using trace of full smoother matrix (Wood 2017, eq 6.18)
# edf_total = trace(A^{{-1}} X'X)
edf_total <- sum(diag(Ainv %*% XtX))
phi_v3 <- rss / (n - edf_total)

# Print all components
cat('trace1:', sprintf('%.10f', trace1), '\\n')
cat('trace2:', sprintf('%.10f', trace2), '\\n')
cat('penalty1:', sprintf('%.10f', penalty1), '\\n')
cat('penalty2:', sprintf('%.10f', penalty2), '\\n')
cat('rank1:', rank1, '\\n')
cat('rank2:', rank2, '\\n')
cat('edf1:', sprintf('%.10f', edf1), '\\n')
cat('edf2:', sprintf('%.10f', edf2), '\\n')
cat('edf_total:', sprintf('%.10f', edf_total), '\\n')
cat('rss:', sprintf('%.10f', rss), '\\n')
cat('phi_v1:', sprintf('%.10f', phi_v1), '\\n')
cat('phi_v2:', sprintf('%.10f', phi_v2), '\\n')
cat('phi_v3:', sprintf('%.10f', phi_v3), '\\n')
cat('n:', n, '\\n')

# Compute gradients with each phi version
grad1_v1 <- (trace1 - rank1 + penalty1/phi_v1) / 2
grad2_v1 <- (trace2 - rank2 + penalty2/phi_v1) / 2

grad1_v2 <- (trace1 - rank1 + penalty1/phi_v2) / 2
grad2_v2 <- (trace2 - rank2 + penalty2/phi_v2) / 2

grad1_v3 <- (trace1 - rank1 + penalty1/phi_v3) / 2
grad2_v3 <- (trace2 - rank2 + penalty2/phi_v3) / 2

cat('grad1_v1:', sprintf('%.10f', grad1_v1), '\\n')
cat('grad2_v1:', sprintf('%.10f', grad2_v1), '\\n')
cat('grad1_v2:', sprintf('%.10f', grad1_v2), '\\n')
cat('grad2_v2:', sprintf('%.10f', grad2_v2), '\\n')
cat('grad1_v3:', sprintf('%.10f', grad1_v3), '\\n')
cat('grad2_v3:', sprintf('%.10f', grad2_v3), '\\n')
"""

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True,
    text=True
)

print("=" * 80)
print("mgcv Gradient Component Breakdown at λ=[2.0, 3.0]")
print("=" * 80)
print()
print(result.stdout)

# Now compute in Rust/Python
print("=" * 80)
print("Our Implementation Component Breakdown")
print("=" * 80)
print()

# Compute using our method
n, p = X.shape
XtX = X.T @ X
A = XtX + lambda1 * S1_full + lambda2 * S2_full
Ainv = np.linalg.inv(A)
beta = Ainv @ (X.T @ y)

fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)

trace1 = np.sum(np.diag(Ainv @ (lambda1 * S1_full)))
trace2 = np.sum(np.diag(Ainv @ (lambda2 * S2_full)))

penalty1 = lambda1 * beta.T @ S1_full @ beta
penalty2 = lambda2 * beta.T @ S2_full @ beta

# Rank estimation
eigs1 = np.linalg.eigvalsh(S1_full)
eigs2 = np.linalg.eigvalsh(S2_full)
rank1 = np.sum(eigs1 > 1e-10 * np.max(eigs1))
rank2 = np.sum(eigs2 > 1e-10 * np.max(eigs2))

edf1 = trace1
edf2 = trace2
edf_total = np.sum(np.diag(Ainv @ XtX))

phi_v1 = rss / (n - edf1 - edf2 - 1)
phi_v2 = rss / (n - rank1 - rank2 - 1)
phi_v3 = rss / (n - edf_total)

print(f'trace1: {trace1:.10f}')
print(f'trace2: {trace2:.10f}')
print(f'penalty1: {penalty1:.10f}')
print(f'penalty2: {penalty2:.10f}')
print(f'rank1: {rank1}')
print(f'rank2: {rank2}')
print(f'edf1: {edf1:.10f}')
print(f'edf2: {edf2:.10f}')
print(f'edf_total: {edf_total:.10f}')
print(f'rss: {rss:.10f}')
print(f'phi_v1: {phi_v1:.10f}')
print(f'phi_v2: {phi_v2:.10f}')
print(f'phi_v3: {phi_v3:.10f}')
print(f'n: {n}')

grad1_v1 = (trace1 - rank1 + penalty1/phi_v1) / 2
grad2_v1 = (trace2 - rank2 + penalty2/phi_v1) / 2

grad1_v2 = (trace1 - rank1 + penalty1/phi_v2) / 2
grad2_v2 = (trace2 - rank2 + penalty2/phi_v2) / 2

grad1_v3 = (trace1 - rank1 + penalty1/phi_v3) / 2
grad2_v3 = (trace2 - rank2 + penalty2/phi_v3) / 2

print(f'grad1_v1: {grad1_v1:.10f}')
print(f'grad2_v1: {grad2_v1:.10f}')
print(f'grad1_v2: {grad1_v2:.10f}')
print(f'grad2_v2: {grad2_v2:.10f}')
print(f'grad1_v3: {grad1_v3:.10f}')
print(f'grad2_v3: {grad2_v3:.10f}')
