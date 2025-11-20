"""
Identify EXACT divergence point between Rust and mgcv gradient computation
Compare every single intermediate value step-by-step
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

print("=" * 80)
print("EXACT DIVERGENCE POINT ANALYSIS")
print("=" * 80)
print()

# Get mgcv's detailed computation
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

# STEP 1: Compute A = X'X + λ1·S1 + λ2·S2
XtX <- t(X) %*% X
A <- XtX + lambda[1] * S1_full + lambda[2] * S2_full

# STEP 2: Compute A^{{-1}}
Ainv <- solve(A)

# STEP 3: Compute beta = A^{{-1}} X'y
beta <- Ainv %*% (t(X) %*% y)

# STEP 4: Compute fitted and residuals
fitted <- X %*% beta
residuals <- y - fitted
rss <- sum(residuals^2)

# STEP 5: Compute trace terms
trace1 <- sum(diag(Ainv %*% (lambda[1] * S1_full)))
trace2 <- sum(diag(Ainv %*% (lambda[2] * S2_full)))

# STEP 6: Compute penalty terms β'·S·β
penalty1 <- lambda[1] * t(beta) %*% S1_full %*% beta
penalty2 <- lambda[2] * t(beta) %*% S2_full %*% beta

# STEP 7: Rank
rank1 <- sum(eigen(S1_full, only.values=TRUE)$values > 1e-10 * max(eigen(S1_full, only.values=TRUE)$values))
rank2 <- sum(eigen(S2_full, only.values=TRUE)$values > 1e-10 * max(eigen(S2_full, only.values=TRUE)$values))

# STEP 8: edf and phi
edf_total <- sum(diag(Ainv %*% XtX))
phi <- rss / (n - edf_total)

# STEP 9: Gradient components (BEFORE division by 2)
grad1_numerator <- trace1 - rank1 + penalty1/phi
grad2_numerator <- trace2 - rank2 + penalty2/phi

# STEP 10: Final gradient (AFTER division by 2)
grad1 <- grad1_numerator / 2
grad2 <- grad2_numerator / 2

# Print everything
cat('STEP 1: A matrix\\n')
cat('A[1,1]:', sprintf('%.10f', A[1,1]), '\\n')
cat('A[2,2]:', sprintf('%.10f', A[2,2]), '\\n')
cat('\\n')

cat('STEP 2: A^{{-1}}\\n')
cat('Ainv[1,1]:', sprintf('%.10f', Ainv[1,1]), '\\n')
cat('Ainv[2,2]:', sprintf('%.10f', Ainv[2,2]), '\\n')
cat('\\n')

cat('STEP 3: beta\\n')
cat('beta[1]:', sprintf('%.10f', beta[1]), '\\n')
cat('beta[2]:', sprintf('%.10f', beta[2]), '\\n')
cat('beta[19]:', sprintf('%.10f', beta[19]), '\\n')
cat('||beta||^2:', sprintf('%.10f', sum(beta^2)), '\\n')
cat('\\n')

cat('STEP 4: Residuals\\n')
cat('rss:', sprintf('%.10f', rss), '\\n')
cat('\\n')

cat('STEP 5: Trace terms\\n')
cat('trace1:', sprintf('%.10f', trace1), '\\n')
cat('trace2:', sprintf('%.10f', trace2), '\\n')
cat('\\n')

cat('STEP 6: Penalty terms\\n')
cat('penalty1:', sprintf('%.10f', penalty1), '\\n')
cat('penalty2:', sprintf('%.10f', penalty2), '\\n')
cat('\\n')

cat('STEP 7: Ranks\\n')
cat('rank1:', rank1, '\\n')
cat('rank2:', rank2, '\\n')
cat('\\n')

cat('STEP 8: edf and phi\\n')
cat('edf_total:', sprintf('%.10f', edf_total), '\\n')
cat('phi:', sprintf('%.10f', phi), '\\n')
cat('\\n')

cat('STEP 9: Gradient numerators (before /2)\\n')
cat('grad1_numerator:', sprintf('%.10f', grad1_numerator), '\\n')
cat('grad2_numerator:', sprintf('%.10f', grad2_numerator), '\\n')
cat('  = trace1 - rank1 + penalty1/phi\\n')
cat('  = ', sprintf('%.10f', trace1), ' - ', rank1, ' + ', sprintf('%.10f', penalty1/phi), '\\n')
cat('  = ', sprintf('%.10f', trace2), ' - ', rank2, ' + ', sprintf('%.10f', penalty2/phi), '\\n')
cat('\\n')

cat('STEP 10: Final gradients (after /2)\\n')
cat('grad1:', sprintf('%.10f', grad1), '\\n')
cat('grad2:', sprintf('%.10f', grad2), '\\n')
"""

result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)
print("MGCV COMPUTATION:")
print("=" * 80)
print(result.stdout)

print("\n" + "=" * 80)
print("OUR COMPUTATION:")
print("=" * 80)
print()

# Now compute with our method
n, p = X.shape

# STEP 1: Compute A
XtX = X.T @ X
A = XtX + lambda1 * S1_full + lambda2 * S2_full
print('STEP 1: A matrix')
print(f'A[1,1]: {A[0,0]:.10f}')
print(f'A[2,2]: {A[1,1]:.10f}')
print()

# STEP 2: A^{-1}
Ainv = np.linalg.inv(A)
print('STEP 2: A^{-1}')
print(f'Ainv[1,1]: {Ainv[0,0]:.10f}')
print(f'Ainv[2,2]: {Ainv[1,1]:.10f}')
print()

# STEP 3: beta
beta = Ainv @ (X.T @ y)
print('STEP 3: beta')
print(f'beta[1]: {beta[0]:.10f}')
print(f'beta[2]: {beta[1]:.10f}')
print(f'beta[19]: {beta[18]:.10f}')
print(f'||beta||^2: {np.sum(beta**2):.10f}')
print()

# STEP 4: Residuals
fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)
print('STEP 4: Residuals')
print(f'rss: {rss:.10f}')
print()

# STEP 5: Trace terms
trace1 = np.sum(np.diag(Ainv @ (lambda1 * S1_full)))
trace2 = np.sum(np.diag(Ainv @ (lambda2 * S2_full)))
print('STEP 5: Trace terms')
print(f'trace1: {trace1:.10f}')
print(f'trace2: {trace2:.10f}')
print()

# STEP 6: Penalty terms
penalty1 = lambda1 * beta.T @ S1_full @ beta
penalty2 = lambda2 * beta.T @ S2_full @ beta
print('STEP 6: Penalty terms')
print(f'penalty1: {penalty1:.10f}')
print(f'penalty2: {penalty2:.10f}')
print()

# STEP 7: Ranks
eigs1 = np.linalg.eigvalsh(S1_full)
eigs2 = np.linalg.eigvalsh(S2_full)
rank1 = np.sum(eigs1 > 1e-10 * np.max(eigs1))
rank2 = np.sum(eigs2 > 1e-10 * np.max(eigs2))
print('STEP 7: Ranks')
print(f'rank1: {rank1}')
print(f'rank2: {rank2}')
print()

# STEP 8: edf and phi
edf_total = np.sum(np.diag(Ainv @ XtX))
phi = rss / (n - edf_total)
print('STEP 8: edf and phi')
print(f'edf_total: {edf_total:.10f}')
print(f'phi: {phi:.10f}')
print()

# STEP 9: Gradient numerators
grad1_numerator = trace1 - rank1 + penalty1/phi
grad2_numerator = trace2 - rank2 + penalty2/phi
print('STEP 9: Gradient numerators (before /2)')
print(f'grad1_numerator: {grad1_numerator:.10f}')
print(f'grad2_numerator: {grad2_numerator:.10f}')
print(f'  = trace1 - rank1 + penalty1/phi')
print(f'  = {trace1:.10f} - {rank1} + {penalty1/phi:.10f}')
print(f'  = {trace2:.10f} - {rank2} + {penalty2/phi:.10f}')
print()

# STEP 10: Final gradients
grad1 = grad1_numerator / 2
grad2 = grad2_numerator / 2
print('STEP 10: Final gradients (after /2)')
print(f'grad1: {grad1:.10f}')
print(f'grad2: {grad2:.10f}')
print()

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print("All values should match exactly. Any difference indicates divergence point.")
