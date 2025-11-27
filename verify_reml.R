# Verification script for REML optimization correctness
# This script tests the REML criterion and gradient formulations against mgcv

library(mgcv)

set.seed(123)

# Generate test data
n <- 500
x <- runif(n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)
w <- rep(1, n)

cat("=== REML Optimization Verification ===\n\n")

# Fit GAM with REML
cat("Fitting GAM with method='REML'...\n")
m_reml <- gam(y ~ s(x, k=20, bs="cr"), method="REML", weights=w)

cat("\nREML Results:\n")
cat("  Smoothing parameter (lambda):", m_reml$sp, "\n")
cat("  REML criterion:", m_reml$gcv.ubre, "\n")
cat("  Scale parameter (phi):", m_reml$sig2, "\n")
cat("  Effective DoF:", sum(m_reml$edf), "\n")

# Extract internals for validation
X <- model.matrix(m_reml)
beta <- coef(m_reml)

# Get the penalty matrix in the right parameterization
# mgcv returns S in the original space, we need to map it properly
S_list <- m_reml$smooth[[1]]$S
S <- S_list[[1]]

# The penalty matrix dimensions
cat("\nDimensions:\n")
cat("  n =", n, "\n")
cat("  p (model matrix cols) =", ncol(X), "\n")
cat("  S dims =", paste(dim(S), collapse=" x "), "\n")

# Get lambda
lambda <- m_reml$sp[1]

# Build full penalty matrix matching X dimensions
p <- ncol(X)
S_full <- matrix(0, p, p)

# Map the smooth-specific penalty to the full parameter space
first_col <- m_reml$smooth[[1]]$first.para
last_col <- m_reml$smooth[[1]]$last.para
S_full[first_col:last_col, first_col:last_col] <- S

cat("  Penalty mapped to columns:", first_col, "to", last_col, "\n")
cat("  rank(S) =", sum(eigen(S, only.values=TRUE)$values > 1e-10 * max(eigen(S, only.values=TRUE)$values)), "\n")

# Compute REML criterion manually
XtWX <- t(X) %*% diag(w) %*% X
A <- XtWX + lambda * S_full

# Fit coefficients
beta_computed <- solve(A, t(X) %*% diag(w) %*% y)

# Residuals and RSS
fitted_vals <- X %*% beta_computed
residuals <- y - fitted_vals
RSS <- sum(w * residuals^2)

# Penalty term
penalty_term <- sum(beta_computed * (S_full %*% beta_computed))

# Scale parameter
rank_S <- sum(eigen(S, only.values=TRUE)$values > 1e-10 * max(eigen(S, only.values=TRUE)$values))
phi <- RSS / (n - rank_S)

# REML criterion components
log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
log_lambda_term <- rank_S * log(lambda)

# Full REML (Wood 2011 formulation)
P <- RSS + lambda * penalty_term
reml_manual <- ((P/phi) + (n - rank_S) * log(2 * pi * phi) + log_det_A - log_lambda_term) / 2

cat("\nManual REML computation:\n")
cat("  RSS:", RSS, "\n")
cat("  Penalty term (beta'*S*beta):", penalty_term, "\n")
cat("  P (RSS + lambda*penalty):", P, "\n")
cat("  phi (RSS/(n-rank)):", phi, "\n")
cat("  log|A|:", log_det_A, "\n")
cat("  rank(S) * log(lambda):", log_lambda_term, "\n")
cat("  REML (manual):", reml_manual, "\n")
cat("  REML (mgcv):", m_reml$gcv.ubre, "\n")
cat("  Difference:", abs(reml_manual - m_reml$gcv.ubre), "\n")

# Test Fellner-Schall method
cat("\n\n=== Fellner-Schall Method Verification ===\n\n")

cat("Fitting GAM with optimizer='efs' (Fellner-Schall)...\n")
m_fs <- gam(y ~ s(x, k=20, bs="cr"), method="REML", optimizer="efs", weights=w)

cat("\nFellner-Schall Results:\n")
cat("  Smoothing parameter (lambda):", m_fs$sp, "\n")
cat("  REML criterion:", m_fs$gcv.ubre, "\n")
cat("  Scale parameter (phi):", m_fs$sig2, "\n")
cat("  Effective DoF:", sum(m_fs$edf), "\n")
cat("  Convergence iterations:", m_fs$outer.info$iter, "\n")

# Compute trace(A^{-1} * S) at optimal lambda
S_full_fs <- matrix(0, p, p)
S_full_fs[first_col:last_col, first_col:last_col] <- S
A_fs <- XtWX + m_fs$sp[1] * S_full_fs
A_inv <- solve(A_fs)
trace_AinvS <- sum(diag(A_inv %*% S_full_fs))

cat("\nFellner-Schall trace condition:\n")
cat("  trace(A^{-1} * S):", trace_AinvS, "\n")
cat("  rank(S):", rank_S, "\n")
cat("  Ratio (should be ~1 at optimum):", trace_AinvS / rank_S, "\n")

# Compare methods
cat("\n\n=== Method Comparison ===\n\n")
cat("Newton vs Fellner-Schall:\n")
cat("  Lambda difference:", abs(m_reml$sp[1] - m_fs$sp[1]), "\n")
cat("  Lambda ratio:", m_reml$sp[1] / m_fs$sp[1], "\n")
cat("  REML difference:", abs(m_reml$gcv.ubre - m_fs$gcv.ubre), "\n")

# Test gradient formula
cat("\n\n=== Gradient Verification ===\n\n")

compute_reml_gradient <- function(lambda_val, X, y, w, S_full, rank_S) {
  n <- nrow(X)
  p <- ncol(X)

  XtWX <- t(X) %*% diag(w) %*% X
  A <- XtWX + lambda_val * S_full
  A_inv <- solve(A)

  # Coefficients
  beta <- solve(A, t(X) %*% diag(w) %*% y)

  # Residuals and RSS
  fitted <- X %*% beta
  residuals <- y - fitted
  RSS <- sum(w * residuals^2)

  # Scale and P
  phi <- RSS / (n - rank_S)
  penalty_term <- sum(beta * (S_full %*% beta))
  P <- RSS + lambda_val * penalty_term

  # Gradient terms
  trace_term <- sum(diag(A_inv %*% (lambda_val * S_full)))

  # Derivatives
  d_beta_d_rho <- -A_inv %*% (lambda_val * S_full) %*% beta
  d_RSS_d_rho <- -2 * sum((w * residuals) * (X %*% d_beta_d_rho))
  d_phi_d_rho <- d_RSS_d_rho / (n - rank_S)

  # d(P/phi)/d_rho
  d_P_d_rho <- d_RSS_d_rho + lambda_val * sum(beta * (S_full %*% beta)) +
               2 * lambda_val * sum(d_beta_d_rho * (S_full %*% beta))
  d_P_phi_d_rho <- (1/phi) * d_P_d_rho - (P/phi^2) * d_phi_d_rho

  # Full gradient
  gradient <- (trace_term - rank_S + d_P_phi_d_rho + (n - rank_S) * (1/phi) * d_phi_d_rho) / 2

  return(gradient)
}

# Test gradient at optimal lambda
grad_at_opt <- compute_reml_gradient(m_reml$sp[1], X, y, w, S_full, rank_S)
cat("Gradient at optimal lambda:", grad_at_opt, "\n")
cat("  (Should be near 0 at optimum)\n")

# Test gradient at nearby points
lambda_test <- m_reml$sp[1] * c(0.5, 0.8, 1.0, 1.2, 1.5)
cat("\nGradient at different lambda values:\n")
for (lam in lambda_test) {
  grad <- compute_reml_gradient(lam, X, y, w, S_full, rank_S)
  cat("  lambda =", sprintf("%.4f", lam), ", gradient =", sprintf("%.6f", grad), "\n")
}

# Save data for Rust testing
cat("\n\nSaving data for Rust verification...\n")
saveRDS(list(
  x = x,
  y = y,
  w = w,
  X = X,
  S = S,
  lambda_reml = m_reml$sp[1],
  lambda_fs = m_fs$sp[1],
  beta_reml = coef(m_reml),
  beta_fs = coef(m_fs),
  reml_criterion = m_reml$gcv.ubre,
  fs_criterion = m_fs$gcv.ubre,
  phi_reml = m_reml$sig2,
  phi_fs = m_fs$sig2,
  rank_S = rank_S,
  trace_at_opt = trace_AinvS,
  fs_iterations = m_fs$outer.info$iter
), "/tmp/reml_test_data.rds")

cat("Data saved to /tmp/reml_test_data.rds\n")
cat("\n=== Verification Complete ===\n")
