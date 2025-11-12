#!/usr/bin/env Rscript
# Extract R mgcv's REML computation details for debugging

library(mgcv)

cat(strrep("=", 70), "\n")
cat("EXTRACTING REML INTERMEDIATE VALUES FROM MGCV\n")
cat(strrep("=", 70), "\n\n")

# Generate same data as Python test
set.seed(42)
n <- 100
x <- seq(0, 1, length.out = n)
y_true <- sin(2 * pi * x)
y <- y_true + rnorm(n, 0, 0.1)

k <- 10  # Number of basis functions

cat("Data:\n")
cat("  n =", n, "\n")
cat("  k =", k, "\n")
cat("  x range: [", min(x), ",", max(x), "]\n")
cat("  y mean:", mean(y), "\n")
cat("  y sd:", sd(y), "\n\n")

# Fit GAM with CR splines
gam_fit <- gam(y ~ s(x, k=k, bs="cr"), method="REML")

# Extract key components
lambda_mgcv <- gam_fit$sp
deviance_mgcv <- deviance(gam_fit)
edf_mgcv <- sum(gam_fit$edf)

cat("mgcv Results:\n")
cat("  Lambda (smoothing parameter):", lambda_mgcv, "\n")
cat("  Deviance:", deviance_mgcv, "\n")
cat("  EDF (effective degrees of freedom):", edf_mgcv, "\n\n")

# Extract design matrix and penalty
sm <- gam_fit$smooth[[1]]
X <- model.matrix(gam_fit)  # Design matrix
S <- sm$S[[1]]  # Penalty matrix

cat("Design Matrix X:\n")
cat("  Shape:", dim(X), "\n")
cat("  X[1, 1:5]:", X[1, 1:min(5, ncol(X))], "\n")
cat("  X[n, 1:5]:", X[n, 1:min(5, ncol(X))], "\n\n")

cat("Penalty Matrix S:\n")
cat("  Shape:", dim(S), "\n")
cat("  Frobenius norm:", norm(S, "F"), "\n")
cat("  Trace:", sum(diag(S)), "\n")
cat("  S[1, 1:5]:", S[1, 1:min(5, ncol(S))], "\n\n")

# Extract fitted values and coefficients
beta <- coef(gam_fit)
fitted <- fitted(gam_fit)
residuals <- residuals(gam_fit)
rss <- sum(residuals^2)

cat("Coefficients and Fit:\n")
cat("  Number of coefficients:", length(beta), "\n")
cat("  beta[1:5]:", beta[1:min(5, length(beta))], "\n")
cat("  RSS (residual sum of squares):", rss, "\n")
cat("  RSS/n:", rss/n, "\n\n")

# Compute REML components manually
cat("REML Components:\n")

# Compute X'X + lambda*S
w <- rep(1, n)  # Weights (all 1 for Gaussian)
XtX <- t(X) %*% X

# Embed S in a larger matrix to match X'X dimensions
# S applies only to smooth term (columns 2-10), not intercept (column 1)
S_full <- matrix(0, nrow = ncol(X), ncol = ncol(X))
S_full[2:ncol(X), 2:ncol(X)] <- S

A <- XtX + lambda_mgcv * S_full

cat("  X'X shape:", dim(XtX), "\n")
cat("  S shape (smooth only):", dim(S), "\n")
cat("  S_full shape (with intercept):", dim(S_full), "\n")
cat("  A = X'X + lambda*S shape:", dim(A), "\n")

# Log determinants
log_det_A <- determinant(A, logarithm = TRUE)$modulus[1]
cat("  log|A| = log|X'X + lambda*S|:", log_det_A, "\n")

# Rank of penalty matrix
rank_S <- Matrix::rankMatrix(S)[1]
cat("  rank(S):", rank_S, "\n")

# REML formula components
log_rss_n <- log(rss / n)
log_lambda_term <- rank_S * log(lambda_mgcv)

cat("  log(RSS/n):", log_rss_n, "\n")
cat("  rank(S) * log(lambda):", log_lambda_term, "\n")

# Full REML criterion
reml_manual <- n * log_rss_n + log_det_A - log_lambda_term
cat("  REML (manual) = n*log(RSS/n) + log|A| - r*log(lambda):\n")
cat("    ", reml_manual, "\n\n")

# Compare with mgcv's internal REML
cat("mgcv's REML criterion:\n")
cat("  REML score from fit:", gam_fit$gcv.ubre, "\n")
cat("  (Note: mgcv may use a different constant/scale)\n\n")

# Save key values for Rust comparison
cat(strrep("=", 70), "\n")
cat("KEY VALUES FOR RUST COMPARISON\n")
cat(strrep("=", 70), "\n")
cat("Lambda:", lambda_mgcv, "\n")
cat("RSS:", rss, "\n")
cat("log|X'X + lambda*S|:", log_det_A, "\n")
cat("rank(S):", rank_S, "\n")
cat("n:", n, "\n")
cat("Penalty Frobenius norm (smooth only):", norm(S, "F"), "\n")
cat("Penalty Frobenius norm (with intercept):", norm(S_full, "F"), "\n")
cat("Deviance:", deviance_mgcv, "\n")

# Save to CSV for easy loading
write.csv(data.frame(
    lambda = lambda_mgcv,
    rss = rss,
    log_det_A = log_det_A,
    rank_S = rank_S,
    n = n,
    penalty_frob = norm(S, "F"),
    penalty_frob_full = norm(S_full, "F"),
    deviance = deviance_mgcv
), "/tmp/mgcv_reml_values.csv", row.names = FALSE)

cat("\nSaved to /tmp/mgcv_reml_values.csv\n")
