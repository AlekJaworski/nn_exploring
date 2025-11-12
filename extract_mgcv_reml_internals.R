#!/usr/bin/env Rscript
# Extract mgcv's REML computation internals for comparison

library(mgcv)

# Test data - same as Python test
set.seed(42)
n <- 100
x <- seq(0, 1, length.out = n)
y_true <- sin(2 * pi * x)
y <- y_true + rnorm(n, 0, 0.2)

# Function to extract all internal values for a given k
extract_mgcv_internals <- function(k_val) {
  cat("\n", rep("=", 80), "\n", sep="")
  cat("k =", k_val, "\n")
  cat(rep("=", 80), "\n", sep="")

  # Fit GAM
  gam_fit <- gam(y ~ s(x, k=k_val, bs="bs"), method="REML")

  # Extract smooth object
  smooth_obj <- gam_fit$smooth[[1]]

  # Get penalty matrix
  S <- smooth_obj$S[[1]]

  # Get design matrix for smooth term only (exclude intercept)
  X_full <- predict(gam_fit, type="lpmatrix")

  # Extract only the smooth columns (first column is intercept)
  smooth_first <- smooth_obj$first.para
  smooth_last <- smooth_obj$last.para
  X <- X_full[, smooth_first:smooth_last, drop=FALSE]

  # Get fitted coefficients for smooth term only
  beta <- coef(gam_fit)[smooth_first:smooth_last]

  # Get lambda (smoothing parameter)
  lambda <- gam_fit$sp

  # Get weights (from IRLS)
  w <- gam_fit$prior.weights
  if (is.null(w)) w <- rep(1, n)

  cat("\n--- Basic Info ---\n")
  cat("Number of basis functions:", ncol(X), "\n")
  cat("Lambda (sp):", lambda, "\n")
  cat("Deviance:", gam_fit$deviance, "\n")
  cat("Scale:", gam_fit$scale, "\n")
  cat("Rank of S:", qr(S)$rank, "\n")

  # Compute REML criterion manually
  cat("\n--- REML Computation ---\n")

  # Weighted design matrix
  W <- diag(w)
  XtWX <- t(X) %*% W %*% X

  # Penalized system matrix
  A <- XtWX + lambda * S

  # Fitted values and residuals
  fitted <- X %*% beta
  residuals <- y - fitted

  # RSS
  RSS <- sum(w * residuals^2)
  cat("RSS:", RSS, "\n")

  # Penalty term
  penalty <- as.numeric(t(beta) %*% S %*% beta)
  cat("beta'S beta:", penalty, "\n")

  # RSS + lambda * penalty
  RSS_penalized <- RSS + lambda * penalty
  cat("RSS + lambda*beta'S*beta:", RSS_penalized, "\n")

  # Rank of S
  rank_S <- qr(S)$rank
  cat("Rank(S):", rank_S, "\n")

  # Scale parameter phi
  phi <- RSS / (n - rank_S)
  cat("phi = RSS/(n-rank(S)):", phi, "\n")

  # Log determinant of A
  log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
  cat("log|A|:", log_det_A, "\n")

  # Log lambda term
  log_lambda_term <- rank_S * log(lambda)
  cat("rank(S) * log(lambda):", log_lambda_term, "\n")

  # REML criterion
  REML <- (RSS_penalized/phi + (n - rank_S) * log(2*pi*phi) + log_det_A - log_lambda_term) / 2
  cat("REML (manual):", REML, "\n")
  cat("REML (mgcv reported):", gam_fit$gcv.ubre, "\n")

  # Check penalty matrix properties
  cat("\n--- Penalty Matrix Properties ---\n")
  cat("S dimensions:", dim(S), "\n")
  cat("S trace:", sum(diag(S)), "\n")
  cat("S norm (Frobenius):", sqrt(sum(S^2)), "\n")
  cat("S max eigenvalue:", max(eigen(S)$values), "\n")
  cat("S min eigenvalue:", min(eigen(S)$values), "\n")

  # Examine eigenvalues
  S_eigen <- eigen(S)
  cat("Eigenvalues of S (first 5):", head(S_eigen$values, 5), "\n")
  cat("Number of near-zero eigenvalues (<1e-6):", sum(S_eigen$values < 1e-6), "\n")

  # Check X'WX properties
  cat("\n--- X'WX Properties ---\n")
  cat("X'WX trace:", sum(diag(XtWX)), "\n")
  cat("X'WX norm (Frobenius):", sqrt(sum(XtWX^2)), "\n")

  # Compare magnitudes
  cat("\n--- Magnitude Comparison ---\n")
  cat("||X'WX|| / ||S||:", sqrt(sum(XtWX^2)) / sqrt(sum(S^2)), "\n")
  cat("lambda * ||S|| / ||X'WX||:", lambda * sqrt(sum(S^2)) / sqrt(sum(XtWX^2)), "\n")

  # Return everything
  list(
    k = k_val,
    lambda = lambda,
    beta = beta,
    S = S,
    X = X,
    XtWX = XtWX,
    RSS = RSS,
    penalty = penalty,
    rank_S = rank_S,
    phi = phi,
    REML = REML,
    deviance = gam_fit$deviance,
    S_eigenvalues = S_eigen$values
  )
}

# Extract for multiple k values
k_values <- c(5, 10, 20, 30)
results <- lapply(k_values, extract_mgcv_internals)

# Summary comparison
cat("\n\n", rep("=", 80), "\n", sep="")
cat("SUMMARY COMPARISON ACROSS k VALUES\n")
cat(rep("=", 80), "\n", sep="")

cat("\nk\tlambda\t\tRSS\t\tphi\t\trank(S)\t||S||\n")
cat(rep("-", 80), "\n", sep="")
for (r in results) {
  cat(sprintf("%d\t%.6e\t%.4f\t%.4f\t%d\t%.4f\n",
              r$k, r$lambda, r$RSS, r$phi, r$rank_S, sqrt(sum(r$S^2))))
}

# Save results for Python analysis
save(results, file="mgcv_reml_internals.RData")
cat("\nSaved results to mgcv_reml_internals.RData\n")
