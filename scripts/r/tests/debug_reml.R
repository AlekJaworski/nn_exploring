# Debug REML formula - investigate discrepancies

library(mgcv)

set.seed(123)
n <- 500
x <- runif(n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)
w <- rep(1, n)

# Fit with REML
m <- gam(y ~ s(x, k=20, bs="cr"), method="REML", weights=w)

cat("=== Investigating REML Discrepancy ===\n\n")
cat("mgcv REML score:", m$gcv.ubre, "\n\n")

# Extract components
X <- model.matrix(m)
beta <- coef(m)
S_list <- m$smooth[[1]]$S
S <- S_list[[1]]
lambda <- m$sp[1]

# Map S to full space
p <- ncol(X)
first_col <- m$smooth[[1]]$first.para
last_col <- m$smooth[[1]]$last.para
S_full <- matrix(0, p, p)
S_full[first_col:last_col, first_col:last_col] <- S

cat("Dimensions:\n")
cat("  n =", n, "\n")
cat("  p =", p, "\n")
cat("  Smooth uses columns", first_col, "to", last_col, "\n")
cat("  S is", nrow(S), "x", ncol(S), "\n\n")

# Compute X'WX
XtWX <- t(X) %*% diag(w) %*% X
A <- XtWX + lambda * S_full

# Fit coefficients
xtwy <- t(X) %*% (w * y)
beta_refit <- solve(A, xtwy)

cat("Beta comparison:\n")
cat("  Max difference:", max(abs(beta - beta_refit)), "\n\n")

# Residuals
fitted_vals <- X %*% beta
residuals <- y - fitted_vals
RSS <- sum(w * residuals^2)

# Penalty
penalty_term <- sum(beta * (S_full %*% beta))

# Rank
rank_S <- sum(eigen(S, only.values=TRUE)$values > 1e-10 * max(eigen(S, only.values=TRUE)$values))

# Scale
phi <- RSS / (n - rank_S)

cat("RSS and scale:\n")
cat("  RSS:", RSS, "\n")
cat("  n - rank(S):", n - rank_S, "\n")
cat("  phi (computed):", phi, "\n")
cat("  phi (mgcv):", m$sig2, "\n")
cat("  Difference:", abs(phi - m$sig2), "\n\n")

# Log determinant
log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
log_lambda_term <- rank_S * log(lambda)

cat("REML components:\n")
cat("  log|A|:", log_det_A, "\n")
cat("  rank*log(lambda):", log_lambda_term, "\n")
cat("  log|X'WX|:", determinant(XtWX, logarithm=TRUE)$modulus[1], "\n\n")

# Try different REML formulas

# Formula 1: Profiled REML (concentrating out phi)
# REML = 0.5 * (n-p) * log(RSS/(n-p)) + 0.5 * log|A| - 0.5 * log|X'WX|
reml1 <- 0.5 * ((n - rank_S) * log(RSS/(n - rank_S)) + log_det_A - determinant(XtWX, logarithm=TRUE)$modulus[1])

# Formula 2: Full REML with explicit phi
# REML = 0.5 * ((RSS + lambda*penalty)/phi + (n-rank)*log(2*pi*phi) + log|A| - rank*log(lambda))
P <- RSS + lambda * penalty_term
reml2 <- 0.5 * (P/phi + (n - rank_S) * log(2 * pi * phi) + log_det_A - log_lambda_term)

# Formula 3: Laplace approximation form (Wood 2011)
# Uses deviance instead of RSS
deviance <- -2 * sum(dnorm(y, fitted_vals, sqrt(m$sig2), log=TRUE))
reml3 <- 0.5 * (deviance + log_det_A - determinant(XtWX, logarithm=TRUE)$modulus[1])

# Formula 4: Marginal likelihood form
# REML = -0.5 * (log|X'WX + lambda*S| - log|X'WX| + (n-p)*log(RSS/(n-p)))
reml4 <- -0.5 * (log_det_A - determinant(XtWX, logarithm=TRUE)$modulus[1] + (n - rank_S) * log(RSS/(n - rank_S)))

# Formula 5: Try mgcv's exact formula from source
# mgcv uses: -0.5 * (log|X'WX + lambda*S| - log|lambda*S| + n*log(RSS/n))
# But need to handle singular S properly
log_det_S <- tryCatch({
  determinant(S_full, logarithm=TRUE)$modulus[1]
}, error = function(e) {
  # S is singular, use pseudo-determinant
  ev <- eigen(S_full, only.values=TRUE)$values
  sum(log(ev[ev > 1e-10]))
})
reml5 <- -0.5 * (log_det_A - log_det_S + n * log(RSS/n))

cat("Testing different REML formulas:\n")
cat("  mgcv value:", m$gcv.ubre, "\n")
cat("  Formula 1 (profiled):", reml1, "  diff:", abs(reml1 - m$gcv.ubre), "\n")
cat("  Formula 2 (full with phi):", reml2, "  diff:", abs(reml2 - m$gcv.ubre), "\n")
cat("  Formula 3 (deviance):", reml3, "  diff:", abs(reml3 - m$gcv.ubre), "\n")
cat("  Formula 4 (marginal):", reml4, "  diff:", abs(reml4 - m$gcv.ubre), "\n")
cat("  Formula 5 (with log|S|):", reml5, "  diff:", abs(reml5 - m$gcv.ubre), "\n\n")

# Check what mgcv actually computes
cat("=== Checking mgcv internals ===\n\n")

# The gcv.ubre field stores the REML score
# Let's check what mgcv::gam.fit3 or mgcv::magic does

# Try to extract the actual REML computation
# mgcv uses: -0.5 * (ldeta - ldetS0 + n * log(V * scale / n))
# where V is the deviance

# Get scale estimate
scale_est <- m$scale

cat("Scale estimates:\n")
cat("  m$sig2:", m$sig2, "\n")
cat("  m$scale:", scale_est, "\n\n")

# Check if mgcv uses deviance vs RSS
deviance_check <- sum((y - fitted_vals)^2 / scale_est)
cat("Deviance check:", deviance_check, "\n")
cat("RSS:", RSS, "\n")
cat("RSS / scale:", RSS / scale_est, "\n\n")

# Final attempt: use mgcv's exact n*log(rss/n) form
reml_exact <- -0.5 * (log_det_A - log_det_S + n * log(RSS/n))
cat("Using n*log(RSS/n) form:\n")
cat("  REML:", reml_exact, "\n")
cat("  mgcv:", m$gcv.ubre, "\n")
cat("  Difference:", abs(reml_exact - m$gcv.ubre), "\n\n")

# Check the Fellner-Schall condition more carefully
cat("=== Fellner-Schall Trace Check ===\n\n")

# The trace should be computed on the smooth-specific S, not S_full!
A_inv_full <- solve(A)
trace_full <- sum(diag(A_inv_full %*% S_full))
trace_smooth <- sum(diag(A_inv_full[first_col:last_col, first_col:last_col] %*% S))

cat("Trace computations:\n")
cat("  trace(A^{-1} * S_full):", trace_full, "\n")
cat("  trace(A^{-1}[smooth,smooth] * S):", trace_smooth, "\n")
cat("  rank(S):", rank_S, "\n")
cat("  Ratio (full):", trace_full / rank_S, "\n")
cat("  Ratio (smooth):", trace_smooth / rank_S, "\n\n")

# The correct formulation: trace should be on lambda*S
trace_lambdaS <- sum(diag(A_inv_full %*% (lambda * S_full)))
cat("  trace(A^{-1} * lambda*S):", trace_lambdaS, "\n")
cat("  Ratio to rank:", trace_lambdaS / rank_S, "\n")

cat("\n=== End Debug ===\n")
