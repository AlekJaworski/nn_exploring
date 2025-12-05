# Check if mgcv uses effective DoF instead of rank(S)

library(mgcv)

set.seed(123)
n <- 500
x <- runif(n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)
w <- rep(1, n)

m <- gam(y ~ s(x, k=20, bs="cr"), method="REML", weights=w)

cat("=== EDF Investigation ===\n\n")

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

XtWX <- t(X) %*% diag(w) %*% X
A <- XtWX + lambda * S_full

# Compute influence/hat matrix
fitted_vals <- X %*% beta
residuals <- y - fitted_vals
RSS <- sum(w * residuals^2)

# Rank
rank_S <- sum(eigen(S, only.values=TRUE)$values > 1e-10 * max(eigen(S, only.values=TRUE)$values))

# EDF from mgcv
edf_total <- sum(m$edf)
edf_smooth <- m$edf[2]  # First is intercept

cat("Degrees of freedom:\n")
cat("  rank(S):", rank_S, "\n")
cat("  EDF (total from mgcv):", edf_total, "\n")
cat("  EDF (smooth only):", edf_smooth, "\n")
cat("  n - rank(S):", n - rank_S, "\n")
cat("  n - EDF:", n - edf_total, "\n\n")

# Compute EDF manually
# EDF = trace(X * (X'WX + lambda*S)^{-1} * X'W)
# For influence matrix H = X * A^{-1} * X'W
A_inv <- solve(A)
# Simplified: trace(H) = trace(X'W * X * A^{-1}) = trace(X'WX * A^{-1})
edf_manual <- sum(diag(XtWX %*% A_inv))

cat("Manual EDF computation:\n")
cat("  trace(X'WX * A^{-1}):", edf_manual, "\n")
cat("  mgcv EDF:", edf_total, "\n")
cat("  Difference:", abs(edf_manual - edf_total), "\n\n")

# Try phi with EDF
phi_with_edf <- RSS / (n - edf_total)
phi_with_rank <- RSS / (n - rank_S)

cat("Scale parameter (phi):\n")
cat("  phi (using EDF):", phi_with_edf, "\n")
cat("  phi (using rank):", phi_with_rank, "\n")
cat("  mgcv phi:", m$sig2, "\n")
cat("  Matches EDF?", abs(phi_with_edf - m$sig2) < 1e-6, "\n")
cat("  Matches rank?", abs(phi_with_rank - m$sig2) < 1e-6, "\n\n")

# Now try REML with EDF
penalty_term <- sum(beta * (S_full %*% beta))
P <- RSS + lambda * penalty_term
log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
log_lambda_term <- rank_S * log(lambda)

# Use EDF instead of rank for the n-r term
reml_with_edf <- 0.5 * (P/phi_with_edf + (n - edf_total) * log(2 * pi * phi_with_edf) + log_det_A - log_lambda_term)

cat("REML with EDF:\n")
cat("  REML (using EDF in denominator):", reml_with_edf, "\n")
cat("  mgcv REML:", m$gcv.ubre, "\n")
cat("  Difference:", abs(reml_with_edf - m$gcv.ubre), "\n\n")

# Check the Fellner-Schall condition with correct understanding
# At optimum, trace(A^{-1} * lambda * S) should equal EDF of the smooth
cat("Fellner-Schall condition:\n")
trace_lambdaS <- sum(diag(A_inv %*% (lambda * S_full)))
cat("  trace(A^{-1} * lambda*S):", trace_lambdaS, "\n")
cat("  EDF of smooth:", edf_smooth, "\n")
cat("  Ratio:", trace_lambdaS / edf_smooth, "\n")
cat("  (Should be ~1 at optimum)\n\n")

# Wood's formulation: effective rank
# rho_i = trace(A^{-1} * lambda_i * S_i) represents the effective rank
cat("Effective ranks:\n")
cat("  Nominal rank(S):", rank_S, "\n")
cat("  Effective rank (trace):", trace_lambdaS, "\n")
cat("  mgcv EDF:", edf_smooth, "\n\n")

cat("=== End EDF Investigation ===\n")
