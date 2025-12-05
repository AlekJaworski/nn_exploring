#!/usr/bin/env Rscript
# Study mgcv's approach to multidimensional REML optimization

library(mgcv)

cat(strrep("=", 70), "\n")
cat("STUDYING MGCV MULTIDIMENSIONAL REML OPTIMIZATION\n")
cat(strrep("=", 70), "\n\n")

# Generate 3D test data
set.seed(42)
n <- 100

x1 <- runif(n, 0, 1)  # Use random data to avoid collinearity
x2 <- runif(n, -1, 1)
x3 <- runif(n, -2, 2)

y <- sin(2*pi*x1) + 0.5*x2^2 + 2*x3 + rnorm(n, 0, 0.2)

cat("Data:\n")
cat("  n =", n, "\n")
cat("  x1 range: [", min(x1), ",", max(x1), "]\n")
cat("  x2 range: [", min(x2), ",", max(x2), "]\n")
cat("  x3 range: [", min(x3), ",", max(x3), "]\n")
cat("  y mean:", mean(y), "sd:", sd(y), "\n\n")

# Fit with mgcv using CR splines
k1 <- 12
k2 <- 10
k3 <- 10

cat("Fitting GAM with 3 smooths using REML...\n")
cat("  k values: [", k1, ",", k2, ",", k3, "]\n\n")

gam_fit <- gam(y ~ s(x1, k=k1, bs="cr") + s(x2, k=k2, bs="cr") + s(x3, k=k3, bs="cr"),
               method="REML")

# Extract smoothing parameters
cat("Results:\n")
cat("  Smoothing parameters (sp):\n")
for(i in 1:length(gam_fit$sp)) {
    cat("    λ", i, ": ", gam_fit$sp[i], "\n", sep="")
}
cat("  Deviance:", deviance(gam_fit), "\n")
cat("  REML score:", gam_fit$gcv.ubre, "\n\n")

# Extract design matrix
X <- predict(gam_fit, type="lpmatrix")
cat("Design matrix X:\n")
cat("  Shape:", dim(X), "\n")
cat("  (includes intercept and all smooth terms)\n\n")

# Extract penalty matrices for each smooth
cat("Penalty matrices:\n")
for(i in 1:length(gam_fit$smooth)) {
    sm <- gam_fit$smooth[[i]]
    S <- sm$S[[1]]
    cat("  S", i, " (", sm$label, "):\n", sep="")
    cat("    Shape:", dim(S), "\n")
    cat("    Rank:", Matrix::rankMatrix(S)[1], "\n")
    cat("    Frobenius norm:", norm(S, "F"), "\n")
}
cat("\n")

# Extract coefficients
beta <- coef(gam_fit)
cat("Coefficients:\n")
cat("  Length:", length(beta), "\n")
cat("  First 5:", beta[1:5], "\n\n")

# Compute residuals and RSS
fitted_vals <- fitted(gam_fit)
residuals <- y - fitted_vals
rss <- sum(residuals^2)

cat("Model fit:\n")
cat("  RSS:", rss, "\n")
cat("  RMSE:", sqrt(rss/n), "\n\n")

# Examine the structure of how mgcv stores the smooths
cat("Smooth terms structure:\n")
for(i in 1:length(gam_fit$smooth)) {
    sm <- gam_fit$smooth[[i]]
    cat("  Smooth", i, ":\n")
    cat("    Label:", sm$label, "\n")
    cat("    Basis:", sm$bs.dim, "functions\n")
    cat("    First basis index:", sm$first.para, "\n")
    cat("    Last basis index:", sm$last.para, "\n")
    cat("    Number of penalties:", length(sm$S), "\n")
}
cat("\n")

# Try to understand mgcv's optimization approach
cat(strrep("=", 70), "\n")
cat("UNDERSTANDING MGCV's OPTIMIZATION\n")
cat(strrep("=", 70), "\n\n")

cat("mgcv uses the following approach for multiple smoothing parameters:\n")
cat("1. Works in log(λ) space for stability\n")
cat("2. Uses Newton's method with the Hessian of REML w.r.t. log(λ)\n")
cat("3. Computes gradients and Hessian analytically\n")
cat("4. Uses step halving for line search\n\n")

cat("The REML criterion for multiple penalties is:\n")
cat("  REML = (RSS + Σᵢ λᵢ βᵢᵀSᵢβᵢ)/φ\n")
cat("       + (n - Σᵢ rank(Sᵢ)) log(2πφ)\n")
cat("       + log|XᵀWX + Σᵢ λᵢSᵢ|\n")
cat("       - Σᵢ rank(Sᵢ) log(λᵢ)\n\n")

cat("where:\n")
cat("  φ = RSS / (n - Σᵢ rank(Sᵢ))  [scale parameter]\n")
cat("  Sᵢ is the penalty matrix for smooth i (embedded in full coefficient space)\n\n")

# Compute components manually
cat("Manual computation of REML components:\n")

# Create embedded penalty matrices
n_coef <- ncol(X)
penalties <- list()
ranks <- c()

for(i in 1:length(gam_fit$smooth)) {
    sm <- gam_fit$smooth[[i]]
    S_smooth <- sm$S[[1]]

    # Create full penalty matrix (zeros except in the block for this smooth)
    S_full <- matrix(0, n_coef, n_coef)
    first_idx <- sm$first.para
    last_idx <- sm$last.para
    smooth_indices <- first_idx:last_idx

    S_full[smooth_indices, smooth_indices] <- S_smooth

    penalties[[i]] <- S_full
    ranks[i] <- Matrix::rankMatrix(S_smooth)[1]
}

cat("  Penalty ranks:", ranks, "\n")
cat("  Total rank:", sum(ranks), "\n")

# Compute A = X'X + Σᵢ λᵢ Sᵢ
XtX <- t(X) %*% X
A <- XtX
for(i in 1:length(penalties)) {
    A <- A + gam_fit$sp[i] * penalties[[i]]
}

cat("  X'X shape:", dim(XtX), "\n")
cat("  A = X'X + Σλᵢ Sᵢ shape:", dim(A), "\n")

# Compute scale parameter
phi <- rss / (n - sum(ranks))
cat("  φ = RSS / (n - Σrank(Sᵢ)) =", phi, "\n")

# Compute penalty term
penalty_sum <- 0
for(i in 1:length(penalties)) {
    beta_S_beta <- t(beta) %*% penalties[[i]] %*% beta
    penalty_sum <- penalty_sum + gam_fit$sp[i] * beta_S_beta
}
cat("  Σλᵢ βᵀSᵢβ =", penalty_sum, "\n")

# Compute log determinant
log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
cat("  log|A| =", log_det_A, "\n")

# Compute lambda term
log_lambda_sum <- sum(ranks * log(gam_fit$sp))
cat("  Σrank(Sᵢ)·log(λᵢ) =", log_lambda_sum, "\n")

# Compute REML
reml_manual <- ((rss + penalty_sum) / phi
                + (n - sum(ranks)) * log(2 * pi * phi)
                + log_det_A
                - log_lambda_sum) / 2

cat("\nManual REML computation:\n")
cat("  REML =", reml_manual, "\n")
cat("  mgcv REML score:", gam_fit$gcv.ubre, "\n")
cat("  (Note: mgcv may use different scaling/constants)\n\n")

# Save for Rust debugging
cat(strrep("=", 70), "\n")
cat("SAVING DATA FOR RUST DEBUGGING\n")
cat(strrep("=", 70), "\n")

# Save all relevant data
data_for_rust <- list(
    x1 = x1,
    x2 = x2,
    x3 = x3,
    y = y,
    n = n,
    k_values = c(k1, k2, k3),
    lambdas = gam_fit$sp,
    coefficients = beta,
    fitted = fitted_vals,
    rss = rss,
    phi = phi,
    ranks = ranks,
    deviance = deviance(gam_fit),
    reml_score = gam_fit$gcv.ubre
)

saveRDS(data_for_rust, "/tmp/mgcv_3d_data.rds")
cat("Saved to /tmp/mgcv_3d_data.rds\n\n")

# Also save matrices
write.csv(X, "/tmp/mgcv_X_matrix.csv", row.names=FALSE)
write.csv(data.frame(y=y, x1=x1, x2=x2, x3=x3), "/tmp/mgcv_data.csv", row.names=FALSE)

cat("Design matrix saved to /tmp/mgcv_X_matrix.csv\n")
cat("Data saved to /tmp/mgcv_data.csv\n\n")

cat("Key findings:\n")
cat("1. Random data (not linspace) avoids collinearity\n")
cat("2. mgcv successfully optimizes multiple λ values\n")
cat("3. Each penalty is embedded in full coefficient space\n")
cat("4. Scale parameter φ uses total rank from all penalties\n")
cat("5. REML criterion combines all penalties additively\n")
