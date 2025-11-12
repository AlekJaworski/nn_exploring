#!/usr/bin/env Rscript

# Extract penalty matrix and lambda from R mgcv for comparison

library(mgcv)

# Set seed for reproducibility
set.seed(42)

# Test parameters
n <- 500
k <- 20
x <- seq(0, 1, length.out = n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.1)

cat(strrep("=", 70), "\n")
cat("R mgcv Penalty Matrix and Lambda Extraction\n")
cat(strrep("=", 70), "\n\n")

cat("Test case: n =", n, ", k =", k, "\n\n")

# --- CR splines ---
cat("1. CR (Cubic Regression) Splines:\n")
gam_cr <- gam(y ~ s(x, k=k, bs="cr"), method="REML")
lambda_cr <- gam_cr$sp
S_cr <- gam_cr$smooth[[1]]$S[[1]]

cat("   Lambda:", lambda_cr, "\n")
cat("   Penalty shape:", dim(S_cr), "\n")
cat("   Penalty trace:", sum(diag(S_cr)), "\n")
cat("   Penalty max row sum:", max(rowSums(abs(S_cr))), "\n")
cat("   Penalty Frobenius norm:", norm(S_cr, "F"), "\n")
cat("   Deviance:", deviance(gam_cr), "\n\n")

# Write to file for Python to read
write.csv(S_cr, "/tmp/mgcv_cr_penalty.csv", row.names=FALSE)
write.table(data.frame(lambda=lambda_cr, deviance=deviance(gam_cr)),
            "/tmp/mgcv_cr_lambda.txt", row.names=FALSE, quote=FALSE)

# --- BS splines ---
cat("2. BS (B-spline) Splines:\n")
gam_bs <- gam(y ~ s(x, k=k, bs="bs"), method="REML")
lambda_bs <- gam_bs$sp
S_bs <- gam_bs$smooth[[1]]$S[[1]]

cat("   Lambda:", lambda_bs, "\n")
cat("   Penalty shape:", dim(S_bs), "\n")
cat("   Penalty trace:", sum(diag(S_bs)), "\n")
cat("   Penalty max row sum:", max(rowSums(abs(S_bs))), "\n")
cat("   Penalty Frobenius norm:", norm(S_bs, "F"), "\n")
cat("   Deviance:", deviance(gam_bs), "\n\n")

# Write to file for Python to read
write.csv(S_bs, "/tmp/mgcv_bs_penalty.csv", row.names=FALSE)
write.table(data.frame(lambda=lambda_bs, deviance=deviance(gam_bs)),
            "/tmp/mgcv_bs_lambda.txt", row.names=FALSE, quote=FALSE)

# Display first 5x5 of penalty matrices
cat("\nCR Penalty (first 5x5):\n")
print(S_cr[1:5, 1:5])

cat("\nBS Penalty (first 5x5):\n")
print(S_bs[1:5, 1:5])

cat("\nData saved to /tmp/mgcv_*_*.{csv,txt}\n")
