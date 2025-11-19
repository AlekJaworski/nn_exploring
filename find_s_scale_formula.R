#!/usr/bin/env Rscript
# Try to figure out mgcv's S.scale formula

library(mgcv)

set.seed(42)
n <- 100
X1 <- rnorm(n)
dat <- data.frame(x1=X1)
sm <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL)[[1]]

# Check various properties
cat("Trying to reverse-engineer S.scale formula...\n\n")

S <- sm$S[[1]]
X_basis <- sm$X

cat("S.scale from mgcv:", sm$S.scale, "\n")
cat("\nTrying different formulas:\n")
cat(sprintf("||S||_F (Frobenius): %.6e, S.scale/||S||_F = %.6e\n", 
            sqrt(sum(S^2)), sm$S.scale / sqrt(sum(S^2))))
cat(sprintf("||S||_inf: %.6e, S.scale/||S||_inf = %.6e\n",
            max(rowSums(abs(S))), sm$S.scale / max(rowSums(abs(S)))))
cat(sprintf("trace(S): %.6e, S.scale/trace(S) = %.6e\n",
            sum(diag(S)), sm$S.scale / sum(diag(S))))
cat(sprintf("||X||_inf^2: %.6e, ||X||_inf^2 / S.scale = %.6e\n",
            max(rowSums(abs(X_basis)))^2, max(rowSums(abs(X_basis)))^2 / sm$S.scale))

# Check if it's related to penalty rank
rk <- sm$rank
cat(sprintf("\nPenalty rank: %d\n", rk))
cat(sprintf("S.scale / rank: %.6e\n", sm$S.scale / rk))

# Check if related to data range
cat(sprintf("\nData range: [%.6f, %.6f]\n", min(X1), max(X1)))
cat(sprintf("Data span: %.6f\n", max(X1) - min(X1)))

# Check eigenvalues
S_eigen <- eigen(S, symmetric=TRUE)
cat(sprintf("\nS eigenvalues (non-zero): %s\n", paste(sprintf("%.6e", S_eigen$values[S_eigen$values > 1e-10]), collapse=", ")))
cat(sprintf("Sum of eigenvalues: %.6e\n", sum(S_eigen$values)))
cat(sprintf("S.scale / sum(eigenvalues): %.6e\n", sm$S.scale / sum(S_eigen$values)))

# Try the mysterious formula from mgcv source
# S.scale might be related to the scale of the covariate
cat("\n\nChecking covariate-based scaling...\n")
# mgcv often uses: S.scale = 1 / median_abs_second_diff(knots) or similar
cat(sprintf("n * S.scale: %.6e\n", n * sm$S.scale))
cat(sprintf("n^2 * S.scale: %.6e\n", n^2 * sm$S.scale))
