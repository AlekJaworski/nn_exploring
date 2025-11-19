#!/usr/bin/env Rscript  
# Check what penalty norm we produce vs mgcv

library(mgcv)

set.seed(42)
n <- 100
x <- rnorm(n)

# mgcv's penalty
dat <- data.frame(x=x)
sm_mgcv <- smoothCon(s(x, k=8, bs='cr'), data=dat, knots=NULL, scale.penalty=FALSE)[[1]]
S_mgcv_raw <- sm_mgcv$S[[1]]

cat("=== mgcv's raw penalty ===\n")
cat("||S||_inf:", max(rowSums(abs(S_mgcv_raw))), "\n")
cat("||S||_F:", sqrt(sum(S_mgcv_raw^2)), "\n")
cat("trace(S):", sum(diag(S_mgcv_raw)), "\n")
cat("S[1,1]:", S_mgcv_raw[1,1], "\n")
cat("S[1,2]:", S_mgcv_raw[1,2], "\n")
cat("\nFirst 3x3 block:\n")
print(S_mgcv_raw[1:3, 1:3])

# Now let's trace through what mgcv does in the C code
# The penalty is computed from basis second derivatives
cat("\n=== Checking basis function details ===\n")
cat("Number of basis functions:", ncol(sm_mgcv$X), "\n")
cat("Number of knots:", length(sm_mgcv$xp), "\n")
cat("Knots:", sm_mgcv$xp, "\n")

# The penalty comes from integrating (f'')^2
# For cubic splines, this is related to knot spacing
knots <- sm_mgcv$xp
if (length(knots) > 1) {
  cat("\nKnot spacing:\n")
  diffs <- diff(knots)
  cat("  Mean:", mean(diffs), "\n")
  cat("  Min:", min(diffs), "\n")
  cat("  Max:", max(diffs), "\n")
}
