#!/usr/bin/env Rscript

# Extract and understand S.scale computation from mgcv

library(mgcv)

set.seed(42)
n <- 100
x <- runif(n)
y <- sin(2*pi*x) + rnorm(n, 0, 0.1)

cat("=== Understanding S.scale ===\n\n")

# Create smooth directly to see internals
sm <- smoothCon(s(x, k=12, bs="cr"), data=data.frame(x=x), knots=NULL)[[1]]

cat("What smoothCon returns:\n")
cat("  S.scale:", sm$S.scale, "\n")
cat("  rank:", sm$rank, "\n")
cat("  null.space.dim:", sm$null.space.dim, "\n\n")

# Look at the actual penalty matrix
S <- sm$S[[1]]
X <- sm$X

cat("Penalty matrix S:\n")
cat("  Dimensions:", dim(S), "\n")
cat("  Max element:", max(abs(S)), "\n")
cat("  Frobenius norm:", norm(S, "F"), "\n")
cat("  ||S||_inf:", max(rowSums(abs(S))), "\n\n")

cat("Design matrix X:\n")
cat("  Dimensions:", dim(X), "\n")
cat("  ||X||_inf:", max(rowSums(abs(X))), "\n\n")

# According to mgcv source, S.scale is related to X'X
XtX <- t(X) %*% X
cat("X'X:\n")
cat("  Max element:", max(abs(XtX)), "\n")
cat("  Trace:", sum(diag(XtX)), "\n")
cat("  Frobenius norm:", norm(XtX, "F"), "\n\n")

# Check if S.scale = trace(X'X)
cat("Checking formulas:\n")
cat("  trace(X'X) =", sum(diag(XtX)), "\n")
cat("  ||X'X||_F^2 =", sum(XtX^2), "\n")
cat("  n * mean(diag(X'X)) =", n * mean(diag(XtX)), "\n\n")

# Looking at mgcv source code, S.scale should be related to
# the scale of X'X
cat("Likely formula (from mgcv source):\n")
cat("  S.scale appears to be sum(X^2) or similar\n")
cat("  sum(X^2) =", sum(X^2), "\n")
cat("  sum(diag(X'X)) =", sum(diag(XtX)), "\n\n")

# Check alternative formulations
cat("Alternative checks:\n")
cat("  Ratio S.scale / trace(X'X) =", sm$S.scale / sum(diag(XtX)), "\n")
cat("  Ratio S.scale / sum(X^2) =", sm$S.scale / sum(X^2), "\n\n")

# The actual source says S.scale is used to make lambda values
# have consistent interpretation across different basis sizes
cat("Purpose of S.scale (from mgcv docs):\n")
cat("  Makes lambda values comparable across different smooths\n")
cat("  Scales penalties so lambda ~ 1 is 'typical' smoothness\n\n")
