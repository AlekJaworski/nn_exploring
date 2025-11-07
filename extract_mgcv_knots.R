#!/usr/bin/env Rscript

library(mgcv)

set.seed(42)
n <- 500
k <- 20
x <- seq(0, 1, length.out = n)
y <- sin(2 * pi * x) + rnorm(n, 0, 0.1)

cat("R mgcv Knot Investigation\n")
cat(strrep("=", 70), "\n\n")

# BS splines
cat("BS (B-splines):\n")
gam_bs <- gam(y ~ s(x, k=k, bs="bs"), method="REML")

# Extract smooth object
sm_bs <- gam_bs$smooth[[1]]

cat("  k:", k, "\n")
cat("  Number of coefficients:", length(gam_bs$coefficients) - 1, "\n")  # -1 for intercept
cat("  Penalty matrix shape:", dim(sm_bs$S[[1]]), "\n")

# Get knots from smooth object
if (!is.null(sm_bs$knots)) {
  cat("  Knots:\n")
  print(sm_bs$knots)
}

# Check if there's an extended knot vector
if (!is.null(sm_bs$xp)) {
  cat("  Extended knot vector (xp):\n")
  print(sm_bs$xp)
}

cat("\n")

# CR splines
cat("CR (Cubic Regression Splines):\n")
gam_cr <- gam(y ~ s(x, k=k, bs="cr"), method="REML")

sm_cr <- gam_cr$smooth[[1]]

cat("  k:", k, "\n")
cat("  Number of coefficients:", length(gam_cr$coefficients) - 1, "\n")
cat("  Penalty matrix shape:", dim(sm_cr$S[[1]]), "\n")

if (!is.null(sm_cr$knots)) {
  cat("  Knots:\n")
  print(sm_cr$knots)
}

if (!is.null(sm_cr$xp)) {
  cat("  Extended knot vector (xp):\n")
  print(sm_cr$xp)
}

cat("\n")
cat(strrep("=", 70), "\n")
cat("Key Question: Are knots evenly spaced in [0, 1]?\n")
cat(strrep("=", 70), "\n")
