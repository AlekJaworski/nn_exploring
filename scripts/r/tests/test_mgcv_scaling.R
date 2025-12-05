#!/usr/bin/env Rscript
library(mgcv)

# Test 1: [0, 1] range
dat1 <- data.frame(x = seq(0, 1, length=100))
sm1 <- smoothCon(s(x, k=10, bs="cr"), data=dat1, knots=NULL)[[1]]
S1 <- sm1$S[[1]]

# Test 2: [0, 2] range
dat2 <- data.frame(x = seq(0, 2, length=100))
sm2 <- smoothCon(s(x, k=10, bs="cr"), data=dat2, knots=NULL)[[1]]
S2 <- sm2$S[[1]]

cat("mgcv CR spline penalty scaling test:\n")
cat(strrep("=", 70), "\n")
cat("\nTest 1 ([0,1]):\n")
cat("  Frobenius:", norm(S1, "F"), "\n")
cat("  S[1,1]:", S1[1,1], "\n")

cat("\nTest 2 ([0,2]):\n")
cat("  Frobenius:", norm(S2, "F"), "\n")
cat("  S[1,1]:", S2[1,1], "\n")

cat("\nRatio (S1/S2):", norm(S1, "F") / norm(S2, "F"), "\n")
cat("Expected if proportional to 1/L^3:", (2/1)^3, "\n")
