#!/usr/bin/env Rscript
library(mgcv)

set.seed(42)
n <- 100
X1 <- rnorm(n)
X2 <- rnorm(n)

# Build smooth terms properly
dat <- data.frame(x1=X1, x2=X2)
sm1 <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL)[[1]]
sm2 <- smoothCon(s(x2, k=8, bs='cr'), data=dat, knots=NULL)[[1]]

cat("Smooth 1 basis matrix:\n")
cat(sprintf("  Shape: %d x %d\n", nrow(sm1$X), ncol(sm1$X)))
cat(sprintf("  Mean abs: %.6e\n", mean(abs(sm1$X))))
cat(sprintf("  Max abs: %.6e\n", max(abs(sm1$X))))
cat(sprintf("  Max row sum: %.6e\n", max(rowSums(abs(sm1$X)))))
cat(sprintf("  maXX: %.6e\n", max(rowSums(abs(sm1$X)))^2))

cat("\nSmooth 2 basis matrix:\n")
cat(sprintf("  Shape: %d x %d\n", nrow(sm2$X), ncol(sm2$X)))
cat(sprintf("  Mean abs: %.6e\n", mean(abs(sm2$X))))
cat(sprintf("  Max abs: %.6e\n", max(abs(sm2$X))))
cat(sprintf("  Max row sum: %.6e\n", max(rowSums(abs(sm2$X)))))
cat(sprintf("  maXX: %.6e\n", max(rowSums(abs(sm2$X)))^2))

cat("\nPenalty matrices:\n")
cat(sprintf("Smooth 1 S inf_norm: %.6e\n", max(rowSums(abs(sm1$S[[1]])))))
cat(sprintf("Smooth 2 S inf_norm: %.6e\n", max(rowSums(abs(sm2$S[[1]])))))

cat("\nComputed S.scale:\n")
cat(sprintf("Smooth 1: %.6e\n", max(rowSums(abs(sm1$X)))^2 / max(rowSums(abs(sm1$S[[1]])))))
cat(sprintf("Smooth 2: %.6e\n", max(rowSums(abs(sm2$X)))^2 / max(rowSums(abs(sm2$S[[1]])))))

cat("\nActual S.scale from mgcv:\n")
cat(sprintf("Smooth 1: %.6e\n", sm1$S.scale))
cat(sprintf("Smooth 2: %.6e\n", sm2$S.scale))
