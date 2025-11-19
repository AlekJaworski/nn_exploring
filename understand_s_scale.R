#!/usr/bin/env Rscript
library(mgcv)

set.seed(42)
n <- 100
X1 <- rnorm(n)
y <- sin(X1) + 0.1*rnorm(n)
dat <- data.frame(x1=X1, y=y)

# Fit a GAM
fit <- gam(y ~ s(x1, k=8, bs='cr'), data=dat, method="REML")

cat("From fitted GAM:\n")
cat("sp (optimized):", fit$sp, "\n")
cat("S.scale:", fit$smooth[[1]]$S.scale, "\n")
cat("Effective lambda (sp * S.scale):", fit$sp * fit$smooth[[1]]$S.scale, "\n")

# Now check smoothCon
sm <- smoothCon(s(x1, k=8, bs='cr'), data=dat, knots=NULL)[[1]]

cat("\nFrom smoothCon:\n")
cat("S.scale:", sm$S.scale, "\n")

# Check if the penalty in smoothCon is different from the fitted one
S_smoothcon <- sm$S[[1]]
S_fitted <- fit$smooth[[1]]$S[[1]]

cat("\nPenalty matrix comparison:\n")
cat("smoothCon S norm:", max(rowSums(abs(S_smoothcon))), "\n")
cat("fitted S norm:", max(rowSums(abs(S_fitted))), "\n")
cat("Ratio:", max(rowSums(abs(S_fitted))) / max(rowSums(abs(S_smoothcon))), "\n")

# Maybe S.scale is applied AFTER?
cat("\nIf S is scaled by S.scale:\n")
cat("||S * S.scale||_inf:", max(rowSums(abs(S_smoothcon * sm$S.scale))), "\n")

# Check knot-related scaling
cat("\nKnot information:\n")
cat("Number of knots:", length(sm$knots), "\n")
if (length(sm$knots) > 0) {
  cat("Knot range: [", min(sm$knots), ",", max(sm$knots), "]\n")
  cat("Knot span:", max(sm$knots) - min(sm$knots), "\n")
  if (length(sm$knots) > 1) {
    diffs <- diff(sm$knots)
    cat("Mean knot spacing:", mean(diffs), "\n")
    cat("Median knot spacing:", median(diffs), "\n") 
  }
}

# Check if S.scale relates to the integration weights or similar
cat("\nData-dependent properties:\n")
cat("Data range: [", min(X1), ",", max(X1), "]\n")
cat("Data span:", max(X1) - min(X1), "\n")
cat("Data span^2:", (max(X1) - min(X1))^2, "\n")
cat("n:", n, "\n")

# Try common scaling factors
cat("\nTrying to find S.scale formula:\n")
cat("S.scale * ||S||_inf / n:", sm$S.scale * max(rowSums(abs(S_smoothcon))) / n, "\n")
cat("S.scale / (data_span^(-2)):", sm$S.scale / (1/(max(X1) - min(X1))^2), "\n")
cat("S.scale * data_span^2:", sm$S.scale * (max(X1) - min(X1))^2, "\n")

