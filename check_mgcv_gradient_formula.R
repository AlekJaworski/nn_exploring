#!/usr/bin/env Rscript
# Check mgcv's gradient computation details

library(mgcv)

# Enable debugging
options(mgcv.vc.logLik=TRUE)

set.seed(42)
n <- 100
X <- matrix(rnorm(n*2), n, 2)
y <- sin(X[,1]) + 0.5*X[,2]^2 + 0.1*rnorm(n)

# Fit with profiling
fit <- gam(y ~ s(X[,1], k=8, bs='cr') + s(X[,2], k=8, bs='cr'), method="REML")

cat("\nmgcv results:\n")
cat("sp (smoothing parameters):", fit$sp, "\n")
cat("S.scale values:", sapply(fit$smooth, function(s) s$S.scale), "\n")
cat("Effective lambda (sp × S.scale):", fit$sp * sapply(fit$smooth, function(s) s$S.scale), "\n")
cat("\nGradient at solution:\n")
print(fit$outer.info$grad)

# Check if penalty matrices are scaled
cat("\nPenalty matrix norms:\n")
for (i in 1:length(fit$smooth)) {
  S <- fit$smooth[[i]]$S[[1]]
  cat(sprintf("Smooth %d: ||S||_inf = %.6e, S.scale = %.6e\n", 
              i, max(rowSums(abs(S))), fit$smooth[[i]]$S.scale))
}
