#!/usr/bin/env Rscript
library(mgcv)

# Same data as our test
set.seed(42)
n <- 500
X <- matrix(rnorm(n*4), n, 4)
y <- sin(X[,1]) + 0.5*X[,2]^2 + cos(X[,3]) + 0.3*X[,4] + 0.1*rnorm(n)

cat("Fitting 4D GAM with mgcv (n=500, k=16)...\n")

# Enable mgcv profiling to see iterations
options(mgcv.vc.logLik=TRUE)

fit <- gam(y ~ s(X[,1], k=16, bs='cr') + s(X[,2], k=16, bs='cr') +
             s(X[,3], k=16, bs='cr') + s(X[,4], k=16, bs='cr'),
           method="REML")

cat("\nmgcv Results:\n")
cat("Iterations:", fit$outer.info$iter, "\n")
cat("Smoothing parameters:", fit$sp, "\n")
cat("Convergence code:", fit$outer.info$conv, "\n")

# Check gradient at solution
cat("\nGradient at solution:\n")
print(fit$outer.info$grad)
