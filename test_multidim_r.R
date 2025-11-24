#!/usr/bin/env Rscript
library(mgcv)

set.seed(42)
n <- 1000
n_dims <- 3

X <- matrix(runif(n * n_dims), nrow=n, ncol=n_dims)
y <- numeric(n)

y <- y + sin(2 * pi * X[,1])
y <- y + 0.5 * cos(3 * pi * X[,2])
y <- y + 0.3 * (X[,3]^2)
y <- y + rnorm(n, 0, 0.2)

df <- data.frame(y=y, x1=X[,1], x2=X[,2], x3=X[,3])

cat("Fitting R's mgcv...\n")
fit <- gam(y ~ s(x1, bs='cr', k=10) + s(x2, bs='cr', k=10) + s(x3, bs='cr', k=10),
           data=df, method='REML')

cat("\nResults:\n")
cat(sprintf("Lambdas: %s\n", paste(sprintf("%.6f", fit$sp), collapse=", ")))
cat(sprintf("EDF: %.2f\n", sum(fit$edf)))
cat(sprintf("REML: %.6f\n", fit$gcv.ubre))
cat(sprintf("Iterations: %d\n", fit$outer.info$iter))
cat(sprintf("Converged: %s\n", fit$outer.info$conv))
