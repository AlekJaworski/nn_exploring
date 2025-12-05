#!/usr/bin/env Rscript
# Check mgcv's values at iteration 1 (initial guess)

library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

cat("========== MGV ITERATION 1 ==========\n")

# Fit and capture trace output
ctrl <- gam.control(trace=TRUE)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\n========== INITIAL SP GUESS ==========\n")
# mgcv's initial guess is sp = 0.1 for each smooth
cat("mgcv typically starts with sp ~ 0.1 or initial.sp if provided\n")
cat("Our implementation starts with lambda ~ 0.003\n\n")

cat("Conversion: sp is the actual smoothing parameter\n")
cat("lambda = exp(rho) where rho = log(sp)\n")
cat("If mgcv starts at sp=0.1, rho=log(0.1)=-2.30\n")
cat("If we start at lambda=0.003, rho=log(0.003)=-5.81\n\n")

cat("This is a 3.5 unit difference in log-space!\n")
