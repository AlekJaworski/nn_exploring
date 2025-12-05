#!/usr/bin/env python3
"""
Test gradient at our NEW converged λ after the fix.
"""

import numpy as np
import subprocess

our_new_lambda = [4.06, 2.07]

r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# At our NEW converged λ
lambda_new <- c({our_new_lambda[0]}, {our_new_lambda[1]})

fit0 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda_new)

cat("\\nAt our NEW λ=[{our_new_lambda[0]}, {our_new_lambda[1]}]:\\n")
cat("REML:", fit0$gcv.ubre, "\\n")

# Finite difference gradient
delta <- 1e-6
reml0 <- fit0$gcv.ubre

lambda1_plus <- lambda_new * exp(c(delta, 0))
fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda1_plus)
grad1 <- (fit1$gcv.ubre - reml0) / delta

lambda2_plus <- lambda_new * exp(c(0, delta))
fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda2_plus)
grad2 <- (fit2$gcv.ubre - reml0) / delta

cat("mgcv FD gradient:\\n")
cat("  [", grad1, ",", grad2, "]\\n")
cat("\\nAt OPTIMAL λ:\\n")
fit_opt <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
              data=df, method="REML")
cat("Optimal λ:", fit_opt$sp, "\\n")
cat("Optimal REML:", fit_opt$gcv.ubre, "\\n")
'''

result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

print(result.stdout)

print("\nCONCLUSION:")
print(f"If mgcv FD gradient is still large/negative at λ={our_new_lambda},")
print(f"then our gradient is STILL wrong (but less wrong than before).")
