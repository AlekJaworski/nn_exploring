#!/usr/bin/env python3
"""
Test just the trace term of the Hessian to see if scaling is the issue.
"""

import numpy as np
import subprocess
import json

# Run mgcv
print("Getting mgcv's Hessian...")
result = subprocess.run(['Rscript', '-e', '''
library(mgcv)
set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"), data=df, method="REML")
cat("mgcv Hessian:\\n")
print(fit$outer.info$hess[1:2,1:2])
cat("\\nDiagonal:", diag(fit$outer.info$hess)[1:2], "\\n")
cat("sp:", fit$sp, "\\n")
'''], capture_output=True, text=True)

print(result.stdout)

print("\nAnalysis:")
print("mgcv's Hessian diagonal is POSITIVE ~2.8")
print("Our Hessian diagonal is NEGATIVE ~-1e-4")
print("\nPossible issues:")
print("1. Missing diagonal gradient term in chain rule conversion")
print("2. Wrong sign in one of the terms")
print("3. Using wrong formula (maybe mgcv uses approximate Hessian)")
