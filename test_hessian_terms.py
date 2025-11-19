#!/usr/bin/env python3
"""
Test individual Hessian terms against mgcv's values.
This will help us understand which terms we're missing.
"""

import numpy as np
import subprocess
import sys

# Run mgcv to get Hessian at a specific λ
print("=" * 80)
print("Getting mgcv's Hessian at λ=[0.1, 0.1]")
print("=" * 80)

result = subprocess.run(['Rscript', '-e', '''
library(mgcv)
set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Fit with optimization
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML")

cat("\\nmgcv converged to:\\n")
cat("Lambda:", fit$sp, "\\n")
cat("REML:", fit$gcv.ubre, "\\n")

# Check if Hessian was computed
if (!is.null(fit$outer.info$hess)) {
    cat("\\nmgcv Hessian:\\n")
    print(fit$outer.info$hess)
    cat("\\nDiagonal:", diag(fit$outer.info$hess), "\\n")
    if (nrow(fit$outer.info$hess) >= 2) {
        cat("Off-diagonal [1,2]:", fit$outer.info$hess[1,2], "\\n")
    }
} else {
    cat("\\nNo Hessian in outer.info\\n")
}

# Check gradient
if (!is.null(fit$outer.info$grad)) {
    cat("\\nGradient:", fit$outer.info$grad, "\\n")
} else {
    cat("\\nNo gradient in outer.info\\n")
}
'''], capture_output=True, text=True, timeout=30)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr, file=sys.stderr)
    sys.exit(1)

print("\n" + "=" * 80)
print("Now run our implementation at same λ with debug output")
print("=" * 80)
print("\nWe should see:")
print("1. Our trace_term values")
print("2. Compare against mgcv Hessian elements")
print("3. Identify which additional terms we need")
