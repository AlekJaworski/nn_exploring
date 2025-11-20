#!/usr/bin/env python3
"""
Component-by-component comparison of gradient calculation with mgcv.

Extract EVERY intermediate value from mgcv's gradient computation and compare
against ours to find the exact discrepancy.
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("COMPONENT-BY-COMPONENT GRADIENT COMPARISON")
print("=" * 80)

# Test at a fixed λ where we can inspect internals
test_lambda = [1.0, 1.0]

r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1

# Fit at fixed λ
lambda <- c({test_lambda[0]}, {test_lambda[1]})
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data.frame(x1=x[,1], x2=x[,2], y=y),
           method="REML", sp=lambda)

cat("\\n=== MGCV AT λ=[{test_lambda[0]}, {test_lambda[1]}] ===\\n")

# Basic values
cat("COEFFICIENTS:", coef(fit), "\\n")
cat("SCALE:", fit$scale, "\\n")
cat("RSS:", sum(residuals(fit)^2), "\\n")
cat("EDF_TOTAL:", sum(fit$edf), "\\n")

# Penalty matrices - extract from smooth objects
S1 <- fit$smooth[[1]]$S[[1]]
S2 <- fit$smooth[[2]]$S[[1]]

# Get ranks
cat("RANK_S1:", fit$smooth[[1]]$rank, "\\n")
cat("RANK_S2:", fit$smooth[[2]]$rank, "\\n")

# Compute beta'·S·beta terms
beta <- coef(fit)
cat("BETA_S1_BETA:", as.numeric(t(beta) %*% S1 %*% beta), "\\n")
cat("BETA_S2_BETA:", as.numeric(t(beta) %*% S2 %*% beta), "\\n")

# Finite difference gradient for reference
delta <- 1e-6
reml0 <- fit$gcv.ubre

fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data.frame(x1=x[,1], x2=x[,2], y=y),
           method="REML", sp=lambda * exp(c(delta, 0)))
grad1_fd <- (fit1$gcv.ubre - reml0) / delta

fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data.frame(x1=x[,1], x2=x[,2], y=y),
           method="REML", sp=lambda * exp(c(0, delta)))
grad2_fd <- (fit2$gcv.ubre - reml0) / delta

cat("GRADIENT_FD:", grad1_fd, grad2_fd, "\\n")

# Save data
write.csv(data.frame(x1=x[,1], x2=x[,2], y=y), "/tmp/grad_compare_data.csv", row.names=FALSE)

# Try to access mgcv internals for trace computation
# This requires looking at the Hessian from Newton iteration
cat("\\nREML:", fit$gcv.ubre, "\\n")
'''

print("\n1. Getting mgcv values...")
result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)

print(result.stdout)

# Parse mgcv results
mgcv = {}
for line in result.stdout.split('\n'):
    parts = line.split(':', 1)
    if len(parts) == 2:
        key = parts[0].strip().lower()
        values = parts[1].strip().split()
        if key and values:
            try:
                if len(values) == 1:
                    mgcv[key] = float(values[0])
                else:
                    mgcv[key] = np.array([float(v) for v in values])
            except:
                pass

print("\n" + "=" * 80)
print("2. Computing our gradient components at same λ...")
print("=" * 80)

# Now compute OUR gradient at the same λ
python_code = f'''
import mgcv_rust
import pandas as pd
import numpy as np
import os

# Enable debug output
os.environ["MGCV_GRAD_DEBUG"] = "1"

data = pd.read_csv('/tmp/grad_compare_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

# Create GAM and fit with our λ
gam = mgcv_rust.GAM()

# Add smooths manually with same λ
# (We need access to internal gradient computation)
# For now, let's fit and extract final values
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr')

print(f"\\nOUR_LAMBDA: {{result.get('lambda', 'N/A')}}")
'''

# Run our computation with debug output
result2 = subprocess.run([
    'bash', '-c',
    f'source /home/user/nn_exploring/.venv/bin/activate && python3 -c "{python_code}" 2>&1'
], capture_output=True, text=True, timeout=30, cwd='/home/user/nn_exploring')

print(result2.stdout)

# Look for gradient debug output
grad_debug_lines = [line for line in result2.stdout.split('\n') if 'QR_GRAD_DEBUG' in line or 'smooth=' in line]

print("\n" + "=" * 80)
print("3. OUR GRADIENT DEBUG OUTPUT")
print("=" * 80)
for line in grad_debug_lines[:20]:  # First few iterations
    print(line)

print("\n" + "=" * 80)
print("4. COMPARISON")
print("=" * 80)

if 'scale' in mgcv and 'rss' in mgcv:
    print(f"\\nScale parameter φ:")
    print(f"  mgcv: {mgcv['scale']:.6e}")
    print(f"  (from RSS={mgcv['rss']:.6f}, n=100, edf_total={mgcv.get('edf_total', 'N/A')})")

if 'rank_s1' in mgcv and 'rank_s2' in mgcv:
    print(f"\\nPenalty ranks:")
    print(f"  S1: {mgcv['rank_s1']}")
    print(f"  S2: {mgcv['rank_s2']}")

if 'beta_s1_beta' in mgcv and 'beta_s2_beta' in mgcv:
    print(f"\\nPenalty quadratic forms β'·S·β:")
    print(f"  β'·S1·β: {mgcv['beta_s1_beta']:.6e}")
    print(f"  β'·S2·β: {mgcv['beta_s2_beta']:.6e}")

if 'gradient_fd' in mgcv:
    print(f"\\nmgcv FD gradient at λ={test_lambda}:")
    print(f"  {mgcv['gradient_fd']}")

print(f"\\n" + "=" * 80)
print("KEY QUESTION:")
print("=" * 80)
print(f"Do our trace values match what mgcv should have?")
print(f"Look for 'trace=' in our debug output above.")
print(f"\\nThe formula is: gradient = (trace - rank + penalty_term/φ) / 2")
print(f"If our trace is much smaller than it should be, that's the bug!")
