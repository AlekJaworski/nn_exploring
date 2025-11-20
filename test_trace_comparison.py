#!/usr/bin/env python3
"""
Direct trace comparison: compute tr(A^{-1}·λ·S) and compare with mgcv.

This is the KEY term in the gradient formula.
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("TRACE TERM COMPARISON: tr(A^{-1}·λ·S)")
print("=" * 80)

# Use a simple λ for clarity
test_lambda = [2.0, 3.0]

r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Fit at fixed λ
lambda <- c({test_lambda[0]}, {test_lambda[1]})
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda)

cat("\\n=== AT λ=[{test_lambda[0]}, {test_lambda[1]}] ===\\n")
cat("REML:", fit$gcv.ubre, "\\n")
cat("SCALE:", fit$scale, "\\n")
cat("RANK1:", fit$smooth[[1]]$rank, "\\n")
cat("RANK2:", fit$smooth[[2]]$rank, "\\n")

# Finite difference gradient (TRUE gradient)
delta <- 1e-6
reml0 <- fit$gcv.ubre

fit1_plus <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
                data=df, method="REML", sp=lambda * exp(c(delta, 0)))
grad1_true <- (fit1_plus$gcv.ubre - reml0) / delta

fit2_plus <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
                data=df, method="REML", sp=lambda * exp(c(0, delta)))
grad2_true <- (fit2_plus$gcv.ubre - reml0) / delta

cat("TRUE_GRADIENT:", grad1_true, grad2_true, "\\n")

# These are the gradients mgcv SHOULD compute analytically
# Our gradient should match these!

# Save data
write.csv(df, "/tmp/trace_test_data.csv", row.names=FALSE)
'''

print("\n1. Getting TRUE gradient from mgcv (via finite differences)...")
result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

print(result.stdout)

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)

# Parse
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
print("2. Computing OUR gradient at same λ...")
print("=" * 80)

# Create a test that evaluates gradient at FIXED λ
python_test = f'''
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/tmp/trace_test_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

print(f"Loaded data: n={{len(y)}}, p={{x.shape[1]}}")

# We need to call our REML gradient function directly
# But it's not exposed to Python! We need to add a test function

# For now, let's see what gradient we get from fitting
import mgcv_rust
gam = mgcv_rust.GAM()

# Fit will optimize, but we'll see the initial gradient in debug output
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr')
print(f"\\nOUR_CONVERGED_LAMBDA: {{result.get('lambda', result.keys())}}")
'''

print("\nRunning our code (will show debug output with trace values)...")
result2 = subprocess.run([
    'bash', '-c',
    f'cd /home/user/nn_exploring && source .venv/bin/activate && export MGCV_GRAD_DEBUG=1 && python3 -c "{python_test}" 2>&1 | head -80'
], capture_output=True, text=True, timeout=30)

print(result2.stdout)

# Extract our gradient components from debug output
our_trace1 = None
our_trace2 = None
our_grad1 = None
our_grad2 = None

for line in result2.stdout.split('\n'):
    if 'smooth=0:' in line and 'trace=' in line:
        parts = line.split('trace=')
        if len(parts) > 1:
            try:
                our_trace1 = float(parts[1].split(',')[0])
            except:
                pass
    if 'smooth=1:' in line and 'trace=' in line:
        parts = line.split('trace=')
        if len(parts) > 1:
            try:
                our_trace2 = float(parts[1].split(',')[0])
            except:
                pass
    if 'smooth=0:' in line and 'gradient =' in line:
        parts = line.split('=')
        if len(parts) > 1:
            try:
                our_grad1 = float(parts[-1])
            except:
                pass
    if 'smooth=1:' in line and 'gradient =' in line:
        parts = line.split('=')
        if len(parts) > 1:
            try:
                our_grad2 = float(parts[-1])
            except:
                pass

print("\n" + "=" * 80)
print("3. COMPARISON")
print("=" * 80)

if 'true_gradient' in mgcv:
    print(f"\\nTRUE gradient (mgcv finite difference):")
    print(f"  {mgcv['true_gradient']}")
    print(f"\\nThis is what our analytical gradient SHOULD be!")

if our_grad1 is not None:
    print(f"\\nOUR initial gradient (at λ~0.003):")
    print(f"  [{our_grad1:.6f}, {our_grad2 if our_grad2 else 'N/A'}]")
    print(f"\\n(Note: This is at DIFFERENT λ than mgcv test above)")

print(f"\\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print(f"""
The gradient formula is:
  ∂REML/∂log(λᵢ) = [tr(A^{{-1}}·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2

At λ={test_lambda}:
  - TRUE gradient (FD): {mgcv.get('true_gradient', 'see above')}
  - rank: {mgcv.get('rank1', '?')}, {mgcv.get('rank2', '?')}

If our trace values are correct, then:
  our_gradient = (trace - rank + penalty/φ) / 2
should match the TRUE gradient above.

If they don't match, the bug is in one of these terms:
  1. trace = λ·tr(A^{{-1}}·S)
  2. rank = rank(S)
  3. penalty/φ = λ·β'·S·β / φ
""")
