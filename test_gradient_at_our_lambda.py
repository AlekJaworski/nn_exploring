#!/usr/bin/env python3
"""
KEY TEST: Compare gradients at OUR converged λ=[4.11, 2.32]

If mgcv's finite-difference gradient is large/negative at our λ,
but our analytical gradient is near zero, then OUR GRADIENT IS WRONG!
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("GRADIENT COMPARISON AT OUR CONVERGED λ=[4.11, 2.32]")
print("=" * 80)

our_lambda = [4.11, 2.32]

r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Evaluate at OUR λ
our_lambda <- c({our_lambda[0]}, {our_lambda[1]})

# Fit at our λ
fit0 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=our_lambda)

cat("\\n=== AT OUR λ=[{our_lambda[0]}, {our_lambda[1]}] ===\\n")
cat("REML:", fit0$gcv.ubre, "\\n")

# Compute mgcv's finite difference gradient w.r.t. LOG(λ)
delta <- 1e-6
reml0 <- fit0$gcv.ubre

# d(REML)/d(log λ_1)
lambda1_plus <- our_lambda * exp(c(delta, 0))
fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda1_plus)
grad1_mgcv_fd <- (fit1$gcv.ubre - reml0) / delta

# d(REML)/d(log λ_2)
lambda2_plus <- our_lambda * exp(c(0, delta))
fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda2_plus)
grad2_mgcv_fd <- (fit2$gcv.ubre - reml0) / delta

cat("\\nMGCV FINITE DIFFERENCE GRADIENT:\\n")
cat("  dREML/d(log λ_1):", grad1_mgcv_fd, "\\n")
cat("  dREML/d(log λ_2):", grad2_mgcv_fd, "\\n")

# Also check at mgcv's optimal
cat("\\n=== AT MGCV OPTIMAL ===\\n")
fit_opt <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
              data=df, method="REML")
cat("Optimal λ:", fit_opt$sp, "\\n")
cat("Optimal REML:", fit_opt$gcv.ubre, "\\n")

# Save data
write.csv(data.frame(x1=x[,1], x2=x[,2], y=y), "/tmp/grad_test_data.csv", row.names=FALSE)
'''

print("\n1. Computing mgcv's finite-difference gradient at our λ...")
result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)

print(result.stdout)

print("\n" + "=" * 80)
print("2. Computing OUR analytical gradient at same λ...")
print("=" * 80)

# Run our code
python_code = f'''
import mgcv_rust
import pandas as pd
import numpy as np

data = pd.read_csv('/tmp/grad_test_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

# Fit and get our gradient at convergence
gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr')

print("\\nOUR RESULTS:")
print(f"  Converged λ: {{result['lambdas']}}")
print(f"  Our REML: {{result['reml']:.6f}}")
print(f"  Our gradient: {{result['gradient']}}")
'''

result2 = subprocess.run([
    'bash', '-c',
    f'source /home/user/nn_exploring/.venv/bin/activate && python3 -c "{python_code}"'
], capture_output=True, text=True, timeout=30, cwd='/home/user/nn_exploring')

print(result2.stdout)
if result2.stderr and 'warning' not in result2.stderr.lower():
    print("STDERR:", result2.stderr)

print("\n" + "=" * 80)
print("3. ANALYSIS")
print("=" * 80)

print(f"""
CRITICAL QUESTION:
At our λ=[4.11, 2.32], is mgcv's FD gradient large and negative?

If YES: Our gradient formula is WRONG (it says stop, should say keep going)
If NO:  Our gradient is correct, problem is elsewhere

From output above:
- mgcv FD gradient at our λ: see output
- Our analytical gradient: see output

If mgcv FD grad is ~[-2.0, -2.0] or similar (large/negative),
but ours is ~[0.05, 0.05] (small), then WE HAVE A GRADIENT BUG!
""")
