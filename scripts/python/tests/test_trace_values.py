#!/usr/bin/env python3
"""
Check if our trace computation tr(A^{-1}·λ·S) is correct by comparing
intermediate values.
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("TRACE VALUE ANALYSIS")
print("=" * 80)

# At converged λ
test_lambda = [4.06, 2.07]

r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

lambda <- c({test_lambda[0]}, {test_lambda[1]})
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda)

cat("AT λ=[{test_lambda[0]}, {test_lambda[1]}]:\\n")
cat("REML:", fit$gcv.ubre, "\\n")

# True gradient via FD
delta <- 1e-6
reml0 <- fit$gcv.ubre

fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda * exp(c(delta, 0)))
grad1 <- (fit1$gcv.ubre - reml0) / delta

fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda * exp(c(0, delta)))
grad2 <- (fit2$gcv.ubre - reml0) / delta

cat("TRUE_GRAD:", grad1, grad2, "\\n")

# From the gradient formula:
# gradient_i = [tr(A^-1·λ_i·S_i) - rank_i + (λ_i·β'·S_i·β)/φ] / 2
#
# We can solve for the trace term:
# trace_i = 2*gradient_i + rank_i - (λ_i·β'·S_i·β)/φ

cat("RANK:", fit$smooth[[1]]$rank, fit$smooth[[2]]$rank, "\\n")
cat("SCALE:", fit$scale, "\\n")

# Get β'·S·β for each smooth
# Note: fit$smooth[[i]]$S[[1]] is the penalty matrix for smooth i
# but it's in the smooth's own basis, need to map to full coefficient vector

# This is tricky - mgcv stores penalties in a compressed format
# Let's just compute what trace SHOULD be from the gradient formula

phi <- fit$scale
rank1 <- fit$smooth[[1]]$rank
rank2 <- fit$smooth[[2]]$rank

# Expected trace from gradient formula:
# trace_i = 2*grad_i + rank_i - penalty_term_i/phi
# where penalty_term_i = λ_i·β'·S_i·β

cat("\\nEXPECTED VALUES (from gradient formula):\\n")
cat("If gradient_1 =", grad1, "and rank_1 =", rank1, "\\n")
cat("Then trace_1 should be around:", 2*grad1 + rank1, "(minus penalty/phi term)\\n")
cat("If gradient_2 =", grad2, "and rank_2 =", rank2, "\\n")
cat("Then trace_2 should be around:", 2*grad2 + rank2, "(minus penalty/phi term)\\n")

write.csv(df, "/tmp/trace_analysis_data.csv", row.names=FALSE)
'''

print("\n1. Getting expected trace values from mgcv...")
result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

print(result.stdout)

print("\n" + "=" * 80)
print("2. Looking at OUR trace values...")
print("=" * 80)

# Just run our fit and extract trace from debug output
python_code = '''
import mgcv_rust
import pandas as pd
import os

os.environ["MGCV_GRAD_DEBUG"] = "1"

data = pd.read_csv('/tmp/trace_analysis_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

gam = mgcv_rust.GAM()
# This will show debug output with trace values
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr', max_iter=2)
'''

result2 = subprocess.run([
    'bash', '-c',
    f'cd /home/user/nn_exploring && source .venv/bin/activate && python3 -c "{python_code}" 2>&1'
], capture_output=True, text=True, timeout=60)

# Extract trace values from our first iteration (where λ is small)
print("\\nOUR TRACE VALUES (from debug output):")
for line in result2.stdout.split('\\n'):
    if 'smooth=' in line and 'trace=' in line and 'gradient' in line:
        print(line)

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print(f"""
The gradient formula is:
  gradient_i = [trace_i - rank_i + penalty_i/φ] / 2

Where:
  trace_i = λ_i · tr(A^{{-1}}·S_i)

From mgcv's TRUE gradient (via FD), we can compute what trace SHOULD be:
  trace_i = 2*gradient_i + rank_i - penalty_i/φ

Compare the EXPECTED trace values (from mgcv) with OUR trace values (from debug output).

If they differ significantly, that's the bug!

Note: penalty_i/φ is usually small, so approximately:
  trace_i ≈ 2*gradient_i + rank_i
""")
