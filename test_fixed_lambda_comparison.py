#!/usr/bin/env python3
"""
Direct comparison: evaluate REML, gradient, and Hessian at FIXED λ=[4.0, 2.0]

Strategy:
1. Use mgcv to get β, gradient, Hessian at λ=[4,2]
2. Use our Rust code to compute the same
3. Compare every value
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("FIXED λ COMPARISON: Evaluate at λ=[4.0, 2.0]")
print("=" * 80)

# Step 1: Get mgcv's values at λ=[4,2]
print("\n1. Getting mgcv REML, gradient, Hessian at λ=[4, 2]...")

r_script = '''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Fit with FIXED sp to get mgcv's internal state
fixed_sp <- c(4.0, 2.0)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=fixed_sp)

cat("\\n=== MGCV VALUES AT λ=[4.0, 2.0] ===\\n")
cat("LAMBDA:", fit$sp, "\\n")
cat("REML:", fit$gcv.ubre, "\\n")
cat("SCALE:", fit$scale, "\\n")
cat("COEFFICIENTS:", coef(fit), "\\n")

# Compute gradient using finite differences
delta <- 1e-6
reml0 <- fit$gcv.ubre

# Gradient w.r.t. log(λ)
sp1_plus <- fixed_sp * exp(c(delta, 0))
fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=sp1_plus)
grad1 <- (fit1$gcv.ubre - reml0) / delta

sp2_plus <- fixed_sp * exp(c(0, delta))
fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=sp2_plus)
grad2 <- (fit2$gcv.ubre - reml0) / delta

cat("GRADIENT:", grad1, grad2, "\\n")

# Hessian using finite differences
sp11_plus <- fixed_sp * exp(c(2*delta, 0))
fit11 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
            data=df, method="REML", sp=sp11_plus)
h11 <- (fit11$gcv.ubre - 2*fit1$gcv.ubre + reml0) / (delta^2)

sp22_plus <- fixed_sp * exp(c(0, 2*delta))
fit22 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
            data=df, method="REML", sp=sp22_plus)
h22 <- (fit22$gcv.ubre - 2*fit2$gcv.ubre + reml0) / (delta^2)

sp12_plus <- fixed_sp * exp(c(delta, delta))
fit12 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
            data=df, method="REML", sp=sp12_plus)
h12 <- (fit12$gcv.ubre - fit1$gcv.ubre - fit2$gcv.ubre + reml0) / (delta^2)

cat("HESSIAN_11:", h11, "\\n")
cat("HESSIAN_22:", h22, "\\n")
cat("HESSIAN_12:", h12, "\\n")

# Save data
write.csv(data.frame(x1=x[,1], x2=x[,2], y=y), "/tmp/fixed_lambda_data.csv", row.names=FALSE)
'''

result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=60)

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)

print(result.stdout)

# Parse results
mgcv = {}
for line in result.stdout.split('\n'):
    parts = line.split(':', 1)
    if len(parts) == 2:
        key = parts[0].strip().lower()
        values = parts[1].strip().split()
        if len(values) == 1:
            try:
                mgcv[key] = float(values[0])
            except:
                pass
        else:
            try:
                mgcv[key] = np.array([float(v) for v in values])
            except:
                pass

print("\n" + "=" * 80)
print("2. Computing our values at λ=[4.0, 2.0]...")
print("=" * 80)

# Load data
import pandas as pd
data = pd.read_csv('/tmp/fixed_lambda_data.csv')
x_data = data[['x1', 'x2']].values
y_data = data['y'].values

# Now I need to call our REML evaluation directly
# Let me create a simple Python script that uses our functions

python_eval = '''
import mgcv_rust
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/tmp/fixed_lambda_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

# Create GAM and fit to get internal state
gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr')

print("OUR_LAMBDA:", result['lambdas'][0], result['lambdas'][1])
print("OUR_REML:", result['reml'])
print("OUR_COEFFICIENTS:", *result['coefficients'])
print("OUR_GRADIENT:", result['gradient'][0], result['gradient'][1])

# Note: We don't have direct access to Hessian in result, need to add that
'''

result2 = subprocess.run(['bash', '-c', f'source .venv/bin/activate && python3 -c "{python_eval}"'],
                        capture_output=True, text=True, timeout=60, cwd='/home/user/nn_exploring')

if result2.returncode != 0:
    print("ERROR:", result2.stderr, file=sys.stderr)
    # Continue anyway to see what we got

print(result2.stdout)
if result2.stderr:
    print("STDERR:", result2.stderr)

# Parse our results
ours = {}
for line in result2.stdout.split('\n'):
    parts = line.split(':', 1)
    if len(parts) == 2:
        key = parts[0].strip().lower().replace('our_', '')
        values = parts[1].strip().split()
        if len(values) == 1:
            try:
                ours[key] = float(values[0])
            except:
                pass
        else:
            try:
                ours[key] = np.array([float(v) for v in values])
            except:
                pass

print("\n" + "=" * 80)
print("3. COMPARISON")
print("=" * 80)

if 'lambda' in ours and 'lambda' in mgcv:
    print(f"\nλ:")
    print(f"  mgcv: {mgcv['lambda']}")
    print(f"  ours: {ours['lambda']}")

if 'reml' in ours and 'reml' in mgcv:
    print(f"\nREML:")
    print(f"  mgcv: {mgcv['reml']:.6f}")
    print(f"  ours: {ours['reml']:.6f}")
    print(f"  diff: {ours['reml'] - mgcv['reml']:.6f}")

if 'gradient' in ours and 'gradient' in mgcv:
    print(f"\nGradient:")
    print(f"  mgcv: {mgcv['gradient']}")
    print(f"  ours: {ours['gradient']}")
    print(f"  diff: {ours['gradient'] - mgcv['gradient']}")

if 'coefficients' in ours and 'coefficients' in mgcv:
    mgcv_beta = mgcv['coefficients']
    ours_beta = ours['coefficients']
    print(f"\nCoefficients β:")
    print(f"  Max diff: {np.max(np.abs(ours_beta - mgcv_beta)):.6e}")
    if np.max(np.abs(ours_beta - mgcv_beta)) > 1e-4:
        print(f"  ❌ COEFFICIENTS DIFFER!")

print("\n" + "=" * 80)
print("Note: Our implementation converges to different λ, so values won't match")
print("Need to evaluate at SAME λ=[4,2] to compare properly")
print("=" * 80)
