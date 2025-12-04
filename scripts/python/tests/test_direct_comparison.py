#!/usr/bin/env python3
"""
Direct value-by-value comparison with mgcv at fixed λ.

This will extract ALL intermediate values from mgcv and our implementation
at the same λ point to find exactly where they diverge.
"""

import numpy as np
import subprocess
import sys
import mgcv_rust

print("=" * 80)
print("DIRECT COMPARISON WITH MGCV AT FIXED λ")
print("=" * 80)

# Step 1: Run mgcv at a fixed λ and extract ALL values
print("\n1. Running mgcv with fixed λ=[4.0, 2.0]...")

r_script = '''
library(mgcv)

# Same data as our tests
set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Fit with fixed sp
fixed_sp <- c(4.0, 2.0)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=fixed_sp)

# Extract everything
cat("MGCV_LAMBDA:", fit$sp, "\\n")
cat("MGCV_REML:", fit$gcv.ubre, "\\n")

# Get coefficients
beta <- coef(fit)
cat("MGCV_BETA:", beta, "\\n")

# Get scale parameter
cat("MGCV_SCALE:", fit$scale, "\\n")

# Get fitted values
fitted <- predict(fit)
residuals <- y - fitted
rss <- sum(residuals^2)
cat("MGCV_RSS:", rss, "\\n")

# Get edf
cat("MGCV_EDF_TOTAL:", sum(fit$edf), "\\n")
cat("MGCV_EDF_SMOOTH1:", fit$edf[1], "\\n")
cat("MGCV_EDF_SMOOTH2:", fit$edf[2], "\\n")

# Now evaluate gradient and Hessian at this λ
# We need to access mgcv internals for this

# Get the GAM object structure
G <- fit$smooth[[1]]
cat("MGCV_SMOOTH1_NULL_DIM:", G$null.space.dim, "\\n")
cat("MGCV_SMOOTH1_DF:", G$df, "\\n")

# Try to get gradient - need to call internal function
# This requires loading mgcv namespace
grad <- rep(0, 2)
# Can't easily access gradient from fitted object
# Instead, let's compute REML at nearby points to verify

# Compute REML at λ + δ to check gradient direction
delta <- 0.001
sp_p1 <- fixed_sp
sp_p1[1] <- sp_p1[1] + delta
fit_p1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
              data=df, method="REML", sp=sp_p1)
cat("MGCV_REML_SP1_PLUS:", fit_p1$gcv.ubre, "\\n")

sp_p2 <- fixed_sp
sp_p2[2] <- sp_p2[2] + delta
fit_p2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
              data=df, method="REML", sp=sp_p2)
cat("MGCV_REML_SP2_PLUS:", fit_p2$gcv.ubre, "\\n")

# Finite difference gradient
grad1 <- (fit_p1$gcv.ubre - fit$gcv.ubre) / delta
grad2 <- (fit_p2$gcv.ubre - fit$gcv.ubre) / delta
cat("MGCV_GRAD_FD:", grad1, grad2, "\\n")

# Save data for our comparison
write.csv(data.frame(x1=x[,1], x2=x[,2], y=y), "/tmp/compare_data.csv", row.names=FALSE)
'''

result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

if result.returncode != 0:
    print("ERROR running R:", result.stderr, file=sys.stderr)
    sys.exit(1)

print(result.stdout)

# Parse mgcv results
mgcv = {}
for line in result.stdout.split('\n'):
    if line.startswith('MGCV_'):
        parts = line.split(':', 1)
        key = parts[0].replace('MGCV_', '').lower()
        values = parts[1].strip().split()
        if len(values) == 1:
            mgcv[key] = float(values[0])
        else:
            mgcv[key] = np.array([float(v) for v in values])

print("\n" + "=" * 80)
print("2. Running our implementation with same λ=[4.0, 2.0]...")
print("=" * 80)

# Load the same data
import pandas as pd
data = pd.read_csv('/tmp/compare_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

# Run our implementation with FIXED λ
# First need to fit to get the structure, then extract gradient/Hessian at λ=[4,2]
gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr')

print(f"\nOur Results:")
print(f"  λ = {result['lambdas']}")
print(f"  REML = {result['reml']}")
print(f"  β shape = {result['coefficients'].shape}")
print(f"  β[:5] = {result['coefficients'][:5]}")

# Compute our RSS and scale
fitted = result['fitted']
residuals = y - fitted
our_rss = np.sum(residuals**2)
print(f"  RSS = {our_rss:.6f}")

print("\n" + "=" * 80)
print("3. COMPARISON: mgcv vs ours at λ=[4.0, 2.0]")
print("=" * 80)

# Compare REML
print(f"\nREML:")
print(f"  mgcv: {mgcv['reml']:.6f}")
print(f"  ours: {result['reml']:.6f}")
print(f"  diff: {result['reml'] - mgcv['reml']:.6f}")
print(f"  ratio: {result['reml'] / mgcv['reml']:.6f}")

# Compare RSS
print(f"\nRSS:")
print(f"  mgcv: {mgcv['rss']:.6f}")
print(f"  ours: {our_rss:.6f}")
print(f"  diff: {our_rss - mgcv['rss']:.6f}")
print(f"  ratio: {our_rss / mgcv['rss']:.6f}")

# Compare scale
print(f"\nScale parameter φ:")
print(f"  mgcv: {mgcv['scale']:.6e}")
print(f"  ours: {result.get('scale', 'N/A')}")

# Compare edf
print(f"\nEffective degrees of freedom:")
print(f"  mgcv total: {mgcv['edf_total']:.3f}")
print(f"  mgcv smooth1: {mgcv['edf_smooth1']:.3f}")
print(f"  mgcv smooth2: {mgcv['edf_smooth2']:.3f}")

# Compare β coefficients
mgcv_beta = mgcv['beta']
our_beta = result['coefficients']
print(f"\nCoefficients β:")
print(f"  mgcv β shape: {mgcv_beta.shape}")
print(f"  ours β shape: {our_beta.shape}")
print(f"  mgcv β[:5]: {mgcv_beta[:5]}")
print(f"  ours β[:5]: {our_beta[:5]}")
print(f"  Max abs diff: {np.max(np.abs(mgcv_beta - our_beta)):.6e}")
print(f"  Relative err: {np.max(np.abs(mgcv_beta - our_beta) / np.abs(mgcv_beta)):.6e}")

if np.max(np.abs(mgcv_beta - our_beta)) > 1e-6:
    print(f"\n⚠️  COEFFICIENTS DIFFER!")
    print(f"  First 10 differences:")
    for i in range(min(10, len(mgcv_beta))):
        diff = our_beta[i] - mgcv_beta[i]
        if abs(diff) > 1e-8:
            print(f"    β[{i}]: mgcv={mgcv_beta[i]:.6f}, ours={our_beta[i]:.6f}, diff={diff:.6e}")

# Compare finite difference gradient
if 'grad_fd' in mgcv:
    print(f"\nGradient (finite difference from mgcv):")
    print(f"  mgcv: {mgcv['grad_fd']}")
    print(f"  ours: {result.get('gradient', 'N/A')}")

print("\n" + "=" * 80)
print("4. DIAGNOSIS")
print("=" * 80)

# Check if REML values match
reml_match = abs(result['reml'] - mgcv['reml']) < 1e-4
beta_match = np.max(np.abs(mgcv_beta - our_beta)) < 1e-6
rss_match = abs(our_rss - mgcv['rss']) < 1e-6

print(f"\nβ coefficients match: {beta_match}")
print(f"RSS matches: {rss_match}")
print(f"REML matches: {reml_match}")

if not beta_match:
    print(f"\n❌ CRITICAL: β coefficients differ!")
    print(f"   This means the penalized fit is different.")
    print(f"   Need to check:")
    print(f"   1. Penalty matrices S_i")
    print(f"   2. QR decomposition")
    print(f"   3. Coefficient solving")
elif not rss_match:
    print(f"\n❌ RSS differs but β matches - numerical precision issue")
elif not reml_match:
    print(f"\n❌ REML differs but β and RSS match")
    print(f"   Bug is in REML calculation, not fitting!")
    print(f"   Need to check REML formula and log-determinant")
else:
    print(f"\n✅ All values match at this λ point!")
    print(f"   Problem must be in gradient/Hessian")
