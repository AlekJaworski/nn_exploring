#!/usr/bin/env python3
"""
Detailed investigation of k=20 mismatch
"""

import numpy as np
import sys
import subprocess
import csv

sys.path.insert(0, "target/release")

np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, n)

print("=" * 70)
print("Detailed k=20 Investigation")
print("=" * 70)

# Save data
with open('/tmp/test_k20.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

# Rust fit
import mgcv_rust

X = x.reshape(-1, 1)
gam = mgcv_rust.GAM()
result_rust = gam.fit_auto(X, y, k=[20], method='REML', bs='cr')
pred_rust = gam.predict(X)

print(f"\nRust (k=20, bs='cr', REML):")
print(f"  Lambda:    {result_rust['lambda']:.6f}")
print(f"  Deviance:  {result_rust['deviance']:.4f}")

# mgcv fit with detailed diagnostics
r_script = """
library(mgcv)
data <- read.csv('/tmp/test_k20.csv')

set.seed(42)
gam_fit <- gam(y ~ s(x, k=20, bs='cr'), data=data, method='REML')

# Get the smooth object
sm <- smoothCon(s(x, k=20, bs='cr'), data=data, knots=NULL)[[1]]

cat('\\n=== Basis Properties ===\\n')
cat('Basis dimensions:', dim(sm$X), '\\n')
cat('Penalty dimensions:', dim(sm$S[[1]]), '\\n')
cat('Penalty rank:', qr(sm$S[[1]])$rank, '\\n')

cat('\\n=== Fit Results ===\\n')
cat('Lambda (sp):', gam_fit$sp, '\\n')
cat('Deviance:', deviance(gam_fit), '\\n')
cat('Scale estimate:', gam_fit$scale, '\\n')
cat('EDF:', sum(gam_fit$edf), '\\n')

# Check penalty scaling
cat('\\n=== Penalty Matrix Stats ===\\n')
cat('Penalty trace:', sum(diag(sm$S[[1]])), '\\n')
cat('Penalty max eigenvalue:', max(eigen(sm$S[[1]])$values), '\\n')
cat('Penalty norm:', norm(sm$S[[1]], 'F'), '\\n')

# Get knots
cat('\\n=== Knots ===\\n')
cat('Number of knots:', length(sm$knots), '\\n')
cat('First 5 knots:', head(sm$knots, 5), '\\n')
cat('Last 5 knots:', tail(sm$knots, 5), '\\n')

# Save predictions
write.csv(data.frame(pred=predict(gam_fit)), '/tmp/mgcv_pred_k20.csv', row.names=FALSE)
"""

result_r = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                         capture_output=True, timeout=30)

if result_r.returncode == 0:
    print(f"\nmgcv (k=20, bs='cr', REML):")
    print(result_r.stdout)

    mgcv_pred = np.loadtxt('/tmp/mgcv_pred_k20.csv', delimiter=',', skiprows=1)

    print(f"\n{'=' * 70}")
    print("Comparison")
    print("=" * 70)

    rust_lambda = result_rust['lambda']
    # Extract mgcv lambda from output
    mgcv_lambda = None
    for line in result_r.stdout.split('\n'):
        if 'Lambda (sp):' in line:
            mgcv_lambda = float(line.split(':')[1].strip())
            break

    if mgcv_lambda:
        ratio = rust_lambda / mgcv_lambda
        print(f"\nLambda:")
        print(f"  Rust:  {rust_lambda:.6f}")
        print(f"  mgcv:  {mgcv_lambda:.6f}")
        print(f"  Ratio: {ratio:.4f}x")

        if ratio < 0.5 or ratio > 2.0:
            print(f"\n⚠️  MAJOR MISMATCH! Lambda is {ratio:.1f}x different!")
        else:
            print(f"\n✓ Lambda reasonably close")

    pred_corr = np.corrcoef(pred_rust, mgcv_pred)[0,1]
    pred_rmse = np.sqrt(np.mean((pred_rust - mgcv_pred)**2))

    print(f"\nPredictions:")
    print(f"  Correlation: {pred_corr:.8f}")
    print(f"  RMSE:        {pred_rmse:.6f}")

    if pred_corr > 0.999:
        print(f"  ✓ Predictions match very well")
    else:
        print(f"  ⚠️  Predictions differ!")

else:
    print(f"R error: {result_r.stderr}")
