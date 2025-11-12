#!/usr/bin/env python3
"""
Simple test: Verify k=10 CR splines work and compare with mgcv
"""

import numpy as np
import sys
import subprocess
import csv

sys.path.insert(0, "target/release")

print("=" * 70)
print("Testing k=10 CR Splines vs mgcv (both using REML)")
print("=" * 70)

# Test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, n)

# Save for R
with open('/tmp/test_k10.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

print("\n[1/3] Fitting with Rust (k=10, bs='cr', method='REML')...")

import mgcv_rust

X = x.reshape(-1, 1)
gam = mgcv_rust.GAM()
result_rust = gam.fit_auto(X, y, k=[10], method='REML', bs='cr')
pred_rust = gam.predict(X)

print(f"✓ Rust fit complete")
print(f"  Lambda:    {result_rust['lambda']:.6f}")
print(f"  Deviance:  {result_rust['deviance']:.4f}")

print("\n[2/3] Fitting with mgcv (k=10, bs='cr', method='REML')...")

r_script = """
library(mgcv)
data <- read.csv('/tmp/test_k10.csv')

set.seed(42)
gam_fit <- gam(y ~ s(x, k=10, bs='cr'), data=data, method='REML')

# Extract smooth to check dimensions
sm <- smoothCon(s(x, k=10, bs='cr'), data=data, knots=NULL)[[1]]

cat('Basis dimensions:', dim(sm$X), '\\n')
cat('Penalty dimensions:', dim(sm$S[[1]]), '\\n')
cat('Lambda:', gam_fit$sp, '\\n')
cat('Deviance:', deviance(gam_fit), '\\n')
cat('EDF:', sum(gam_fit$edf), '\\n')

write.csv(data.frame(pred=predict(gam_fit)), '/tmp/mgcv_pred_k10.csv', row.names=FALSE)
"""

result_r = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                         capture_output=True, timeout=30)

if result_r.returncode != 0:
    print(f"❌ R failed: {result_r.stderr}")
    sys.exit(1)

# Parse
mgcv_lambda = None
mgcv_deviance = None
mgcv_edf = None
mgcv_basis_dim = None
mgcv_penalty_dim = None

for line in result_r.stdout.split('\n'):
    if 'Basis dimensions:' in line:
        parts = line.split()
        mgcv_basis_dim = (int(parts[-2]), int(parts[-1]))
    elif 'Penalty dimensions:' in line:
        parts = line.split()
        mgcv_penalty_dim = (int(parts[-2]), int(parts[-1]))
    elif line.startswith('Lambda:'):
        mgcv_lambda = float(line.split()[1])
    elif line.startswith('Deviance:'):
        mgcv_deviance = float(line.split()[1])
    elif line.startswith('EDF:'):
        mgcv_edf = float(line.split()[1])

mgcv_pred = np.loadtxt('/tmp/mgcv_pred_k10.csv', delimiter=',', skiprows=1)

print(f"✓ mgcv fit complete")
print(f"  Basis dim:  {mgcv_basis_dim}")
print(f"  Penalty dim: {mgcv_penalty_dim}")
print(f"  Lambda:    {mgcv_lambda:.6f}")
print(f"  Deviance:  {mgcv_deviance:.4f}")
print(f"  EDF:       {mgcv_edf:.4f}")

print("\n[3/3] Comparing results...")

lambda_ratio = result_rust['lambda'] / mgcv_lambda
deviance_ratio = result_rust['deviance'] / mgcv_deviance
pred_corr = np.corrcoef(pred_rust, mgcv_pred)[0, 1]

print(f"\nComparison:")
print(f"  Lambda ratio:    {lambda_ratio:.4f}x")
print(f"  Deviance ratio:  {deviance_ratio:.4f}x")
print(f"  Pred correlation: {pred_corr:.6f}")

print("\n" + "=" * 70)
print("ASSESSMENT")
print("=" * 70)

if mgcv_basis_dim[1] == 10 and mgcv_penalty_dim == (10, 10):
    print("✅ mgcv uses k=10 basis and 10x10 penalty (confirmed)")
else:
    print(f"❌ mgcv dimensions unexpected: {mgcv_basis_dim}, {mgcv_penalty_dim}")

if 0.9 < lambda_ratio < 1.1:
    print("✅ Lambda matches within 10%!")
elif 0.5 < lambda_ratio < 2.0:
    print(f"✓  Lambda reasonably close ({lambda_ratio:.2f}x)")
else:
    print(f"⚠️  Lambda still off by {lambda_ratio:.2f}x - REML formula needs investigation")

if 0.9 < deviance_ratio < 1.1:
    print("✅ Deviance matches within 10%!")
elif 0.5 < deviance_ratio < 2.0:
    print(f"✓  Deviance reasonably close ({deviance_ratio:.2f}x)")
else:
    print(f"⚠️  Deviance still off by {deviance_ratio:.2f}x")

if pred_corr > 0.99:
    print("✅ Predictions match very well")
else:
    print(f"⚠️  Prediction correlation: {pred_corr:.4f}")

print("\n" + "=" * 70)
if lambda_ratio > 2.0 or deviance_ratio > 2.0:
    print("📊 Next step: Investigate REML formula in src/reml.rs")
else:
    print("🎉 Results match! Problem solved!")
print("=" * 70)
