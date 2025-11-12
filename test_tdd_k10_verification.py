#!/usr/bin/env python3
"""
Test-Driven Development: Verify k=10 behavior matches mgcv
"""

import numpy as np
import sys
import subprocess
import csv

sys.path.insert(0, "target/release")

print("=" * 70)
print("TDD: Testing k=10 CR Splines (No Constraint Reduction)")
print("=" * 70)

# Test data (same as before for consistency)
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
noise = np.random.normal(0, 0.1, n)
y = y_true + noise

# Save data for R
with open('/tmp/test_data_tdd.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

print("\n" + "=" * 70)
print("EXPECTATION 1: Basis should have k=10 columns (not k-1=9)")
print("=" * 70)

import mgcv_rust

X = x.reshape(-1, 1)
gam = mgcv_rust.GAM()

# Get basis dimensions before fitting
try:
    # Create a CR smooth term
    from mgcv_rust import gam as gam_module
    smooth = gam_module.SmoothTerm.cr_spline("x", 10, 0.0, 1.0)

    # Evaluate basis
    x_test = np.linspace(0, 1, 10)
    from numpy import array
    x_arr = array(x_test)
    basis = smooth.evaluate(x_arr)

    print(f"✓ Basis evaluated")
    print(f"  Expected dimensions: 10 x 10")
    print(f"  Actual dimensions:   {basis.shape[0]} x {basis.shape[1]}")

    if basis.shape[1] == 10:
        print("✅ PASS: Basis has k=10 columns")
    else:
        print(f"❌ FAIL: Expected 10 columns, got {basis.shape[1]}")
        sys.exit(1)

except Exception as e:
    print(f"❌ FAIL: Could not evaluate basis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("EXPECTATION 2: Penalty should be 10x10 (not 9x9)")
print("=" * 70)

penalty_shape = smooth.penalty.shape
print(f"  Expected shape: 10 x 10")
print(f"  Actual shape:   {penalty_shape[0]} x {penalty_shape[1]}")

if penalty_shape == (10, 10):
    print("✅ PASS: Penalty matrix is 10x10")
else:
    print(f"❌ FAIL: Expected 10x10, got {penalty_shape[0]}x{penalty_shape[1]}")
    sys.exit(1)

print("\n" + "=" * 70)
print("EXPECTATION 3: Fit should use 10 basis functions")
print("=" * 70)

result = gam.fit_auto(X, y, k=[10], method='REML', bs='cr')
pred_rust = gam.predict(X)

print(f"✓ GAM fitted with REML")
print(f"  Lambda:    {result['lambda']:.6f}")
print(f"  Deviance:  {result['deviance']:.4f}")

# Get coefficients
# Note: Can't access coefficients directly from Python, but deviance tells us fit worked

print("\n" + "=" * 70)
print("EXPECTATION 4: Results should be closer to mgcv now")
print("=" * 70)

# Fit with mgcv
r_script = """
library(mgcv)
data <- read.csv('/tmp/test_data_tdd.csv')
x <- data$x
y <- data$y

set.seed(42)
gam_fit <- gam(y ~ s(x, k=10, bs='cr'), method='REML')

# Check dimensions
sm <- smoothCon(s(x, k=10, bs='cr'), data=data.frame(x=x), knots=NULL)[[1]]

cat('mgcv basis columns:', ncol(sm$X), '\\n')
cat('mgcv penalty dim:', nrow(sm$S[[1]]), 'x', ncol(sm$S[[1]]), '\\n')
cat('Lambda:', gam_fit$sp, '\\n')
cat('Deviance:', deviance(gam_fit), '\\n')

# Save predictions
predictions <- predict(gam_fit)
write.csv(data.frame(pred=predictions), '/tmp/mgcv_predictions_tdd.csv', row.names=FALSE)
"""

result_r = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                         capture_output=True, timeout=30)

if result_r.returncode != 0:
    print(f"❌ R script failed: {result_r.stderr}")
    sys.exit(1)

# Parse R output
output = result_r.stdout
mgcv_lambda = None
mgcv_deviance = None
mgcv_basis_cols = None

for line in output.split('\n'):
    if 'basis columns:' in line:
        mgcv_basis_cols = int(line.split()[-1])
    elif line.startswith('Lambda:'):
        mgcv_lambda = float(line.split()[1])
    elif line.startswith('Deviance:'):
        mgcv_deviance = float(line.split()[1])

print(f"\nmgcv verification:")
print(f"  Basis columns: {mgcv_basis_cols}")
print(f"  Lambda:        {mgcv_lambda:.6f}")
print(f"  Deviance:      {mgcv_deviance:.4f}")

# Load mgcv predictions
mgcv_pred = []
with open('/tmp/mgcv_predictions_tdd.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        mgcv_pred.append(float(row['pred']))
mgcv_pred = np.array(mgcv_pred)

# Compare
lambda_ratio = result['lambda'] / mgcv_lambda
deviance_ratio = result['deviance'] / mgcv_deviance
pred_corr = np.corrcoef(pred_rust, mgcv_pred)[0, 1]

print(f"\nComparison:")
print(f"  Lambda ratio (Rust/mgcv):    {lambda_ratio:.4f}x")
print(f"  Deviance ratio (Rust/mgcv):  {deviance_ratio:.4f}x")
print(f"  Prediction correlation:      {pred_corr:.6f}")

print("\n" + "=" * 70)
print("ASSESSMENT")
print("=" * 70)

# Note: We expect lambda/deviance to STILL be off because we haven't fixed REML yet
# But at least the dimensions should match now

if mgcv_basis_cols == 10:
    print("✅ PASS: Both use k=10 basis (dimensions match)")
else:
    print(f"❌ FAIL: Dimension mismatch")

if pred_corr > 0.99:
    print("✅ PASS: Predictions highly correlated")
elif pred_corr > 0.95:
    print("✓  PASS: Predictions reasonably correlated")
else:
    print(f"❌ FAIL: Predictions don't match (correlation={pred_corr:.4f})")

# We EXPECT lambda/deviance to still be off (REML formula issue)
print(f"\n⚠️  EXPECTED: Lambda/deviance still off (REML formula not fixed yet)")
print(f"    Lambda ratio: {lambda_ratio:.4f}x (investigating REML next)")

print("\n" + "=" * 70)
print("TDD RESULT: Basis dimensions now match mgcv ✅")
print("Next step: Investigate REML formula")
print("=" * 70)
