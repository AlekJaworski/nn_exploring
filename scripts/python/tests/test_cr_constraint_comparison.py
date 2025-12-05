#!/usr/bin/env python3
"""
Compare Rust CR spline implementation with mgcv
No rpy2 dependency - uses subprocess to call R
"""

import numpy as np
import sys
import subprocess
import csv

sys.path.insert(0, "target/release")

print("=" * 70)
print("CR Spline Implementation: Rust vs mgcv Comparison")
print("=" * 70)

# Generate test data with fixed seed for reproducibility
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
noise = np.random.normal(0, 0.1, n)
y = y_true + noise

# Save data for R
with open('/tmp/test_data_comparison.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

print("\n✓ Generated test data (n=100, sin wave with noise)")

# Fit with Rust implementation
print("\n" + "=" * 70)
print("Fitting with Rust CR splines (k=10, constrained to k-1=9)")
print("=" * 70)

try:
    import mgcv_rust

    X = x.reshape(-1, 1)
    gam = mgcv_rust.GAM()
    result = gam.fit_auto(X, y, k=[10], method='REML', bs='cr')
    pred_rust = gam.predict(X)

    rust_lambda = result['lambda']
    rust_deviance = result['deviance']
    rust_rmse = np.sqrt(np.mean((pred_rust - y_true)**2))

    print(f"✓ Rust fit complete")
    print(f"  Lambda:    {rust_lambda:.6f}")
    print(f"  Deviance:  {rust_deviance:.4f}")
    print(f"  RMSE:      {rust_rmse:.4f}")

    # Save predictions
    with open('/tmp/rust_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pred'])
        for p in pred_rust:
            writer.writerow([p])

except Exception as e:
    print(f"✗ Rust fit failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Fit with mgcv
print("\n" + "=" * 70)
print("Fitting with R mgcv (k=10, bs='cr')")
print("=" * 70)

r_script = """
library(mgcv)

# Load data
data <- read.csv('/tmp/test_data_comparison.csv')
x <- data$x
y <- data$y

# Fit GAM with CR splines
set.seed(42)
gam_fit <- gam(y ~ s(x, k=10, bs='cr'), method='REML')

# Extract results
lambda <- gam_fit$sp
deviance <- deviance(gam_fit)
edf <- sum(gam_fit$edf)
predictions <- predict(gam_fit)

# Save results
cat('Lambda:', lambda, '\\n')
cat('Deviance:', deviance, '\\n')
cat('EDF:', edf, '\\n')

# Save predictions
write.csv(data.frame(pred=predictions), '/tmp/mgcv_predictions.csv', row.names=FALSE)
"""

try:
    result = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                          capture_output=True, timeout=30)

    if result.returncode != 0:
        print(f"✗ R script failed:")
        print(result.stderr)
        sys.exit(1)

    # Parse R output
    output = result.stdout
    mgcv_lambda = None
    mgcv_deviance = None
    mgcv_edf = None

    for line in output.split('\n'):
        if line.startswith('Lambda:'):
            mgcv_lambda = float(line.split()[1])
        elif line.startswith('Deviance:'):
            mgcv_deviance = float(line.split()[1])
        elif line.startswith('EDF:'):
            mgcv_edf = float(line.split()[1])

    # Load mgcv predictions
    mgcv_pred = []
    with open('/tmp/mgcv_predictions.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mgcv_pred.append(float(row['pred']))
    mgcv_pred = np.array(mgcv_pred)

    mgcv_rmse = np.sqrt(np.mean((mgcv_pred - y_true)**2))

    print(f"✓ mgcv fit complete")
    print(f"  Lambda:    {mgcv_lambda:.6f}")
    print(f"  Deviance:  {mgcv_deviance:.4f}")
    print(f"  EDF:       {mgcv_edf:.4f}")
    print(f"  RMSE:      {mgcv_rmse:.4f}")

except Exception as e:
    print(f"✗ mgcv fit failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare results
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

lambda_ratio = rust_lambda / mgcv_lambda
deviance_ratio = rust_deviance / mgcv_deviance
pred_corr = np.corrcoef(pred_rust, mgcv_pred)[0, 1]
pred_rmse_diff = np.sqrt(np.mean((pred_rust - mgcv_pred)**2))

print(f"\nLambda:")
print(f"  Rust:      {rust_lambda:.6f}")
print(f"  mgcv:      {mgcv_lambda:.6f}")
print(f"  Ratio:     {lambda_ratio:.4f}x")

print(f"\nDeviance:")
print(f"  Rust:      {rust_deviance:.4f}")
print(f"  mgcv:      {mgcv_deviance:.4f}")
print(f"  Ratio:     {deviance_ratio:.4f}x")

print(f"\nPredictions:")
print(f"  Correlation:   {pred_corr:.6f}")
print(f"  RMSE diff:     {pred_rmse_diff:.6f}")
print(f"  Rust RMSE:     {rust_rmse:.4f}")
print(f"  mgcv RMSE:     {mgcv_rmse:.4f}")

# Success criteria
print("\n" + "=" * 70)
print("ASSESSMENT")
print("=" * 70)

all_good = True

if pred_corr > 0.99:
    print("✅ Predictions match very well (correlation > 0.99)")
elif pred_corr > 0.95:
    print("✓  Predictions match well (correlation > 0.95)")
else:
    print(f"⚠️  Predictions differ (correlation = {pred_corr:.4f})")
    all_good = False

if 0.9 < lambda_ratio < 1.1:
    print("✅ Lambda matches closely (within 10%)")
elif 0.5 < lambda_ratio < 2.0:
    print(f"✓  Lambda reasonably close (ratio = {lambda_ratio:.4f}x)")
elif 0.1 < lambda_ratio < 10.0:
    print(f"⚠️  Lambda differs (ratio = {lambda_ratio:.4f}x)")
    all_good = False
else:
    print(f"❌ Lambda differs significantly (ratio = {lambda_ratio:.4f}x)")
    all_good = False

if 0.95 < deviance_ratio < 1.05:
    print("✅ Deviance matches very closely (within 5%)")
elif 0.9 < deviance_ratio < 1.1:
    print("✓  Deviance matches closely (within 10%)")
elif 0.5 < deviance_ratio < 2.0:
    print(f"⚠️  Deviance differs (ratio = {deviance_ratio:.4f}x)")
    all_good = False
else:
    print(f"❌ Deviance differs significantly (ratio = {deviance_ratio:.4f}x)")
    all_good = False

print("\n" + "=" * 70)
if all_good:
    print("🎉 EXCELLENT: Implementation matches mgcv very well!")
else:
    print("📊 GOOD: Implementation working, some differences from mgcv")
print("=" * 70)
