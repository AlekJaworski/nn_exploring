#!/usr/bin/env python3
"""
Simple debug: Check what's actually happening with CR spline fit
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

k = 10

print(f"Testing k={k} CR spline")
print(f"Data: n={n}, x in [0,1], y = sin(2πx) + noise")
print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

# Rust fit
import mgcv_rust

X = x.reshape(-1, 1)
gam = mgcv_rust.GAM()
result_rust = gam.fit_auto(X, y, k=[k], method='REML', bs='cr')
pred_rust = gam.predict(X)

residuals_rust = y - pred_rust
ss_res_rust = np.sum(residuals_rust**2)

print(f"\nRust:")
print(f"  Lambda: {result_rust['lambda']:.6f}")
print(f"  Deviance (from result): {result_rust['deviance']:.4f}")
print(f"  Sum squared residuals: {ss_res_rust:.4f}")
print(f"  Predictions: mean={pred_rust.mean():.4f}, std={pred_rust.std():.4f}, min={pred_rust.min():.4f}, max={pred_rust.max():.4f}")

# Save for R
with open('/tmp/test_cr_debug.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

# mgcv fit
r_script = f"""
library(mgcv)
data <- read.csv('/tmp/test_cr_debug.csv')

set.seed(42)
gam_fit <- gam(y ~ s(x, k={k}, bs='cr'), data=data, method='REML')

pred <- predict(gam_fit)
residuals <- data$y - pred
ss_res <- sum(residuals^2)

cat('Lambda:', gam_fit$sp, '\\n')
cat('Deviance:', deviance(gam_fit), '\\n')
cat('Sum squared residuals:', ss_res, '\\n')
cat('Predictions: mean=', mean(pred), ' std=', sd(pred), ' min=', min(pred), ' max=', max(pred), '\\n', sep='')

# Check if there's an intercept
cat('Intercept coef:', coef(gam_fit)[1], '\\n')
cat('Has intercept in formula?', '(Intercept)' %in% names(coef(gam_fit)), '\\n')

write.csv(data.frame(pred=pred), '/tmp/mgcv_pred_cr_debug.csv', row.names=FALSE)
"""

result_r = subprocess.run(['Rscript', '-'], input=r_script, text=True,
                         capture_output=True, timeout=30)

if result_r.returncode == 0:
    print(f"\nmgcv:")
    print(result_r.stdout)

    mgcv_pred = np.loadtxt('/tmp/mgcv_pred_cr_debug.csv', delimiter=',', skiprows=1)

    print("\nComparison:")
    print(f"  Prediction correlation: {np.corrcoef(pred_rust, mgcv_pred)[0,1]:.8f}")
    print(f"  Prediction RMSE: {np.sqrt(np.mean((pred_rust - mgcv_pred)**2)):.6f}")
    print(f"  Prediction max diff: {np.max(np.abs(pred_rust - mgcv_pred)):.6f}")

    # Check if predictions are shifted
    mean_diff = pred_rust.mean() - mgcv_pred.mean()
    print(f"  Mean difference (Rust - mgcv): {mean_diff:.6f}")

    # Try correlation after centering
    pred_rust_centered = pred_rust - pred_rust.mean()
    mgcv_pred_centered = mgcv_pred - mgcv_pred.mean()
    centered_rmse = np.sqrt(np.mean((pred_rust_centered - mgcv_pred_centered)**2))
    print(f"  Centered RMSE: {centered_rmse:.6f}")

else:
    print(f"R error: {result_r.stderr}")
