#!/usr/bin/env python3
"""Debug extrapolation behavior"""

import numpy as np
import mgcv_rust

np.random.seed(42)

# Simple test
x_train = np.linspace(0.3, 0.7, 50)
y_train = 2 * x_train + 1 + 0.1 * np.random.randn(50)

print(f"Training range: [{x_train.min():.3f}, {x_train.max():.3f}]")

gam = mgcv_rust.GAM()
result = gam.fit_auto(x_train.reshape(-1, 1), y_train, k=[6], method='REML')
print(f"Fit successful, λ = {result['lambda']:.6f}")

# Test at specific points
test_points = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0])
X_test = test_points.reshape(-1, 1)
y_pred = gam.predict(X_test)
y_true = 2 * test_points + 1

print(f"\nPredictions:")
for i, x in enumerate(test_points):
    status = "LEFT" if x < 0.3 else ("RIGHT" if x > 0.7 else "IN")
    print(f"  x={x:.1f} ({status:5s}): pred={y_pred[i]:.4f}, true={y_true[i]:.4f}, error={abs(y_pred[i]-y_true[i]):.4f}")

# Check if RIGHT extrapolation is working
if abs(y_pred[5]) < 1e-10:  # x=0.8
    print(f"\n⚠ RIGHT extrapolation NOT working (getting zeros)")
else:
    print(f"\n✓ RIGHT extrapolation working!")

if abs(y_pred[0]) < 1e-10:  # x=0.0
    print(f"⚠ LEFT extrapolation NOT working (getting zeros)")
else:
    print(f"✓ LEFT extrapolation working!")
