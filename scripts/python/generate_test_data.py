#!/usr/bin/env python3
"""
Generate test data and save to CSV for consistent comparison
"""
import numpy as np
import pandas as pd

# Generate test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + np.random.normal(0, 0.1, n)

# Save to CSV
df = pd.DataFrame({'x': x, 'y': y, 'y_true': y_true})
df.to_csv('/tmp/test_data.csv', index=False)

print(f"Generated n={n} data points")
print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
print(f"y mean: {y.mean():.6f}")
print(f"y std: {y.std():.6f}")
print(f"Saved to /tmp/test_data.csv")
