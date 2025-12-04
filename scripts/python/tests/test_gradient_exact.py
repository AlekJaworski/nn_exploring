#!/usr/bin/env python3
"""
EXACT gradient comparison at SAME λ.
"""

import numpy as np
import pandas as pd
import mgcv_rust
import subprocess

# Test λ
test_lambda = [2.0, 3.0]

# Load data
data = pd.read_csv('/tmp/trace_test_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

print("=" * 80)
print(f"EXACT COMPARISON AT λ={test_lambda}")
print("=" * 80)

# Get mgcv's FD gradient
r_script = f'''
library(mgcv)
data <- read.csv("/tmp/trace_test_data.csv")
lambda <- c({test_lambda[0]}, {test_lambda[1]})
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data, method="REML", sp=lambda)

delta <- 1e-6
reml0 <- fit$gcv.ubre

fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data, method="REML", sp=lambda * exp(c(delta, 0)))
grad1 <- (fit1$gcv.ubre - reml0) / delta

fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=data, method="REML", sp=lambda * exp(c(0, delta)))
grad2 <- (fit2$gcv.ubre - reml0) / delta

cat(grad1, grad2, "\\n")
'''

result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)
mgcv_grad = np.array([float(x) for x in result.stdout.strip().split()])

print(f"\\n1. mgcv TRUE gradient (finite difference):")
print(f"   {mgcv_grad}")

# Compute OUR gradient at same λ
print(f"\\n2. Computing OUR gradient at λ={test_lambda}...")

import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

our_grad = mgcv_rust.evaluate_gradient(x, y, test_lambda, [10, 10])

print(f"   {our_grad}")

# Compare
print(f"\\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\\nmgcv gradient:  {mgcv_grad}")
print(f"our gradient:   {our_grad}")
print(f"difference:     {our_grad - mgcv_grad}")
print(f"ratio:          {our_grad / mgcv_grad}")

error = np.abs(our_grad - mgcv_grad) / np.abs(mgcv_grad)
print(f"relative error: {error * 100}%")

if np.all(error < 0.01):
    print(f"\\n✅ GRADIENTS MATCH! (< 1% error)")
else:
    print(f"\\n❌ GRADIENTS DIFFER by {error.max()*100:.1f}%")
    print(f"\\nThis confirms our gradient formula is still wrong.")
