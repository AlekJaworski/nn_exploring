#!/usr/bin/env python3
"""
Deterministic test to compare our gradient calculation with mgcv's.
We'll use the EXACT same data and check if gradients match.
"""

import numpy as np
import subprocess
import json

# Fixed seed for determinism
np.random.seed(42)
n = 1000
x = np.random.randn(n, 4)
y = np.sin(x[:, 0]) + 0.5*x[:, 1]**2 + np.cos(x[:, 2]) + 0.3*x[:, 3] + np.random.randn(n)*0.1

# Save data for R (CSV format)
np.savetxt('/tmp/test_x.csv', x, delimiter=',')
np.savetxt('/tmp/test_y.csv', y)

print("=" * 70)
print("DETERMINISTIC GRADIENT COMPARISON: Rust vs R")
print("=" * 70)
print(f"\nData: n={n}, p=4, k=16 (CR splines)")
print(f"Seed: 42 (fixed for determinism)")

# Get R's gradient at specific lambda values
r_script = '''
library(mgcv)

# Load exact same data
x <- as.matrix(read.csv("/tmp/test_x.csv", header=FALSE))
y <- as.numeric(read.csv("/tmp/test_y.csv", header=FALSE)$V1)

# Build GAM with proper data frames
df <- data.frame(x1=x[,1], x2=x[,2], x3=x[,3], x4=x[,4], y=y)
sm1 <- smoothCon(s(x1, k=16, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=16, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm3 <- smoothCon(s(x3, k=16, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm4 <- smoothCon(s(x4, k=16, bs="cr"), df, absorb.cons=TRUE)[[1]]

# Design matrix and penalties
X <- cbind(sm1$X, sm2$X, sm3$X, sm4$X)
S <- list(sm1$S[[1]], sm2$S[[1]], sm3$S[[1]], sm4$S[[1]])

# Check rank of penalty matrices
cat("\n=== Rank of each smooth S matrix ===\n")
for (i in 1:4) {
    cat(sprintf("S%d: %dx%d, rank = %d\n",
        i, nrow(S[[i]]), ncol(S[[i]]),
        qr(S[[i]])$rank))
}

# Fit GAM with REML to see optimization trace
cat("\n=== Fitting GAM with REML ===\n")
ctrl <- gam.control(trace=TRUE, epsilon=1e-7)
fit <- gam(y ~ s(x1, k=16, bs="cr") + s(x2, k=16, bs="cr") +
             s(x3, k=16, bs="cr") + s(x4, k=16, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\n=== Final Results ===\n")
cat("Final lambda: ", fit$sp, "\n")
cat("REML value: ", fit$gcv.ubre, "\n")
cat("Iterations: (see trace above)\n")
'''

print("\n" + "=" * 70)
print("R's gradient calculation:")
print("=" * 70)

result = subprocess.run(['R', '--vanilla', '--slave', '-e', r_script],
                       capture_output=True, text=True, timeout=60)
print(result.stdout)
if result.stderr:
    lines = result.stderr.split('\n')
    for line in lines[-20:]:  # Last 20 lines
        if line.strip():
            print(line)

# Now test our implementation
print("\n" + "=" * 70)
print("Our (Rust) gradient calculation:")
print("=" * 70)

import mgcv_rust
import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

# Start with lambda=[1,1,1,1] explicitly
gam = mgcv_rust.GAM()

# We need to manually set initial lambda to [1,1,1,1]
# This requires modifying the optimization to accept initial lambda
# For now, let's just see what our initialization gives us

result = gam.fit_auto_optimized(x, y, k=[16]*4, method='REML', bs='cr')

print(f"\nFinal lambda: {result['lambda']}")
print("\n" + "=" * 70)
print("Analysis:")
print("=" * 70)
print("""
KEY QUESTIONS:
1. Does our rank estimation match R's qr(S)$rank?
2. At the same lambda values, do our gradients match?
3. Is there a scaling difference in the gradient formula?
4. Does initialization differ significantly?
""")
