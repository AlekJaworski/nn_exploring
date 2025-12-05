#!/usr/bin/env python3
"""
Count iterations for Rust vs R
"""

import numpy as np
import mgcv_rust
import subprocess
import os

# Same data as test_4d_multidim_inference.py
np.random.seed(42)
n, d, k = 500, 4, 12

X = np.random.uniform(0, 1, size=(n, d))

effect_1 = np.sin(2 * np.pi * X[:, 0])
effect_2 = (X[:, 1] - 0.5) ** 2
effect_3 = X[:, 2]
y = effect_1 + effect_2 + effect_3 + np.random.normal(0, 0.3, n)

print("="*70)
print("Iteration Count Comparison")
print("="*70)
print(f"Data: n={n}, d={d}, k={k}")

# Fit with Rust (with profiling)
print("\n[Rust] Fitting with profiling...")
os.environ['MGCV_PROFILE'] = '1'

gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr', max_iter=20)

# Count Newton iterations from stderr (profiling output)
# The profiling output goes to stderr, so we can't capture it here
# But we can see it in the output above

del os.environ['MGCV_PROFILE']

print(f"\nRust results:")
print(f"  Deviance: {result['deviance']:.6f}")
print(f"  Lambda: {result['lambda']}")
print(f"  (Check stderr output above for iteration count)")

# Save data for R
np.savetxt('/tmp/test_x.csv', X, delimiter=',')
np.savetxt('/tmp/test_y.csv', y, delimiter=',')

# R script
r_script = """
library(mgcv)

# Load data
X <- as.matrix(read.csv('/tmp/test_x.csv', header=FALSE))
y <- as.matrix(read.csv('/tmp/test_y.csv', header=FALSE))[,1]

cat('\\n[R] Fitting with mgcv...\\n')

# Fit
fit <- gam(
  y ~ s(X[,1], k=12, bs='cr') + s(X[,2], k=12, bs='cr') +
      s(X[,3], k=12, bs='cr') + s(X[,4], k=12, bs='cr'),
  method='REML'
)

cat('\\nR results:\\n')
cat('  Deviance:', deviance(fit), '\\n')
cat('  Lambda:', fit$sp, '\\n')
cat('  Outer iterations:', fit$outer.info$iter, '\\n')
cat('  Converged:', fit$outer.info$conv, '\\n')
"""

with open('/tmp/count_iters.R', 'w') as f:
    f.write(r_script)

print("\n" + "="*70)
print("Running R comparison...")
print("="*70)

result = subprocess.run(['Rscript', '/tmp/count_iters.R'],
                       capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("R stderr:", result.stderr)
