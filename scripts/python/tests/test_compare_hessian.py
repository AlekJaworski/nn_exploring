#!/usr/bin/env python3
"""
Compare our Hessian computation against mgcv's at optimal λ values.

This test:
1. Uses mgcv's optimal λ = [5.69, 5.20]
2. Computes our det2, bSb2, and total Hessian
3. Compares against mgcv's total Hessian [2.81, 3.19, 0.023]
4. Identifies discrepancies
"""

import numpy as np
import mgcv_rust
import subprocess
import sys

print("=" * 80)
print("Comparing Hessian at mgcv's optimal λ = [5.69, 5.20]")
print("=" * 80)

# Get mgcv's Hessian at optimal λ
print("\n1. Getting mgcv's Hessian at optimal λ...")
result = subprocess.run(['Rscript', '-e', '''
library(mgcv)
set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# First get optimal λ by fitting normally
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML")

cat("mgcv optimal λ:", fit$sp, "\\n")
cat("mgcv REML:", fit$gcv.ubre, "\\n")

# Get Hessian at optimal
if (!is.null(fit$outer.info$hess)) {
    h <- fit$outer.info$hess
    cat("mgcv Hessian[0,0]:", h[1,1], "\\n")
    cat("mgcv Hessian[1,1]:", h[2,2], "\\n")
    cat("mgcv Hessian[0,1]:", h[1,2], "\\n")
}

# Also get gradient (should be near zero)
if (!is.null(fit$outer.info$grad)) {
    cat("mgcv gradient[0]:", fit$outer.info$grad[1], "\\n")
    cat("mgcv gradient[1]:", fit$outer.info$grad[2], "\\n")
}
'''], capture_output=True, text=True, timeout=30)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr, file=sys.stderr)
    sys.exit(1)

# Parse mgcv results
mgcv_lambda = None
mgcv_hess_00 = None
mgcv_hess_11 = None
mgcv_hess_01 = None
mgcv_grad = []

for line in result.stdout.split('\n'):
    if 'mgcv optimal λ:' in line:
        parts = line.split(':')[1].strip().split()
        mgcv_lambda = [float(parts[0]), float(parts[1])]
    elif 'mgcv Hessian[0,0]:' in line:
        mgcv_hess_00 = float(line.split(':')[1].strip())
    elif 'mgcv Hessian[1,1]:' in line:
        mgcv_hess_11 = float(line.split(':')[1].strip())
    elif 'mgcv Hessian[0,1]:' in line:
        mgcv_hess_01 = float(line.split(':')[1].strip())
    elif 'mgcv gradient[0]:' in line:
        mgcv_grad.append(float(line.split(':')[1].strip()))
    elif 'mgcv gradient[1]:' in line:
        mgcv_grad.append(float(line.split(':')[1].strip()))

print(f"\nParsed mgcv results:")
print(f"  λ = {mgcv_lambda}")
print(f"  Hessian[0,0] = {mgcv_hess_00:.6f}")
print(f"  Hessian[1,1] = {mgcv_hess_11:.6f}")
print(f"  Hessian[0,1] = {mgcv_hess_01:.6f}")
print(f"  Gradient = {mgcv_grad}")

# Now run our implementation at the same λ
print("\n" + "=" * 80)
print("2. Running our implementation at same λ...")
print("=" * 80)

# Generate same data
np.random.seed(42)
n = 100
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

# Create GAM with fixed λ
import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

# We need to run one iteration with fixed λ to get our Hessian
# The Rust code will compute and print debug info
print("\nNote: Look for [HESS_DEBUG] lines in output to see our values\n")

# Create a simple test that fits with one iteration from mgcv's λ
result_rust = subprocess.run([
    'python3', '-c', f'''
import numpy as np
import mgcv_rust
import os

os.environ["MGCV_GRAD_DEBUG"] = "1"

np.random.seed(42)
n = 100
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

# Initialize with mgcv's optimal λ
initial_sp = {mgcv_lambda}

result = mgcv_rust.fit_gam(
    x=x,
    y=y,
    smooths=[
        {{"type": "cr", "vars": [0], "k": 10}},
        {{"type": "cr", "vars": [1], "k": 10}}
    ],
    initial_sp=initial_sp,
    max_outer_iter=1  # Just one iteration to see Hessian
)

print("Our λ after 1 iter:", result["sp"])
'''
], capture_output=True, text=True)

print(result_rust.stdout)
if result_rust.stderr:
    # Look for HESS_DEBUG lines
    for line in result_rust.stderr.split('\n'):
        if 'HESS_DEBUG' in line or 'gradient[' in line:
            print(line)

print("\n" + "=" * 80)
print("3. Analysis")
print("=" * 80)

print(f"""
Look for the [HESS_DEBUG] output above to see:
- trace_a_inv_m: diagonal term tr(A^{{-1}}·M_i)
- trace_term: tr[(A^{{-1}}·M_i)·(A^{{-1}}·M_j)]
- det2: log-determinant Hessian component
- term2, term3: bSb2 components (incomplete)
- bSb2: total penalty Hessian
- total hessian: det2 + bSb2

Compare these against mgcv's total:
  Hessian[0,0] = {mgcv_hess_00:.6f} (mgcv)
  Hessian[1,1] = {mgcv_hess_11:.6f} (mgcv)
  Hessian[0,1] = {mgcv_hess_01:.6f} (mgcv)
""")

print("""
Expected findings:
- If det2 alone ≈ mgcv total: bSb2 terms are negligible
- If det2 + bSb2 << mgcv total: we're missing major components
- If det2 >> mgcv total: det2 computation is wrong
- If bSb2 is huge negative: term2/term3 formulas are wrong
""")
