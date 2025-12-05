#!/usr/bin/env python3
"""
Compare our det2 computation against mgcv's total Hessian at optimal λ.

This will tell us:
1. Is our det2 correct?
2. Is bSb2 significant or negligible?
3. What bSb2 values are needed to match mgcv?
"""

import numpy as np
import subprocess
import sys
import re

print("=" * 80)
print("det2 VALIDATION: Compare our det2 against mgcv at optimal λ")
print("=" * 80)

# Step 1: Get mgcv's optimal λ and Hessian
print("\n1. Getting mgcv's optimal values...")
result = subprocess.run(['Rscript', '-e', '''
library(mgcv)
set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML")

cat("MGCV_LAMBDA:", fit$sp, "\\n")
cat("MGCV_REML:", fit$gcv.ubre, "\\n")

h <- fit$outer.info$hess
cat("MGCV_H00:", h[1,1], "\\n")
cat("MGCV_H11:", h[2,2], "\\n")
cat("MGCV_H01:", h[1,2], "\\n")

g <- fit$outer.info$grad
cat("MGCV_G0:", g[1], "\\n")
cat("MGCV_G1:", g[2], "\\n")
'''], capture_output=True, text=True, timeout=30)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)
    sys.exit(1)

# Parse mgcv results
mgcv = {}
for line in result.stdout.split('\n'):
    if line.startswith('MGCV_'):
        parts = line.split(':', 1)
        key = parts[0].replace('MGCV_', '').lower()
        values = parts[1].strip().split()
        if len(values) == 1:
            mgcv[key] = float(values[0])
        else:
            mgcv[key] = [float(v) for v in values]

print("\nParsed mgcv:")
print(f"  λ = {mgcv['lambda']}")
print(f"  H[0,0] = {mgcv['h00']:.6f}")
print(f"  H[1,1] = {mgcv['h11']:.6f}")
print(f"  H[0,1] = {mgcv['h01']:.6f}")
print(f"  REML = {mgcv['reml']:.6f}")

# Step 2: Run our implementation starting slightly below optimal
# This ensures we compute at least one Hessian
print("\n" + "=" * 80)
print("2. Running our implementation starting near mgcv's optimal λ...")
print("=" * 80)

# Start slightly below optimal to trigger Newton step
near_optimal = [mgcv['lambda'][0] * 0.85, mgcv['lambda'][1] * 0.85]
print(f"Starting from λ = {near_optimal} (85% of optimal)")
print(f"Will converge toward λ = {mgcv['lambda']}")

import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

result_rust = subprocess.run(['python3', '-c', f'''
import numpy as np
import mgcv_rust
import os
os.environ["MGCV_GRAD_DEBUG"] = "1"

np.random.seed(42)
n = 100
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

result = mgcv_rust.fit_gam(
    x=x,
    y=y,
    smooths=[
        {{"type": "cr", "vars": [0], "k": 10}},
        {{"type": "cr", "vars": [1], "k": 10}}
    ],
    initial_sp={near_optimal},
    max_iter=3
)

print("OUR_LAMBDA:", result["sp"])
print("OUR_DEVIANCE:", result["deviance"])
'''], capture_output=True, text=True)

print(result_rust.stdout)

# Step 3: Parse our det2 values from debug output
print("\n" + "=" * 80)
print("3. Extracting our det2 values from debug output...")
print("=" * 80)

# Store all Hessian values from all iterations
all_det2 = []
all_hess = []
all_lambdas = []

current_lambda = None
current_det2 = {}
current_hess = {}

for line in result_rust.stderr.split('\n'):
    # Track lambda values
    if '[SMOOTH_DEBUG] Raw Hessian at λ=' in line:
        if current_det2:  # Save previous iteration
            all_det2.append(current_det2.copy())
            all_hess.append(current_hess.copy())
            all_lambdas.append(current_lambda)

        # Extract lambda
        match = re.search(r'λ=\[([\d.e+-]+), ([\d.e+-]+)\]', line)
        if match:
            current_lambda = [float(match.group(1)), float(match.group(2))]
        current_det2 = {}
        current_hess = {}

    # Look for Hessian debug output
    if '[HESS_DEBUG] Hessian[' in line:
        match = re.search(r'Hessian\[(\d+),(\d+)\]', line)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            key = f'{i}{j}'
    elif '[HESS_DEBUG]   det2 = ' in line:
        val = float(line.split('=')[1].strip())
        current_det2[key] = val
    elif '[HESS_DEBUG]   total hessian = ' in line:
        val = float(line.split('=')[1].strip())
        current_hess[key] = val

# Save final iteration
if current_det2:
    all_det2.append(current_det2.copy())
    all_hess.append(current_hess.copy())
    all_lambdas.append(current_lambda)

# Use the last iteration (closest to optimal)
if all_det2:
    our_det2 = all_det2[-1]
    our_hess = all_hess[-1]
    our_lambda = all_lambdas[-1]

    print(f"\nUsing Hessian from last iteration:")
    print(f"  λ = {our_lambda}")
    print(f"  (vs mgcv optimal = {mgcv['lambda']})")
else:
    print("\n⚠️  WARNING: No Hessian debug output found!")
    print("Check stderr below:")
    print(result_rust.stderr[:2000])
    our_det2 = {}
    our_hess = {}
    our_lambda = None

print("\nOur det2 values (log-determinant Hessian only):")
print(f"  det2[0,0] = {our_det2.get('00', 0):.6e}")
print(f"  det2[1,1] = {our_det2.get('11', 0):.6e}")
print(f"  det2[0,1] = {our_det2.get('01', 0):.6e}")

print("\nOur total Hessian (det2 + bSb2, where bSb2=0):")
print(f"  H[0,0] = {our_hess.get('00', 0):.6e}")
print(f"  H[1,1] = {our_hess.get('11', 0):.6e}")
print(f"  H[0,1] = {our_hess.get('01', 0):.6e}")

# Step 4: Compare and compute required bSb2
print("\n" + "=" * 80)
print("4. COMPARISON AND ANALYSIS")
print("=" * 80)

print("\n### Diagonal Terms ###")
for idx, name in [(('00', '0,0')), (('11', '1,1'))]:
    mgcv_val = mgcv[f'h{idx}']
    our_det2_val = our_det2.get(idx, 0)
    our_total = our_hess.get(idx, 0)

    ratio = our_total / mgcv_val if mgcv_val != 0 else 0
    bSb2_needed = (mgcv_val - our_det2_val) * 2  # *2 because we divide by 2 in formula

    print(f"\nH[{name}]:")
    print(f"  mgcv total:     {mgcv_val:10.6f}")
    print(f"  our det2/2:     {our_total:10.6f} ({ratio*100:5.1f}% of mgcv)")
    print(f"  difference:     {mgcv_val - our_total:10.6f}")
    print(f"  bSb2 needed:    {bSb2_needed:10.6f} (before /2)")

print(f"\n### Off-Diagonal Terms ###")
mgcv_val = mgcv['h01']
our_det2_val = our_det2.get('01', 0)
our_total = our_hess.get('01', 0)

ratio = our_total / mgcv_val if mgcv_val != 0 else 0
bSb2_needed = (mgcv_val - our_det2_val) * 2

print(f"\nH[0,1]:")
print(f"  mgcv total:     {mgcv_val:10.6f}")
print(f"  our det2/2:     {our_total:10.6f}")
print(f"  difference:     {mgcv_val - our_total:10.6f}")
print(f"  bSb2 needed:    {bSb2_needed:10.6f} (before /2)")

print("\n" + "=" * 80)
print("5. CONCLUSIONS")
print("=" * 80)

# Calculate average ratio
ratios = []
for idx in ['00', '11', '01']:
    mgcv_val = mgcv[f'h{idx}']
    our_total = our_hess.get(idx, 0)
    if mgcv_val != 0:
        ratios.append(abs(our_total / mgcv_val))

avg_ratio = np.mean(ratios) if ratios else 0

print(f"""
Our det2-only Hessian is {avg_ratio*100:.1f}% of mgcv's total Hessian.

**Interpretation**:
""")

if avg_ratio > 0.9:
    print("✅ det2 is SUFFICIENT - bSb2 contributes < 10%")
    print("   → Our det2 formula is correct")
    print("   → bSb2 is negligible at optimal λ")
    print("   → Convergence issues must be elsewhere")
elif avg_ratio > 0.5:
    print("⚠️  det2 is DOMINANT but bSb2 is SIGNIFICANT")
    print("   → Our det2 is mostly correct")
    print("   → bSb2 contributes 10-50%, should be implemented")
    print("   → This explains partial convergence")
elif avg_ratio > 0.1:
    print("⚠️  det2 and bSb2 are COMPARABLE")
    print("   → Both terms are important")
    print("   → Need proper bSb2 implementation")
    print("   → det2 might also have errors")
else:
    print("❌ det2 is TOO SMALL - formula likely wrong")
    print("   → Our det2 formula has errors")
    print("   → Fix det2 before implementing bSb2")

print(f"""
**Next Steps**:
""")

if avg_ratio < 0.5:
    print("1. DEBUG det2 formula - compare trace terms against mgcv")
    print("2. Check: tr(A^{-1}·M_i) computation")
    print("3. Check: tr[(A^{-1}·M_i)·(A^{-1}·M_j)] computation")
else:
    print("1. IMPLEMENT proper bSb2 with β derivatives")
    print("2. Compute dβ/dρ_i = -A^{-1}·M_i·β")
    print("3. Implement 4-term bSb2 formula from mgcv C code")

print("\nSee output above for exact bSb2 values needed.")
