#!/usr/bin/env python3
"""
Test to verify the φ bug hypothesis.

Compare:
1. Our current approach: φ = RSS / (n - Σrank(S_i))
2. Correct approach: φ = RSS / (n - tr(A^{-1}·X'WX))
3. mgcv's actual φ value

This will tell us if the φ bug explains our convergence issues.
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("φ (SCALE PARAMETER) BUG VERIFICATION")
print("=" * 80)

# Step 1: Get mgcv's φ at optimal λ
print("\n1. Getting mgcv's φ and edf at optimal λ...")
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
cat("MGCV_EDF:", sum(fit$edf), "\\n")  # Total effective df
cat("MGCV_EDF1:", fit$edf[1], "\\n")   # Smooth 1 edf
cat("MGCV_EDF2:", fit$edf[2], "\\n")   # Smooth 2 edf
cat("MGCV_SCALE:", fit$scale, "\\n")   # This is φ (should be same as sig2)
cat("MGCV_SIG2:", fit$sig2, "\\n")     # Alternative name for scale parameter
cat("MGCV_REML:", fit$gcv.ubre, "\\n")

# Compute RSS
fitted_vals <- predict(fit)
residuals <- y - fitted_vals
rss <- sum(residuals^2)
cat("MGCV_RSS:", rss, "\\n")
cat("MGCV_N:", n, "\\n")

# Verify: φ should equal RSS / (n - total_edf)
total_edf <- sum(fit$edf)
phi_check <- rss / (n - total_edf)
cat("MGCV_PHI_CHECK:", phi_check, "\\n")
cat("MGCV_RATIO:", fit$scale / phi_check, "\\n")  # Should be ~1.0
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

print("\n" + "=" * 80)
print("2. MGCV Results Summary")
print("=" * 80)
print(f"λ = {mgcv['lambda']}")
print(f"Total edf = {mgcv['edf']:.3f}")
print(f"  Smooth 1 edf = {mgcv['edf1']:.3f}")
print(f"  Smooth 2 edf = {mgcv['edf2']:.3f}")
print(f"φ (scale) = {mgcv['scale']:.6f}")
print(f"RSS = {mgcv['rss']:.6f}")
print(f"n = {mgcv['n']}")
print(f"")
print(f"Verification: φ = RSS/(n-edf)")
print(f"  RSS/(n-edf) = {mgcv['rss']:.6f} / ({mgcv['n']} - {mgcv['edf']:.3f})")
print(f"              = {mgcv['rss'] / (mgcv['n'] - mgcv['edf']):.6f}")
print(f"  mgcv φ      = {mgcv['scale']:.6f}")
print(f"  Ratio       = {mgcv['ratio']:.6f} (should be 1.0)")

# Step 2: Estimate what our WRONG approach gives
print("\n" + "=" * 80)
print("3. Our WRONG Approach (using penalty rank)")
print("=" * 80)

# For CR splines with k=10: rank = k-2 = 8
# (One null space dimension for the penalty)
# We have 2 smooths, so total_rank = 8 + 8 = 16
our_total_rank = 16

our_phi = mgcv['rss'] / (mgcv['n'] - our_total_rank)
print(f"Penalty ranks: S_1 has rank 8, S_2 has rank 8")
print(f"total_rank = {our_total_rank}")
print(f"Our φ = RSS / (n - total_rank)")
print(f"      = {mgcv['rss']:.6f} / ({mgcv['n']} - {our_total_rank})")
print(f"      = {our_phi:.6f}")

print("\n" + "=" * 80)
print("4. COMPARISON: Correct vs Wrong φ")
print("=" * 80)

correct_edf = mgcv['edf']
wrong_edf = our_total_rank
phi_correct = mgcv['scale']
phi_wrong = our_phi

print(f"Effective df:")
print(f"  Correct (mgcv): {correct_edf:.3f}")
print(f"  Wrong (ours):   {wrong_edf}")
print(f"  Difference:     {correct_edf - wrong_edf:.3f}")
print(f"  Error:          {(correct_edf - wrong_edf) / correct_edf * 100:.1f}%")
print(f"")
print(f"Scale parameter φ:")
print(f"  Correct (mgcv): {phi_correct:.6f}")
print(f"  Wrong (ours):   {phi_wrong:.6f}")
print(f"  Ratio:          {phi_wrong / phi_correct:.3f}")
print(f"  Error:          {(phi_wrong - phi_correct) / phi_correct * 100:.1f}%")

print("\n" + "=" * 80)
print("5. IMPACT ON HESSIAN")
print("=" * 80)

print(f"""
Since we divide bSb1 and bSb2 by φ:
  bSb1_correct = raw_bSb1 / φ_correct
  bSb1_wrong   = raw_bSb1 / φ_wrong

The scaling factor is:
  bSb1_wrong / bSb1_correct = φ_correct / φ_wrong
                             = {phi_correct / phi_wrong:.3f}

Similarly for bSb2 (Hessian penalty term):
  bSb2_wrong / bSb2_correct = {phi_correct / phi_wrong:.3f}
""")

if phi_wrong > phi_correct:
    factor = phi_wrong / phi_correct
    print(f"❌ Our φ is TOO LARGE by {(factor-1)*100:.1f}%")
    print(f"   → We UNDER-estimate bSb2 by factor of {factor:.2f}")
    print(f"   → Hessian is too small")
    print(f"   → Newton steps are too large")
    print(f"   → Explains why we overshoot and converge to wrong minimum!")
else:
    factor = phi_correct / phi_wrong
    print(f"❌ Our φ is TOO SMALL by {(1-1/factor)*100:.1f}%")
    print(f"   → We OVER-estimate bSb2 by factor of {factor:.2f}")
    print(f"   → Hessian is too large")
    print(f"   → Newton steps are too small")
    print(f"   → Would cause slow convergence")

print("\n" + "=" * 80)
print("6. CONCLUSIONS")
print("=" * 80)

if abs(correct_edf - wrong_edf) > 1.0:
    print(f"""
✅ BUG CONFIRMED: Our edf estimate is wrong by {abs(correct_edf - wrong_edf):.1f}

**Root Cause**:
  We use total_rank = Σrank(S_i) = {wrong_edf}
  Should use edf = tr(A^{{-1}}·X'WX) = {correct_edf:.3f}

**Impact**:
  φ error of {abs(phi_wrong - phi_correct) / phi_correct * 100:.1f}%
  → bSb2 scaling error of {abs(phi_wrong / phi_correct - 1) * 100:.1f}%
  → Wrong Hessian curvature
  → Incorrect Newton steps
  → Convergence to wrong λ

**Fix Required**:
  Compute edf = tr(A^{{-1}}·X'WX) at each Newton iteration
  Use correct φ = RSS / (n - edf)
""")
else:
    print(f"""
⚠️  Edf difference is small ({abs(correct_edf - wrong_edf):.3f})

This may not fully explain our convergence issues.
Need to investigate other potential sources of error.
""")

print("\nNext step: Implement correct edf computation in src/reml.rs")
