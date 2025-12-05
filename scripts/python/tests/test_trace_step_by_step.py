#!/usr/bin/env python3
"""
Step-by-step validation: where does our trace diverge from mgcv?

We'll compute the trace at each intermediate step and compare.
"""

import numpy as np
import subprocess
import sys

print("=" * 80)
print("STEP-BY-STEP TRACE VALIDATION")
print("=" * 80)

# Test at a simple λ
test_lambda = [2.0, 3.0]

# First get mgcv's expected values
r_script = f'''
library(mgcv)

set.seed(42)
n <- 100
x <- matrix(rnorm(n*2), n, 2)
y <- sin(x[,1]) + 0.5*x[,2]^2 + rnorm(n)*0.1
df <- data.frame(x1=x[,1], x2=x[,2], y=y)

lambda <- c({test_lambda[0]}, {test_lambda[1]})
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda)

# Get coefficient vector
beta <- coef(fit)

# Get penalty matrices from smooth objects
# Each smooth has its penalty in fit$smooth[[i]]$S[[1]]
S1 <- fit$smooth[[1]]$S[[1]]
S2 <- fit$smooth[[2]]$S[[1]]

# But these are in the smooth's own basis (k × k), not full coefficient space!
# Need to map them to full coefficient vector

# mgcv stores the "map" in smooth$first.para and smooth$last.para
start1 <- fit$smooth[[1]]$first.para
end1 <- fit$smooth[[1]]$last.para
start2 <- fit$smooth[[2]]$first.para
end2 <- fit$smooth[[2]]$last.para

cat("SMOOTH1: params", start1, "to", end1, "\\n")
cat("SMOOTH2: params", start2, "to", end2, "\\n")
cat("BETA_LENGTH:", length(beta), "\\n")

# Build design matrix
X <- predict(fit, type="lpmatrix")
cat("X_SHAPE:", dim(X), "\\n")

# Build full penalty matrix for smooth 1
p <- length(beta)
S1_full <- matrix(0, p, p)
S1_full[start1:end1, start1:end1] <- S1

# Build full penalty matrix for smooth 2
S2_full <- matrix(0, p, p)
S2_full[start2:end2, start2:end2] <- S2

# Compute A = X'X + λ1·S1 + λ2·S2
XtX <- t(X) %*% X
A <- XtX + lambda[1] * S1_full + lambda[2] * S2_full

# Compute A^{-1}
Ainv <- solve(A)

# Compute trace terms: tr(A^{-1}·λi·Si)
trace1_direct <- sum(diag(Ainv %*% (lambda[1] * S1_full)))
trace2_direct <- sum(diag(Ainv %*% (lambda[2] * S2_full)))

cat("\\nDIRECT TRACE COMPUTATION (mgcv internals):\\n")
cat("tr(A^-1 · λ1·S1) =", trace1_direct, "\\n")
cat("tr(A^-1 · λ2·S2) =", trace2_direct, "\\n")

# Also compute via alternative: tr(A^{-1}·S) = tr(S·A^{-1})
trace1_alt <- sum(diag(S1_full %*% Ainv)) * lambda[1]
trace2_alt <- sum(diag(S2_full %*% Ainv)) * lambda[2]
cat("\\nALTERNATIVE (tr(S·A^-1)·λ):\\n")
cat("tr(S1·A^-1) · λ1 =", trace1_alt, "\\n")
cat("tr(S2·A^-1) · λ2 =", trace2_alt, "\\n")

# Now let's verify via QR decomposition like our code does
# Build augmented matrix Z
Z <- rbind(X, sqrt(lambda[1]) * chol(S1_full), sqrt(lambda[2]) * chol(S2_full))
cat("\\nZ_SHAPE:", dim(Z), "\\n")

# QR decomposition
qr_result <- qr(Z)
R <- qr.R(qr_result)
cat("R_SHAPE:", dim(R), "\\n")

# P = R^{-1}
P <- solve(R)

# Verify: P'P should equal A^{-1}
PtP <- t(P) %*% P
error_norm <- max(abs(PtP - Ainv))
cat("\\nVERIFICATION: ||P'P - A^-1|| =", error_norm, "\\n")

if (error_norm < 1e-10) {{
  cat("✓ P'P = A^-1 (verified!)\\n")
}} else {{
  cat("✗ P'P ≠ A^-1 (ERROR!)\\n")
}}

# Now compute trace via QR method: tr(P'·S·P)
# For S1
PtS1P <- t(P) %*% S1_full %*% P
trace1_qr <- sum(diag(PtS1P)) * lambda[1]

# For S2
PtS2P <- t(P) %*% S2_full %*% P
trace2_qr <- sum(diag(PtS2P)) * lambda[2]

cat("\\nTRACE VIA QR METHOD (tr(P'·S·P)·λ):\\n")
cat("tr(P'·S1·P) · λ1 =", trace1_qr, "\\n")
cat("tr(P'·S2·P) · λ2 =", trace2_qr, "\\n")

cat("\\nCOMPARISON:\\n")
cat("Method 1 (direct): [", trace1_direct, ",", trace2_direct, "]\\n")
cat("Method 2 (QR):     [", trace1_qr, ",", trace2_qr, "]\\n")
cat("Difference:        [", trace1_direct - trace1_qr, ",", trace2_direct - trace2_qr, "]\\n")

# Save data and matrices
write.csv(df, "/tmp/trace_step_data.csv", row.names=FALSE)
write.csv(X, "/tmp/X_matrix.csv", row.names=FALSE)
write.csv(S1_full, "/tmp/S1_full.csv", row.names=FALSE)
write.csv(S2_full, "/tmp/S2_full.csv", row.names=FALSE)

# Finite difference gradient for reference
delta <- 1e-6
reml0 <- fit$gcv.ubre

fit1 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda * exp(c(delta, 0)))
grad1 <- (fit1$gcv.ubre - reml0) / delta

fit2 <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df, method="REML", sp=lambda * exp(c(0, delta)))
grad2 <- (fit2$gcv.ubre - reml0) / delta

cat("\\nTRUE GRADIENT (FD): [", grad1, ",", grad2, "]\\n")

# From gradient formula: grad = (trace - rank + penalty/phi) / 2
# So: expected_trace = 2*grad + rank - penalty/phi

phi <- fit$scale
rank1 <- fit$smooth[[1]]$rank
rank2 <- fit$smooth[[2]]$rank

# Get penalty terms
beta_vec <- as.vector(beta)
penalty1 <- lambda[1] * sum(beta_vec * (S1_full %*% beta_vec)) / phi
penalty2 <- lambda[2] * sum(beta_vec * (S2_full %*% beta_vec)) / phi

expected_trace1 <- 2*grad1 + rank1 - penalty1
expected_trace2 <- 2*grad2 + rank2 - penalty2

cat("\\nEXPECTED TRACE (from gradient formula):\\n")
cat("trace1 should be:", expected_trace1, "\\n")
cat("trace2 should be:", expected_trace2, "\\n")
cat("\\nACTUAL TRACE (direct computation):\\n")
cat("trace1 is:", trace1_direct, "\\n")
cat("trace2 is:", trace2_direct, "\\n")
'''

print("\n1. Computing mgcv trace values...")
result = subprocess.run(['Rscript', '-e', r_script],
                       capture_output=True, text=True, timeout=30)

print(result.stdout)

if result.returncode != 0:
    print("ERROR:", result.stderr, file=sys.stderr)

print("\n" + "=" * 80)
print("2. Computing OUR trace values at same λ...")
print("=" * 80)

# Parse expected values
expected_trace = [None, None]
for line in result.stdout.split('\n'):
    if 'tr(A^-1 · λ1·S1) =' in line:
        expected_trace[0] = float(line.split('=')[1].strip())
    if 'tr(A^-1 · λ2·S2) =' in line:
        expected_trace[1] = float(line.split('=')[1].strip())

print(f"\nExpected trace (mgcv): {expected_trace}")

# Now run our code and extract trace values
python_code = '''
import mgcv_rust
import pandas as pd
import os

os.environ["MGCV_GRAD_DEBUG"] = "1"

data = pd.read_csv('/tmp/trace_step_data.csv')
x = data[['x1', 'x2']].values
y = data['y'].values

gam = mgcv_rust.GAM()
result = gam.fit_auto(x, y, k=[10, 10], method='REML', bs='cr', max_iter=1)
'''

result2 = subprocess.run([
    'bash', '-c',
    f'cd /home/user/nn_exploring && source .venv/bin/activate && python3 -c "{python_code}" 2>&1'
], capture_output=True, text=True, timeout=30)

# Extract our trace values
our_trace = [None, None]
for line in result2.stdout.split('\n'):
    if 'smooth=0:' in line and 'trace=' in line:
        parts = line.split('trace=')
        if len(parts) > 1:
            try:
                our_trace[0] = float(parts[1].split(',')[0])
            except:
                pass
    if 'smooth=1:' in line and 'trace=' in line:
        parts = line.split('trace=')
        if len(parts) > 1:
            try:
                our_trace[1] = float(parts[1].split(',')[0])
            except:
                pass

print(f"Our trace (first iter): {our_trace}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

if expected_trace[0] and our_trace[0]:
    ratio1 = our_trace[0] / expected_trace[0]
    ratio2 = our_trace[1] / expected_trace[1]
    print(f"\nTrace comparison:")
    print(f"  Smooth 1: mgcv={expected_trace[0]:.4f}, ours={our_trace[0]:.4f}, ratio={ratio1:.3f}")
    print(f"  Smooth 2: mgcv={expected_trace[1]:.4f}, ours={our_trace[1]:.4f}, ratio={ratio2:.3f}")

    if abs(ratio1 - 1.0) < 0.01 and abs(ratio2 - 1.0) < 0.01:
        print(f"\n✅ TRACES MATCH! (< 1% error)")
    else:
        print(f"\n❌ TRACES DIFFER!")
        print(f"\nOur trace is {ratio1:.1f}x and {ratio2:.1f}x of what it should be.")
        print(f"\nThis confirms the trace computation has a systematic error.")

print(f"\n" + "=" * 80)
print("NEXT: Check each computation step")
print("=" * 80)
print("""
The trace computation has these steps:
1. sqrt_penalties[i] = sqrt(S_i) via eigenvalue decomposition
2. p_matrix = R^{-1} from QR decomposition of Z
3. p_t_l = p_matrix.t() · sqrt_penalties[i]
4. trace_term = sum(p_t_l²)
5. trace = λ_i · trace_term

We need to check:
- Is sqrt(S) computed correctly? (check L·L' = S)
- Is P computed correctly? (check P'P = A^{-1})
- Is the sum computed correctly?

Let's validate each step...
""")
