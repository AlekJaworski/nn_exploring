#!/usr/bin/env python3
"""
Comprehensive diagnostic to compare Newton method inputs with mgcv.

We'll check:
1. Penalty matrix values and scaling
2. Gradient computation at specific lambda values
3. Hessian computation at specific lambda values
4. Newton step sizes
5. Convergence behavior
"""

import numpy as np
import sys

print("=" * 80)
print("NEWTON METHOD DIAGNOSTICS")
print("=" * 80)

# Simple test case: n=500, 2 variables
np.random.seed(42)
n = 500
x = np.random.randn(n, 2)
y = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.random.randn(n) * 0.1

print(f"\nTest case: n={n}, p=2, k=10 (CR splines)")
print(f"Data range: x0=[{x[:,0].min():.2f}, {x[:,0].max():.2f}], x1=[{x[:,1].min():.2f}, {x[:,1].max():.2f}]")
print(f"Response: y=[{y.min():.2f}, {y.max():.2f}]")

# Save data for R comparison
np.savetxt('/tmp/diag_x.csv', x, delimiter=',')
np.savetxt('/tmp/diag_y.csv', y)

print("\n" + "=" * 80)
print("STEP 1: Extract penalty matrices from Rust")
print("=" * 80)

try:
    import mgcv_rust

    # We need to extract penalty matrices - let's see if we can access them
    # For now, let's just run the fit and see what happens
    gam = mgcv_rust.GAM()

    # Enable debug output
    import os
    os.environ['MGCV_GRAD_DEBUG'] = '1'
    os.environ['MGCV_PROFILE'] = '1'

    print("\nRunning Rust GAM fit with profiling enabled...")
    result = gam.fit_auto_optimized(x, y, k=[10, 10], method='REML', bs='cr')

    print(f"\nRust Results:")
    print(f"  Final lambda: {result['lambda']}")
    print(f"  Deviance: {result.get('deviance', 'N/A')}")

except ImportError as e:
    print(f"ERROR: Cannot import mgcv_rust: {e}")
    print("Need to build Rust code first!")
    sys.exit(1)
except Exception as e:
    print(f"ERROR during Rust fit: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("STEP 2: Compare with R's mgcv")
print("=" * 80)

r_script = '''
library(mgcv)

# Load data
x <- as.matrix(read.csv("/tmp/diag_x.csv", header=FALSE))
y <- as.numeric(read.csv("/tmp/diag_y.csv", header=FALSE)$V1)

df <- data.frame(x1=x[,1], x2=x[,2], y=y)

# Build smooths
sm1 <- smoothCon(s(x1, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]
sm2 <- smoothCon(s(x2, k=10, bs="cr"), df, absorb.cons=TRUE)[[1]]

cat("\\n=== Penalty Matrix Info ===\\n")
cat(sprintf("S1: %dx%d, max=%.6f, frobenius=%.6f, rank=%d\\n",
    nrow(sm1$S[[1]]), ncol(sm1$S[[1]]),
    max(abs(sm1$S[[1]])),
    sqrt(sum(sm1$S[[1]]^2)),
    qr(sm1$S[[1]])$rank))

cat(sprintf("S2: %dx%d, max=%.6f, frobenius=%.6f, rank=%d\\n",
    nrow(sm2$S[[1]]), ncol(sm2$S[[1]]),
    max(abs(sm2$S[[1]])),
    sqrt(sum(sm2$S[[1]]^2)),
    qr(sm2$S[[1]])$rank))

# Show first few elements
cat("\\nS1[1:3, 1:3]:\\n")
print(sm1$S[[1]][1:3, 1:3])

cat("\\nS2[1:3, 1:3]:\\n")
print(sm2$S[[1]][1:3, 1:3])

# Fit with trace
cat("\\n=== Fitting GAM with REML (trace enabled) ===\\n")
ctrl <- gam.control(trace=TRUE, epsilon=1e-7)
fit <- gam(y ~ s(x1, k=10, bs="cr") + s(x2, k=10, bs="cr"),
           data=df,
           method="REML",
           control=ctrl)

cat("\\n=== Final Results ===\\n")
cat(sprintf("Final lambda: [%.6f, %.6f]\\n", fit$sp[1], fit$sp[2]))
cat(sprintf("REML value: %.6f\\n", fit$gcv.ubre))
cat(sprintf("Converged: %s\\n", fit$converged))

# Extract iteration count from optimizer info
cat("\\nOptimization method: ", fit$optimizer, "\\n")
'''

print("\nRunning R script for comparison...")
import subprocess
try:
    result = subprocess.run(['R', '--version'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("R is available:", result.stdout.split('\n')[0])

        result = subprocess.run(['R', '--vanilla', '--slave', '-e', r_script],
                              capture_output=True, text=True, timeout=60)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Show last 30 lines of stderr (contains trace output)
            lines = result.stderr.split('\n')
            for line in lines[-30:]:
                if line.strip():
                    print(line)
    else:
        print("R not available - install with: apt-get install r-base r-base-dev")
        print("Then install mgcv: R -e 'install.packages(\"mgcv\", repos=\"https://cloud.r-project.org\")'")
except FileNotFoundError:
    print("R not found - need to install it first")
except Exception as e:
    print(f"Error running R: {e}")

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print("""
Key things to check:

1. PENALTY MATRICES:
   - Do the Frobenius norms match?
   - Do the matrix structures match?
   - Are the values in the same range?

2. GRADIENT COMPUTATION:
   - Check the debug output from Rust
   - Look for gradient values at each iteration
   - Compare with R's trace output

3. ITERATION COUNT:
   - R typically converges in 3-8 iterations
   - Are we taking 20-30+ iterations?

4. LAMBDA VALUES:
   - Do final lambdas match approximately?
   - Large discrepancies suggest different penalty scales

5. CONVERGENCE CRITERIA:
   - What gradient norm triggers convergence?
   - Are we using the same tolerance as R?
""")
