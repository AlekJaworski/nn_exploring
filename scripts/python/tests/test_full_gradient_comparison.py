"""
Compare full REML gradients between Rust and mgcv at various lambda values
"""
import numpy as np
import pandas as pd
import subprocess
import os

# Disable debug output for cleaner comparison
if 'MGCV_GRAD_DEBUG' in os.environ:
    del os.environ['MGCV_GRAD_DEBUG']

import mgcv_rust

# Load matrices
X = pd.read_csv('/tmp/X_matrix.csv').values
S1_full = pd.read_csv('/tmp/S1_full.csv').values
S2_full = pd.read_csv('/tmp/S2_full.csv').values
y = pd.read_csv('/tmp/trace_step_data.csv')['y'].values

n = len(y)
w = np.ones(n)

def get_mgcv_gradient(lambda1, lambda2):
    """Get gradient from mgcv at given lambda values"""
    r_code = f"""
    library(mgcv)
    df <- read.csv('/tmp/trace_step_data.csv')
    X <- as.matrix(read.csv('/tmp/X_matrix.csv'))
    S1_full <- as.matrix(read.csv('/tmp/S1_full.csv'))
    S2_full <- as.matrix(read.csv('/tmp/S2_full.csv'))

    lambda <- c({lambda1}, {lambda2})
    y <- df$y
    n <- nrow(X)
    p <- ncol(X)

    # Compute A = X'X + λ1·S1 + λ2·S2
    XtX <- t(X) %*% X
    A <- XtX + lambda[1] * S1_full + lambda[2] * S2_full
    Ainv <- solve(A)

    # Compute beta
    beta <- Ainv %*% (t(X) %*% y)

    # Residuals and phi
    fitted <- X %*% beta
    residuals <- y - fitted

    # edf for each smooth (trace of smoother matrix for that term)
    edf1 <- sum(diag(Ainv %*% (lambda[1] * S1_full)))
    edf2 <- sum(diag(Ainv %*% (lambda[2] * S2_full)))

    rss <- sum(residuals^2)
    phi <- rss / (n - edf1 - edf2 - 1)  # Use sum of edfs

    # Gradient components
    trace1 <- sum(diag(Ainv %*% (lambda[1] * S1_full)))
    trace2 <- sum(diag(Ainv %*% (lambda[2] * S2_full)))

    penalty1 <- lambda[1] * t(beta) %*% S1_full %*% beta
    penalty2 <- lambda[2] * t(beta) %*% S2_full %*% beta

    # Assuming rank 7 for CR splines (k=10 gives 10-2-1=7 rank)
    rank1 <- 7
    rank2 <- 7

    grad1 <- (trace1 - rank1 + penalty1/phi) / 2
    grad2 <- (trace2 - rank2 + penalty2/phi) / 2

    cat(sprintf("%.10f,%.10f\\n", grad1, grad2))
    """

    result = subprocess.run(
        ['Rscript', '-e', r_code],
        capture_output=True,
        text=True
    )

    for line in result.stdout.strip().split('\n'):
        if ',' in line and not line.startswith('['):
            parts = line.split(',')
            return np.array([float(parts[0]), float(parts[1])])

    raise ValueError(f"Could not parse R output: {result.stdout}")

# Test at multiple lambda values
test_cases = [
    (0.1, 0.1),
    (1.0, 1.0),
    (2.0, 3.0),
    (10.0, 10.0),
    (100.0, 100.0),
]

print("=" * 80)
print("REML Gradient Comparison: Rust vs mgcv")
print("=" * 80)
print()

for lambda1, lambda2 in test_cases:
    lambdas = np.array([lambda1, lambda2])

    # Get Rust gradient
    grad_rust = mgcv_rust.reml_gradient_multi_qr_py(
        y, X, w, lambdas, [S1_full, S2_full]
    )

    # Get mgcv gradient
    grad_mgcv = get_mgcv_gradient(lambda1, lambda2)

    # Compute error
    abs_error = np.abs(grad_rust - grad_mgcv)
    rel_error = abs_error / (np.abs(grad_mgcv) + 1e-10)

    print(f"λ = [{lambda1:.1f}, {lambda2:.1f}]")
    print(f"  Rust: [{grad_rust[0]:+.6f}, {grad_rust[1]:+.6f}]")
    print(f"  mgcv: [{grad_mgcv[0]:+.6f}, {grad_mgcv[1]:+.6f}]")
    print(f"  Abs error: [{abs_error[0]:.2e}, {abs_error[1]:.2e}]")
    print(f"  Rel error: [{rel_error[0]:.2%}, {rel_error[1]:.2%}]")

    if np.all(rel_error < 0.01):
        print("  ✅ EXCELLENT (< 1% error)")
    elif np.all(rel_error < 0.05):
        print("  ✓ Good (< 5% error)")
    elif np.all(rel_error < 0.10):
        print("  ~ Acceptable (< 10% error)")
    else:
        print("  ⚠ Needs improvement (> 10% error)")

    print()

print("=" * 80)
