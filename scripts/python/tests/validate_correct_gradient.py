"""
Validate that Rust gradient matches numerical derivative of REML criterion
This is the TRUE test of correctness
"""
import numpy as np
import pandas as pd
import mgcv_rust

# Load matrices
X = pd.read_csv('/tmp/X_matrix.csv').values
S1_full = pd.read_csv('/tmp/S1_full.csv').values
S2_full = pd.read_csv('/tmp/S2_full.csv').values
y = pd.read_csv('/tmp/trace_step_data.csv')['y'].values

n, p = X.shape
w = np.ones(n)

def reml_criterion(log_lambda):
    """Compute REML criterion at given log(lambda)"""
    lambda1, lambda2 = np.exp(log_lambda)
    XtX = X.T @ X
    A = XtX + lambda1 * S1_full + lambda2 * S2_full
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (X.T @ y)
    fitted = X @ beta
    residuals = y - fitted
    rss = np.sum(residuals**2)
    edf_total = np.sum(np.diag(Ainv @ XtX))
    phi = rss / (n - edf_total)
    sign_A, logdet_A = np.linalg.slogdet(A)
    reml = (n - edf_total) * np.log(phi) + logdet_A
    return reml

def numerical_gradient(log_lambda, eps=1e-7):
    """Compute numerical gradient via finite differences"""
    grad = np.zeros(2)
    for i in range(2):
        log_lambda_plus = log_lambda.copy()
        log_lambda_plus[i] += eps
        log_lambda_minus = log_lambda.copy()
        log_lambda_minus[i] -= eps
        grad[i] = (reml_criterion(log_lambda_plus) - reml_criterion(log_lambda_minus)) / (2 * eps)
    return grad

print("=" * 80)
print("VALIDATION: Rust Gradient vs Numerical REML Derivative")
print("=" * 80)
print()

test_cases = [
    (0.1, 0.1),
    (1.0, 1.0),
    (2.0, 3.0),
    (10.0, 10.0),
    (100.0, 100.0),
]

all_passed = True

for lambda1, lambda2 in test_cases:
    lambdas = np.array([lambda1, lambda2])
    log_lambdas = np.log(lambdas)

    # Get Rust gradient
    grad_rust = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, [S1_full, S2_full])

    # Get numerical gradient
    grad_numeric = numerical_gradient(log_lambdas)

    # Compute error
    abs_error = np.abs(grad_rust - grad_numeric)
    rel_error = abs_error / (np.abs(grad_numeric) + 1e-10)

    print(f"λ = [{lambda1:.1f}, {lambda2:.1f}]")
    print(f"  Rust:     [{grad_rust[0]:+.10f}, {grad_rust[1]:+.10f}]")
    print(f"  Numeric:  [{grad_numeric[0]:+.10f}, {grad_numeric[1]:+.10f}]")
    print(f"  Abs err:  [{abs_error[0]:.2e}, {abs_error[1]:.2e}]")
    print(f"  Rel err:  [{rel_error[0]:.2e}, {rel_error[1]:.2e}]")

    # Check if within tolerance (accounting for numerical precision)
    if np.all(abs_error < 1e-4):
        print("  ✅ EXCELLENT (matches to numerical precision)")
    elif np.all(rel_error < 1e-4):
        print("  ✅ EXCELLENT (< 0.01% relative error)")
    elif np.all(rel_error < 1e-3):
        print("  ✓ Good (< 0.1% relative error)")
    else:
        print("  ❌ FAILED (gradient does not match numerical derivative)")
        all_passed = False

    print()

print("=" * 80)
if all_passed:
    print("✅ ALL TESTS PASSED")
    print("Rust gradient correctly computes ∂REML/∂log(λ)")
else:
    print("❌ SOME TESTS FAILED")
    print("Gradient implementation has bugs")
print("=" * 80)
