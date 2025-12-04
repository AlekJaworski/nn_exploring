"""
Diagnostic script to understand why λ converges to near-zero values
"""
import numpy as np
import pandas as pd
import mgcv_rust
from scipy.optimize import minimize

# Load test data
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
penalties = []
for i in range(5):
    S = pd.read_csv(f'/tmp/bench_S{i+1}.csv').values
    penalties.append(S)

n, p = X.shape
d = len(penalties)
w = np.ones(n)

print("="*80)
print("DIAGNOSTIC: Understanding λ Convergence")
print("="*80)
print(f"Problem: n={n}, p={p}, d={d} smooths")
print()

# ============================================================================
# Test 1: What does Newton-PIRLS converge to?
# ============================================================================
print("="*80)
print("TEST 1: Newton-PIRLS Solution")
print("="*80)

log_lambda0 = np.log([1.0] * d)
log_lambda_opt, lambda_opt, reml_value, iterations, converged, message = \
    mgcv_rust.newton_pirls_py(y, X, w, log_lambda0, penalties,
                              max_iter=100, grad_tol=1e-6, verbose=False)

print(f"Final λ: {lambda_opt}")
print(f"Final log(λ): {log_lambda_opt}")
print(f"REML: {reml_value:.6f}")
print(f"Iterations: {iterations}")
print()

# Check gradient at solution
grad_at_solution = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambda_opt.tolist(), penalties)
print(f"Gradient at solution: {grad_at_solution}")
print(f"Gradient norm: {np.linalg.norm(grad_at_solution):.6e}")
print()

# ============================================================================
# Test 2: What does L-BFGS-B converge to?
# ============================================================================
print("="*80)
print("TEST 2: L-BFGS-B Solution (for comparison)")
print("="*80)

def reml_criterion(log_lambda):
    """REML criterion to minimize"""
    lambdas = np.exp(log_lambda)
    XtX = X.T @ X
    A = XtX.copy()
    for i, lam in enumerate(lambdas):
        A += lam * penalties[i]
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (X.T @ y)
    fitted = X @ beta
    residuals = y - fitted
    rss = np.sum(residuals**2)
    edf_total = np.trace(Ainv @ XtX)
    phi = rss / (n - edf_total)
    sign_A, logdet_A = np.linalg.slogdet(A)
    reml = (n - edf_total) * np.log(phi) + logdet_A
    return reml

def reml_gradient_rust(log_lambda):
    """Gradient using Rust implementation"""
    lambdas = np.exp(log_lambda)
    grad = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, penalties)
    return grad

result_lbfgsb = minimize(
    reml_criterion,
    log_lambda0,
    method='L-BFGS-B',
    jac=reml_gradient_rust,
    options={'maxiter': 100, 'disp': False}
)

print(f"Final λ: {np.exp(result_lbfgsb.x)}")
print(f"Final log(λ): {result_lbfgsb.x}")
print(f"REML: {result_lbfgsb.fun:.6f}")
print(f"Iterations: {result_lbfgsb.nit}")
print()

# ============================================================================
# Test 3: REML landscape - is there a better solution?
# ============================================================================
print("="*80)
print("TEST 3: Exploring REML Landscape")
print("="*80)

# Test REML at various log(λ) values
test_points = [
    ("Near zero (current)", log_lambda_opt),
    ("log(λ) = 0 (λ=1)", np.zeros(d)),
    ("log(λ) = -5 (λ≈0.007)", -5 * np.ones(d)),
    ("log(λ) = -10 (λ≈4.5e-5)", -10 * np.ones(d)),
    ("log(λ) = 5 (λ≈148)", 5 * np.ones(d)),
]

print(f"{'Test Point':<30} {'REML Value':<15} {'Gradient Norm':<15}")
print("-" * 60)

for name, log_lam in test_points:
    lam = np.exp(log_lam)
    try:
        reml = reml_criterion(log_lam)
        grad = reml_gradient_rust(log_lam)
        grad_norm = np.linalg.norm(grad)
        print(f"{name:<30} {reml:<15.6f} {grad_norm:<15.6e}")
    except Exception as e:
        print(f"{name:<30} {'ERROR':<15} {str(e)}")

print()

# ============================================================================
# Test 4: What if we start from different initial points?
# ============================================================================
print("="*80)
print("TEST 4: Sensitivity to Initial Point")
print("="*80)

initial_points = [
    ("log(λ) = 0", np.zeros(d)),
    ("log(λ) = -2", -2 * np.ones(d)),
    ("log(λ) = -5", -5 * np.ones(d)),
    ("log(λ) = 2", 2 * np.ones(d)),
    ("Random", np.random.randn(d)),
]

print(f"{'Initial Point':<20} {'Final λ[0]':<15} {'REML':<15} {'Iterations':<12}")
print("-" * 65)

for name, log_lam_init in initial_points:
    try:
        log_lam_opt, lam_opt, reml_val, iters, conv, msg = \
            mgcv_rust.newton_pirls_py(y, X, w, log_lam_init, penalties,
                                      max_iter=100, grad_tol=1e-6, verbose=False)
        print(f"{name:<20} {lam_opt[0]:<15.6e} {reml_val:<15.6f} {iters:<12}")
    except Exception as e:
        print(f"{name:<20} {'ERROR':<15} {str(e)}")

print()

# ============================================================================
# Test 5: What penalty values do we have?
# ============================================================================
print("="*80)
print("TEST 5: Examining Penalty Matrices")
print("="*80)

for i, S in enumerate(penalties):
    rank = np.linalg.matrix_rank(S)
    evals = np.linalg.eigvalsh(S)
    max_eval = np.max(evals)
    min_nonzero_eval = np.min(evals[evals > 1e-10])

    print(f"Penalty {i+1}:")
    print(f"  Shape: {S.shape}")
    print(f"  Rank: {rank}")
    print(f"  Max eigenvalue: {max_eval:.6e}")
    print(f"  Min nonzero eigenvalue: {min_nonzero_eval:.6e}")
    print(f"  Condition number: {max_eval / min_nonzero_eval:.6e}")
    print()

# ============================================================================
# Test 6: What is the effective degrees of freedom?
# ============================================================================
print("="*80)
print("TEST 6: EDF Analysis at Different λ")
print("="*80)

def compute_edf_rss(lambdas):
    """Compute EDF and RSS for given λ"""
    XtX = X.T @ X
    A = XtX.copy()
    for i, lam in enumerate(lambdas):
        A += lam * penalties[i]
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (X.T @ y)
    fitted = X @ beta
    residuals = y - fitted
    rss = np.sum(residuals**2)
    edf_total = np.trace(Ainv @ XtX)
    return edf_total, rss

test_lambda_scales = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]

print(f"{'λ (uniform)':<15} {'EDF':<15} {'RSS':<15} {'REML':<15}")
print("-" * 60)

for lam_scale in test_lambda_scales:
    lams = lam_scale * np.ones(d)
    try:
        edf, rss = compute_edf_rss(lams)
        reml = reml_criterion(np.log(lams))
        print(f"{lam_scale:<15.6e} {edf:<15.6f} {rss:<15.6f} {reml:<15.6f}")
    except Exception as e:
        print(f"{lam_scale:<15.6e} {'ERROR':<15}")

print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print()
print("Key findings:")
print(f"1. Newton-PIRLS converges to λ ≈ {lambda_opt[0]:.2e}")
print(f"2. L-BFGS-B converges to λ ≈ {np.exp(result_lbfgsb.x[0]):.2e}")
print(f"3. Gradient norm at solution: {np.linalg.norm(grad_at_solution):.2e}")
print()
print("Questions to answer:")
print("- Are both methods converging to the same solution?")
print("- Is the REML minimum actually at near-zero λ?")
print("- Do we need bounds on λ to prevent it from going too small?")
print("- What λ values does mgcv find for this data?")
