"""
Derive the correct REML gradient including implicit dependencies

The REML criterion is:
  REML = (n - edf) * log(φ) + log|A|

where:
  - A = X'X + ∑λᵢ·Sᵢ
  - β = A^{-1}·X'y  (depends on λ implicitly)
  - rss = ||y - X·β||²  (depends on β)
  - edf = tr(A^{-1}·X'X)  (depends on A)
  - φ = rss / (n - edf)  (depends on rss and edf)

Taking ∂/∂ρᵢ where ρᵢ = log(λᵢ):

1. ∂log|A|/∂ρᵢ = tr(A^{-1}·∂A/∂ρᵢ) = tr(A^{-1}·λᵢ·Sᵢ) = traceᵢ

2. ∂[(n-edf)·log(φ)]/∂ρᵢ needs chain rule:

   Let L = (n-edf)·log(φ)

   ∂L/∂ρᵢ = -∂edf/∂ρᵢ · log(φ) + (n-edf) · 1/φ · ∂φ/∂ρᵢ

   Now: φ = rss / (n-edf)

   ∂φ/∂ρᵢ = 1/(n-edf) · ∂rss/∂ρᵢ - rss/(n-edf)² · (-∂edf/∂ρᵢ)
          = 1/(n-edf) · ∂rss/∂ρᵢ + rss/(n-edf)² · ∂edf/∂ρᵢ

   Substituting:
   ∂L/∂ρᵢ = -∂edf/∂ρᵢ · log(φ) + (n-edf)/φ · [1/(n-edf) · ∂rss/∂ρᵢ + rss/(n-edf)² · ∂edf/∂ρᵢ]
          = -∂edf/∂ρᵢ · log(φ) + 1/φ · ∂rss/∂ρᵢ + rss/[φ(n-edf)] · ∂edf/∂ρᵢ
          = ∂edf/∂ρᵢ · [-log(φ) + rss/(φ(n-edf))] + ∂rss/∂ρᵢ/φ

   But rss/(n-edf) = φ, so:
          = ∂edf/∂ρᵢ · [-log(φ) + 1] + ∂rss/∂ρᵢ/φ

Now we need ∂edf/∂ρᵢ and ∂rss/∂ρᵢ:

∂edf/∂ρᵢ = ∂/∂ρᵢ tr(A^{-1}·X'X)
         = tr(∂A^{-1}/∂ρᵢ · X'X)
         = -tr(A^{-1} · ∂A/∂ρᵢ · A^{-1} · X'X)  (using ∂A^{-1}/∂ρ = -A^{-1}·∂A/∂ρ·A^{-1})
         = -tr(A^{-1} · λᵢ·Sᵢ · A^{-1} · X'X)
         = -λᵢ · tr(Sᵢ · A^{-1} · X'X · A^{-1})  (cyclic property)

∂rss/∂ρᵢ = ∂/∂ρᵢ ||y - X·β||²
         = -2(y - X·β)' · X · ∂β/∂ρᵢ

But (y - X·β) = residuals, and X'·residuals = X'y - X'X·β = X'y - X'X·A^{-1}·X'y

Wait, actually we have A·β = X'y, so X'·residuals = X'y - X'X·β = X'y - X'X·A^{-1}·X'y

Hmm, let me use the fact that ∂β/∂ρᵢ comes from differentiating A·β = X'y:

∂A/∂ρᵢ · β + A · ∂β/∂ρᵢ = 0
∂β/∂ρᵢ = -A^{-1} · ∂A/∂ρᵢ · β
       = -A^{-1} · λᵢ·Sᵢ · β

So:
∂rss/∂ρᵢ = -2(y - X·β)' · X · ∂β/∂ρᵢ
         = -2·residuals' · X · (-A^{-1} · λᵢ·Sᵢ · β)
         = 2·residuals' · X · A^{-1} · λᵢ·Sᵢ · β

But X'·residuals = 0 at the solution! (from normal equations: X'y = X'X·β + penalties·β ≈ X'·fitted)

Actually, with penalized regression: X'W·residuals = -∑λᵢ·Sᵢ·β

So ∂rss/∂ρᵢ involves X'·residuals which is NOT zero in penalized case.

Let me compute this numerically to verify...
"""
import numpy as np
import pandas as pd

# Load matrices
X = pd.read_csv('/tmp/X_matrix.csv').values
S1_full = pd.read_csv('/tmp/S1_full.csv').values
S2_full = pd.read_csv('/tmp/S2_full.csv').values
y = pd.read_csv('/tmp/trace_step_data.csv')['y'].values

lambda1, lambda2 = 2.0, 3.0
n, p = X.shape

# Compute current state
XtX = X.T @ X
A = XtX + lambda1 * S1_full + lambda2 * S2_full
Ainv = np.linalg.inv(A)
beta = Ainv @ (X.T @ y)
fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)
edf_total = np.sum(np.diag(Ainv @ XtX))
phi = rss / (n - edf_total)

print("=" * 80)
print("COMPUTING IMPLICIT DERIVATIVES")
print("=" * 80)
print()

# Compute ∂β/∂ρᵢ = -A^{-1}·λᵢ·Sᵢ·β
dbeta_drho1 = -Ainv @ (lambda1 * S1_full) @ beta
dbeta_drho2 = -Ainv @ (lambda2 * S2_full) @ beta

print("∂β/∂ρ₁ (first 3 elements):", dbeta_drho1[:3])
print("∂β/∂ρ₂ (first 3 elements):", dbeta_drho2[:3])
print()

# Compute ∂rss/∂ρᵢ = -2·residuals'·X·∂β/∂ρᵢ
drss_drho1 = -2 * residuals @ X @ dbeta_drho1
drss_drho2 = -2 * residuals @ X @ dbeta_drho2

print(f"∂rss/∂ρ₁: {drss_drho1:.10f}")
print(f"∂rss/∂ρ₂: {drss_drho2:.10f}")
print()

# Compute ∂edf/∂ρᵢ = -tr(A^{-1}·λᵢ·Sᵢ·A^{-1}·X'X)
dedf_drho1 = -np.trace(Ainv @ (lambda1 * S1_full) @ Ainv @ XtX)
dedf_drho2 = -np.trace(Ainv @ (lambda2 * S2_full) @ Ainv @ XtX)

print(f"∂edf/∂ρ₁: {dedf_drho1:.10f}")
print(f"∂edf/∂ρ₂: {dedf_drho2:.10f}")
print()

# Now compute full gradient
trace1 = np.sum(np.diag(Ainv @ (lambda1 * S1_full)))
trace2 = np.sum(np.diag(Ainv @ (lambda2 * S2_full)))

# d(log|A|)/dρᵢ
dlogdetA_drho1 = trace1
dlogdetA_drho2 = trace2

# d[(n-edf)·log(φ)]/dρᵢ
dL_drho1 = dedf_drho1 * (-np.log(phi) + 1) + drss_drho1 / phi
dL_drho2 = dedf_drho2 * (-np.log(phi) + 1) + drss_drho2 / phi

# Total gradient
grad1_correct = dlogdetA_drho1 + dL_drho1
grad2_correct = dlogdetA_drho2 + dL_drho2

print("=" * 80)
print("GRADIENT COMPONENTS")
print("=" * 80)
print()

print(f"Component 1: d(log|A|)/dρ")
print(f"  grad1: {dlogdetA_drho1:.10f}")
print(f"  grad2: {dlogdetA_drho2:.10f}")
print()

print(f"Component 2: d[(n-edf)·log(φ)]/dρ")
print(f"  grad1: {dL_drho1:.10f}")
print(f"  grad2: {dL_drho2:.10f}")
print()

print(f"Total gradient (including implicit terms):")
print(f"  grad1: {grad1_correct:.10f}")
print(f"  grad2: {grad2_correct:.10f}")
print()

# Compare with numerical
print("=" * 80)
print("COMPARISON WITH NUMERICAL GRADIENT")
print("=" * 80)
print()

def reml_criterion(log_lambda):
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

log_lambda = np.log([lambda1, lambda2])
eps = 1e-7

# Numerical gradient
log_lambda_plus = log_lambda.copy()
log_lambda_plus[0] += eps
log_lambda_minus = log_lambda.copy()
log_lambda_minus[0] -= eps
grad1_numeric = (reml_criterion(log_lambda_plus) - reml_criterion(log_lambda_minus)) / (2 * eps)

log_lambda_plus = log_lambda.copy()
log_lambda_plus[1] += eps
log_lambda_minus = log_lambda.copy()
log_lambda_minus[1] -= eps
grad2_numeric = (reml_criterion(log_lambda_plus) - reml_criterion(log_lambda_minus)) / (2 * eps)

print(f"Numerical gradient:")
print(f"  grad1: {grad1_numeric:.10f}")
print(f"  grad2: {grad2_numeric:.10f}")
print()

print(f"Analytical gradient (with implicit terms):")
print(f"  grad1: {grad1_correct:.10f}")
print(f"  grad2: {grad2_correct:.10f}")
print()

print(f"Difference:")
print(f"  Δgrad1: {grad1_numeric - grad1_correct:.10e}")
print(f"  Δgrad2: {grad2_numeric - grad2_correct:.10e}")
print()

if np.allclose([grad1_numeric, grad2_numeric], [grad1_correct, grad2_correct], rtol=1e-5):
    print("✅ ANALYTICAL GRADIENT (with implicit terms) MATCHES NUMERICAL!")
else:
    print("❌ Still differs - need to check derivation")
