"""
Test trust region with better initialization
"""
import numpy as np
import pandas as pd
import mgcv_rust

# Load the fresh data
df = pd.read_csv('/tmp/fresh_data.csv')
X = pd.read_csv('/tmp/fresh_X.csv').values
y = df['y'].values
penalties = []
for i in range(5):
    S = pd.read_csv(f'/tmp/fresh_S{i+1}.csv').values
    penalties.append(S)

n, p = X.shape
w = np.ones(n)

# mgcv's solution
mgcv_lambda = np.array([0.2705535, 9038.71, 150.8265, 400.144, 13747035])

print("="*80)
print("TEST: Better initialization")
print("="*80)
print(f"mgcv λ: {mgcv_lambda}")
print()

# Try starting from mgcv's solution / 10
initial_lambda = mgcv_lambda / 10.0
initial_log_lambda = np.log(initial_lambda)

print(f"Starting λ: {initial_lambda}")
print(f"Starting log(λ): {initial_log_lambda}")
print()

log_lambda_opt, lambda_opt, reml_value, iterations, converged, message = \
    mgcv_rust.newton_pirls_py(y, X, w, initial_log_lambda, penalties,
                              max_iter=50, grad_tol=1e-6, verbose=True)

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Final λ: {lambda_opt}")
print(f"mgcv λ:  {mgcv_lambda}")
print(f"REML: {reml_value:.6f}")
print(f"Iterations: {iterations}")
print(f"Converged: {converged}")
print(f"Message: {message}")
