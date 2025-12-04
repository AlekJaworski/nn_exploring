"""
Detailed comparison of Hessian terms between Rust and Python
"""
import numpy as np
import pandas as pd
import mgcv_rust
import os

os.environ['MGCV_HESS_DEBUG'] = '1'

# Load test data
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
S1 = pd.read_csv('/tmp/bench_S1.csv').values
S2 = pd.read_csv('/tmp/bench_S2.csv').values

n, p = X.shape
w = np.ones(n)

lambdas = np.array([1.0, 2.0])
penalties = [S1, S2]

print("Testing H[0,0] (diagonal element)...")
print()

# Compute using Rust
print("="*80)
print("RUST COMPUTATION:")
print("="*80)
hessian_rust = mgcv_rust.reml_hessian_multi_qr_py(y, X, w, lambdas.tolist(), penalties)
print()

# Compute using Python with detailed breakdown
print("="*80)
print("PYTHON COMPUTATION:")
print("="*80)

XtX = X.T @ X
A = XtX + lambdas[0] * S1 + lambdas[1] * S2
Ainv = np.linalg.inv(A)
beta = Ainv @ (X.T @ y)

fitted = X @ beta
residuals = y - fitted
rss = np.sum(residuals**2)

ainv_xtx = Ainv @ XtX
edf_total = np.trace(ainv_xtx)
phi = rss / (n - edf_total)
log_phi = np.log(phi)
ainv_xtx_ainv = ainv_xtx @ Ainv

print(f"phi = {phi:.6e}")
print(f"log_phi = {log_phi:.6f}")
print(f"edf_total = {edf_total:.6f}")
print()

# H[0,0] detailed computation
i, j = 0, 0
lambda_i, lambda_j = lambdas[i], lambdas[j]
Si, Sj = S1, S1

# Term 1
term1a = -lambda_i * lambda_j * np.trace(Ainv @ Sj @ Ainv @ Si)
term1b = lambda_i * np.trace(Ainv @ Si)  # diagonal correction
term1_total = term1a + term1b

print(f"TERM 1 (log|A|):")
print(f"  1a (main): {term1a:.6e}")
print(f"  1b (diag correction): {term1b:.6e}")
print(f"  Total: {term1_total:.6e}")
print()

# Term 2
d2edf_part1 = lambda_i * lambda_j * (
    np.trace(Ainv @ Sj @ Ainv @ Si @ ainv_xtx) +
    np.trace(Ainv @ Si @ Ainv @ Sj @ ainv_xtx)
)
d2edf_part2 = -lambda_i * np.trace(Ainv @ Si @ ainv_xtx)  # diagonal correction
d2edf = d2edf_part1 + d2edf_part2
term2a = d2edf * (-log_phi + 1.0)

# For term2b we need dedf_drho
dedf_drho_i = -lambda_i * np.sum(ainv_xtx_ainv * Si.T)
dbeta_drho_i = -Ainv @ (lambda_i * Si @ beta)
drss_drho_i = -2 * residuals @ X @ dbeta_drho_i

dphi_drho_i = (drss_drho_i + phi * dedf_drho_i) / (n - edf_total)
dlogphi_drho_i = -dphi_drho_i / phi
term2b = dedf_drho_i * dlogphi_drho_i

term2_total = term2a + term2b

print(f"TERM 2 (edf):")
print(f"  2a (d2edf part1): {d2edf_part1:.6e}")
print(f"  2a (d2edf part2, diag corr): {d2edf_part2:.6e}")
print(f"  2a (d2edf total): {d2edf:.6e}")
print(f"  2a (d2edf * factor): {term2a:.6e}")
print(f"  2b (dedf * dlogphi): {term2b:.6e}")
print(f"  Total: {term2_total:.6e}")
print()

# Term 3
d2rss_part1 = 2 * (X @ dbeta_drho_i) @ (X @ dbeta_drho_i)

d2beta = (Ainv @ (lambda_i * Si) @ Ainv @ (lambda_i * Si @ beta) -
          Ainv @ (lambda_i * Si @ dbeta_drho_i))
d2beta = d2beta - dbeta_drho_i  # diagonal correction

d2rss_part2 = -2 * residuals @ X @ d2beta
d2rss = d2rss_part1 + d2rss_part2
term3a = d2rss / phi
term3b = -drss_drho_i * dphi_drho_i / phi**2
term3_total = term3a + term3b

print(f"TERM 3 (rss):")
print(f"  3a (d2rss part1): {d2rss_part1:.6e}")
print(f"  3a (d2rss part2): {d2rss_part2:.6e}")
print(f"  3a (d2rss total): {d2rss:.6e}")
print(f"  3a (d2rss / phi): {term3a:.6e}")
print(f"  3b (-drss * dphi / phi²): {term3b:.6e}")
print(f"  Total: {term3_total:.6e}")
print()

h_python = term1_total + term2_total + term3_total
print(f"TOTAL H[0,0] (Python): {h_python:.6e}")
print(f"TOTAL H[0,0] (Rust):   {hessian_rust[0,0]:.6e}")
print(f"Difference:            {hessian_rust[0,0] - h_python:.6e}")
print(f"Relative error:        {abs(hessian_rust[0,0] - h_python) / abs(h_python):.2%}")
