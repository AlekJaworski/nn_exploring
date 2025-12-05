"""
Test Rust trace computation with debug output enabled
Compares against Python validation values
"""
import numpy as np
import pandas as pd
import os

# Set debug flag
os.environ['MGCV_GRAD_DEBUG'] = '1'

# Now import Rust module
import mgcv_rust

# Load matrices
X = pd.read_csv('/tmp/X_matrix.csv').values
S1_full = pd.read_csv('/tmp/S1_full.csv').values
S2_full = pd.read_csv('/tmp/S2_full.csv').values
y = pd.read_csv('/tmp/trace_step_data.csv')['y'].values

lambdas = np.array([2.0, 3.0])
n = len(y)
w = np.ones(n)

print("=" * 80)
print("Testing Rust trace computation with debug output")
print("=" * 80)
print(f"X shape: {X.shape}")
print(f"S1 shape: {S1_full.shape}")
print(f"S2 shape: {S2_full.shape}")
print(f"λ = {lambdas}")
print()
print("Expected trace values from Python validation:")
print("  trace1 = 1.340574 (at λ1=2.0)")
print("  trace2 = 1.437015 (at λ2=3.0)")
print()
print("-" * 80)
print("Rust debug output:")
print("-" * 80)

try:
    gradient = mgcv_rust.reml_gradient_multi_qr_py(
        y, X, w, lambdas, [S1_full, S2_full]
    )

    print()
    print("-" * 80)
    print(f"Gradient returned: {gradient}")
    print("=" * 80)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
