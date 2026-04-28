#!/usr/bin/env python3
"""
Test eigenvalue modification fix.
Compare mgcv_rust with and without the fix.
"""
import numpy as np
import mgcv_rust

np.random.seed(42)

print("="*70)
print("TESTING EIGENVALUE MODIFICATION FIX")
print("="*70)

# Setup 4D test case
n = 1000
d = 4
k = 10

X = np.random.uniform(0, 1, size=(n, d))
y = (np.sin(2 * np.pi * X[:, 0])
     + 0.5 * np.cos(3 * np.pi * X[:, 1])
     + 0.3 * (X[:, 2] ** 2)
     + 0.2 * np.exp(-5 * (X[:, 3] - 0.5) ** 2)
     + np.random.normal(0, 0.2, n))

print(f"\nData: n={n}, d={d}, k={k}")

# Test with eigenvalue modification (current build)
print("\n" + "="*70)
print("MGCV_RUST WITH EIGENVALUE MODIFICATION")
print("="*70)

gam = mgcv_rust.GAM()
result = gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr', max_iter=100)
rust_lambda = result['lambda']

print(f"\nLambdas: [{', '.join([f'{l:.2f}' for l in rust_lambda])}]")

# Compare with target (from previous analysis)
mgcv_lambda = [5.08, 5.79, 9043.05, 660.41]
mgcv_reml = -119.094291

print(f"\nComparison with mgcv target:")
print(f"{'Dim':>4} {'mgcv':>10} {'rust':>10} {'diff %':>10}")
for i in range(d):
    diff_pct = abs(rust_lambda[i] - mgcv_lambda[i]) / mgcv_lambda[i] * 100
    print(f"{i:4d} {mgcv_lambda[i]:10.2f} {rust_lambda[i]:10.2f} {diff_pct:10.1f}%")

# Check fit quality
y_pred = gam.predict(X)
y_true = (np.sin(2 * np.pi * X[:, 0])
          + 0.5 * np.cos(3 * np.pi * X[:, 1])
          + 0.3 * (X[:, 2] ** 2)
          + 0.2 * np.exp(-5 * (X[:, 3] - 0.5) ** 2))

r2 = 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)
print(f"\nR² vs true function: {r2:.6f}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Check if we're closer to mgcv
old_rust_lambda = [10.40, 8.63, 198.63, 180.83]  # From before fix

print(f"\nImprovement over previous mgcv_rust:")
for i in range(d):
    old_diff = abs(old_rust_lambda[i] - mgcv_lambda[i]) / mgcv_lambda[i] * 100
    new_diff = abs(rust_lambda[i] - mgcv_lambda[i]) / mgcv_lambda[i] * 100
    improvement = old_diff - new_diff
    status = "✓ BETTER" if improvement > 0 else "✗ WORSE" if improvement < -5 else "~ SAME"
    print(f"  Dim {i}: was {old_diff:.1f}% off, now {new_diff:.1f}% off ({status})")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Check if dim 2 and 3 improved (these were the problematic ones)
dim2_improved = abs(rust_lambda[2] - mgcv_lambda[2]) < abs(old_rust_lambda[2] - mgcv_lambda[2])
dim3_improved = abs(rust_lambda[3] - mgcv_lambda[3]) < abs(old_rust_lambda[3] - mgcv_lambda[3])

if dim2_improved or dim3_improved:
    print("\n✓ SUCCESS! Eigenvalue modification is helping:")
    if dim2_improved:
        print(f"  - Dim 2: Got closer to target (was {old_rust_lambda[2]:.1f}, now {rust_lambda[2]:.1f}, target {mgcv_lambda[2]:.1f})")
    if dim3_improved:
        print(f"  - Dim 3: Got closer to target (was {old_rust_lambda[3]:.1f}, now {rust_lambda[3]:.1f}, target {mgcv_lambda[3]:.1f})")
else:
    print("\n✗ No significant improvement yet. May need:")
    print("  - Additional fixes (quadratic error check)")
    print("  - Multiple starting points")
    print("  - Better step acceptance criteria")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
If this shows improvement, we should also implement:
1. Quadratic approximation error checking (mgcv: qerror < 0.8)
2. Better steepest descent fallback
3. Multiple starting point strategy

If no improvement, the eigenvalue modification alone may not be
sufficient - we need to combine it with other techniques.
""")
