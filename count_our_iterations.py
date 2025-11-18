"""
Count OUR REML iterations to compare with R
"""
import numpy as np
import mgcv_rust

# We need to instrument the code to count iterations
# For now, let's check convergence by looking at lambda values

np.random.seed(123)  # Same seed as R

print("=" * 60)
print("Checking our convergence behavior")
print("=" * 60)

for n in [1500, 3000, 5000]:
    x = np.random.randn(n, 4)
    y_true = np.sin(x[:, 0]) + 0.5 * x[:, 1]**2 + np.cos(x[:, 2]) + 0.3 * x[:, 3]
    y = y_true + np.random.randn(n) * 0.1

    k_list = [16] * 4
    gam = mgcv_rust.GAM()
    result = gam.fit_auto_optimized(x, y, k=k_list, method='REML', bs='cr')

    print(f"\nn={n}:")
    print(f"  Lambda: {result['lambda']}")
    print(f"  Deviance: {result.get('deviance', 'N/A')}")

# Compare with R's lambdas
print("\n" + "=" * 60)
print("R's mgcv smoothing parameters for comparison:")
print("=" * 60)
print("n=1500: [16.83, 9.85, 19.66, 68588600.00]")
print("n=3000: (run R to get)")
print("n=5000: (run R to get)")
