"""
Profile the Rust gradient computation to identify bottlenecks
"""
import numpy as np
import pandas as pd
import time
import mgcv_rust

# Load the largest test case
n, d = 1000, 5

# Generate test data
np.random.seed(42)
X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
penalties = []
for i in range(d):
    S = pd.read_csv(f'/tmp/bench_S{i+1}.csv').values
    penalties.append(S)

lambdas = np.array([6.99504494, 2.93277942, 2.34582939, 5.55801621, 7.2227428])
w = np.ones(n)

print("="*80)
print("PROFILING RUST GRADIENT COMPUTATION")
print("="*80)
print(f"Problem size: n={n}, p={X.shape[1]}, d={d}")
print()

# Run with detailed timing
import subprocess

# Create a Python script that calls Rust with detailed debug output
profile_script = """
import mgcv_rust
import numpy as np
import pandas as pd
import time

X = pd.read_csv('/tmp/perf_X.csv').values
y = pd.read_csv('/tmp/bench_y.csv')['y'].values
penalties = []
for i in range(5):
    S = pd.read_csv(f'/tmp/bench_S{{i+1}}.csv').values
    penalties.append(S)

lambdas = np.array([6.99504494, 2.93277942, 2.34582939, 5.55801621, 7.2227428])
w = np.ones(len(y))

# Single call with timing
start = time.time()
gradient = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, penalties)
end = time.time()

print(f"Total time: {(end-start)*1000:.2f} ms")
print(f"Gradient: {gradient}")
"""

with open('/tmp/profile_rust.py', 'w') as f:
    f.write(profile_script)

# Run multiple times to get consistent timing
print("Running 50 iterations to get stable timing...")
times = []
for i in range(50):
    start = time.time()
    gradient = mgcv_rust.reml_gradient_multi_qr_py(y, X, w, lambdas, penalties)
    end = time.time()
    times.append((end - start) * 1000)

mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)

print(f"\nTiming statistics (50 runs):")
print(f"  Mean: {mean_time:.2f} ms")
print(f"  Std:  {std_time:.2f} ms")
print(f"  Min:  {min_time:.2f} ms")
print(f"  Max:  {max_time:.2f} ms")

print(f"\nGradient: {gradient}")

# Now let's analyze computational complexity
print("\n" + "="*80)
print("COMPLEXITY ANALYSIS")
print("="*80)

n, p = X.shape
print(f"\nProblem parameters:")
print(f"  n = {n} (observations)")
print(f"  p = {p} (parameters)")
print(f"  d = {d} (smooths)")

print(f"\nExpected operations:")
print(f"  QR decomposition: O(n·p²) = O({n}·{p}²) ≈ {n*p*p:.0e} ops")
print(f"  Matrix inverse: O(p³) = O({p}³) ≈ {p**3:.0e} ops")
print(f"  Per smooth (×{d}):")
print(f"    - Matrix mult: O(p²) ≈ {p*p:.0e} ops")
print(f"    - Trace: O(p²) ≈ {p*p:.0e} ops")
print(f"  Total per smooth: O(p²) × {d} ≈ {d*p*p:.0e} ops")

total_ops = n * p * p + p**3 + d * p * p
print(f"\nTotal operations: ≈ {total_ops:.2e}")
print(f"Time per operation: ≈ {mean_time * 1e-3 / total_ops * 1e9:.2f} ns")

# Compare to mgcv
print("\n" + "="*80)
print("COMPARISON TO MGCV")
print("="*80)
print("\nmgcv time: 4.04 ms (from benchmark)")
print(f"Rust time: {mean_time:.2f} ms")
print(f"Ratio: {mean_time / 4.04:.2f}x")

print("\nPossible reasons for slower performance:")
print("  1. Python/Rust FFI overhead")
print("  2. Memory allocation patterns")
print("  3. BLAS/LAPACK optimizations in R")
print("  4. Matrix operation implementations")

# Let's check if the issue is in the IFT terms
print("\n" + "="*80)
print("BOTTLENECK IDENTIFICATION")
print("="*80)

# The gradient computation has these main steps:
# 1. QR decomposition: Z matrix is (n + sum(ranks)) × p
ranks = [8, 8, 8, 8, 8]  # From earlier debug output
z_rows = n + sum(ranks)
print(f"\nQR decomposition:")
print(f"  Z matrix: {z_rows}×{p}")
print(f"  Operations: O({z_rows}·{p}²) ≈ {z_rows * p * p:.2e}")

print(f"\nMatrix inverse (R⁻¹):")
print(f"  R matrix: {p}×{p}")
print(f"  Operations: O({p}³) ≈ {p**3:.2e}")

print(f"\nPer-smooth gradient ({d} smooths):")
print(f"  Each smooth requires:")
print(f"    - Trace computation: O(p·rank) ≈ {p * 8:.2e}")
print(f"    - IFT β derivative: O(p²) ≈ {p**2:.2e}")
print(f"    - ∂rss/∂ρ: O(n·p) ≈ {n * p:.2e}")
print(f"    - ∂edf/∂ρ: O(p³) ≈ {p**3:.2e} ⚠️ EXPENSIVE!")

print(f"\nTotal IFT overhead per smooth: O(p³)")
print(f"Total for {d} smooths: {d} × O(p³) ≈ {d * p**3:.2e} ops")

print(f"\n⚠️  The ∂edf/∂ρ computation is O(p³) per smooth!")
print(f"   This dominates for large p and d.")
print(f"   Total: {d * p**3:.2e} ops just for ∂edf/∂ρ terms")
