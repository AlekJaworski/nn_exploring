#!/usr/bin/env python3
"""Final performance comparison"""
import numpy as np
import mgcv_rust
import time
import subprocess

# Same data
np.random.seed(42)
n, d, k = 500, 4, 12
X = np.random.uniform(0, 1, size=(n, d))
effect_1 = np.sin(2 * np.pi * X[:, 0])
effect_2 = (X[:, 1] - 0.5) ** 2
effect_3 = X[:, 2]
y = effect_1 + effect_2 + effect_3 + np.random.normal(0, 0.3, n)

print("="*70)
print("FINAL PERFORMANCE COMPARISON")
print("="*70)
print(f"Data: n={n}, d={d}, k={k}")

# Test Rust
print("\n[Rust] Running 50 iterations...")
times_rust = []
for i in range(50):
    gam = mgcv_rust.GAM()
    start = time.perf_counter()
    gam.fit_auto(X, y, k=[k]*d, method='REML', bs='cr')
    times_rust.append((time.perf_counter() - start) * 1000)

rust_mean = np.mean(times_rust)
rust_std = np.std(times_rust)

print(f"Rust: {rust_mean:.2f} ± {rust_std:.2f} ms")

# Save data for R
np.savetxt('/tmp/test_x.csv', X, delimiter=',')
np.savetxt('/tmp/test_y.csv', y, delimiter=',')

# R timing script
r_script = """
library(mgcv)
X <- as.matrix(read.csv('/tmp/test_x.csv', header=FALSE))
y <- as.matrix(read.csv('/tmp/test_y.csv', header=FALSE))[,1]

times <- numeric(50)
for (i in 1:50) {
  start <- proc.time()
  fit <- gam(
    y ~ s(X[,1], k=12, bs='cr') + s(X[,2], k=12, bs='cr') +
        s(X[,3], k=12, bs='cr') + s(X[,4], k=12, bs='cr'),
    method='REML'
  )
  elapsed <- (proc.time() - start)['elapsed']
  times[i] <- elapsed * 1000
}

cat(sprintf('R: %.2f ± %.2f ms\\n', mean(times), sd(times)))
"""

with open('/tmp/benchmark_r.R', 'w') as f:
    f.write(r_script)

print("\n[R] Running 50 iterations...")
result = subprocess.run(['Rscript', '/tmp/benchmark_r.R'],
                       capture_output=True, text=True)
r_output = result.stdout.strip()
print(r_output)

# Parse R time
if 'R:' in r_output:
    r_mean = float(r_output.split('±')[0].split(':')[1].strip())

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Rust:  {rust_mean:.2f} ms")
    print(f"R:     {r_mean:.2f} ms")
    print(f"Ratio: {rust_mean/r_mean:.2f}x")
    print(f"")
    print(f"Rust is {rust_mean/r_mean:.2f}x slower than R")
