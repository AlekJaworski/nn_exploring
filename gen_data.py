#!/usr/bin/env python3
import numpy as np

np.random.seed(123)
n = 500
d = 1
k = 20

x = np.random.uniform(0, 1, n)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.3, n)

with open('test_simple.txt', 'w') as f:
    f.write(f"{n} {d} {k}\n")
    for xi in x:
        f.write(f"{xi}\n")
    for yi in y:
        f.write(f"{yi}\n")

print(f"Generated test_simple.txt: n={n}, d={d}, k={k}")
