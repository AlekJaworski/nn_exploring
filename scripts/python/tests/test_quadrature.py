#!/usr/bin/env python3
"""
Test if Gaussian quadrature integration is correct
"""

import numpy as np

# Test integrating x^2 from 0 to 1
# Analytical result: 1/3

# 10-point Gauss-Legendre on [-1, 1]
gauss_points = [
    (-0.9739065285171717, 0.0666713443086881),
    (-0.8650633666889845, 0.1494513491505806),
    (-0.6794095682990244, 0.2190863625159820),
    (-0.4333953941292472, 0.2692667193099963),
    (-0.1488743389816312, 0.2955242247147529),
    (0.1488743389816312, 0.2955242247147529),
    (0.4333953941292472, 0.2692667193099963),
    (0.6794095682990244, 0.2190863625159820),
    (0.8650633666889845, 0.1494513491505806),
    (0.9739065285171717, 0.0666713443086881),
]

# Transform from [-1, 1] to [0, 1]
a, b = 0.0, 1.0
h = b - a

result = 0.0
for xi, wi in gauss_points:
    # Transform point from [-1, 1] to [a, b]
    x = a + 0.5 * h * (xi + 1.0)
    f = x**2
    result += wi * f * 0.5 * h

print(f"Gauss quadrature: {result:.10f}")
print(f"Analytical:       {1/3:.10f}")
print(f"Error:            {abs(result - 1/3):.2e}")

print("\n" + "="*50)
print("Testing constant function (should be exact)")
result = 0.0
for xi, wi in gauss_points:
    x = a + 0.5 * h * (xi + 1.0)
    f = 1.0
    result += wi * f * 0.5 * h

print(f"Gauss quadrature: {result:.10f}")
print(f"Analytical:       1.0000000000")
print(f"Error:            {abs(result - 1.0):.2e}")
