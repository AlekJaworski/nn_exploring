#!/usr/bin/env python3
"""
Get unnormalized penalty traces for different k values from our code
"""

import numpy as np
import sys

sys.path.insert(0, "target/release")
import mgcv_rust

# Test different k values
for k in [5, 10, 15, 20, 25, 30]:
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 100)

    X = x.reshape(-1, 1)
    gam = mgcv_rust.GAM()

    # This will trigger the debug output showing unnormalized trace
    result = gam.fit_auto(X, y, k=[k], method='REML', bs='cr')
