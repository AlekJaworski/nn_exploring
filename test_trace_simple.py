#!/usr/bin/env python3
import os
os.environ['MGCV_GRAD_DEBUG'] = '1'

import numpy as np
import mgcv_rust

np.random.seed(42)
n = 1000
x = np.random.randn(n, 4)
y = np.sin(x[:, 0]) + 0.5*x[:, 1]**2 + np.cos(x[:, 2]) + 0.3*x[:, 3] + np.random.randn(n)*0.1

print("Running single optimization iteration to check trace...")
gam = mgcv_rust.GAM()
result = gam.fit_auto_optimized(x, y, k=[16]*4, method='REML', bs='cr')
print(f"\nFinal lambdas: {result['lambda']}")
