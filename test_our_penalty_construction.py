#!/usr/bin/env python3
"""Check our penalty matrix norm"""
import numpy as np
import sys
sys.path.insert(0, '/home/user/nn_exploring/target/release')

# Import Rust library directly to access penalty computation
import mgcv_rust

set.seed(42)
n = 100
x = np.random.randn(n)

# Create GAM to trigger penalty construction  
gam = mgcv_rust.GAM()

# We need to check the Rust code's penalty construction
# Let me look at the penalty.rs file instead
print("Need to check src/penalty.rs for cubic_regression_spline_penalty")
print("The formula should match mgcv's integral of (f'')^2")
