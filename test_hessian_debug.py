#!/usr/bin/env python3
"""
Debug Hessian computation by comparing with mgcv at specific sp values.
"""

import numpy as np
import subprocess
import json

# Run mgcv to get Hessian at specific sp values
print("=" * 80)
print("Running mgcv to get Hessian values...")
print("=" * 80)

result = subprocess.run(['Rscript', '/home/user/nn_exploring/unit_test_gradient_exact.R'],
                       capture_output=True, text=True, timeout=60)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# Load and compare results
test_sps = [(1.0, 1.0), (5.69, 5.20), (0.1, 0.1), (10.0, 1.0)]

for sp1, sp2 in test_sps:
    fname = f"/tmp/mgcv_sp_{sp1:.4f}_{sp2:.4f}.json"
    try:
        with open(fname, 'r') as f:
            mgcv_result = json.load(f)

        print("\n" + "=" * 80)
        print(f"sp = [{sp1:.4f}, {sp2:.4f}]")
        print("=" * 80)
        print(f"mgcv REML: {mgcv_result['REML']:.6f}")
        print(f"mgcv gradient: {mgcv_result['gradient']}")
        print(f"mgcv Hessian diagonal: {mgcv_result['hessian_diag']}")

    except FileNotFoundError:
        print(f"\nFile not found: {fname}")
