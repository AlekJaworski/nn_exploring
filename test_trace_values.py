#!/usr/bin/env python
import subprocess
import sys

result = subprocess.run(['/root/miniforge3/bin/python', 'test_gradient_fix.py'],
                       capture_output=True, text=True,
                       env={'MGCV_GRAD_DEBUG': '1', 'PATH': '/root/miniforge3/bin',
                            'LD_LIBRARY_PATH': '/root/miniforge3/lib'})

lines = (result.stdout + result.stderr).split('\n')
for i, line in enumerate(lines):
    if ('trace_unscaled' in line or 'gradient[0]' in line) and i < 30:
        print(line)
