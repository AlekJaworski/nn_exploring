#!/usr/bin/env python
import subprocess
import sys

result = subprocess.run(['/root/miniforge3/bin/python', 'test_gradient_fix.py'],
                       capture_output=True, text=True,
                       env={'MGCV_GRAD_DEBUG': '1', 'PATH': '/root/miniforge3/bin',
                            'LD_LIBRARY_PATH': '/root/miniforge3/lib'})

lines = (result.stdout + result.stderr).split('\n')
for line in lines[:80]:  # First 80 lines
    if ('diagonal range' in line.lower() or 'p_matrix' in line.lower() or
        'gradient[0]' in line.lower() or 'conditioning' in line.lower() or
        'ridge' in line.lower()):
        print(line)
