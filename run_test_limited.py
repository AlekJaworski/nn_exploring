#!/usr/bin/env python
import subprocess
import sys

result = subprocess.run(['python', 'test_gradient_fix.py'],
                       capture_output=True, text=True)

lines = (result.stdout + result.stderr).split('\n')
for i, line in enumerate(lines[:50]):  # First 50 lines
    print(line)

sys.exit(result.returncode)
