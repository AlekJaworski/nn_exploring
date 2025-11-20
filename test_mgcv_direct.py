#!/usr/bin/env python3
"""
Use rpy2 to access mgcv internals and extract exact gradient/Hessian at our converged λ.

We'll compare mgcv's gradient/Hessian at OUR λ=[4.11, 2.32] vs what we compute.
