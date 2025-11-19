# SP Parameterization Investigation

## Problem
After implementing sp parameterization (λ = sp × S.scale), gradients are still catastrophic (4.3e27).

## Root Cause Found

### S.scale Formula is Wrong!

Our formula: `S.scale = maXX / ||S||_inf` gives:
- **Our S.scale**: 0.013 - 0.015
- **mgcv's S.scale**: 70 - 173
- **Ratio**: ~5000x difference!

### Key Discovery

When applying the formula `maXX / ||S||_inf` to mgcv's stored penalty matrices:
- **Result**: 1.0 (exactly!)  
- **Conclusion**: mgcv's stored S matrices are ALREADY normalized

This means:
1. mgcv computes S.scale using some other formula (not maXX/||S||)
2. They scale the stored penalty: `S_stored = S_original × S.scale`
3. Our formula gives 1.0 because we're computing on already-scaled matrices!

### Evidence from mgcv

smoothCon output shows:
```
maXX = 10.76
||S||_inf = 10.76  
maXX / ||S||_inf = 1.0
But S.scale = 70.87!
```

## Impact

We're applying penalty normalization TWICE:
1. First: multiply by scale_factor (our computation)  
2. Second: treat it as if S.scale should be scale_factor

This creates huge numerical issues because our S.scale values are 5000x too small!

## Next Steps

Need to find mgcv's actual S.scale formula by:
1. Checking mgcv source for where S.scale is set (not in smooth.construct.cr)
2. May be in gam.setup or penalty absorption code  
3. Might be related to knot spacing or data range

Suspect it's related to the integral of the squared second derivative or similar.
