# Fellner-Schall Convergence Issue

## Current Status
- Version: 8d63abb (tagged: working-fellner-schall-v1)
- Iterations: 22 (goal: 7 like R's bam)
- Accuracy: ✓ correlation >0.999
- Convergence: ✓ converges but λ → 10^-7 instead of optimal ~1.0

## The Dilemma

Penalty normalization creates a fundamental conflict:

**WITH normalization** (current):
- ✓ Numerical accuracy: correlation >0.999
- ✓ RSS matches R
- ✗ trace/rank ≈ 0.13 (expected 1.0)
- ✗ λ converges to lower bound (10^-7)
- ✗ Takes 22 iterations

**WITHOUT normalization**:
- ✗ Numerical accuracy broken: correlation 0.58
- ✗ RSS 67x worse than R
- ✓ trace/rank ≈ 1.0 (correct ratio)
- ✗ Takes 30+ iterations without converging

## Root Cause

Penalty normalization: `S → S * (||X||²_∞ / ||S||_∞)` scales penalties by ~0.000078.

This changes the expected Fellner-Schall ratio:
- Before norm: tr(A^{-1}·S) ≈ rank = 8 at optimal λ
- After norm: tr(A^{-1}·(c·S)) = c·tr(A^{-1}·S) ≈ c·rank ≈ 0.0006

But we observe tr(A^{-1}·S_norm) ≈ 1.07 at R's optimal λ ≈ 1.2, not 0.0006!

This suggests either:
1. Our understanding of the Fellner-Schall criterion is wrong
2. R applies normalization differently (before/after optimization)
3. The effective rank calculation needs adjustment we haven't figured out

## What Works

Despite λ → 10^-7 being "wrong", the fitted values are correct! This suggests:
- Either the fit is insensitive to λ in the range [10^-7, 1.0]
- Or there's compensation happening elsewhere in the algorithm

## Next Steps

To reduce iterations from 22 → 7:
1. Understand R's actual Fellner-Schall implementation
2. Figure out how they handle penalty normalization
3. Find the correct "effective rank" after normalization
4. Or use a different convergence criterion that's scale-invariant

## References
- Working commit: 8d63abb
- Penalty normalization added: 700ed48
- Performance analysis: 6023a9e (documents the 22 vs 7 iteration issue)
