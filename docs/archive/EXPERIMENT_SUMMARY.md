# Eigenvalue Modification Experiment

## Changes Made to src/smooth.rs

### 1. Eigenvalue Modification (lines 556-622)
Added eigendecomposition and modified eigenvalues like mgcv does:
- Compute eigendecomposition of Hessian
- Flip negative eigenvalues to positive (absolute value)
- Set floor for tiny eigenvalues
- Reconstruct Hessian from modified eigenvalues

This is the key technique mgcv uses to handle indefinite Hessians.

### 2. Quadratic Approximation Error Check (lines 755-775)
Added check for how well the quadratic model predicts the actual REML change:
- Compute predicted change: grad·step + 0.5·step'·Hessian·step
- Compute actual change: new_reml - current_reml
- Compute qerror = |pred - actual| / (max(|pred|, |actual|) + scale·tol)
- Only accept step if qerror < 0.8

This prevents accepting steps where the model is a poor fit.

## Build Issues

The code compiles successfully but there are linking issues with the Python extension:
- Undefined symbol: dgeqrf_ (LAPACK routine)
- This is needed for the eigendecomposition we added
- The wheel build has issues with static vs dynamic linking

## Next Steps

To complete the experiment:
1. Fix the linking issue (may need to adjust build flags)
2. Test with the 4D example
3. Compare lambdas before/after the fix
4. If no improvement, implement additional mgcv techniques:
   - Steepest descent fallback after 3 halvings
   - Multiple starting points strategy
   - Better initialization

## Expected Outcome

If these fixes work, we should see:
- mgcv_rust lambdas: [5, 6, 9000, 660] (closer to mgcv)
- Better REML score: closer to -119.09
- Improved smoothness on dims 2 & 3

## Code Changes Location

See git diff for exact changes to src/smooth.rs:
- Lines 556-622: Eigenvalue modification
- Lines 755-775: Quadratic error check
