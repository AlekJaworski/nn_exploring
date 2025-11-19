# Checkpoint: Simplified Hessian (Partial Success)

## Status: 2025-01-XX

This checkpoint captures the state where we have a **working but incomplete** Newton optimizer.

## What Works
- ✅ Gradient formula correct (Wood 2011)
- ✅ Simplified Hessian: `H[i,j] = -[λ_i·λ_j·trace_term/2 + δ_{ij}·λ_i·grad_i]`
- ✅ Newton makes progress initially (iterations 1-4)
- ✅ REML improves monotonically
- ✅ No crashes or numerical instability

## What's Wrong
- ❌ Converges to **suboptimal point**: REML=-63.26 (should be -64.64)
- ❌ Takes 100+ iterations (should be ~5)
- ❌ Gradient stuck at ~0.3 (should converge to ~0)
- ❌ Final λ=[2.30, 1.76] (should be [5.69, 5.20])

## Performance
```
Iteration 1: λ=0.003, |grad|=3.99, REML=-20.04
Iteration 2: λ=0.051, |grad|=3.82, REML=-42.39
Iteration 3: λ=0.906, |grad|=2.10, REML=-61.07
Iteration 4: λ=2.46,  |grad|=0.76, REML=-63.20
--- Effective progress stops here ---
Iteration 5-100: λ oscillates around 2.2, |grad| ~0.3, REML crawls to -63.26
```

## Implementation Details

### Gradient (src/reml.rs:614)
```rust
gradient[i] = (trace - (rank_i as f64) + penalty_term / phi) / 2.0;
```
Where `trace *= lambda_i` earlier to account for ρ=log(λ) parameterization.

### Hessian (src/reml.rs:897-915)
```rust
// Chain rule scaling for ρ-space
let mut h_val = lambda_i * lambda_j * trace_term / 2.0;

// Add diagonal gradient term (chain rule correction)
if i == j {
    let grad_lambda_i = (trace_term - (rank_i as f64) + penalty_term_i / phi) / 2.0;
    h_val += lambda_i * grad_lambda_i;
}

// CRITICAL: Negate for correct Newton direction
hessian[[i, j]] = -h_val;
```

This is the **trace-only approximation** with chain rule scaling.

### Why It Fails
mgcv uses **6 terms** in the Hessian (from gdi.c:get_ddetXWXpS):
1. `tr(Tkm KK')` - baseline
2. `-tr(KTkKK'TmK)` - our trace term
3. `+sp[k]*tr(P'S_kP)` - diagonal only, scaled by λ_k
4. `-sp[m]*tr(K'T_kKP'S_mP)` - scaled by λ_m
5. `-sp[k]*tr(K'T_mKP'S_kP)` - scaled by λ_k
6. `-sp[m]*sp[k]*tr(P'S_kPP'S_mP)` - scaled by λ_m·λ_k

We only have term 2 (approximately). Missing terms cause wrong curvature.

## Test Case
```python
# Data: n=100, 2 cubic splines k=10
# True optimum (mgcv): λ=[5.69, 5.20], REML=-64.64
# Our result: λ=[2.30, 1.76], REML=-63.26
```

## Files State
- `src/reml.rs`: Lines 614 (gradient), 897-915 (simplified Hessian)
- `src/smooth.rs`: Line 287 (REML convergence disabled for testing)

## Next Steps
1. Implement mgcv's complete 6-term Hessian
2. Test term-by-term against mgcv
3. Verify convergence to correct minimum

## How to Test
```bash
pip install -e .
python3 test_compare_gradients.py
# Should see convergence to λ≈[5.69, 5.20] in ~5 iterations
```

## Git State
Branch: `claude/investigate-penalty-gradient-01HySHcZuTwJx4QTEaVVBfXJ`
Last commit: "Diagnosis: Simplified Hessian converges to wrong minimum"
Commit hash: 3cd2b99
