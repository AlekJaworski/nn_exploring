# GRADIENT BUG FOUND: Block vs Full Matrix Trace

## The Smoking Gun

At λ=[4.11, 2.32]:
- **mgcv FD gradient**: [-0.70, -1.81]
- **Our gradient**: ~[0.05, 0.05]
- **Error**: ~10-40x too small!

## Root Cause: Block Trace Instead of Full Trace

### What the Formula Should Be

```
∂REML/∂log(λᵢ) = [tr(A^{-1}·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2
```

Where:
- S_i is FULL p×p penalty matrix (block-diagonal, mostly zeros)
- A^{-1} is FULL p×p matrix
- tr(A^{-1}·λᵢ·Sᵢ) should use FULL matrices

### What We're Actually Computing

From `src/reml.rs` lines 549-602:

```rust
// Extract non-zero BLOCK from penalty (lines 549-563)
let mut block_start = 0;
let mut block_end = p;
// ... find non-zero block ...
let block_size = block_end - block_start;

// Extract penalty BLOCK (lines 571-577)
let mut penalty_block = Array2::<f64>::zeros((block_size, block_size));
// ... copy block ...

// Extract corresponding rows from P matrix (lines 582-588)
let mut p_block = Array2::<f64>::zeros((block_size, p));
for ii in 0..block_size {
    for jj in 0..p {
        p_block[[ii, jj]] = p_matrix[[block_start + ii, jj]];
    }
}

// Compute trace using BLOCK (lines 590-602)
let p_block_t_l = p_block.t().dot(&sqrt_pen_block);  // p × rank_i
let mut trace = 0.0;
for k in 0..p {
    for r in 0..sqrt_pen_block.ncols() {
        trace += p_block_t_l[[k, r]] * p_block_t_l[[k, r]];
    }
}
trace *= lambda_i;
```

This computes:
```
trace = λᵢ · tr(P_block'·S_block·P_block)
```

Where:
- P_block is only block_size rows of P (e.g., rows 0-9 for smooth 1)
- S_block is only the block_size×block_size non-zero block

### The Problem

**We need**: `tr(P'·S·P)` where S is FULL p×p matrix

**We compute**: `tr(P_block'·S_block·P_block)` where both are extracted blocks

These are NOT the same!

### Dimensional Analysis

For smooth 1 (rows 0-9):
- S (full) is 20×20 with 10×10 non-zero block in top-left
- P (full) is 20×20
- S_block is 10×10 (just the non-zero part)
- P_block is 10×20 (rows 0-9 of P)

Computing tr(P_block'·S_block·P_block):
- P_block' is 20×10
- S_block is 10×10
- P_block'·S_block is 20×10
- (P_block'·S_block)·P_block is 20×20
- tr(...) sums the diagonal

But this is NOT tr(P'·S·P) because we're missing the contribution from the OTHER rows/columns of P!

### Correct Computation

We should compute:
```rust
// Don't extract blocks - use FULL matrices
let p_t_s = p_matrix.t().dot(&penalty_i);  // p × p
let p_t_s_p = p_t_s.dot(&p_matrix);        // p × p
let trace = p_t_s_p.diag().sum() * lambda_i;
```

Or using the QR sqrt representation:
```rust
// penalty_i is p×p (full matrix)
let sqrt_pen_i = penalty_sqrt(&penalty_i)?;  // p × rank_i
let p_t_l = p_matrix.t().dot(&sqrt_pen_i);   // p × rank_i
let trace_term: f64 = p_t_l.iter().map(|x| x*x).sum();
let trace = lambda_i * trace_term;
```

### Why This Causes 10-40x Error

The non-zero block is ~10×10 out of full 20×20 matrix.

When we compute tr(P_block'·S_block·P_block), we only get contributions from 10 rows of P, when we should sum over ALL 20 rows!

The missing rows contribute significantly to the trace, causing our gradient to be ~10-40x too small.

### Impact

Wrong gradient → optimizer thinks it's converged when it shouldn't → stops at λ=[4.11, 2.32] instead of [5.69, 5.20].

## Fix Required

Replace block-based trace computation (lines 571-602) with full-matrix computation:

```rust
// Compute tr(A^{-1}·λᵢ·Sᵢ) = λᵢ·tr(P'·Sᵢ·P) using full matrices
let sqrt_pen_i = &sqrt_penalties[i];  // p × rank_i
let p_t_l = p_matrix.t().dot(sqrt_pen_i);  // p × rank_i

// tr(P'·L·L'·P) = sum of squared elements of P'·L
let trace_term: f64 = p_t_l.iter().map(|x| x * x).sum();
let trace = lambda_i * trace_term;
```

This will give the correct gradient and fix convergence!
