# MGCV vs MGCV_RUST: Optimization Analysis

## Repository Cloned
- Location: `mgcv_analysis/mgcv/`
- Source: https://github.com/cran/mgcv

## Key Finding: Why mgcv Finds Better Solutions

### 1. EIGENVALUE MODIFICATION (Critical Difference)

**mgcv** (gam.fit3.r:1447-1455):
```r
# Check if Hessian is indefinite
indef <- (sum(-d > abs(d[1])*.Machine$double.eps^.5)>0)

# Heuristic: set negative eigenvalues to their ABSOLUTE VALUE
# This is different from adding a ridge!
ind <- d < 0
d[ind] <- -d[ind]  # FLIP sign of negative eigenvalues
```

**mgcv_rust** (smooth.rs:553-556):
```rust
// Add small ridge for numerical stability
let ridge = 1e-7;
for i in 0..m {
    hessian[[i, i]] += ridge;  // Just adds ridge, doesn't flip
}
```

**Impact**: mgcv's eigenvalue flip allows it to navigate indefinite regions better,
escaping shallow local minima that mgcv_rust gets stuck in.

---

### 2. QUADRATIC APPROXIMATION ERROR CHECK

**mgcv** (gam.fit3.r:1498-1499):
```r
# Compute quadratic approximation error
qerror <- abs(pred.change-score.change)/(max(abs(pred.change),abs(score.change))+score.scale*conv.tol)

# Only accept if qerror < 0.8 (threshold)
if (is.finite(score1) && score.change<0 && pdef && qerror < qerror.thresh) {
    # Accept step
}
```

**mgcv_rust**: No quadratic error check! Just uses Armijo condition.

**Impact**: mgcv rejects steps where the quadratic model is poor,
preventing convergence to bad local minima.

---

### 3. STEEPEST DESCENT FALLBACK

**mgcv** (gam.fit3.r:1521-1524, 1580-1608):
```r
# After 3 halvings, try steepest descent
if (ii==3&&i<10) {
    step <- Sstep*s.length/sum(Sstep^2)^.5  # Switch to SD
}

# Also tries SD separately and picks best of Newton vs SD
if (!pdef&&sd.unused) {
    # Try steepest descent with different step lengths
    # Pick best between Newton and SD steps
}
```

**mgcv_rust**: Has steepest descent fallback but doesn't compare both.

**Impact**: mgcv's exploration of both directions helps escape local minima.

---

### 4. CONVERGENCE CRITERIA

**mgcv** (gam.fit3.r:1381-1385):
```r
# Scale gradient tolerance by score scale
score.scale <- abs(log(b$scale.est)) + abs(score)
uconv.ind <- abs(grad) > score.scale*conv.tol
```

**mgcv_rust** (smooth.rs:561-562):
```rust
let grad_norm_linf: f64 = gradient.iter().map(|g| g.abs()).fold(0.0f64, f64::max);
// Fixed tolerance of 0.05
```

**Impact**: mgcv uses adaptive tolerance based on problem scale.

---

## ROOT CAUSE

mgcv_rust gets stuck because:

1. **No eigenvalue flipping** → Can't navigate indefinite Hessians well
2. **No quadratic error check** → Accepts steps where model is poor
3. **Limited exploration** → Doesn't try steepest descent as thoroughly
4. **Fixed starting point** → Always starts at λ=1

The combination of eigenvalue modification + quadratic error checking
allows mgcv to escape the local minimum that mgcv_rust converges to.

---

## SOLUTION: Implement in mgcv_rust

### Priority 1: Eigenvalue Modification (HIGH IMPACT)
```rust
// In optimize_reml_newton_multi_with_xtwx()
// After computing Hessian:

// 1. Compute eigen decomposition
let (eigenvalues, eigenvectors) = hessian.eig()?;

// 2. Check if indefinite
let mut is_indefinite = false;
for val in &eigenvalues {
    if *val < -1e-10 {
        is_indefinite = true;
        break;
    }
}

// 3. Modify eigenvalues (mgcv heuristic)
let mut modified_eigenvalues = eigenvalues.clone();
for i in 0..m {
    if modified_eigenvalues[i] < 0.0 {
        modified_eigenvalues[i] = -modified_eigenvalues[i]; // Flip sign
    }
    if modified_eigenvalues[i] < 1e-7 {
        modified_eigenvalues[i] = 1e-7; // Small floor
    }
}

// 4. Reconstruct Hessian from modified eigenvalues
let mut modified_hessian = Array2::zeros((m, m));
for i in 0..m {
    for j in 0..m {
        for k in 0..m {
            modified_hessian[[i, j]] += eigenvectors[[i, k]] 
                * modified_eigenvalues[k] 
                * eigenvectors[[j, k]];
        }
    }
}
```

### Priority 2: Quadratic Approximation Error (MEDIUM IMPACT)
```rust
// After computing trial step and new REML:

// Predicted change from quadratic model
let pred_change: f64 = gradient.iter()
    .zip(step.iter())
    .map(|(g, s)| g * s)
    .sum::<f64>() 
    + 0.5 * step.dot(&hessian.dot(&step));

let score_change = new_reml - current_reml;
let score_scale = current_reml.abs() + 1.0; // Scale factor

// Quadratic approximation error
let qerror = (pred_change - score_change).abs() 
    / (pred_change.abs().max(score_change.abs()) + score_scale * tolerance);

// Only accept if qerror < 0.8 AND score improved
if qerror < 0.8 && score_change < 0.0 {
    // Accept step
} else {
    // Step halve or try steepest descent
}
```

### Priority 3: Better Steepest Descent Integration (LOWER IMPACT)
- After 3 step halvings, switch to SD direction
- Try both Newton and SD, pick best

---

## VERIFICATION

To verify the fix works:

1. Run 4D test case
2. Compare REML scores:
   - mgcv: -119.09 (target)
   - Current mgcv_rust: -113.51 (4.69% worse)
   - Fixed mgcv_rust: should approach -119.09

3. Lambda comparison:
   - Dim 2: mgcv=9043, current=199 → fixed should be closer to 9043
   - Dim 3: mgcv=660, current=181 → fixed should be closer to 660

## FILES TO MODIFY

1. `src/smooth.rs`:
   - `optimize_reml_newton_multi_with_xtwx()` (lines 362-888)
   - Add eigenvalue modification
   - Add quadratic error check
   - Improve steepest descent fallback

2. Dependencies needed:
   - Eigenvalue decomposition (use `ndarray-linalg` or pure Rust)

## REFERENCES

- mgcv: `mgcv_analysis/mgcv/R/gam.fit3.r` lines 1290-1700
- Key functions: `newton()`, eigenvalue modification at lines 1447-1455
