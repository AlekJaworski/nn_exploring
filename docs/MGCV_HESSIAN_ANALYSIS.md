# mgcv C Source Analysis: Hessian Computation

## Key Finding: Gaussian Case Simplification

For **Gaussian family with constant weights**:
- `Tk = 0` (first derivative of weights = 0)
- `Tkm = 0` (second derivative of weights = 0)

This causes **Terms 1, 2, 4, 5 to vanish**, leaving only:

## Simplified Hessian (Gaussian Case)

```c
det2[k,m] = 0;  // Initialize

// Term 3: Diagonal correction (only when k==m)
if (k >= M0 && k==m)
    det2[k,m] += trPtSP[m - M0];

// Term 6: Penalty interaction
if (k >= M0 && m >= M0)
    det2[k,m] -= sp[m - M0] * sp[k - M0] *
                 diagABt(work, PtSP[k-M0], PtSP[m-M0], r, r);
```

Where:
```c
trPtSP[m] = sp[m] * tr(P' * S_m * P)      // Precomputed
PtSP[m] = P' * S_m * P                    // r×r matrix
```

## Variable Mapping

| C Variable | Dimension | Mathematical Meaning |
|------------|-----------|---------------------|
| `P` | q×r or r×r | R^{-1} from QR decomposition |
| `K` | n×r | From QR decomposition |
| `sp[m]` | scalar | λ_m (smoothing parameter) |
| `S_m` | q×q | Penalty matrix m |
| `PtSP[m]` | r×r | P'·S_m·P |
| `trPtSP[m]` | scalar | λ_m · tr(P'·S_m·P) |
| `M0` | int | # of theta parameters (dispersion, not smoothing) |
| `M` | int | # of smoothing parameters |
| `r` | int | Rank of fit |
| `q` | int | # of coefficients |

## QR Decomposition Context

From Wood (2008) JRSSB 70:495-518:

The augmented matrix for QR decomposition is:
```
Z = [sqrt(W)·X    ]    (n×q)
    [sqrt(λ₁)·L₁' ]    (rank₁×q)
    [sqrt(λ₂)·L₂' ]    (rank₂×q)
    [    ...      ]
```

Where L_m' is sqrt(S_m) transposed (rank_m × q).

QR decomposition: Z = Q·R
- R is q×q upper triangular
- R'·R = X'·W·X + Σλ_m·S_m = A (penalized information matrix)
- P = R^{-1}, so P'·P = (R'·R)^{-1} = A^{-1}

## Formula in Our Notation

The Hessian `det2[k,m]` for Gaussian case is:

```
∂²V/∂ρ_k∂ρ_m = δ_{k,m} · λ_m · tr(A^{-1}·S_m)
                - λ_k · λ_m · tr[(A^{-1}·S_k)·(A^{-1}·S_m)]
```

Where:
- ρ_m = log(λ_m) (log smoothing parameter)
- V = REML criterion (to be minimized)
- A = X'WX + Σλ_m·S_m

### Expanding with P'·P = A^{-1}:

```
tr[(A^{-1}·S_k)·(A^{-1}·S_m)] = tr[(P'·P·S_k)·(P'·P·S_m)]
```

Using P (which is r×r after rank reduction):
```
= tr[(P'·S_k·P)·(P'·S_m·P)]
= tr[PtSP[k]·PtSP[m]]
```

This matches the C code!

## Implementation Plan

For Gaussian case, we need:
1. Compute A^{-1} via QR: P = R^{-1} where R'R = A
2. For each penalty m: compute PtSP[m] = P'·S_m·P
3. For diagonal: add λ_m · tr(PtSP[m])
4. For all terms: subtract λ_k·λ_m · tr(PtSP[k]·PtSP[m])

## Cross-Reference with Wood (2011)

Wood (2011) Section 3 gives the REML criterion:
```
V_R = log|Z'Z| + (n-q)*log(φ) + ...
```

The Hessian with respect to log smoothing parameters involves:
- ∂²log|Z'Z|/∂ρ_k∂ρ_m  (determinant term - this is what gdi.c computes!)
- Additional terms from φ derivatives (handled elsewhere)

The `get_ddetXWXpS` function computes **only the log-determinant Hessian**, not the complete REML Hessian!

## CRITICAL INSIGHT

**mgcv splits the Hessian computation**:
1. `get_ddetXWXpS`: Computes ∂²log|A|/∂ρ_k∂ρ_m
2. Other functions: Add penalty derivative terms ∂²(β'Sβ/φ)/∂ρ_k∂ρ_m

We need to find where the penalty terms get added!
