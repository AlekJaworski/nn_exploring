# mgcv Penalty Matrix Algorithm - COMPLETE AND VERIFIED

## Summary

After extensive investigation, we've successfully reverse-engineered and verified mgcv's penalty matrix computation algorithm. The key was correctly implementing the band Cholesky weighting scheme, including the subtle detail of properly constructing the P matrix with column-major layout and transposition.

## The Problem

- Initial Rust implementation used analytical integration of B''_i(x) * B''_j(x)
- This produced penalty matrices with magnitudes ~29,000x too large
- Lambda estimates were consequently off by 31-1500x
- Simple scaling hacks (like `/1000`) were non-generalizable

## The Solution: mgcv's Band Cholesky Algorithm

### Step 1: Extended Knot Vector

```python
k = num_basis + 1
nk = k - degree + 1
xl = x_min - (x_max - x_min) * 0.001
xu = x_max + (x_max - x_min) * 0.001
dx = (xu - xl) / (nk - 1)
knots_ext = linspace(xl - degree*dx, xu + degree*dx, nk + 2*degree)
```

**Key points:**
- Extends knot vector beyond data range
- For k=20, degree=3, creates 24 knots from ~-0.178 to ~1.178
- ✅ VERIFIED: Matches mgcv exactly (max diff < 5e-9)

### Step 2: Evaluation Points

```python
k0 = knots_ext[degree:(degree + nk)]  # Interior knots
h_unscaled = diff(k0)  # Knot spacings (DON'T scale yet!)
pord = degree - m2_order  # For degree=3, m2=2: pord=1

# Create evaluation points
h1 = repeat(h_unscaled / pord, pord)
k1 = cumsum([k0[0], ...h1])

# NOW scale h for use in weights
h_scaled = h_unscaled / 2.0
```

**Key points:**
- pord = degree - penalty_order (NOT penalty_order!)
- h is scaled AFTER creating k1, not before
- For cubic splines with 2nd derivative penalty: pord=1

### Step 3: Compute Derivative Matrix D

```python
D = zeros((len(k1), num_basis))
for i in range(num_basis):
    c = zeros(len(knots_ext) - degree - 1)
    c[i] = 1.0
    D[:, i] = BSpline(knots_ext, c, degree)(k1, nu=m2_order)
```

**Key points:**
- Evaluate m2_order-th derivatives of basis functions
- At the k1 evaluation points (not the full knot vector)
- num_basis basis functions (typically k-1 for identifiability)

### Step 4: Build W1 Weight Matrix

**CRITICAL: This is where we had the bug!**

```python
seq_vals = linspace(-1, 1, pord + 1)

# Build powers matrix EXACTLY as R does (column-major, then transpose)
vec = repeat(seq_vals, pord + 1) ** tile(arange(pord + 1), len(seq_vals))
powers_matrix = vec.reshape((pord + 1, pord + 1), order='F').T  # TRANSPOSE!
P = linalg.inv(powers_matrix)

# Build H matrix
i1 = add.outer(arange(1, pord + 2), arange(1, pord + 2))
H = (1 + (-1)**(i1 - 2)) / (i1 - 1)

# Compute W1
W1 = P.T @ H @ P
```

**Key points:**
- The reshape with `order='F'` (column-major) followed by `.T` (transpose) is ESSENTIAL
- Without the transpose, W1 has wrong signs on off-diagonals
- This propagates through to B_chol and causes 2.3x error in final penalty
- ✅ VERIFIED: W1 now matches mgcv exactly

For pord=1, this produces:
```
W1 = [[0.666667, 0.333333],
      [0.333333, 0.666667]]
```

NOT:
```
W1 = [[0.666667, -0.333333],   # ❌ WRONG SIGNS
      [-0.333333, 0.666667]]
```

### Step 5: Build ld Vector

```python
diag_W1 = diag(W1)
ld0 = tile(diag_W1, len(h_scaled)) * repeat(h_scaled, pord + 1)

# Reindex
indices = concatenate([
    repeat(arange(1, pord + 1), len(h_scaled)) +
    tile(arange(len(h_scaled)) * (pord + 1), pord),
    [len(ld0)]
]) - 1
ld = ld0[indices.astype(int)]

# Handle overlaps
if len(h_scaled) > 1:
    i0 = arange(1, len(h_scaled)) * pord
    i2 = arange(1, len(h_scaled)) * (pord + 1) - 1
    ld[i0] += ld0[i2]
```

**Key points:**
- ld0 uses SCALED h values (h/2)
- Reindexing selects elements for band matrix diagonal
- Overlaps add contributions from adjacent intervals
- ✅ VERIFIED: ld matches mgcv exactly (max diff 4.9e-17)

### Step 6: Build Banded B Matrix

```python
B = zeros((pord + 1, len(ld)))
B[0, :] = ld

for kk in range(1, pord + 1):
    if kk < W1.shape[0]:
        diwk = diag(W1, kk)  # kk-th super-diagonal
        ind_len = len(ld) - kk
        pattern = concatenate([diwk, zeros(kk - 1)])
        B[kk, :ind_len] = (repeat(h_scaled, pord) * tile(pattern, len(h_scaled)))[:ind_len]
```

**Key points:**
- B is stored in banded form: (pord+1) x len(ld)
- B[0, :] is the main diagonal
- B[1, :] is the first super-diagonal
- For pord=1, B is 2 x 18

### Step 7: Apply Banded Cholesky

```python
# Reconstruct full symmetric matrix
B_full = zeros((len(ld), len(ld)))
for i in range(pord + 1):
    for j in range(len(ld) - i):
        B_full[j, j + i] = B_full[j + i, j] = B[i, j]

# Cholesky decomposition
L_upper = cholesky(B_full, lower=False)

# Extract banded form
B_chol = zeros_like(B)
for i in range(pord + 1):
    for j in range(len(ld) - i):
        B_chol[i, j] = L_upper[j, j + i]
```

**Key points:**
- Standard Cholesky on full matrix, then extract bands
- L_upper is upper triangular
- B_chol stores the bands
- ✅ VERIFIED: B_chol matches mgcv exactly (max diff 4.7e-16)

### Step 8: Apply Weights to Derivatives

```python
D1 = D * B_chol[0, :len(D), newaxis]
for kk in range(1, pord + 1):
    ind = len(D) - kk
    if ind > 0:
        D1[:ind, :] += D[kk:, :] * B_chol[kk, :ind, newaxis]
```

**Key points:**
- Multiply D by Cholesky-weighted bands
- Main diagonal: D * B_chol[0, :]
- Super-diagonals: Add shifted contributions
- This transforms derivatives to weighted form

### Step 9: Compute Penalty

```python
S = D1.T @ D1
```

**Key points:**
- Simple matrix multiplication
- ✅ VERIFIED: Frobenius norm = 66901.7 (matches mgcv exactly!)
- ✅ VERIFIED: Trace = 221391.7 (matches mgcv exactly!)

## Verification Results

### Python Implementation
- ✅ Knots match mgcv (max diff < 5e-9)
- ✅ W1 matrix correct (after transpose fix)
- ✅ ld vector matches (max diff 4.9e-17)
- ✅ B_chol matches (max diff 4.7e-16)
- ✅ Final penalty Frobenius: 66901.7 ✓
- ✅ Final penalty Trace: 221391.7 ✓

### Test Files
- `test_exact_r_sequence.py` - ✅ PASSING
- `test_final_penalty.py` - ✅ PASSING
- `debug_h_values.py` - ✅ PASSING

## Critical Bug That Was Fixed

The original Python implementation had:
```python
powers_matrix = np.power.outer(np.arange(pord + 1), seq_vals)
P = np.linalg.inv(powers_matrix.T)
```

This produced wrong W1 off-diagonal signs. The correct implementation is:
```python
vec = np.repeat(seq_vals, pord + 1) ** np.tile(np.arange(pord + 1), len(seq_vals))
powers_matrix = vec.reshape((pord + 1, pord + 1), order='F').T
P = np.linalg.inv(powers_matrix)
```

The difference is subtle but crucial - R's `matrix()` function uses column-major layout, and we print it row-wise, which requires the transpose.

## Next Steps for Rust Implementation

1. Replace current analytical integration with band Cholesky algorithm
2. Implement column-major matrix construction for P
3. Add banded matrix support
4. Implement banded or full Cholesky (nalgebra supports both)
5. Add comprehensive unit tests comparing with mgcv outputs

## References

- Python implementation: `test_exact_r_sequence.py`, `test_final_penalty.py`
- R extraction scripts: `extract_mgcv_internals.R`, `extract_h_values.R`
- Investigation notes: `PENALTY_INVESTIGATION_SUMMARY.md`
