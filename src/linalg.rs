//! Basic linear algebra operations for GAM fitting

use crate::{GAMError, Result};
use ndarray::{s, Array1, Array2};

#[cfg(feature = "blas")]
use ndarray_linalg::{Determinant, Inverse, Solve};

#[cfg(feature = "blas")]
use lapack_sys::dpstrf_;

/// Result of [`pivoted_cholesky`] — mirrors mgcv's `Sl.fitChol` use of LAPACK's
/// `dpstrf` (full pivoted Cholesky with rank revelation).
///
/// On entry the input `A` is taken to be symmetric (only upper triangle is
/// read). On exit:
///
/// * `r` is the **upper-triangular** Cholesky factor of `A[piv, piv]` truncated
///   to the rank-revealing prefix. Specifically, if the LAPACK call returns
///   numerical rank `rank`, then `R[..rank, ..rank]` is upper triangular with
///   positive diagonal and `R' · R = A[piv[..rank], piv[..rank]]` to machine
///   precision. The trailing `(p − rank)` columns of `R` reflect the unfactored
///   Schur complement and **must not be used** in subsequent solves.
///
/// * `piv` is the **0-based** pivot permutation: `piv[k]` gives the original
///   column index that lands at position `k` after pivoting (LAPACK's
///   `dpstrf` writes 1-based indices, which we shift on the way out so callers
///   can index ndarray rows/cols directly).
///
/// * `rank` is the numerical rank reported by LAPACK (after thresholding small
///   pivots at `tol`).
///
/// The output is byte-faithful to mgcv's `R::chol(... , pivot = TRUE)` modulo
/// the 0- vs 1-based convention. mgcv uses this same routine internally
/// (see R/fast-REML.r:1606 → R's `chol(... pivot=TRUE)` → LAPACK `dpstrf`).
#[cfg(feature = "blas")]
#[derive(Debug, Clone)]
pub struct PivotedCholesky {
    /// Upper-triangular factor — `R[..rank, ..rank] · R[..rank, ..rank]' =
    /// A[piv[..rank], piv[..rank]]`. Trailing block is **garbage**, do not use.
    pub r: Array2<f64>,
    /// 0-based pivot permutation. `piv[k]` is the original column index of
    /// row/col `k` in the factored permuted matrix.
    pub piv: Vec<usize>,
    /// Numerical rank as reported by `dpstrf` after thresholding small pivots
    /// at `tol`. Equal to `n` for full-rank inputs.
    pub rank: usize,
}

/// Pivoted Cholesky factorisation of a symmetric (possibly rank-deficient)
/// matrix via LAPACK `dpstrf`.
///
/// Computes `P' · A · P = R' · R` where `P` is a permutation matrix and `R`
/// is upper triangular. The pivot is chosen to maximise the diagonal pivot at
/// every step — the same rule mgcv invokes through `chol(... pivot=TRUE)`
/// (R/fast-REML.r:1606).
///
/// ## Why pivoted (vs plain Cholesky + ridge)
///
/// Plain `chol` fails the instant a leading principal minor is non-positive.
/// On indefinite `X'WX + S(ρ)` matrices that arise when the scat/InvGauss
/// IRLS uses observed-info weights, this triggers a numerical breakdown
/// even though the system is well-defined in the rank-deficient sense.
/// `dpstrf` walks the diagonal in pivot-decreasing order, stops when the
/// remaining pivots drop below `tol`, and exposes the working rank — which
/// gives the caller a clean drop-of-rank-deficient-terms recovery path
/// (mgcv R/fast-REML.r:1610-1613).
///
/// ## Arguments
///
/// * `a` — input matrix (square, symmetric; only upper triangle is read).
///   Must be `n × n` for some `n ≥ 1`.
/// * `tol` — pivot tolerance. Pass `tol < 0` to let LAPACK choose
///   `tol = n · ε · max_pivot` (R's `chol` default; recommended). Pass
///   `tol = 0.0` to factor down to the first zero pivot only. Strictly
///   positive values give a custom rank-revealing threshold.
///
/// ## Returns
///
/// A [`PivotedCholesky`] with the upper-triangular factor, 0-based pivot
/// permutation, and numerical rank. See struct doc for which entries of
/// `R` are valid.
///
/// ## Errors
///
/// * `DimensionMismatch` — `a` is not square.
/// * `SingularMatrix` — LAPACK reported `info < 0` (illegal argument; should
///   never happen if `a` is square and `tol` is sane).
#[cfg(feature = "blas")]
pub fn pivoted_cholesky(a: &Array2<f64>, tol: f64) -> Result<PivotedCholesky> {
    use std::os::raw::{c_char, c_int};

    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square for pivoted Cholesky".to_string(),
        ));
    }
    if n == 0 {
        return Ok(PivotedCholesky {
            r: Array2::<f64>::zeros((0, 0)),
            piv: Vec::new(),
            rank: 0,
        });
    }

    // LAPACK works in column-major. We supply a column-major buffer of the
    // upper-triangle of A (only the upper triangle is read with UPLO='U').
    //
    // ndarray's default is row-major: A[i,j] is at offset i·n + j. We want
    // column-major A[i,j] at offset i + j·n. We materialise an owned vec.
    let mut a_col: Vec<f64> = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            a_col.push(a[[i, j]]);
        }
    }
    let mut piv_lapack: Vec<c_int> = vec![0; n];
    let mut rank_out: c_int = 0;
    let mut info: c_int = 0;
    // `dpstrf` workspace: 2n doubles.
    let mut work: Vec<f64> = vec![0.0; 2 * n];

    let uplo: c_char = b'U' as c_char;
    let n_i: c_int = n as c_int;
    let lda: c_int = n as c_int;
    let tol_in: f64 = tol;

    // SAFETY: all pointers refer to live owned buffers of the correct length;
    // LAPACK is single-threaded for one call and only reads `UPLO=U` triangle
    // and writes to `A`, `piv_lapack`, `rank_out`, `info`, `work`.
    unsafe {
        dpstrf_(
            &uplo,
            &n_i,
            a_col.as_mut_ptr(),
            &lda,
            piv_lapack.as_mut_ptr(),
            &mut rank_out,
            &tol_in,
            work.as_mut_ptr(),
            &mut info,
        );
    }

    if info < 0 {
        return Err(GAMError::InvalidParameter(format!(
            "dpstrf: illegal argument at position {} (info={})",
            -info, info
        )));
    }
    // info > 0 means rank-deficient (the i-th leading minor was not PD).
    // That's fine — `rank_out` reflects the working rank. Caller must drop
    // the trailing columns of R (we leave them in place; doc says
    // they're garbage).

    // Convert back to row-major `Array2<f64>`.
    let mut r = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        for i in 0..n {
            // Zero out the strictly lower-triangular part; LAPACK writes the
            // upper triangle of R to the upper triangle of A and leaves the
            // strict lower part as scratch.
            if i <= j {
                r[[i, j]] = a_col[i + j * n];
            }
        }
    }

    let piv: Vec<usize> = piv_lapack
        .iter()
        .map(|&k| (k as usize).saturating_sub(1))
        .collect();

    Ok(PivotedCholesky {
        r,
        piv,
        rank: rank_out as usize,
    })
}

/// Solve `(A_{piv, piv}) · x = b` using a [`PivotedCholesky`] factor, dropping
/// rank-deficient trailing rows.
///
/// Performs the mgcv `Sl.fitChol` solve recipe (R/fast-REML.r:1615):
///
/// ```text
///   x[piv] = backsolve(R, forwardsolve(R', b[piv]))    (for piv[..rank])
///   x[piv[rank..]] = 0                                 (drop rank-deficient terms)
/// ```
///
/// where `R` is the rank-`rank` upper-triangular factor returned by
/// [`pivoted_cholesky`]. The remaining `n − rank` entries of `x` (those
/// indexed by `piv[rank..]`) are explicitly set to zero — mirroring mgcv's
/// "drop rank deficient terms" branch.
///
/// ## Arguments
///
/// * `pc` — pivoted Cholesky factor of `A` from [`pivoted_cholesky`].
/// * `b` — right-hand side vector, length `n`.
///
/// ## Returns
///
/// `x` of length `n`. If `pc.rank == n` this is the full-rank solution; if
/// `pc.rank < n` the components indexed by `pc.piv[pc.rank..]` are zero.
#[cfg(feature = "blas")]
pub fn pivoted_cholesky_solve(pc: &PivotedCholesky, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = pc.r.nrows();
    if b.len() != n {
        return Err(GAMError::DimensionMismatch(format!(
            "RHS length {} does not match factor dim {}",
            b.len(),
            n
        )));
    }
    let r_eff = pc.rank;
    let mut x = Array1::<f64>::zeros(n);
    if r_eff == 0 {
        return Ok(x);
    }

    // Permute RHS: b_perm[k] = b[piv[k]] for k in 0..rank.
    let mut bp = Array1::<f64>::zeros(r_eff);
    for k in 0..r_eff {
        bp[k] = b[pc.piv[k]];
    }

    // Forward solve R' · y = b_perm  (R' is lower-triangular, size rank×rank,
    // taken from R[..rank, ..rank]).
    let mut y = Array1::<f64>::zeros(r_eff);
    for i in 0..r_eff {
        let mut s = bp[i];
        for j in 0..i {
            s -= pc.r[[j, i]] * y[j];
        }
        let diag = pc.r[[i, i]];
        if diag.abs() < f64::MIN_POSITIVE {
            return Err(GAMError::SingularMatrix);
        }
        y[i] = s / diag;
    }

    // Back-solve R · z = y.
    let mut z = Array1::<f64>::zeros(r_eff);
    for i in (0..r_eff).rev() {
        let mut s = y[i];
        for j in (i + 1)..r_eff {
            s -= pc.r[[i, j]] * z[j];
        }
        let diag = pc.r[[i, i]];
        if diag.abs() < f64::MIN_POSITIVE {
            return Err(GAMError::SingularMatrix);
        }
        z[i] = s / diag;
    }

    // Unpermute: x[piv[k]] = z[k]. The remaining entries (piv[rank..]) are
    // left at zero, matching mgcv's drop-rank-deficient branch.
    for k in 0..r_eff {
        x[pc.piv[k]] = z[k];
    }

    Ok(x)
}

/// Compute `R⁻¹` for the (truncated) upper-triangular factor in a pivoted
/// Cholesky. Returns an `(rank × rank)` matrix.
///
/// Used by [`pivoted_cholesky_inverse`] to form `PP[piv, piv] = R⁻¹ · R⁻¹'`
/// (mgcv's `chol2inv(R)` at R/fast-REML.r:1625).
#[cfg(feature = "blas")]
fn invert_upper_triangular(r: &Array2<f64>, rank: usize) -> Result<Array2<f64>> {
    if rank == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let mut inv = Array2::<f64>::zeros((rank, rank));
    // Solve R · X = I column by column via back-substitution.
    for col in 0..rank {
        let mut x = Array1::<f64>::zeros(rank);
        // RHS is e_col.
        for i in (0..rank).rev() {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for j in (i + 1)..rank {
                s -= r[[i, j]] * x[j];
            }
            let diag = r[[i, i]];
            if diag.abs() < f64::MIN_POSITIVE {
                return Err(GAMError::SingularMatrix);
            }
            x[i] = s / diag;
        }
        for i in 0..rank {
            inv[[i, col]] = x[i];
        }
    }
    Ok(inv)
}

/// Reconstruct `A⁻¹` from a [`PivotedCholesky`] factor, zero-padding the
/// rank-deficient null-space (mgcv R/fast-REML.r:1625: `PP[piv, piv] <- chol2inv(R)`).
///
/// Returns an `n × n` symmetric positive-semi-definite matrix `PP` such that:
///
/// * If `pc.rank == n` then `A · PP = I` to machine precision.
/// * If `pc.rank < n` then `PP[piv[..rank], piv[..rank]] = (R'R)⁻¹` and the
///   trailing rows/cols (those indexed by `piv[rank..]`) are zero.
#[cfg(feature = "blas")]
pub fn pivoted_cholesky_inverse(pc: &PivotedCholesky) -> Result<Array2<f64>> {
    let n = pc.r.nrows();
    let r_eff = pc.rank;
    let mut pp = Array2::<f64>::zeros((n, n));
    if r_eff == 0 {
        return Ok(pp);
    }
    let r_inv = invert_upper_triangular(&pc.r, r_eff)?;
    // (R'R)⁻¹ = R⁻¹ · R⁻¹' (because R is upper triangular).
    let inv = r_inv.dot(&r_inv.t());
    for a in 0..r_eff {
        for b in 0..r_eff {
            pp[[pc.piv[a], pc.piv[b]]] = inv[[a, b]];
        }
    }
    Ok(pp)
}

/// Log-determinant of an SPSD matrix from its pivoted-Cholesky factor.
///
/// Returns `2 · Σ_{k=0..rank} log(R[k, k])`. When `pc.rank < n`, the trailing
/// null-space contributes zero (since the null pivots are below `tol` and
/// already excluded). This matches mgcv R/fast-REML.r:1627
/// (`ldetXXS <- 2*sum(log(diag(R))+log(d[piv]))` — modulo the per-row
/// preconditioner `d`, which lives at the call site).
#[cfg(feature = "blas")]
pub fn pivoted_cholesky_log_det(pc: &PivotedCholesky) -> f64 {
    let mut s = 0.0;
    for k in 0..pc.rank {
        let v = pc.r[[k, k]];
        if v > 0.0 {
            s += v.ln();
        }
    }
    2.0 * s
}

/// Solve linear system Ax = b
/// Uses BLAS/LAPACK when available for large matrices (n >= 1000)
/// Falls back to Gaussian elimination for small matrices (BLAS overhead dominates for n < 1000)
pub fn solve(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        // BLAS crossover point is around n=1000
        // Below this, Gaussian elimination is faster due to BLAS overhead
        if n >= 1000 {
            solve_blas(a, b)
        } else {
            solve_gaussian(a, b)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        // Fall back to Gaussian elimination
        solve_gaussian(a, b)
    }
}

#[cfg(feature = "blas")]
fn solve_blas(a: Array2<f64>, b: Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square and match RHS".to_string(),
        ));
    }

    // Use general LU solver via Solve trait
    // In ndarray-linalg 0.17, solve() works directly with 1D arrays
    match Solve::solve(&a, &b) {
        Ok(x) => Ok(x),
        Err(_) => Err(GAMError::SingularMatrix),
    }
}

// Always available for hybrid BLAS approach (used for small matrices even when BLAS is enabled)
fn solve_gaussian(mut a: Array2<f64>, mut b: Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square and match RHS".to_string(),
        ));
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = a[[k, k]].abs();

        for i in (k + 1)..n {
            let val = a[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Use more relaxed threshold for ill-conditioned penalized systems
        // mgcv uses even more relaxed thresholds with ridge regularization
        if max_val < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = a.row(k).to_owned();
            let max_row = a.row(max_idx).to_owned();

            for j in 0..n {
                a[[k, j]] = max_row[j];
                a[[max_idx, j]] = temp_row[j];
            }
            b.swap(k, max_idx);
        }

        // Eliminate (optimized but safe)
        let pivot = a[[k, k]];
        let pivot_row = a.row(k).to_owned(); // Cache pivot row

        for i in (k + 1)..n {
            let factor = a[[i, k]] / pivot;

            // Update row i using cached pivot row
            for j in (k + 1)..n {
                a[[i, j]] -= factor * pivot_row[j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += a[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / a[[i, i]];
    }

    Ok(x)
}

/// Compute matrix determinant
/// Uses BLAS/LAPACK for large matrices (n >= 1000), LU decomposition for small matrices
/// log|det(A)| for a symmetric matrix A, allowing indefinite spectra.
///
/// Uses symmetric eigendecomposition and returns `Σ log|λ_i|`, dropping
/// eigenvalues below `max|λ| · 1e-14` as numerical zero. Required for mgcv
/// parity in non-canonical-link GLMs (e.g. InvGauss + log): Newton weights
/// `w = wf · α` can be negative, making `H = X'WX + S` indefinite. mgcv
/// returns this exact quantity in `oo$rank.tol` (gdi.c:2841) via its
/// pivoted-QR `neg_w` handling.
#[cfg(feature = "blas")]
pub fn log_abs_det_symmetric(a: &Array2<f64>) -> Result<f64> {
    use ndarray_linalg::{Eigh, UPLO};
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }
    let (eigvals, _) = a
        .eigh(UPLO::Upper)
        .map_err(|_| GAMError::SingularMatrix)?;
    let max_abs = eigvals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    let zero_thresh = max_abs * 1e-14;
    let mut log_abs_det = 0.0;
    for &lam in eigvals.iter() {
        let a = lam.abs();
        if a > zero_thresh {
            log_abs_det += a.ln();
        }
    }
    Ok(log_abs_det)
}

pub fn determinant(a: &Array2<f64>) -> Result<f64> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        if n >= 1000 {
            determinant_blas(a)
        } else {
            determinant_lu(a)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        determinant_lu(a)
    }
}

#[cfg(feature = "blas")]
fn determinant_blas(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    Determinant::det(&a.to_owned()).map_err(|_| GAMError::SingularMatrix)
}

// Always available for hybrid BLAS approach
fn determinant_lu(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mut lu = a.clone();
    let mut sign = 1.0;

    // LU decomposition with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = lu[[k, k]].abs();

        for i in (k + 1)..n {
            let val = lu[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < 1e-15 {
            return Ok(0.0); // Singular matrix has det = 0
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = lu.row(k).to_owned();
            let max_row = lu.row(max_idx).to_owned();

            for j in 0..n {
                lu[[k, j]] = max_row[j];
                lu[[max_idx, j]] = temp_row[j];
            }
            sign = -sign;
        }

        // Eliminate (safe version with pivot row caching)
        let pivot = lu[[k, k]];
        let pivot_row = lu.row(k).to_owned();

        for i in (k + 1)..n {
            lu[[i, k]] /= pivot;
            let multiplier = lu[[i, k]];

            // Update row i
            for j in (k + 1)..n {
                lu[[i, j]] -= multiplier * pivot_row[j];
            }
        }
    }

    // Determinant is product of diagonal elements
    let mut det = sign;
    for i in 0..n {
        det *= lu[[i, i]];
    }

    Ok(det)
}

/// Compute matrix inverse
/// Uses BLAS/LAPACK for large matrices (n >= 1000), Gauss-Jordan for small matrices
pub fn inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    #[cfg(feature = "blas")]
    {
        let n = a.nrows();
        if n >= 1000 {
            inverse_blas(a)
        } else {
            inverse_gauss_jordan(a)
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        inverse_gauss_jordan(a)
    }
}

#[cfg(feature = "blas")]
fn inverse_blas(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    Inverse::inv(&a.to_owned()).map_err(|_| GAMError::SingularMatrix)
}

// Always available for hybrid BLAS approach
fn inverse_gauss_jordan(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GAMError::DimensionMismatch(
            "Matrix must be square".to_string(),
        ));
    }

    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I] (safe version using slicing)
    aug.slice_mut(s![.., 0..n]).assign(a);
    for i in 0..n {
        aug[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val = aug[[k, k]].abs();

        for i in (k + 1)..n {
            let val = aug[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Use more relaxed threshold for numerical stability
        if max_val < 1e-14 {
            return Err(GAMError::SingularMatrix);
        }

        // Swap rows (safe version using row cloning)
        if max_idx != k {
            let temp_row = aug.row(k).to_owned();
            let max_row = aug.row(max_idx).to_owned();

            for j in 0..(2 * n) {
                aug[[k, j]] = max_row[j];
                aug[[max_idx, j]] = temp_row[j];
            }
        }

        // Scale pivot row (safe)
        let pivot = aug[[k, k]];
        let inv_pivot = 1.0 / pivot;
        for j in 0..(2 * n) {
            aug[[k, j]] *= inv_pivot;
        }

        // Eliminate column (safe with row caching)
        let pivot_row = aug.row(k).to_owned();

        for i in 0..n {
            if i != k {
                let factor = aug[[i, k]];
                for j in 0..(2 * n) {
                    aug[[i, j]] -= factor * pivot_row[j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Apply sum-to-zero identifiability constraint using simplified QR approach
///
/// This implements mgcv's approach to absorbing identifiability constraints
/// into the parameterization. For sum-to-zero constraint Σf(xi) = 0, we:
/// 1. Create constraint matrix C = [1, 1, ..., 1]^T / sqrt(k)  (normalized)
/// 2. Build orthogonal complement using Gram-Schmidt
/// 3. Return Q (k x k-1) matrix where columns are orthonormal basis for constraint complement
///
/// The constrained basis is then: X_constrained = X * Q
/// The constrained penalty is: S_constrained = Q^T * S * Q
///
/// # Arguments
/// * `k` - Number of basis functions (before constraint)
///
/// # Returns
/// * Q matrix (k x k-1) with orthonormal columns orthogonal to [1,1,...,1]
pub fn sum_to_zero_constraint_matrix(k: usize) -> Result<Array2<f64>> {
    // Build orthonormal basis for the orthogonal complement of [1,1,...,1]
    // Start with standard basis vectors and orthogonalize against [1,1,...,1]/sqrt(k)

    let mut q = Array2::zeros((k, k - 1));
    let sqrt_k = (k as f64).sqrt();

    // For each basis vector (except the first which is [1,1,...,1])
    for j in 0..(k - 1) {
        // Start with standard basis vector e_{j+1} (0,0,...,1,0,...,0)
        let mut v = Array1::zeros(k);
        v[j + 1] = 1.0;

        // Subtract projection onto [1,1,...,1]/sqrt(k)
        // projection = (v · [1/sqrt(k),...,1/sqrt(k)]) * [1/sqrt(k),...,1/sqrt(k)]
        let dot_with_ones = v.sum() / sqrt_k; // v · [1/sqrt(k),...]
        for i in 0..k {
            v[i] -= dot_with_ones / sqrt_k; // subtract projection
        }

        // Orthogonalize against previously computed columns
        for prev_j in 0..j {
            let prev_col = q.column(prev_j);
            let dot: f64 = v.iter().zip(prev_col.iter()).map(|(a, b)| a * b).sum();
            for i in 0..k {
                v[i] -= dot * prev_col[i];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(GAMError::SingularMatrix);
        }

        for i in 0..k {
            q[[i, j]] = v[i] / norm;
        }
    }

    Ok(q)
}

/// Build the pc-anchoring constraint matrix Q (k × k-1).
///
/// Given `b = B(pc)` ∈ ℝ^k (basis row evaluated at the point-constraint `pc`),
/// Q spans the null space of `bᵀ` so that `b · Q β̃ = 0` for any β̃.
/// This enforces `f(pc) = 0` identically.
///
/// Construction via Householder reflector (same primitive used by
/// `sum_to_zero_constraint_matrix`):
///
/// 1. Normalise `b` → `b̂ = b / ‖b‖` (first column of the full Q factor).
/// 2. Build the Householder vector `v` that maps `b̂ → ±e_0`.
/// 3. The last k-1 columns of the full Q factor are the null-space basis.
///
/// # Arguments
/// * `basis_at_pc` - Row vector `B(pc)` of length k
///
/// # Returns
/// * Q matrix (k × k-1) with orthonormal columns spanning null(bᵀ)
pub fn pc_constraint_matrix(basis_at_pc: &Array1<f64>) -> Result<Array2<f64>> {
    let k = basis_at_pc.len();
    if k <= 1 {
        return Err(GAMError::SingularMatrix);
    }
    let b_norm: f64 = basis_at_pc.iter().map(|x| x * x).sum::<f64>().sqrt();
    if b_norm < 1e-10 {
        // Basis is zero at pc — fall back to dropping last column (identity-like).
        // This is degenerate but avoids panicking.
        let mut z = Array2::<f64>::zeros((k, k - 1));
        for j in 0..(k - 1) {
            z[[j, j]] = 1.0;
        }
        return Ok(z);
    }

    // Normalised basis vector: b̂ = b / ‖b‖
    // Build Householder reflector that maps b̂ → ±e_0.
    // v = b̂ ± e_0,  H = I - 2 v vᵀ / ‖v‖²
    // Columns 1..k of H span null(b̂ᵀ) = null(bᵀ).
    let mut b_hat = basis_at_pc.clone();
    for x in b_hat.iter_mut() {
        *x /= b_norm;
    }
    let sign = if b_hat[0] >= 0.0 { 1.0 } else { -1.0 };
    let mut v = b_hat.clone();
    v[0] += sign; // v = b̂ + sign * e_0
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();

    let mut q = Array2::<f64>::zeros((k, k - 1));
    // j-th column of H (for j = 1..k-1) = e_j - (2 v_j / ‖v‖²) v
    // stored as (j-1)-th column of Q.
    for j in 1..k {
        let coef = 2.0 * v[j] / v_norm_sq;
        for i in 0..k {
            let e_ij = if i == j { 1.0 } else { 0.0 };
            q[[i, j - 1]] = e_ij - coef * v[i];
        }
    }

    Ok(q)
}

/// Apply identifiability constraint to penalty matrix
///
/// Given penalty matrix S (k x k) and constraint matrix Q (k x k-1),
/// compute: S_constrained = Q^T * S * Q
///
/// # Arguments
/// * `penalty` - Original penalty matrix (k x k)
/// * `q_matrix` - Constraint matrix from QR decomposition (k x k-1)
///
/// # Returns
/// * Constrained penalty matrix (k-1 x k-1)
pub fn apply_constraint_to_penalty(
    penalty: &Array2<f64>,
    q_matrix: &Array2<f64>,
) -> Result<Array2<f64>> {
    // S_constrained = Q^T * S * Q
    let s_q = penalty.dot(q_matrix);
    let constrained_penalty = q_matrix.t().dot(&s_q);

    Ok(constrained_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_solve() {
        let a = Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 2.0])
            .unwrap();

        let b = Array1::from_vec(vec![4.0, 6.0, 5.0]);

        let x = solve(a.clone(), b.clone()).unwrap();

        // Check Ax = b
        let ax = a.dot(&x);
        for i in 0..3 {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_determinant() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let det = determinant(&a).unwrap();
        assert_abs_diff_eq!(det, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse() {
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();

        let inv = inverse(&a).unwrap();
        let product = a.dot(&inv);

        // Check that A * A^(-1) = I
        assert_abs_diff_eq!(product[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(product[[1, 0]], 0.0, epsilon = 1e-10);
    }

    /// R8: pivoted Cholesky factor of a well-conditioned SPD matrix must
    /// reconstruct the original to machine precision after permutation.
    #[cfg(feature = "blas")]
    #[test]
    fn pivoted_chol_spd_full_rank() {
        // SPD 4×4 (Gram of a random-ish matrix).
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                4.0, 1.0, 0.5, 0.2, 1.0, 3.0, 0.7, 0.1, 0.5, 0.7, 5.0, 0.4, 0.2, 0.1, 0.4, 2.5,
            ],
        )
        .unwrap();
        let pc = pivoted_cholesky(&a, -1.0).unwrap();
        assert_eq!(pc.rank, 4, "full-rank SPD must be reported rank=n");
        // Reconstruct A[piv, piv] = R'R.
        let rt_r = pc.r.t().dot(&pc.r);
        for i in 0..4 {
            for j in 0..4 {
                let want = a[[pc.piv[i], pc.piv[j]]];
                assert!(
                    (rt_r[[i, j]] - want).abs() < 1e-10,
                    "R'R[{},{}] = {}, want A[piv[{}],piv[{}]] = {}",
                    i,
                    j,
                    rt_r[[i, j]],
                    i,
                    j,
                    want
                );
            }
        }
    }

    /// R8: pivoted Chol solve agrees with `solve` on a well-conditioned SPD.
    #[cfg(feature = "blas")]
    #[test]
    fn pivoted_chol_solve_matches_solve() {
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                4.0, 1.0, 0.5, 0.2, 1.0, 3.0, 0.7, 0.1, 0.5, 0.7, 5.0, 0.4, 0.2, 0.1, 0.4, 2.5,
            ],
        )
        .unwrap();
        let b = Array1::from_vec(vec![1.0, -2.0, 0.5, 3.0]);
        let x_solve = solve(a.clone(), b.clone()).unwrap();
        let pc = pivoted_cholesky(&a, -1.0).unwrap();
        let x_chol = pivoted_cholesky_solve(&pc, &b).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(x_solve[i], x_chol[i], epsilon = 1e-10);
        }
    }

    /// R8: pivoted Chol inverse matches `inverse` on a well-conditioned SPD.
    #[cfg(feature = "blas")]
    #[test]
    fn pivoted_chol_inverse_matches_inverse() {
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                4.0, 1.0, 0.5, 0.2, 1.0, 3.0, 0.7, 0.1, 0.5, 0.7, 5.0, 0.4, 0.2, 0.1, 0.4, 2.5,
            ],
        )
        .unwrap();
        let inv_ref = inverse(&a).unwrap();
        let pc = pivoted_cholesky(&a, -1.0).unwrap();
        let inv_chol = pivoted_cholesky_inverse(&pc).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(inv_ref[[i, j]], inv_chol[[i, j]], epsilon = 1e-10);
            }
        }
        // Sanity: log|A| from pivoted Chol vs direct determinant.
        let ldet_chol = pivoted_cholesky_log_det(&pc);
        let ldet_ref = determinant(&a).unwrap().ln();
        assert_abs_diff_eq!(ldet_chol, ldet_ref, epsilon = 1e-10);
    }

    /// R8: rank-deficient symmetric matrix (one zero eigenvalue) is detected,
    /// and the solve sets the null direction to zero rather than blowing up.
    #[cfg(feature = "blas")]
    #[test]
    fn pivoted_chol_rank_deficient() {
        // Rank-2 symmetric in R^3: A = v_1 v_1' + v_2 v_2', explicit null
        // direction v_3 = (1,1,1)/sqrt(3) (so we KNOW one zero eigenvalue).
        let v1 = ndarray::arr1(&[1.0, -1.0, 0.0]);
        let v2 = ndarray::arr1(&[0.5, 0.5, -1.0]);
        let mut a = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                a[[i, j]] = v1[i] * v1[j] + v2[i] * v2[j];
            }
        }
        // R's default tolerance picks this up cleanly.
        let pc = pivoted_cholesky(&a, -1.0).unwrap();
        assert_eq!(pc.rank, 2, "rank-2 input must be reported rank=2");
        // The rank-2 prefix must still reconstruct A[piv[..2], piv[..2]].
        // Build R[..2,..2]'·R[..2,..2] and compare.
        let r2 = pc.r.slice(s![..2, ..2]).to_owned();
        let rtr = r2.t().dot(&r2);
        for i in 0..2 {
            for j in 0..2 {
                let want = a[[pc.piv[i], pc.piv[j]]];
                assert!(
                    (rtr[[i, j]] - want).abs() < 1e-10,
                    "rank-deficient prefix mismatch at ({},{})",
                    i,
                    j
                );
            }
        }
    }
}
