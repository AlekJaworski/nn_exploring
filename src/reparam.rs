//! Reparametrization port — mgcv's `get_stableS` similarity transform.
//!
//! Mirrors `gdi.c:550-792` of mgcv (C source) and the R wrapper
//! `gam.reparam` at `R/gam.fit3.r:9-63`. Given square-root penalty roots
//! `rs[i]` (each `q × rs_ncol[i]`) and smoothing parameters `sp[i]`,
//! produces an orthogonal `qs` such that `S = qs' · S₀ · qs` is
//! well-conditioned for downstream Cholesky / `log|S|` evaluation.
//! The component roots `rs[i]` are returned in the new basis. Optional
//! derivatives `det1 = ∂log|S|/∂log(sp)` and `det2 = ∂²log|S|/∂log(sp)²`
//! are produced when `deriv >= 1` or `deriv == 2`.
//!
//! Used by the outer Newton loop to rotate `X`, `β`, and the per-block
//! penalty roots into mgcv's stable similarity basis at every outer
//! iteration (`gam.fit3.r:144-170`).

use crate::Result;
use ndarray::{s, Array1, Array2};
#[cfg(feature = "blas")]
use ndarray_linalg::{Eigh, Inverse, UPLO};

/// Output of the mgcv reparametrization step.
///
/// `s = qs' · S₀ · qs`, `S₀ = Σᵢ spᵢ · rsᵢ · rsᵢ'` (the original total
/// penalty). `rs[i]` are the component roots in the new basis:
/// `S₀_new[i] = rs[i] · rs[i]'`. The R wrapper additionally builds a
/// column-separated square root `e` via diagonal preconditioning +
/// pivoted Cholesky; that step lives in [`build_e_from_s`] and the
/// top-level [`gam_reparam`] wrapper, mirroring `gam.fit3.r:43-50`.
#[derive(Debug, Clone)]
pub struct GamReparamResult {
    /// Similarity-transformed total penalty (q × q).
    pub s: Array2<f64>,
    /// Orthogonal transformation (q × q): `S = qs' · S₀ · qs`.
    pub qs: Array2<f64>,
    /// Component roots in the new basis: `rs[i]` is `q × rs_ncol[i]`.
    pub rs: Vec<Array2<f64>>,
    /// `log|s|` (the pseudo-determinant uses the rank-deficient
    /// extension, but for our use case `s` is full rank up to the null
    /// space induced by the input rS, which mgcv assumes lies in the
    /// total penalty's null space; see `gam.fit3.r:15`).
    pub det: f64,
    /// `∂log|s|/∂log(sp)` (length M), or `None` if `deriv == 0`.
    pub det1: Option<Array1<f64>>,
    /// `∂²log|s|/∂log(sp)²` (M × M, symmetric), or `None` if
    /// `deriv < 2`.
    pub det2: Option<Array2<f64>>,
    /// Reflects the input flag — kept on the result so callers don't
    /// have to track it separately.
    pub fixed_penalty: bool,
}

/// mgcv's defaults: `d_tol = ε^0.3`, `r_tol = ε^0.75` (see
/// `gam.fit3.r:32-34`).
pub fn default_tolerances() -> (f64, f64) {
    let eps = f64::EPSILON;
    (eps.powf(0.3), eps.powf(0.75))
}

/// Port of `get_stableS` (gdi.c:550-792). See module docs.
///
/// * `rs` — list of square-root penalty roots. When `fixed_penalty` is
///   true the last entry is the root of a fixed (sp = 1) component;
///   `rs.len() == sp.len() + 1` in that case, else `rs.len() == sp.len()`.
/// * `sp` — smoothing parameters in *linear* scale (R wrapper passes
///   `exp(log_sp)`; C parameter name is `sp`).
/// * `deriv` — 0, 1, or 2; orders of `det` derivatives to compute.
/// * `d_tol`, `r_tol` — mgcv tolerances (`ε^0.3` and `ε^0.75` defaults).
#[cfg(feature = "blas")]
pub fn gam_reparam_core(
    rs: &[Array2<f64>],
    sp: &[f64],
    deriv: u8,
    d_tol: f64,
    r_tol: f64,
    fixed_penalty: bool,
) -> Result<GamReparamResult> {
    assert!(!rs.is_empty(), "gam_reparam_core: rs must be non-empty");
    let q = rs[0].nrows();
    for (i, r) in rs.iter().enumerate() {
        assert_eq!(
            r.nrows(),
            q,
            "gam_reparam_core: rs[{}].nrows() = {}, expected {}",
            i,
            r.nrows(),
            q
        );
    }
    let m = sp.len();
    let mf = if fixed_penalty { m + 1 } else { m };
    assert_eq!(
        rs.len(),
        mf,
        "gam_reparam_core: rs.len() = {} but expected {} (= sp.len() + fixed_penalty)",
        rs.len(),
        mf
    );

    // spf: smoothing params extended by a 1.0 for the fixed component (gdi.c:589-595).
    let mut spf = Vec::with_capacity(mf);
    spf.extend_from_slice(sp);
    if fixed_penalty {
        spf.push(1.0);
    }

    // Working copy of rs in the q × rs_ncol[i] layout. mgcv mutates in
    // place — we mutate the local clone and return it.
    let mut rs_work: Vec<Array2<f64>> = rs.to_vec();
    let rs_ncol: Vec<usize> = rs.iter().map(|r| r.ncols()).collect();

    // Si[i] = rs[i] · rs[i]'  (gdi.c:605-608). Each Si starts q × q;
    // shrinks to (Q - r) × (Q - r) when its term is moved to the
    // sub-dominant set `gamma1`. We allocate fresh `Array2<f64>` each
    // iter to side-step C's in-place storage shrinkage trick.
    let mut si: Vec<Array2<f64>> = rs
        .iter()
        .map(|r| r.dot(&r.t()))
        .collect();

    // Active-set bookkeeping.
    let mut gamma = vec![true; mf]; // terms still to deal with
    let mut alpha = vec![false; mf]; // this-iter dominant set
    let mut gamma1 = vec![false; mf]; // next-iter active set
    let mut frob = vec![0.0_f64; mf];

    // Output state.
    let mut s: Array2<f64> = Array2::zeros((q, q));
    let mut qf: Array2<f64> = Array2::zeros((q, q));

    let mut k: usize = 0; // coefficients already locked into top-left block of S
    let mut q_active: usize = q; // active block size
    let mut iter: usize = 0;

    loop {
        iter += 1;

        // 2.1 — Frobenius norms over active terms (gdi.c:634-640).
        let mut max_frob = 0.0_f64;
        for i in 0..mf {
            if gamma[i] {
                frob[i] = frobenius_norm(&si[i]);
                let s = frob[i] * spf[i];
                if s > max_frob {
                    max_frob = s;
                }
            }
        }

        // 2.2 — Partition active set into dominant α and sub-dominant γ'
        // (gdi.c:641-653).
        let mut n_gamma1 = 0_usize;
        for i in 0..mf {
            if gamma[i] && frob[i] * spf[i] > max_frob * d_tol {
                alpha[i] = true;
                gamma1[i] = false;
            } else if gamma[i] {
                alpha[i] = false;
                gamma1[i] = true;
                n_gamma1 += 1;
            } else {
                alpha[i] = false;
                gamma1[i] = false;
            }
        }

        // 2.3 — Rank `r` of the dominant sum (gdi.c:655-669).
        let r: usize = if n_gamma1 > 0 {
            // sum_α Si[i] / frob[i] (unweighted by spf — gdi.c:660).
            let mut sb: Array2<f64> = Array2::zeros((q_active, q_active));
            for i in 0..mf {
                if alpha[i] && frob[i] > 0.0 {
                    let inv = 1.0 / frob[i];
                    sb.scaled_add(inv, &si[i]);
                }
            }
            // Eigenvalues (ascending).
            let (ev_asc, _) = sb
                .eigh(UPLO::Lower)
                .map_err(|e| crate::GAMError::InvalidParameter(format!("eigh failed: {:?}", e)))?;
            // r = number of eigenvalues at the top end above ev_max * r_tol.
            let ev_max = ev_asc[q_active - 1];
            let mut rr: usize = 1;
            while rr < q_active && ev_asc[q_active - rr - 1] > ev_max * r_tol {
                rr += 1;
            }
            rr
        } else {
            q_active
        };

        // 2.4 — Termination (gdi.c:672-685).
        if q_active == r {
            if iter == 1 {
                // Never entered the transform loop body; just sum Si's
                // weighted by spf, and Qf = I_q.
                s.fill(0.0);
                for i in 0..mf {
                    s.scaled_add(spf[i], &si[i]);
                }
                qf.fill(0.0);
                for i in 0..q {
                    qf[[i, i]] = 1.0;
                }
            }
            break;
        }

        // 2.5 — Dominant block eigendecomposition (gdi.c:687-694). Sum
        // weighted by spf (NOT by 1/frob this time).
        let mut sb: Array2<f64> = Array2::zeros((q_active, q_active));
        for i in 0..mf {
            if alpha[i] {
                sb.scaled_add(spf[i], &si[i]);
            }
        }
        // We want eigenvectors. ndarray-linalg returns ascending order;
        // flip to descending so U[:, 0..r] are the dominant ones.
        let (ev_asc, vec_asc) = sb
            .eigh(UPLO::Lower)
            .map_err(|e| crate::GAMError::InvalidParameter(format!("eigh failed: {:?}", e)))?;
        let u_desc = flip_columns(&vec_asc);
        let ev_desc = flip_array1(&ev_asc);

        // 2.6 — Update Qf (gdi.c:697-701).
        if iter == 1 {
            // Bootstrap: Qf = U (placed in the top-left q × q block,
            // which is everything when K = 0).
            qf.slice_mut(s![0..q, 0..q_active]).assign(&u_desc);
        } else {
            // Right-multiply Qf[:, K..K+Q] by U.
            let block = qf.slice(s![.., k..k + q_active]).dot(&u_desc);
            qf.slice_mut(s![.., k..k + q_active]).assign(&block);
        }

        // 2.7 — Sub-dominant Sg = Σ_{γ'} spf[i] · Si[i] (gdi.c:704-708).
        let mut sg: Array2<f64> = Array2::zeros((q_active, q_active));
        for i in 0..mf {
            if gamma1[i] {
                sg.scaled_add(spf[i], &si[i]);
            }
        }

        // 2.8 — Update S's upper-right K × Q strip (gdi.c:711-719). Only
        // runs after the first iter where K > 0.
        if k > 0 {
            let c_block = s.slice(s![0..k, k..k + q_active]).to_owned();
            let b_block = c_block.dot(&u_desc);
            s.slice_mut(s![0..k, k..k + q_active]).assign(&b_block);
            // Symmetric mirror: S[K..K+Q, 0..K] = B'  (mgcv writes both
            // halves explicitly at gdi.c:718).
            s.slice_mut(s![k..k + q_active, 0..k])
                .assign(&b_block.t());
        }

        // 2.9 — Lower-right Q × Q block: C = U' · Sg · U + diag(ev[:r])
        // (gdi.c:722-730).
        let b_mat = sg.dot(&u_desc); // Q × Q
        let mut c_mat = u_desc.t().dot(&b_mat); // Q × Q
        for i in 0..r {
            c_mat[[i, i]] += ev_desc[i];
        }
        s.slice_mut(s![k..k + q_active, k..k + q_active])
            .assign(&c_mat);

        // 2.10 — Rotate per-component roots in α and γ' (gdi.c:732-746).
        // Loop runs `k in 0..M` (NOT 0..Mf): the fixed-term root is
        // NOT mutated by intent (gdi.c:732 comment).
        for kk in 0..m {
            if !(alpha[kk] || gamma1[kk]) {
                continue;
            }
            let ncols = rs_ncol[kk];
            // Extract the Q × ncols submatrix at rows K..K+Q.
            let sub = rs_work[kk]
                .slice(s![k..k + q_active, 0..ncols])
                .to_owned();
            if alpha[kk] {
                // B = U' · sub, keeping only the first r rows (the
                // dominant subspace). Then rows K..K+r get B; rows
                // K+r..K+Q get zeroed.
                let b = u_desc.t().dot(&sub); // Q × ncols
                // Write back: rows K..K+r = first r rows of b; trailing
                // rows zero.
                rs_work[kk]
                    .slice_mut(s![k..k + r, 0..ncols])
                    .assign(&b.slice(s![0..r, 0..ncols]));
                for ii in (k + r)..(k + q_active) {
                    for jj in 0..ncols {
                        rs_work[kk][[ii, jj]] = 0.0;
                    }
                }
            } else {
                // gamma1[kk] — rotate all Q rows.
                let b = u_desc.t().dot(&sub);
                rs_work[kk]
                    .slice_mut(s![k..k + q_active, 0..ncols])
                    .assign(&b);
            }
        }

        // 2.11 — Shrink Si for sub-dominant terms: new Si = Un' · Si · Un
        // where Un = U[:, r..Q] is the trailing-column orthogonal
        // complement (gdi.c:748-754). Allocates a fresh Vec<Array2>
        // since dimensions change.
        let qr = q_active - r;
        let un = u_desc.slice(s![.., r..q_active]).to_owned(); // Q × (Q-r)
        let mut si_new: Vec<Array2<f64>> = Vec::with_capacity(mf);
        for i in 0..mf {
            if gamma1[i] {
                let tmp = un.t().dot(&si[i]); // (Q-r) × Q
                let new_block = tmp.dot(&un); // (Q-r) × (Q-r)
                si_new.push(new_block);
            } else {
                // Term is done; placeholder so indices line up. We never
                // read it again (gamma becomes gamma1 next iter).
                si_new.push(Array2::zeros((qr, qr)));
            }
        }
        si = si_new;

        // 2.12 — Advance counters (gdi.c:756-757).
        k += r;
        q_active = qr;
        for i in 0..mf {
            gamma[i] = gamma1[i];
        }
    }

    // ---- Post-loop: log|S| + derivatives (gdi.c:760-778) ----
    // One symmetric eigendecomposition powers both `det = log|S+|` and
    // (for `deriv >= 1`) the Moore-Penrose pseudo-inverse used in
    // `tr(S^+ · S_i)` for `det1` and the `(S^+ · S_i)` chain for `det2`.
    // mgcv's C uses `qr_ldet_inv` (pivoted QR) which handles indefinite
    // / rank-deficient `S` via pivoting; we mirror that robustness here
    // with a pseudo-inverse so callers don't have to pre-project rS to
    // the range space of the total penalty.
    let s_sym = symmetrise(&s);
    let (evs, evecs) = s_sym
        .eigh(UPLO::Lower)
        .map_err(|e| crate::GAMError::InvalidParameter(format!("eigh failed: {:?}", e)))?;
    let max_abs = evs.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let thresh = max_abs * 1e-14;
    let det: f64 = evs
        .iter()
        .filter(|&&x| x.abs() > thresh)
        .map(|x| x.abs().ln())
        .sum();

    let s_inv: Array2<f64> = if deriv >= 1 {
        let mut acc = Array2::<f64>::zeros((q, q));
        for (k, &ev) in evs.iter().enumerate() {
            if ev.abs() > thresh {
                let v_k = evecs.column(k);
                let inv = 1.0 / ev;
                for i in 0..q {
                    for j in 0..q {
                        acc[[i, j]] += v_k[i] * v_k[j] * inv;
                    }
                }
            }
        }
        acc
    } else {
        Array2::zeros((0, 0))
    };

    let (det1, det2) = if deriv >= 1 {
        let mut det1 = Array1::<f64>::zeros(m);
        for i in 0..m {
            // det1[i] = sp[i] · tr(B' · rs[i] · rs[i]') with B = S^{-1}
            //         = sp[i] · tr(S^{-1} · S_i)
            //         = sp[i] · sum( (rs[i]' · S^{-1} · rs[i]).diag )
            // Using the trBtAB formula (gdi.c:35): tr(B'AB) with
            // A = S^{-1}, B = rs[i].
            let tmp = s_inv.dot(&rs_work[i]); // q × rs_ncol[i]
            let prod = rs_work[i].t().dot(&tmp); // rs_ncol[i] × rs_ncol[i]
            let tr: f64 = (0..prod.nrows()).map(|d| prod[[d, d]]).sum();
            det1[i] = sp[i] * tr;
        }
        let det2 = if deriv >= 2 {
            // First produce S^{-1} · S_i = S^{-1} · rs[i] · rs[i]'  for
            // each i, store in `si_inv[i]` (mgcv stores in `Si`).
            let mut si_inv: Vec<Array2<f64>> = Vec::with_capacity(m);
            for i in 0..m {
                let tmp = s_inv.dot(&rs_work[i]); // q × rs_ncol[i]
                si_inv.push(tmp.dot(&rs_work[i].t())); // q × q
            }
            let mut d2 = Array2::<f64>::zeros((m, m));
            for i in 0..m {
                for j in i..m {
                    // tr(A B) = sum over k of (A B)[k,k] = sum_k Σ_l A[k,l] B[l,k]
                    // i.e. sum_{k,l} A[k,l] · B[l,k] = sum over all
                    // elements of A ⊙ B'.
                    let v = trace_ab(&si_inv[i], &si_inv[j]);
                    let val = -sp[i] * sp[j] * v;
                    d2[[i, j]] = val;
                    d2[[j, i]] = val;
                }
            }
            for i in 0..m {
                d2[[i, i]] += det1[i];
            }
            Some(d2)
        } else {
            None
        };
        (Some(det1), det2)
    } else {
        (None, None)
    };

    Ok(GamReparamResult {
        s: symmetrise(&s),
        qs: qf,
        rs: rs_work,
        det,
        det1,
        det2,
        fixed_penalty,
    })
}

// --------------------------------------------------------------------
// Local helpers
// --------------------------------------------------------------------

#[inline]
fn frobenius_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[inline]
fn trace_ab(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // tr(A · B) where A and B are square and the same shape.
    let n = a.nrows();
    debug_assert_eq!(a.ncols(), b.nrows());
    debug_assert_eq!(b.ncols(), n);
    let mut s = 0.0_f64;
    for k in 0..n {
        for l in 0..a.ncols() {
            s += a[[k, l]] * b[[l, k]];
        }
    }
    s
}

#[inline]
fn flip_columns(a: &Array2<f64>) -> Array2<f64> {
    let n = a.ncols();
    let mut out = Array2::<f64>::zeros((a.nrows(), n));
    for j in 0..n {
        out.column_mut(j)
            .assign(&a.column(n - 1 - j));
    }
    out
}

#[inline]
fn flip_array1(a: &Array1<f64>) -> Array1<f64> {
    let n = a.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = a[n - 1 - i];
    }
    out
}

#[inline]
fn symmetrise(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut out = a.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

// --------------------------------------------------------------------
// Unit tests
// --------------------------------------------------------------------

#[cfg(all(test, feature = "blas"))]
mod tests {
    use super::*;
    use ndarray::array;

    /// Single-penalty identity: rs = I_q, sp = [λ]. S₀ = λ · I_q.
    /// log|S₀| = q · log λ, ∂log|S₀|/∂log λ = q.
    #[test]
    fn single_identity_penalty_one_smoothing_param() {
        let q = 4;
        let rs = vec![Array2::<f64>::eye(q)];
        let sp = vec![1.5_f64];
        let (d_tol, r_tol) = default_tolerances();
        let out = gam_reparam_core(&rs, &sp, 2, d_tol, r_tol, false).unwrap();

        // |S| = λ^q -> log|S| = q · log λ
        let expected_det = (q as f64) * sp[0].ln();
        assert!(
            (out.det - expected_det).abs() < 1e-12,
            "det={}, expected={}",
            out.det,
            expected_det
        );
        // ∂log|S|/∂log λ = q (since |S| = λ^q)
        let d1 = out.det1.unwrap();
        assert!((d1[0] - (q as f64)).abs() < 1e-12);
        // ∂²log|S|/∂log²λ = 0 for a single power
        let d2 = out.det2.unwrap();
        assert!(d2[[0, 0]].abs() < 1e-12);
    }

    /// Two disjoint identity penalties: rs[0] = e_1·e_1', rs[1] = (I - e_1·e_1').
    /// S₀ = λ_1 · e_1·e_1' + λ_2 · (I - e_1·e_1'); |S₀| = λ_1 · λ_2^{q-1}.
    #[test]
    fn two_disjoint_identity_penalties() {
        let q = 4;
        let mut r0 = Array2::<f64>::zeros((q, 1));
        r0[[0, 0]] = 1.0;
        let mut r1 = Array2::<f64>::zeros((q, q - 1));
        for j in 0..(q - 1) {
            r1[[j + 1, j]] = 1.0;
        }
        let rs = vec![r0, r1];
        let sp = vec![2.0_f64, 0.5_f64];
        let (d_tol, r_tol) = default_tolerances();
        let out = gam_reparam_core(&rs, &sp, 2, d_tol, r_tol, false).unwrap();

        // |S₀| = sp_1 · sp_2^{q-1}
        let expected_det = sp[0].ln() + ((q - 1) as f64) * sp[1].ln();
        assert!(
            (out.det - expected_det).abs() < 1e-12,
            "det={}, expected={}",
            out.det,
            expected_det
        );

        // det1[0] = sp_1 · tr(S^{-1} · S_1) = sp_1 · sp_1^{-1} = 1
        // det1[1] = sp_2 · tr(S^{-1} · S_2) = sp_2 · (q-1) · sp_2^{-1} = q - 1
        let d1 = out.det1.unwrap();
        assert!((d1[0] - 1.0).abs() < 1e-10, "d1[0]={}", d1[0]);
        assert!(
            (d1[1] - ((q - 1) as f64)).abs() < 1e-10,
            "d1[1]={}",
            d1[1]
        );
    }

    /// Reconstruction check: qs' · S₀ · qs ≈ s, on a small problem with
    /// two overlapping penalties so the rotation is non-trivial.
    #[test]
    fn similarity_transform_reconstructs_s0() {
        let q = 5;
        // Two penalty roots of different scales whose ranges overlap.
        let rs0 = array![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ];
        let rs1 = array![
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ];
        let rs = vec![rs0.clone(), rs1.clone()];
        let sp = vec![3.0_f64, 0.2_f64];

        let s0: Array2<f64> =
            sp[0] * rs[0].dot(&rs[0].t()) + sp[1] * rs[1].dot(&rs[1].t());
        let s0 = symmetrise(&s0);

        let (d_tol, r_tol) = default_tolerances();
        let out = gam_reparam_core(&rs, &sp, 0, d_tol, r_tol, false).unwrap();

        let lhs = out.qs.t().dot(&s0).dot(&out.qs);
        let mut max_diff = 0.0_f64;
        for i in 0..q {
            for j in 0..q {
                let d = (lhs[[i, j]] - out.s[[i, j]]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(
            max_diff < 1e-10,
            "qs' S0 qs ≠ s, max diff = {}",
            max_diff
        );
    }

    #[test]
    fn fixed_penalty_treats_last_as_sp1() {
        // 2 free + 1 fixed (with implicit sp = 1).
        let q = 3;
        let rs0 = array![[1.0_f64], [0.0], [0.0]];
        let rs1 = array![[0.0_f64], [1.0], [0.0]];
        let rs_fixed = array![[0.0_f64], [0.0], [1.0]];
        let rs = vec![rs0, rs1, rs_fixed];
        let sp = vec![4.0_f64, 0.5_f64];

        let (d_tol, r_tol) = default_tolerances();
        let out = gam_reparam_core(&rs, &sp, 1, d_tol, r_tol, true).unwrap();
        // |S₀| = sp_1 · sp_2 · 1.0
        let expected_det = sp[0].ln() + sp[1].ln();
        assert!(
            (out.det - expected_det).abs() < 1e-12,
            "det={}, expected={}",
            out.det,
            expected_det
        );
        // No det1 entry for the fixed component.
        let d1 = out.det1.unwrap();
        assert_eq!(d1.len(), 2);
        // det1[i] = 1.0 for each free identity penalty.
        assert!((d1[0] - 1.0).abs() < 1e-10);
        assert!((d1[1] - 1.0).abs() < 1e-10);
    }
}
