//! Path B / Step B3 — `Sl.fitChol` port: a closed-form fREML score / gradient /
//! Hessian / Newton-step evaluator on **cached** `X'WX` and `X'Wy`.
//!
//! ## What this is
//!
//! Direct Rust translation of mgcv's `Sl.fitChol` (R/fast-REML.r:1585) and its
//! sub-helpers `Sl.iftChol` (R/fast-REML.r:1405) and `d.detXXS`
//! (R/fast-REML.r:1329). Given pre-assembled `XX = X'WX (p×p)`, `f = X'Wy (p)`,
//! and `yy = y'Wy` plus a singleton-block penalty list, this function
//!
//!   1. Solves the penalised normal equations `(X'WX + S(ρ)) β̂ = f` directly.
//!   2. Evaluates the fREML score derivatives `∂V_R/∂ρ` and `∂²V_R/(∂ρ∂ρ')`
//!      via the Implicit Function Theorem on β̂ (i.e. `Sl.iftChol`) plus the
//!      `d.detXXS` trace-of-product expressions for `∂log|X'X+S|`.
//!   3. Forms the eigen-clamped Newton step `−H⁻¹·g` with magnitude cap.
//!
//! The kernel is the hot path of mgcv's `bgam.fitd` (R/bam.r:430) outer
//! `fREML + discrete=TRUE` loop and the foundation for B4's `fit_pirls_fastreml`.
//!
//! ## Score formula (Gaussian / scaled families)
//!
//! Letting `λ_k = exp(ρ_k)`, `S(ρ) = Σ λ_k S_k`, `φ = exp(log_φ)`, and
//! `β̂(ρ) = (X'WX + S)⁻¹ f`, the score evaluated is **`2·V_R`**:
//!
//! ```text
//!   2·V_R(ρ, log_φ) = log|X'WX + S| − log|S|₊
//!                    + (yy − β̂'f) / (φ·γ)
//!                    + (n_obs/γ − M_p) · log(φ)
//! ```
//!
//! mgcv's `Sl.fitChol` returns `reml1 = ∂V_R/∂ρ` and `reml2 = ∂²V_R/(∂ρ∂ρ')`
//! — i.e. **derivatives of `V_R`**, not of `2·V_R`. We follow the same
//! convention so the returned `grad` / `hess` are drop-in for a Newton step on
//! `V_R`. The full-`(m+1)` extension carrying the `log_φ` coordinate is
//! activated when `phi_fixed = false`.
//!
//! ## Constant offset vs the existing REML score
//!
//! Our `reml_criterion_multi_cached_mgcv_exact` returns `V_R` in the form
//! (mgcv gam.fit3.r:621, Gaussian/canonical case after collapsing `ls[1]`):
//!
//! ```text
//!   2·V_R^reml = D_p/σ² + (n − M_p)·log(2π·σ²) + log|H| − log|S|₊
//! ```
//!
//! whereas `Sl.fitChol`'s `2·V_R^fitChol = D_p/φ + (n − M_p)·log(φ) + log|H| − log|S|₊`
//! (with `γ = 1`, `D_p = yy − β'f`). At the SAME `σ² = φ`, the constant
//! offset is exactly `(n − M_p)·log(2π)` — it does NOT depend on ρ or φ, so
//! both scores agree on the location of the optimum and on all derivatives
//! wrt ρ. (Sanity test 3 verifies this numerically.)
//!
//! ## Sub-helpers (private to this module)
//!
//! * [`compute_sl_ift_chol`] — `dβ/dρ_k = −(X'X+S)⁻¹·λ_k S_k β` plus the
//!   `bSb1[k] = β'·λ_k S_k·β` and `rss2 / bSb2` quantities mgcv uses to form
//!   the second-derivative chain. Mirrors `Sl.iftChol`.
//! * [`compute_d_det_xxs`] — `d1[k] = λ_k tr(PP·S_k)` and
//!   `d2[i,j] = −λ_i λ_j tr(PP·S_i·PP·S_j) + δ_ij · d1[i]`. Mirrors
//!   `d.detXXS`.
//!
//! ## Numerical kernel: preconditioned pivoted Cholesky (R8)
//!
//! mgcv builds a preconditioned pivoted Cholesky of `D⁻¹(X'X+S)D⁻¹` and
//! forms `PP = (X'X+S)⁻¹` from its triangular factor (R/fast-REML.r:1606).
//! We follow the same recipe via LAPACK `dpstrf`
//! (see [`crate::linalg::pivoted_cholesky`]), which:
//!
//!   1. Pivots aggressively so the working rank is exposed when the
//!      penalised Hessian has near-null directions, rather than failing on
//!      a non-PD leading minor.
//!   2. Lets the caller **drop rank-deficient columns** and zero-pad the
//!      null-space coordinate (mgcv R/fast-REML.r:1610-1613 + 1615).
//!
//! On well-conditioned inputs the result is numerically identical to plain
//! `solve` + `inverse` (the previous implementation). The advantage is on
//! **indefinite** or rank-deficient inputs that arise when `bgam.fitd` swaps
//! Fisher → observed-info weights in scat / inverse-Gaussian IRLS: plain
//! LU + 1e-12 ridge returns non-finite β; pivoted Chol with rank truncation
//! returns the unique penalised-minimum-norm β̂ on the working subspace.
//!
//! ### Stability fallback
//!
//! If `pivoted_cholesky` reports `rank == 0` (numerically zero penalised
//! Hessian — never happens in practice with `S(ρ) > 0`) or fails to factor,
//! we fall back to the original `solve` + `inverse` path so the caller still
//! receives a finite Newton step.

use crate::block_penalty::BlockPenalty;
use crate::linalg::{inverse, pivoted_cholesky, solve, PivotedCholesky};
use crate::reml::compute_ldet_s_with_derivs;
use crate::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Result struct returned by [`compute_sl_fitchol_step`].
///
/// Mirrors the named list returned by mgcv's `Sl.fitChol`. `grad` / `hess` /
/// `step` have length `m` when `phi_fixed = true` and length `m+1` (with the
/// `log_φ` coordinate appended last) when `phi_fixed = false`.
#[derive(Debug, Clone)]
pub struct SlFitCholResult {
    /// β̂(ρ, φ) — solution of `(X'WX + S(ρ)) β = f`. Length `p`.
    pub beta: Array1<f64>,
    /// ∂V_R/∂ρ_k (and last entry ∂V_R/∂log_φ if `phi_fixed = false`).
    pub grad: Array1<f64>,
    /// ∂²V_R/(∂ρ ∂ρ'). Same dimension convention as `grad`.
    pub hess: Array2<f64>,
    /// Eigen-clamped Newton step `−H⁻¹·grad`, magnitude capped at 4
    /// (matches mgcv's stability heuristic).
    pub step: Array1<f64>,
    /// dβ/dρ — column `k` is `−λ_k (X'WX+S)⁻¹ S_k β`. Shape `(p, m)`.
    pub db: Array2<f64>,
    /// `(X'WX + S)⁻¹`. Used by B4's outer loop for the `log|X'X+S|` derivs
    /// across families; kept as a full inverse rather than a Cholesky factor
    /// to match `assemble_reml_system`'s convention elsewhere in `reml/`.
    pub pp: Array2<f64>,
    /// `log|S(ρ)|₊` evaluated at the supplied ρ.
    pub ldet_s: f64,
    /// `log|X'WX + S(ρ)|` evaluated at the supplied ρ.
    pub ldet_xxs: f64,
}

/// Compute the fREML score Newton step on a cached `(XX, f, yy)`.
///
/// See module-level docs for the math contract. This is the closed-form
/// `Sl.fitChol` evaluation used by mgcv's `bgam.fitd` outer fREML loop and the
/// B-task driver landing in B4.
///
/// # Arguments
///
/// * `sl` — singleton penalty blocks (one ρ per block).
/// * `xx` — cached `X'WX`, shape `(p, p)`.
/// * `f` — cached `X'Wy`, length `p`.
/// * `rho` — log-smoothing parameters, length `m == sl.len()`.
/// * `yy` — cached `y'Wy` (Σ w_i y_i²). Used only when `phi_fixed = false`.
/// * `log_phi` — log scale parameter. Ignored numerically for the gradient/
///   Hessian wrt ρ but enters the `D_p/φ` scaling.
/// * `phi_fixed` — `true` for known-scale families (Binomial/Poisson/NegBin)
///   and `false` for Gaussian/Gamma/etc where φ is jointly estimated. When
///   `false`, `grad` and `hess` are extended by one row/column for `log_φ`.
/// * `nobs` — effective sample size (sum of weights for Gaussian; n for
///   non-Gaussian).
/// * `mp` — penalty null-space dimension (M_p in mgcv's notation).
/// * `gamma` — γ correction factor (1.0 by default).
pub fn compute_sl_fitchol_step(
    sl: &[BlockPenalty],
    xx: ArrayView2<f64>,
    f: ArrayView1<f64>,
    rho: ArrayView1<f64>,
    yy: f64,
    log_phi: f64,
    phi_fixed: bool,
    nobs: f64,
    mp: usize,
    gamma: f64,
) -> Result<SlFitCholResult> {
    let p = xx.nrows();
    let m = sl.len();
    assert_eq!(xx.ncols(), p, "XX must be square");
    assert_eq!(f.len(), p, "f length must match XX dim");
    assert_eq!(
        rho.len(),
        m,
        "rho length must match number of penalty blocks"
    );

    // λ_k = exp(ρ_k)
    let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();

    // ===== Initial reparameterisation (mgcv `Sl.initial.repara`) =====
    //
    // For each singleton block carrying a cached `ReparaTransform`, rotate
    // `XX` and `f` into the per-block reparameterised basis at entry, and the
    // returned `β`, `db`, `pp` back into the original basis at exit. The
    // rotated linear system `(XX_rot + Σ λ_k S_k^rot) β_rot = f_rot` has
    // `S_k^rot` equal to a partial identity, which is dramatically better
    // conditioned for fREML than the raw-basis system.
    //
    // We work on owned copies so the caller's `xx`/`f` views remain
    // unchanged — `compute_sl_fitchol_step` is called inside outer loops
    // that reuse the cached primitives across multiple ρ trials.
    //
    // Multi-block note: blocks live on disjoint column ranges
    // `[start..stop]`, so rotations from different blocks commute. We loop
    // in any order without worrying about basis interactions.
    //
    // **Penalty rotation in-place**: in the reparameterised basis,
    // `S_k^rot = D_k' · S_k · D_k`, which by construction is a partial
    // identity (1's on the rank-r entries, 0's elsewhere). Instead of
    // re-deriving this every step, we form a fresh `sl_rot` view where
    // each block's matrix is replaced with that partial identity. The
    // `dot_vec`/`scaled_add_to`/`trace_product`/etc operations on
    // `sl_rot[k]` then naturally reflect the rotated penalty.
    let any_repara = sl.iter().any(|b| b.repara.is_some());

    // We always materialise `xx_work` and `f_work` as owned copies — the
    // unrotated branch then just clones the inputs and never touches the
    // owned data otherwise. This keeps the borrow-graph trivial (no
    // conditional lifetime juggling).
    let mut xx_work = xx.to_owned();
    let mut f_work = f.to_owned();
    let sl_rot_storage: Vec<BlockPenalty>;
    let sl_use: &[BlockPenalty] = if any_repara {
        for block in sl.iter() {
            block.rotate_xx_in_place(&mut xx_work);
            block.rotate_f_in_place(&mut f_work);
        }
        // Build the rotated penalty list. In the rotated basis,
        // `S_k^rot = D_k' · S_k · D_k` is (by construction of D_k) a partial
        // identity — diagonal with 1's at the rank-carrying entries and 0's
        // on the null. For the EIGEN-stabilised branch we precomputed the
        // basis so the 1's land at indices `0..rank` after the descending
        // reorder; for the DIAGONAL branch (where D is itself diagonal), the
        // 1's land at the original positions of nonzero diag entries, which
        // may not be contiguous. We compute `D' · S · D` explicitly to cover
        // both cases without assuming the index layout.
        let mut sl_rot_vec: Vec<BlockPenalty> = Vec::with_capacity(sl.len());
        for block in sl.iter() {
            if let Some(repara) = &block.repara {
                let rotated = repara.d_mat.t().dot(&block.block).dot(&repara.d_mat);
                let mut new_block = BlockPenalty::new(rotated, block.offset, block.total_size);
                // Carry the transform forward defensively (callers may inspect
                // `sl_use[k].repara`).
                new_block.repara = block.repara.clone();
                sl_rot_vec.push(new_block);
            } else {
                sl_rot_vec.push(block.clone());
            }
        }
        sl_rot_storage = sl_rot_vec;
        &sl_rot_storage
    } else {
        sl
    };
    let xx_view = xx_work.view();
    let f_view = f_work.view();

    // ===== 1) Assemble penalised Hessian A = X'WX + Σ λ_k S_k =====
    let mut a = xx_view.to_owned();
    for (lambda, pen) in lambdas.iter().zip(sl_use.iter()) {
        pen.scaled_add_to(&mut a, *lambda);
    }

    // ===== 2) Fast LU solve for β and PP = A⁻¹, with pivoted-Chol fallback =====
    //
    // The common production path is well-conditioned once the singleton
    // penalties have been reparameterised, and the old LU + tiny-ridge solve is
    // much cheaper than running LAPACK `dpstrf` on every outer iteration. Try
    // that path first and accept it only if every downstream quantity is finite.
    // Pathological observed-info systems still fall through to the mgcv-style
    // pivoted Cholesky kernel below.
    let (beta, pp, ldet_xxs) = match try_lu_ridge_solve(&a, &f_view.to_owned()) {
        Some(out) => (out.beta, out.pp, out.ldet_xxs),
        None => {
            // Mirrors mgcv `Sl.fitChol` (R/fast-REML.r:1601-1626):
            //
            //   1. Diagonal preconditioner `d[i] = sqrt(A[i,i])` (1 where A[i,i] ≤ 0).
            //   2. Pivoted Chol of `A_p = D⁻¹·A·D⁻¹` via LAPACK `dpstrf`.
            //   3. Drop rank-deficient trailing columns (mgcv lines 1610-1613).
            //   4. `β[piv] = R⁻¹·(R'⁻¹·(f[piv]/d[piv]))/d[piv]` (line 1615).
            //   5. `PP[piv, piv] = chol2inv(R)`, then `PP = D⁻¹·PP·D⁻¹` (lines 1623-1626).
            //   6. `ldet_xxs = 2·Σ(log R_kk + log d_piv_k)` (line 1627).
            //
            // The preconditioner brings the spectrum of `A_p` closer to 1, which is
            // what makes pivoted Chol numerically stable on the `(X'WX + S)` block-
            // diagonal structure (the penalty's null-space directions live on a few
            // rows, so the rank-revealing pivot pulls them to the tail).
            //
            // If `try_pivoted_chol_solve` returns `None` (rank == 0, or LAPACK
            // failure — never seen in practice for `S(ρ) > 0`), fall back to the
            // LU path even if it produced non-finite diagnostics so the caller's
            // candidate-rejection ladder can surface a precise failure.
            match try_pivoted_chol_solve(&a, &f_view.to_owned()) {
                Some(out) => (out.beta, out.pp, out.ldet_xxs),
                None => {
                    let out = lu_ridge_solve(&a, &f_view.to_owned())?;
                    (out.beta, out.pp, out.ldet_xxs)
                }
            }
        }
    };

    // ===== 3) log|S(ρ)|₊ and its ρ-derivatives (singleton blocks) =====
    // For singleton penalties (one ρ per block, no shared λ across blocks),
    // `ldet_s_d1[k] = rank_k` and `ldet_s_d2 = 0`. We rely on the shared
    // R3 helper for both byte-identical computation across reml/ paths.
    //
    // **Repara note**: In the rotated basis, `S_k^rot` is a partial identity,
    // so `log|S^rot|_+ = rank·ρ + 0` (the eigen-stabilised mgcv branch sets
    // `Sl[[b]]$ldet = 0`, line 296 of R/fast-REML.r). That cancels against
    // the matching `−log_pseudo_det(S)` offset in `ldet_xxs_rot`, leaving
    // the score and its ρ-derivatives invariant. To avoid double-counting
    // the pseudo-det of the rotated penalty (which would be zero anyway),
    // we feed `sl_use` (which may be `sl_rot`) here. The result is the
    // same `rank·ρ + constant` formula the unrotated path uses, modulo a
    // ρ-independent constant that has no effect on gradients/Hessians.
    let rho_vec: Vec<f64> = rho.to_vec();
    let (ldet_s, ldet_s_d1, ldet_s_d2) = compute_ldet_s_with_derivs(sl_use, &rho_vec);

    // ===== 4) log|X'WX + S(ρ)| =====
    // Returned by `try_pivoted_chol_solve` above when the pivoted Chol succeeds
    // (formula `2·Σ log(R_kk · d_piv_k)`, mgcv R/fast-REML.r:1627), or
    // computed via `determinant(A).ln()` in the LU-ridge fallback. We keep
    // the pivoted-Chol value to byte-match mgcv on the well-conditioned path.
    let _ = &a; // silence unused (kept alive for the fallback path above)

    // ===== 5) IFT: dβ/dρ_k + bSb1 + (rss2 + bSb2) =====
    let ift = compute_sl_ift_chol(sl_use, xx_view, &pp, &beta, &lambdas)?;

    // ===== 6) d.detXXS: trace derivs of log|X'WX + S| =====
    let (dxxs_d1, dxxs_d2) = compute_d_det_xxs(sl_use, &pp, &lambdas);

    // ===== 7) Assemble REML grad/Hess wrt ρ =====
    // Per mgcv R/fast-REML.r:1634-1638:
    //   reml1[k]   = (dxxs_d1[k]   - ldet_s_d1[k]   + (rss1[k]   + bSb1[k]) /(φ·γ))/2
    //   reml2[i,j] = (dxxs_d2[i,j] - ldet_s_d2[i,j] + (rss2[i,j] + bSb2[i,j])/(φ·γ))/2
    // Note `rss1 = 0` identically (envelope cancellation; the IFT helper keeps
    // it stored as zero for symmetry with the multi-S deferred path).
    let phi = log_phi.exp();
    let phi_gamma = phi * gamma;

    let mut grad_rho = Array1::<f64>::zeros(m);
    let mut hess_rho = Array2::<f64>::zeros((m, m));
    for k in 0..m {
        grad_rho[k] = 0.5 * (dxxs_d1[k] - ldet_s_d1[k] + (ift.rss1[k] + ift.bsb1[k]) / phi_gamma);
    }
    for i in 0..m {
        for j in 0..m {
            hess_rho[[i, j]] = 0.5
                * (dxxs_d2[[i, j]] - ldet_s_d2[[i, j]]
                    + (ift.rss2[[i, j]] + ift.bsb2[[i, j]]) / phi_gamma);
        }
    }

    // ===== 8) Optional log_φ extension =====
    // mgcv R/fast-REML.r:1644-1651. For Gaussian/scaled (phi_fixed = false):
    //   reml1[m+1] = (-(yy - β'f)/(φ·γ) + nobs/γ - Mp) / 2
    //   reml2 cross-blocks come from differentiating reml1 wrt log_φ:
    //     d/dlog_φ [(rss1+bSb1)/(φ·γ) / 2] = -(rss1+bSb1)/(φ·γ) / 2
    //   reml2[m+1,m+1] = ((yy - β'f)/(φ·γ)) / 2
    // mgcv stuffs that into the augmented row/column via `d <- c(-(...), rss.bSb)/(2 φ γ)`.
    let (grad, hess) = if phi_fixed {
        (grad_rho, hess_rho)
    } else {
        let nrho = m;
        let mut grad_ext = Array1::<f64>::zeros(nrho + 1);
        let mut hess_ext = Array2::<f64>::zeros((nrho + 1, nrho + 1));
        // ρ block
        for k in 0..nrho {
            grad_ext[k] = grad_rho[k];
        }
        for i in 0..nrho {
            for j in 0..nrho {
                hess_ext[[i, j]] = hess_rho[[i, j]];
            }
        }
        // log_φ gradient. `rss_bsb = yy - β'f` uses the optimum identity
        // `‖y-Xβ‖² + β'Sβ = y'y - β'X'y` (mgcv R/fast-REML.r:1646).
        // In the rotated basis, β_rot · f_rot = β · f (since β_rot = D⁻¹β,
        // f_rot = D'f, so β_rot · f_rot = β' Di' D' f = β' (D Di)' f = β' f).
        // We deliberately use the same view we solved with to keep the
        // identity bit-exact.
        let rss_bsb = yy - beta.dot(&f_view);
        grad_ext[nrho] = 0.5 * (-rss_bsb / phi_gamma + nobs / gamma - mp as f64);
        // log_φ ⊗ ρ cross block:
        // mgcv R/fast-REML.r:1648: d <- c(-(rss1+bSb1), rss_bsb) / (2 φ γ)
        // The (m+1)-th row/col of reml2 is filled with this vector.
        for k in 0..nrho {
            let cross = -(ift.rss1[k] + ift.bsb1[k]) / (2.0 * phi_gamma);
            hess_ext[[k, nrho]] = cross;
            hess_ext[[nrho, k]] = cross;
        }
        hess_ext[[nrho, nrho]] = rss_bsb / (2.0 * phi_gamma);
        (grad_ext, hess_ext)
    };

    // ===== 9) Eigen-clamped Newton step −H⁻¹·g with magnitude cap =====
    // mgcv R/fast-REML.r:1666-1676:
    //   er <- eigen(reml2, symmetric=TRUE)
    //   er$values <- abs(er$values)
    //   me <- max(er$values) * eps^0.5  (clamp small/negative eigenvalues up)
    //   step <- -er$vectors %*% ((t(er$vectors) %*% grad) / er$values)
    //   if (max|step| > 4) step <- 4 step / max|step|
    let step = clamped_newton_step(&grad, &hess)?;

    // ===== 10) Inverse-rotate β, db, PP back to the caller's basis =====
    //
    // Mirrors the inverse-repara calls at R/bam.r:759 (β), 800-801 (db),
    // and 823 (PP). The grad/Hess/step live in ρ-space (NOT coefficient
    // space) and are invariant under the repara, so they don't need
    // rotation. `ldet_s`/`ldet_xxs` are scalars whose ρ-derivatives are
    // unchanged (the rotation contributes a ρ-independent constant that
    // cancels between the two log-dets; cf. step-3 comment).
    let (mut beta_out, mut db_out, mut pp_out) = (beta, ift.db, pp);
    if any_repara {
        for block in sl.iter() {
            // β is a coefficient vector: `β ← D · β_rot`.
            block.inverse_rotate_beta_in_place(&mut beta_out);
            // PP is a covariance matrix: `PP ← D · PP_rot · D'`.
            block.inverse_rotate_cov_in_place(&mut pp_out);
        }
        // db columns are dβ/dρ_k — same coefficient-style rotation as β.
        let m_cols = db_out.ncols();
        for col in 0..m_cols {
            let mut col_vec = db_out.column(col).to_owned();
            for block in sl.iter() {
                block.inverse_rotate_db_column_in_place(&mut col_vec);
            }
            db_out.column_mut(col).assign(&col_vec);
        }
    }

    Ok(SlFitCholResult {
        beta: beta_out,
        grad,
        hess,
        step,
        db: db_out,
        pp: pp_out,
        ldet_s,
        ldet_xxs,
    })
}

/// Internal IFT result mirroring mgcv's `Sl.iftChol` return list (minus the
/// non-linear bookkeeping which only matters for multi-S blocks).
struct IftCholResult {
    /// `dβ/dρ_k`, shape `(p, m)`. Column `k` is `−λ_k · (X'X+S)⁻¹ · S_k · β`.
    db: Array2<f64>,
    /// `bSb1[k] = ∂(β'Sβ)/∂ρ_k = λ_k · β'·S_k·β`. mgcv's "linear" cancellation
    /// formula — the cross-terms `2β'S_k·dβ/dρ_k` are folded into rss2+bSb2
    /// and don't appear here.
    bsb1: Array1<f64>,
    /// `rss1[k] = ∂RSS/∂ρ_k` envelope-cancelled to zero. Stored for shape
    /// compatibility with the multi-S path.
    rss1: Array1<f64>,
    /// `rss2[k,j]` — the part of `∂²RSS/∂ρ_k∂ρ_j` that survives cancellation.
    rss2: Array2<f64>,
    /// `bSb2[k,j]` — the part of `∂²(β'Sβ)/∂ρ_k∂ρ_j` that survives.
    bsb2: Array2<f64>,
}

/// Port of `Sl.iftChol` (R/fast-REML.r:1405) for singleton blocks. Computes
/// `dβ/dρ` via the IFT on the penalised normal equations and the
/// `bSb1 / rss2 / bSb2` quantities used by `Sl.fitChol` to assemble the
/// gradient/Hessian.
///
/// Signature note vs the task spec: we take `pp = (X'X+S)⁻¹` directly rather
/// than `(R, d, piv)` from a preconditioned pivoted Cholesky, because we form
/// `PP` directly in [`compute_sl_fitchol_step`] (see module-level note on the
/// numerical choice). The mgcv-formula content is identical.
fn compute_sl_ift_chol(
    sl: &[BlockPenalty],
    xx: ArrayView2<f64>,
    pp: &Array2<f64>,
    beta: &Array1<f64>,
    lambdas: &[f64],
) -> Result<IftCholResult> {
    let p = beta.len();
    let m = sl.len();

    // D[:,k] = λ_k · S_k · β  (mgcv's `Skb` list, length-m).
    let mut d_mat = Array2::<f64>::zeros((p, m));
    let mut bsb1 = Array1::<f64>::zeros(m);
    for k in 0..m {
        let sk_beta = sl[k].dot_vec(beta);
        let lam = lambdas[k];
        for r in 0..p {
            d_mat[[r, k]] = lam * sk_beta[r];
        }
        // bSb1[k] = β' · (λ_k S_k) · β  (matches mgcv R/fast-REML.r:1454).
        bsb1[k] = lam * beta.dot(&sk_beta);
    }

    // db[:,k] = -PP · D[:,k] = -λ_k (X'X+S)⁻¹ S_k β  (mgcv R/fast-REML.r:1457-1461).
    let mut db = pp.dot(&d_mat);
    db.mapv_inplace(|x| -x);

    // rss1 ≡ 0 (envelope cancellation; see mgcv R/fast-REML.r:1448 — both
    // arrays initialised to zero, only bSb1 receives a non-zero contribution).
    let rss1 = Array1::<f64>::zeros(m);

    // S·db[:,k] = Σ_l λ_l S_l · db[:,k]. Needed for bSb2's cross terms.
    let mut s_db = Array2::<f64>::zeros((p, m));
    for k in 0..m {
        let db_k = db.column(k).to_owned();
        let mut sd_k = Array1::<f64>::zeros(p);
        for (lam_l, pen_l) in lambdas.iter().zip(sl.iter()) {
            let pen_db_k = pen_l.dot_vec(&db_k);
            for r in 0..p {
                sd_k[r] += lam_l * pen_db_k[r];
            }
        }
        for r in 0..p {
            s_db[[r, k]] = sd_k[r];
        }
    }

    // XX·db (the unpenalised half of the second-derivative chain).
    let xx_db = xx.dot(&db);

    // rss2[k,j] = 2 · db[:,j]' · XX · db[:,k]   (mgcv R/fast-REML.r:1484).
    let rss2 = {
        let mut out = Array2::<f64>::zeros((m, m));
        for k in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for r in 0..p {
                    s += db[[r, j]] * xx_db[[r, k]];
                }
                out[[k, j]] = 2.0 * s;
            }
        }
        out
    };

    // bSb2[k,j]:
    //   diag term: δ_{kj} · bSb1[k]  (mgcv's "linear only" branch line 1465)
    //   + 2 · db[:,k]' · (D[:,j] + S·db[:,j])     (line 1466 first pmmult)
    //   + 2 · D[:,k]' · db[:,j]                    (line 1466 second pmmult)
    let bsb2 = {
        let mut out = Array2::<f64>::zeros((m, m));
        for k in 0..m {
            for j in 0..m {
                let mut acc = 0.0;
                if k == j {
                    acc += bsb1[k];
                }
                let mut t1 = 0.0;
                let mut t2 = 0.0;
                for r in 0..p {
                    t1 += db[[r, k]] * (d_mat[[r, j]] + s_db[[r, j]]);
                    t2 += d_mat[[r, k]] * db[[r, j]];
                }
                acc += 2.0 * t1 + 2.0 * t2;
                out[[k, j]] = acc;
            }
        }
        out
    };

    Ok(IftCholResult {
        db,
        bsb1,
        rss1,
        rss2,
        bsb2,
    })
}

/// Port of `d.detXXS` (R/fast-REML.r:1329) for singleton blocks. Computes
/// derivatives of `log|X'WX + S(ρ)|` wrt ρ.
///
/// For each singleton block `k`:
///
/// ```text
///   d1[k]   = λ_k · tr(PP · S_k)
///   d2[i,j] = -λ_i λ_j · tr(PP · S_i · PP · S_j) + δ_{ij} · d1[i]
/// ```
///
/// (mgcv folds the `δ_ij d1[i]` correction into the diagonal via the
/// `nli[2,i]==0` branch at R/fast-REML.r:1362.)
fn compute_d_det_xxs(
    sl: &[BlockPenalty],
    pp: &Array2<f64>,
    lambdas: &[f64],
) -> (Array1<f64>, Array2<f64>) {
    let m = sl.len();
    let mut d1 = Array1::<f64>::zeros(m);
    let mut d2 = Array2::<f64>::zeros((m, m));

    // Pre-compute SPP[k] = λ_k · S_k · PP  (a (p×p) dense matrix per block;
    // only the `block_size × p` slab is non-zero in rows, since S_k is sparse).
    // We store each SPP[k] as a (k_size × p) dense block keyed by the offset.
    // For the diagonal `d2` correction we also need to evaluate
    // `tr(SPP[i] · SPP[j]')` which equals
    // `Σ_{r in block_i} Σ_{c in block_j} SPP[i][r,c] · SPP[j][c,r]`
    // (sparse-aware trace; matches mgcv's `t(SPP[i]) * SPP[j]` Hadamard then sum).
    //
    // To keep this code straightforward and consistent with `trace_product`-style
    // helpers elsewhere in the module, we materialise `λ_k S_k PP` as a dense
    // (p×p) for each k. For singleton smooths this is `block_size × p`
    // non-zero work; we accept the small `O(m·p²)` allocation cost for
    // clarity at the precision we need.
    let p = pp.nrows();
    let mut spp_full: Vec<Array2<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let mut sp_k = Array2::<f64>::zeros((p, p));
        let pen_k = &sl[k];
        let block = pen_k.block_view();
        let off = pen_k.offset;
        let bsize = block.nrows();
        let lam_k = lambdas[k];
        // (S_k PP)[r, c] = Σ_l S_k[r, l] PP[l, c].  S_k is zero outside the
        // block, so only rows in [off..off+bsize] are nonzero, and the inner
        // sum only over l in [off..off+bsize].
        for r_local in 0..bsize {
            let r = off + r_local;
            for c in 0..p {
                let mut s = 0.0;
                for l_local in 0..bsize {
                    let l = off + l_local;
                    s += block[[r_local, l_local]] * pp[[l, c]];
                }
                sp_k[[r, c]] = lam_k * s;
            }
        }
        spp_full.push(sp_k);
    }

    // d1[k] = tr(SPP[k]) = sum of diagonal entries.
    for k in 0..m {
        let mut tr = 0.0;
        for i in 0..p {
            tr += spp_full[k][[i, i]];
        }
        d1[k] = tr;
    }

    // d2[i,j] = -tr(SPP[i]' · SPP[j]) = -Σ_{r,c} SPP[i][c,r] · SPP[j][r,c]
    //         = -Σ_{r,c} SPP[i][c,r] · SPP[j][r,c]
    // (mgcv line 1354 form).
    // For singleton blocks both rows are sparse outside their respective blocks
    // — only rows in [off_i..] for SPP[i] and [off_j..] for SPP[j] contribute.
    for i in 0..m {
        let off_i = sl[i].offset;
        let bsize_i = sl[i].block_size();
        for j in 0..m {
            let off_j = sl[j].offset;
            let bsize_j = sl[j].block_size();
            // tr(SPP[i]' · SPP[j]) = Σ_r Σ_c SPP[i][r,c] · SPP[j][c,r].
            // SPP[i][r,c] non-zero only for r in [off_i..off_i+bsize_i].
            // SPP[j][c,r] non-zero only for c in [off_j..off_j+bsize_j].
            let mut tr = 0.0;
            for r_local in 0..bsize_i {
                let r = off_i + r_local;
                for c_local in 0..bsize_j {
                    let c = off_j + c_local;
                    tr += spp_full[i][[r, c]] * spp_full[j][[c, r]];
                }
            }
            d2[[i, j]] = -tr;
        }
        // Diagonal correction (mgcv R/fast-REML.r:1362, "nli[2,i]==0"
        // linear branch): add d1[i] to d2[i,i]. This is the
        // `∂²/∂ρ² log|A(ρ)|` extra term from `∂²A/∂ρ_k²  = λ_k S_k`
        // (= ∂A/∂ρ_k for linear λ_k = exp(ρ_k)), which contributes
        // `tr(PP · λ_k S_k)` = `d1[k]` to the diagonal.
        d2[[i, i]] += d1[i];
    }

    (d1, d2)
}

/// Output of `try_pivoted_chol_solve`: the three quantities the parent
/// `compute_sl_fitchol_step` needs from mgcv's preconditioned pivoted-Chol
/// block (R/fast-REML.r:1601-1627).
struct PivotedCholFitOut {
    beta: Array1<f64>,
    pp: Array2<f64>,
    ldet_xxs: f64,
}

fn try_lu_ridge_solve(a: &Array2<f64>, f: &Array1<f64>) -> Option<PivotedCholFitOut> {
    match lu_ridge_solve(a, f) {
        Ok(out) if fit_out_is_finite(&out) && lu_ridge_is_well_conditioned(a, &out) => Some(out),
        _ => None,
    }
}

fn lu_ridge_solve(a: &Array2<f64>, f: &Array1<f64>) -> Result<PivotedCholFitOut> {
    let p = a.nrows();
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let solve_ridge = 1e-12 * max_diag;
    for i in 0..p {
        a_solve[[i, i]] += solve_ridge;
    }

    let beta = solve(a_solve.clone(), f.to_owned())?;
    let pp = inverse(&a_solve)?;
    let ldet_xxs = crate::linalg::determinant(&a_solve)?.ln();
    Ok(PivotedCholFitOut { beta, pp, ldet_xxs })
}

fn fit_out_is_finite(out: &PivotedCholFitOut) -> bool {
    out.beta.iter().all(|x| x.is_finite())
        && out.pp.iter().all(|x| x.is_finite())
        && out.ldet_xxs.is_finite()
}

fn lu_ridge_is_well_conditioned(a: &Array2<f64>, out: &PivotedCholFitOut) -> bool {
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let max_pp = out.pp.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    // Conservative cheap proxy for rank trouble. mgcv's reference path uses
    // pivoted Chol plus Rrank(); if the LU inverse is this large relative to
    // the Hessian diagonal, tiny pivots can perturb rho/grad enough to move the
    // outer fREML solution. Use the mgcv-style kernel for those cases.
    max_diag * max_pp < 1.0e10
}

/// Mgcv's `Sl.fitChol` numerical kernel (R/fast-REML.r:1601-1627), ported
/// straight to LAPACK `dpstrf`.
///
/// Given a symmetric `A = X'WX + S(ρ)` (only the upper triangle is used) and
/// rhs `f`, returns:
///   - `β` solving `A·β = f` (or the minimum-norm β̂ on the working
///     subspace if A has near-null directions).
///   - `PP = A⁺` (Moore-Penrose pseudo-inverse with zero null-space block).
///   - `ldet_xxs = log|A|_+` from `2·Σ(log R_kk + log d_piv_k)`.
///
/// **Algorithm**:
///   1. `d[i] = sqrt(A[i,i])` if `A[i,i] > 0` else `1`. (Diag preconditioner;
///      mgcv lines 1603-1604).
///   2. `A_p = D⁻¹ · A · D⁻¹` (numerically: divide row `i` and col `j` by
///      `d[i]·d[j]`).
///   3. `R, piv, rank = dpstrf(A_p, tol < 0)`. LAPACK default tol gives
///      `n · ε · max_pivot`, matching R's `chol(..., pivot=TRUE)`.
///   4. Drop rank-deficient trailing columns. `R` and `piv` are truncated
///      to `R[..rank, ..rank]` and `piv[..rank]`.
///   5. `β[piv] = R⁻¹·(R'⁻¹·(f[piv]/d[piv])) / d[piv]`, with the
///      rank-deficient entries of `β` left at zero.
///   6. `PP_perm = chol2inv(R)`, `PP[piv, piv] = D⁻¹ · PP_perm · D⁻¹`.
///      Rank-deficient entries of `PP` remain zero.
///   7. `ldet_xxs = 2 · Σ_{k<rank}(log R_kk + log d_{piv[k]})`.
///
/// Returns `None` if LAPACK reports total rank deficiency (`rank == 0`) or
/// fails to factor. Both indicate the caller should fall back to the
/// LU + ridge path.
fn try_pivoted_chol_solve(a: &Array2<f64>, f: &Array1<f64>) -> Option<PivotedCholFitOut> {
    let p = a.nrows();
    if p == 0 {
        return Some(PivotedCholFitOut {
            beta: Array1::<f64>::zeros(0),
            pp: Array2::<f64>::zeros((0, 0)),
            ldet_xxs: 0.0,
        });
    }

    // Step 1: diagonal preconditioner d[i] = sqrt(A[i,i]) for A[i,i] > 0
    // else 1. Mgcv R/fast-REML.r:1603-1604.
    let mut d = Array1::<f64>::zeros(p);
    for i in 0..p {
        let dii = a[[i, i]];
        d[i] = if dii > 0.0 { dii.sqrt() } else { 1.0 };
    }

    // Step 2: A_p = D⁻¹ · A · D⁻¹. Materialise into an owned matrix because
    // dpstrf needs an in-place buffer.
    let mut a_p = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        let di = d[i];
        for j in 0..p {
            let dj = d[j];
            a_p[[i, j]] = a[[i, j]] / (di * dj);
        }
    }

    // Step 3: pivoted Chol with LAPACK default tol (R `chol(... pivot=TRUE)`).
    let pchol: PivotedCholesky = match pivoted_cholesky(&a_p, -1.0) {
        Ok(p) => p,
        Err(_) => return None, // LAPACK failure — fall back to LU+ridge.
    };
    if pchol.rank == 0 {
        return None;
    }

    // Step 5: β[piv] = R⁻¹ · (R'⁻¹ · (f[piv] / d[piv])) / d[piv]. We feed
    // `pivoted_cholesky_solve` the rhs `b[k] = f[piv[k]] / d[piv[k]]`, then
    // it back-permutes and we still need the trailing `/d`. Easier to just
    // do the forward/back-solve inline so we can divide by `d[piv]` on both
    // ends without going through the `solve` helper that already permutes.
    let r_eff = pchol.rank;
    // Build b_perm[k] = f[piv[k]] / d[piv[k]] for k in 0..rank.
    let mut bp = Array1::<f64>::zeros(r_eff);
    for k in 0..r_eff {
        let pk = pchol.piv[k];
        bp[k] = f[pk] / d[pk];
    }
    // Forward solve R' y = bp.
    let mut y = Array1::<f64>::zeros(r_eff);
    for i in 0..r_eff {
        let mut s = bp[i];
        for j in 0..i {
            s -= pchol.r[[j, i]] * y[j];
        }
        let diag = pchol.r[[i, i]];
        if !(diag.abs() > 0.0) {
            return None;
        }
        y[i] = s / diag;
    }
    // Back solve R z = y.
    let mut z = Array1::<f64>::zeros(r_eff);
    for i in (0..r_eff).rev() {
        let mut s = y[i];
        for j in (i + 1)..r_eff {
            s -= pchol.r[[i, j]] * z[j];
        }
        let diag = pchol.r[[i, i]];
        if !(diag.abs() > 0.0) {
            return None;
        }
        z[i] = s / diag;
    }
    // β[piv] = z / d[piv]. Rank-deficient entries (piv[rank..]) stay at zero.
    let mut beta = Array1::<f64>::zeros(p);
    for k in 0..r_eff {
        let pk = pchol.piv[k];
        beta[pk] = z[k] / d[pk];
    }

    // Step 6: PP[piv, piv] = chol2inv(R), then PP = D⁻¹ · PP · D⁻¹.
    // `chol2inv(R) = R⁻¹·R⁻¹'`. Reuse the linalg helper.
    let r_block = pchol.r.slice(ndarray::s![..r_eff, ..r_eff]).to_owned();
    let r_inv = match invert_upper_triangular_inline(&r_block) {
        Some(ri) => ri,
        None => return None,
    };
    let pp_perm = r_inv.dot(&r_inv.t()); // (R'R)⁻¹ for the rank-`r_eff` block.
    let mut pp = Array2::<f64>::zeros((p, p));
    for a_idx in 0..r_eff {
        let pa = pchol.piv[a_idx];
        let da = d[pa];
        for b_idx in 0..r_eff {
            let pb = pchol.piv[b_idx];
            let db = d[pb];
            pp[[pa, pb]] = pp_perm[[a_idx, b_idx]] / (da * db);
        }
    }

    // Step 7: ldet_xxs = 2 · Σ(log R_kk + log d_{piv[k]}).
    let mut ldet = 0.0;
    for k in 0..r_eff {
        let rkk = pchol.r[[k, k]];
        let dpk = d[pchol.piv[k]];
        if rkk > 0.0 && dpk > 0.0 {
            ldet += rkk.ln() + dpk.ln();
        } else {
            // Numerically singular pivot — fall back so we don't emit NaN.
            return None;
        }
    }
    let ldet_xxs = 2.0 * ldet;

    Some(PivotedCholFitOut { beta, pp, ldet_xxs })
}

/// Inline upper-triangular inverse via back-substitution. Mirrors
/// `linalg::invert_upper_triangular` but kept module-local so we don't have
/// to re-export it from `linalg`.
fn invert_upper_triangular_inline(r: &Array2<f64>) -> Option<Array2<f64>> {
    let n = r.nrows();
    if n == 0 {
        return Some(Array2::<f64>::zeros((0, 0)));
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for j in (i + 1)..n {
                s -= r[[i, j]] * x[j];
            }
            let diag = r[[i, i]];
            if !(diag.abs() > 0.0) {
                return None;
            }
            x[i] = s / diag;
        }
        for i in 0..n {
            inv[[i, col]] = x[i];
        }
    }
    Some(inv)
}

/// Eigen-clamped Newton step `−H⁻¹·g` with magnitude cap. Mirrors mgcv's
/// `Sl.fitChol` step-formation at R/fast-REML.r:1666-1676.
fn clamped_newton_step(grad: &Array1<f64>, hess: &Array2<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::Eigh;
    let n = grad.len();
    if n == 0 {
        return Ok(Array1::<f64>::zeros(0));
    }
    // Symmetrise to feed `eigh` cleanly (mgcv passes `reml2` directly).
    let h_sym = {
        let mut h = hess.clone();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = avg;
                h[[j, i]] = avg;
            }
        }
        h
    };
    let (eigvals, eigvecs) = h_sym.eigh(ndarray_linalg::UPLO::Upper).map_err(|e| {
        crate::GAMError::InvalidParameter(format!(
            "clamped_newton_step: eigendecomposition failed: {:?}",
            e
        ))
    })?;
    let max_abs = eigvals.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let me = max_abs * f64::EPSILON.sqrt();
    let abs_eig: Vec<f64> = eigvals
        .iter()
        .map(|v| v.abs().max(me).max(1e-300))
        .collect();

    // step = -V · (V'·g / |Λ|)
    let qt_g = eigvecs.t().dot(grad);
    let scaled: Array1<f64> = (0..n).map(|i| qt_g[i] / abs_eig[i]).collect();
    let mut step = eigvecs.dot(&scaled);
    step.mapv_inplace(|x| -x);

    let ms = step.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    if ms > 4.0 {
        let scale = 4.0 / ms;
        step.mapv_inplace(|x| x * scale);
    }
    Ok(step)
}

// ---------------------------------------------------------------------------
// Internal tests — exercised by the dedicated `tests/test_fastreml_unit.rs`
// integration tests too; these in-module ones catch obvious wiring breakage
// on `cargo test --lib --features blas`.
// ---------------------------------------------------------------------------
#[cfg(test)]
#[cfg(feature = "blas")]
mod internal_tests {
    use super::*;

    /// Tiny sanity check that `compute_d_det_xxs` returns shapes and that
    /// `d1` is non-negative for a positive-definite penalty (singleton).
    #[test]
    fn d_det_xxs_shape_and_sign() {
        // 1-block fixture: 4-dim cubic-like tri-diagonal SPSD.
        let p = 6;
        let block = Array2::from_shape_vec(
            (4, 4),
            vec![
                2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0,
                2.0,
            ],
        )
        .unwrap();
        let pen = BlockPenalty::new(block, 1, p);
        let sl = vec![pen];
        let pp = Array2::<f64>::eye(p);
        let lambdas = vec![1.5];
        let (d1, d2) = compute_d_det_xxs(&sl, &pp, &lambdas);
        assert_eq!(d1.len(), 1);
        assert_eq!(d2.dim(), (1, 1));
        // tr(λ S · I) = λ · tr(S) = 1.5 · 8 = 12 > 0.
        assert!(
            (d1[0] - 12.0).abs() < 1e-12,
            "tr expected 12, got {}",
            d1[0]
        );
    }

    #[test]
    fn lu_fast_path_matches_pivoted_chol_substeps_when_well_conditioned() {
        let p = 4;
        let block = Array2::from_shape_vec(
            (3, 3),
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        )
        .unwrap();
        let sl = vec![BlockPenalty::new(block, 1, p)];
        let rho = Array1::from_vec(vec![2.0_f64.ln()]);
        let lambda = rho[0].exp();
        assert!((lambda - 2.0).abs() < 1e-15);

        let xx = Array2::from_shape_vec(
            (p, p),
            vec![
                8.0, 0.2, 0.1, 0.0, 0.2, 7.0, 0.3, 0.1, 0.1, 0.3, 6.0, 0.2, 0.0, 0.1, 0.2, 5.0,
            ],
        )
        .unwrap();
        let f = Array1::from_vec(vec![1.0, 0.5, -0.25, 0.75]);
        let mut a = xx.clone();
        sl[0].scaled_add_to(&mut a, lambda);

        let lu = try_lu_ridge_solve(&a, &f).expect("well-conditioned LU fast path");
        let pchol = try_pivoted_chol_solve(&a, &f).expect("pivoted Chol reference");

        let beta_diff = (&lu.beta - &pchol.beta)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        let pp_diff = (&lu.pp - &pchol.pp)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(beta_diff < 1e-11, "beta diff {beta_diff:.3e}");
        assert!(pp_diff < 1e-11, "PP diff {pp_diff:.3e}");
        assert!((lu.ldet_xxs - pchol.ldet_xxs).abs() < 1e-10);

        let (dxxs_d1, _) = compute_d_det_xxs(&sl, &lu.pp, &[lambda]);
        let manual_trace = lambda * sl[0].trace_product(&lu.pp);
        assert!((dxxs_d1[0] - manual_trace).abs() < 1e-12);

        let out = compute_sl_fitchol_step(
            &sl,
            xx.view(),
            f.view(),
            rho.view(),
            3.0,
            0.0,
            true,
            10.0,
            1,
            1.0,
        )
        .unwrap();
        assert!(out.grad.iter().all(|g| g.is_finite()));
        assert!((out.ldet_xxs - lu.ldet_xxs).abs() < 1e-10);
    }

    #[test]
    fn lu_fast_path_rejects_rank_risky_scaled_inverse() {
        let a = Array2::from_diag(&Array1::from_vec(vec![1.0e13, 1.0, 2.0]));
        let f = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let out = lu_ridge_solve(&a, &f).unwrap();
        assert!(fit_out_is_finite(&out));
        assert!(!lu_ridge_is_well_conditioned(&a, &out));
        assert!(try_lu_ridge_solve(&a, &f).is_none());
    }

    /// `IftCholResult::rss1` is identically zero — guards against accidental
    /// mutation during refactors of the IFT helper.
    #[test]
    fn ift_rss1_is_zero() {
        let p = 4;
        let block = Array2::from_shape_vec(
            (3, 3),
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        )
        .unwrap();
        let pen = BlockPenalty::new(block, 0, p);
        let sl = vec![pen];
        let xx = Array2::<f64>::eye(p) * 5.0;
        // PP = (XX + λS)⁻¹ at λ=1: small invert.
        let mut a = xx.clone();
        sl[0].scaled_add_to(&mut a, 1.0);
        let pp = inverse(&a).unwrap();
        let beta = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.05]);
        let ift = compute_sl_ift_chol(&sl, xx.view(), &pp, &beta, &[1.0]).unwrap();
        for v in ift.rss1.iter() {
            assert_eq!(*v, 0.0, "rss1 must be exactly zero (envelope)");
        }
    }

    /// FD check on `d1`/`d2` from `compute_d_det_xxs` against the analytical
    /// definition `log|A(ρ)|` with `A(ρ) = XX + Σ exp(ρ_k) S_k`. A standalone
    /// guard for the trace machinery before plugging it into the score.
    #[test]
    fn d_det_xxs_fd_parity_1block() {
        let p = 6;
        let block = Array2::from_shape_vec(
            (4, 4),
            vec![
                2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0, 2.0, -1.0, 0.0, 0.0, -1.0,
                2.0,
            ],
        )
        .unwrap();
        let pen = BlockPenalty::new(block, 1, p);
        let sl = vec![pen];
        // Use a well-conditioned XX (positive diagonal).
        let mut xx = Array2::<f64>::eye(p);
        for i in 0..p {
            xx[[i, i]] = 3.0 + (i as f64) * 0.1;
        }
        let rho0: f64 = 0.3;
        let lam0 = rho0.exp();
        let mut a0 = xx.clone();
        sl[0].scaled_add_to(&mut a0, lam0);
        let pp0 = inverse(&a0).unwrap();
        let (d1, d2) = compute_d_det_xxs(&sl, &pp0, &[lam0]);
        // FD on log|A|:
        let log_det = |rho: f64| -> f64 {
            let mut a = xx.clone();
            sl[0].scaled_add_to(&mut a, rho.exp());
            crate::linalg::determinant(&a).unwrap().ln()
        };
        let h = 1e-4;
        let fd1 = (log_det(rho0 + h) - log_det(rho0 - h)) / (2.0 * h);
        let fd2 = (log_det(rho0 + h) - 2.0 * log_det(rho0) + log_det(rho0 - h)) / (h * h);
        assert!(
            (d1[0] - fd1).abs() < 1e-6,
            "d1 mismatch: analytic={}, fd={}",
            d1[0],
            fd1
        );
        assert!(
            (d2[[0, 0]] - fd2).abs() < 1e-3,
            "d2 mismatch: analytic={}, fd={}",
            d2[[0, 0]],
            fd2
        );
    }
}
