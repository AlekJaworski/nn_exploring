//! Tk·KK' contribution to mgcv's analytical REML gradient + Hessian.
//!
//! Ports the W-dependent pieces of `det1[k]` and `det2[k,j]` from
//! mgcv's gdi.c (`get_ddetXWXpS`, lines 817-948). Used by the IFT
//! gradient (`reml_gradient_mgcv_exact_ift_inner`) and the IFT Hessian
//! (`reml_hessian_mgcv_exact_ift`) to match mgcv's
//!   `∂log|H|/∂ρ_k = tr(Tk·KK') + λ_k · tr(A⁻¹·S_k)`
//! and its derivative.

use super::{assemble_reml_system, compute_b1_ift, compute_xtwx, compute_xtwy};
use crate::block_penalty::BlockPenalty;
use crate::linalg::{inverse, solve};
use crate::Result;
use ndarray::{Array1, Array2};

/// Compute the analytical Tk·KK' gradient term at a given λ.
///
/// Returns a length-m vector with
///   `tk_kkt[k] = Σᵢ a1[i] · η₁[i,k] · sign(w[i]) · lev_uw[i]`
/// matching the inline computation in `reml_gradient_mgcv_exact_ift_inner`.
/// This is the missing piece of mgcv's `∂log|H|/∂ρ_k` (gdi.c:857) used by
/// `tk_kkt_hessian_fd` to FD-differentiate ∂tk_kkt/∂ρ_j (legacy diagnostic;
/// production dispatch now uses `tk_kkt_hessian_analytical`).
#[cfg(feature = "blas")]
fn compute_tk_kkt_vec(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    let n = y.len();
    let m = lambdas.len();
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    let system = assemble_reml_system(y, x, w, xtwx, lambdas, penalties_blocks, None)?;
    let p = system.a.nrows();

    let b1 = compute_b1_ift(&system.a_inv, &system.beta, lambdas, penalties_blocks);
    let eta1 = x.dot(&b1);

    // lev_uw[i] = x_i' A⁻¹ x_i
    let xa = x.dot(&system.a_inv);
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    // a1[i] = dw/dη (gdi.c:2535 / 2556).
    let mut a1 = Array1::<f64>::zeros(n);
    if !matches!(family, crate::pirls::Family::Gaussian) {
        let use_fisher = family.is_canonical_link();
        let system_fitted = system.fitted(x);
        for i in 0..n {
            let eta_i = system_fitted[i];
            let mu_i = family.inverse_link(eta_i);
            let dmu_deta = family.d_inverse_link(eta_i);
            if dmu_deta.abs() < 1e-12 {
                continue;
            }
            let g1 = 1.0 / dmu_deta;
            let v = family.variance(mu_i).max(1e-300);
            let v1n = family.dvar(mu_i) / v;
            let v2n = family.d2var(mu_i) / v;
            let g2n = family.d2link(mu_i) * dmu_deta;
            let g3n = family.d3link(mu_i) * dmu_deta;
            if use_fisher {
                a1[i] = -w[i] * (v1n + 2.0 * g2n) / g1;
            } else {
                let y_for_resid = y_original.unwrap_or(y);
                let c_resid = y_for_resid[i] - mu_i;
                let alpha_raw = crate::pirls::newton_irls_alpha(c_resid, v1n, g2n);
                let alpha = if alpha_raw <= 0.0 { 1.0 } else { alpha_raw };
                let xx = v2n - v1n * v1n + g3n - g2n * g2n;
                let alpha1 = (-(v1n + g2n) + c_resid * xx) / alpha;
                a1[i] = w[i] * (alpha1 - v1n - 2.0 * g2n) / g1;
            }
        }
    }

    let mut tk = Array1::<f64>::zeros(m);
    for k in 0..m {
        let mut s = 0.0;
        for i in 0..n {
            s += a1[i] * eta1[[i, k]] * w[i].signum() * lev_uw[i];
        }
        tk[k] = s;
    }
    Ok(tk)
}

/// Central-FD Hessian contribution from the Tk·KK' gradient term.
///
/// Returns an m×m matrix where entry `[k,j] = ∂tk_kkt[k]/∂ρ_j`. The full
/// Hessian-of-REML contribution is this matrix divided by 2 (matching the
/// `tk_kkt/2` factor in the gradient assembly).
///
/// Perturbs each ρ_j = log(λ_j) by ±h=1e-4 and central-differences the
/// analytical `tk_kkt` vector. Cost: 2m extra `compute_tk_kkt_vec` calls per
/// Hessian, each of which is one A⁻¹ build + a few matmuls.
#[cfg(feature = "blas")]
pub fn tk_kkt_hessian_fd(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let m = lambdas.len();
    let mut out = Array2::<f64>::zeros((m, m));
    let h = 1.0e-4_f64;
    let log_lam: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
    for j in 0..m {
        let mut lp = log_lam.clone();
        lp[j] += h;
        let mut lm = log_lam.clone();
        lm[j] -= h;
        let lam_plus: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
        let lam_minus: Vec<f64> = lm.iter().map(|l| l.exp()).collect();
        let tk_plus = compute_tk_kkt_vec(
            y,
            x,
            w,
            &lam_plus,
            penalties_blocks,
            cached_xtwx,
            family,
            y_original,
        )?;
        let tk_minus = compute_tk_kkt_vec(
            y,
            x,
            w,
            &lam_minus,
            penalties_blocks,
            cached_xtwx,
            family,
            y_original,
        )?;
        for k in 0..m {
            out[[k, j]] = (tk_plus[k] - tk_minus[k]) / (2.0 * h);
        }
    }
    Ok(out)
}

/// Analytical Tk·KK' Hessian contribution — full mgcv `get_ddetXWXpS`
/// port (gdi.c:817-948). Computes the four W-dependent pieces of
/// `det2[k,j]` that depend on the per-observation weights:
///
///   `H[k,j] = Σᵢ Tkm[i,k,j]·sign(w_i)·lev_uw[i]   (P1, gdi.c:919-920)`
///             ` − tr(C_k·C_j)                     (P2, gdi.c:922-923)`
///             ` − λ_j·tr(C_k·S_j·A⁻¹)             (P4, gdi.c:928-929)`
///             ` − λ_k·tr(C_j·S_k·A⁻¹)             (P5, gdi.c:931-932)`
///
/// where `C_k = B_k·A⁻¹` and `B_k = Xᵀ·diag(Tk[:,k])·X`. The penalty-only
/// pieces P3 (δ_kj·λ_k·tr(A⁻¹·S_k), gdi.c:925-926) and P6 (−λ_k·λ_j·
/// tr(A⁻¹·S_k·A⁻¹·S_j), gdi.c:934-936) are assembled separately by the
/// caller (`reml_hessian_mgcv_exact_ift`).
///
/// The result is symmetric in (k,j) (P1, P2 by construction; P4+P5
/// together) — matching mgcv's `det2[mk] = det2[km]` enforcement at
/// gdi.c:938.
///
/// The caller adds `H[k,j] / 2` to the REML Hessian to match the score's
/// `+ 0.5·log|H|` factor.
///
/// Math (with reference to gdi.c lines):
///
/// 1. `a1 = dw/dη` (gdi.c:2532 Fisher, 2553 Newton) and `a2 = d²w/dη²`
///    (2537 Fisher, 2555 Newton). Both are IFT-correct: a2 carries the
///    `a1²/w` chunk from differentiating w along the IFT chain.
/// 2. `b1[:,k] = -λ_k·A⁻¹·S_k·β` (gdi.c:1338, ift1) and
///    `b2[:,(k,j)] = A⁻¹·(Xᵀ(-a1·η₁_k·η₁_j) − λ_k·S_k·b1[:,j]
///                       − λ_j·S_j·b1[:,k])  + δ_kj·b1[:,k]`
///    (gdi.c:1343-1356, ift1 second-derivative loop, with the
///    Xᵀ(-a1·η₁·η₁) W-piece that captures the implicit dependence of W
///    on λ).
/// 3. `η₁[:,k] = X·b1[:,k]`, `η₂[:,k,j] = X·b2[:,(k,j)]`.
/// 4. `Tk[i,k] = a1[i]·η₁[i,k]/|w_i|`, gdi.c:2212.
/// 5. `Tkm[i,k,j] = (a2[i]·η₁[i,k]·η₁[i,j] + a1[i]·η₂[i,k,j])/|w_i|`,
///    gdi.c:2184-2202.
///
/// Trace identities used to avoid forming K explicitly:
///   `tr(KtTK[k]·KtTK[j])     = tr(B_k·A⁻¹·B_j·A⁻¹)`
///   `tr(KtTK[k]·PtSP[j])     = tr(B_k·A⁻¹·S_j·A⁻¹) = tr(C_k·S_j·A⁻¹)`
///   `Σᵢ Tkm[i,k,j]·diagKKt[i] = Σᵢ Tkm[i,k,j]·sign(w_i)·lev_uw[i]`
/// with `lev_uw[i] = (X·A⁻¹·Xᵀ)[i,i]`.
#[cfg(feature = "blas")]
pub fn tk_kkt_hessian_analytical(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let n = y.len();
    let m = lambdas.len();
    let p = x.ncols();

    // --- A = X'WX + Σλ_jS_j, β = A⁻¹ X'Wy, A⁻¹ ---
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }
    let xtwy = compute_xtwy(x, w, y);
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let solve_ridge = 1e-12 * max_diag;
    a_solve
        .diag_mut()
        .iter_mut()
        .for_each(|d| *d += solve_ridge);
    let beta = solve(a_solve, xtwy)?;
    let a_inv = inverse(&a)?;

    // --- b1 (p × m) and η₁ (n × m) ---
    let b1 = compute_b1_ift(&a_inv, &beta, lambdas, penalties_blocks);
    let eta1 = x.dot(&b1);

    // --- lev_uw[i] = x_iᵀ A⁻¹ x_i = (X·A⁻¹·Xᵀ)[i,i] ---
    let xa = x.dot(&a_inv);
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    // --- a1, a2 (per-obs, see gdi.c:2532-2556) ---
    let mut a1 = Array1::<f64>::zeros(n);
    let mut a2 = Array1::<f64>::zeros(n);
    if !matches!(family, crate::pirls::Family::Gaussian) {
        let eta = x.dot(&beta);
        let use_fisher = family.is_canonical_link();
        for i in 0..n {
            let mu_i = family.inverse_link(eta[i]);
            let dmu_deta = family.d_inverse_link(eta[i]);
            if dmu_deta.abs() < 1e-12 {
                continue;
            }
            let g1 = 1.0 / dmu_deta;
            let v = family.variance(mu_i).max(1e-300);
            // Normalised variance derivatives V'/V, V''/V, V'''/V (gdi.c:2369)
            let v1n = family.dvar(mu_i) / v;
            let v2n = family.d2var(mu_i) / v;
            let v3n = family.d3var(mu_i) / v;
            // Normalised link derivatives g''/g', g'''/g', g''''/g' (gdi.c:2367)
            let g2n = family.d2link(mu_i) * dmu_deta;
            let g3n = family.d3link(mu_i) * dmu_deta;
            let g4n = family.d4link(mu_i) * dmu_deta;

            if use_fisher {
                // a1 = -w·(V1 + 2g2)/g1   (gdi.c:2532)
                a1[i] = -w[i] * (v1n + 2.0 * g2n) / g1;
                // mgcv's full Fisher a2 (gdi.c:2537):
                //   a2 = a1·(a1/w − g2/g1) − w·(V2−V1²+2g3−2g2²)/g1²
                //      = a1²/w − a1·g2/g1 − w·(...)/g1²
                a2[i] = a1[i] * a1[i] / w[i]
                    - a1[i] * (g2n / g1)
                    - w[i] * (v2n - v1n * v1n + 2.0 * g3n - 2.0 * g2n * g2n) / (g1 * g1);
            } else {
                // Full Newton (gdi.c:2543-2556).
                let y_for_resid = y_original.unwrap_or(y);
                let c_resid = y_for_resid[i] - mu_i;
                let alpha_raw = crate::pirls::newton_irls_alpha(c_resid, v1n, g2n);
                let alpha = if alpha_raw <= 0.0 { 1.0 } else { alpha_raw };
                // xx = V2 − V1² + g3 − g2²
                let xx = v2n - v1n * v1n + g3n - g2n * g2n;
                // xx2 = V3 − 3·V1·V2 + 2·V1³ + g4 − 3·g3·g2 + 2·g2³
                let xx2 = v3n - 3.0 * v1n * v2n + 2.0 * v1n * v1n * v1n
                    + g4n - 3.0 * g3n * g2n + 2.0 * g2n * g2n * g2n;
                let alpha1 = (-(v1n + g2n) + c_resid * xx) / alpha;
                let alpha2 = (-2.0 * xx + c_resid * xx2) / alpha;
                a1[i] = w[i] * (alpha1 - v1n - 2.0 * g2n) / g1;
                // mgcv's full Newton a2 (gdi.c:2555-2556):
                //   a2 = a1·(a1/w − g2/g1)
                //        − w·(α1² − α2 + V2 − V1²+2g3−2g2²)/g1²
                //      = a1²/w − a1·g2/g1 − w·(...)/g1²
                a2[i] = a1[i] * a1[i] / w[i]
                    - a1[i] * (g2n / g1)
                    - w[i]
                        * (alpha1 * alpha1 - alpha2 + v2n - v1n * v1n
                            + 2.0 * g3n - 2.0 * g2n * g2n)
                        / (g1 * g1);
            }
        }
    }

    // --- b2 storage: upper triangle, pair_idx(k,m) for k ≤ m ---
    // We store m·(m+1)/2 columns, indexed [k=0,m=0], [k=0,m=1], ..., [k=0,m=M-1],
    //                                    [k=1,m=1], ..., [k=1,m=M-1], ...
    // This matches mgcv ift1 pp pointer arithmetic.
    let n_pairs = m * (m + 1) / 2;
    let pair_idx = |k: usize, j: usize| -> usize {
        // Caller is responsible for ensuring k ≤ j.
        debug_assert!(k <= j);
        // Number of pairs before row k (rows 0..k each contribute m, m-1, ...,
        // m-k+1) is k·m − k·(k-1)/2. Then offset within row k is (j-k).
        k * m - (k * (k.saturating_sub(1))) / 2 + (j - k)
    };
    // For each pair (k, j) with k ≤ j: build rhs and solve A·b2 = rhs.
    // b2_cols stores p-vectors; eta2_cols stores n-vectors X·b2.
    let mut b2_cols: Vec<Array1<f64>> = Vec::with_capacity(n_pairs);
    let mut eta2_cols: Vec<Array1<f64>> = Vec::with_capacity(n_pairs);
    for k in 0..m {
        for j in k..m {
            // rhs = Xᵀ(-a1·η1_k·η1_j) − λ_k · S_k · b1[:,j] − λ_j · S_j · b1[:,k]
            //   (mgcv ift1 second-derivative loop, gdi.c:1343-1356)
            let b1_j = b1.column(j).to_owned();
            let b1_k = b1.column(k).to_owned();
            let sk_b1j = penalties_blocks[k].dot_vec(&b1_j);
            let sj_b1k = penalties_blocks[j].dot_vec(&b1_k);
            let mut rhs = Array1::<f64>::zeros(p);
            for r in 0..p {
                rhs[r] = -(lambdas[k] * sk_b1j[r] + lambdas[j] * sj_b1k[r]);
            }
            // Full-IFT W-piece (gdi.c:1347-1348): Xᵀ(-a1·η1_k·η1_j).
            // Captures the implicit dependence of W on λ via η.
            {
                let mut v = Array1::<f64>::zeros(n);
                for i in 0..n {
                    v[i] = -a1[i] * eta1[[i, k]] * eta1[[i, j]];
                }
                let xtv = x.t().dot(&v);
                for r in 0..p {
                    rhs[r] += xtv[r];
                }
            }
            // b2_kj = A⁻¹ · rhs
            let mut b2_kj = a_inv.dot(&rhs);
            // δ_kj correction: gdi.c:1355 — `pp[j] += b1[i * r + j]` when i==k.
            if k == j {
                for r in 0..p {
                    b2_kj[r] += b1[[r, k]];
                }
            }
            let eta2_kj = x.dot(&b2_kj);
            b2_cols.push(b2_kj);
            eta2_cols.push(eta2_kj);
            // pair_idx invariant
            debug_assert_eq!(pair_idx(k, j), b2_cols.len() - 1);
        }
    }

    // --- Build B_k = Xᵀ diag(Tk[:,k]) X (p × p) and C_k = B_k · A⁻¹ ---
    // Tk[i,k] = a1[i] · η₁[i,k]·sign(w). Needed by term 4 (always-on) and by
    // term 2 (`-tr(C_k C_j)`, off in FD-target mode).
    let mut c_per_k: Vec<Array2<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        let mut tx = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let tk_i = a1[i] * eta1[[i, k]] * w[i].signum();
            for r in 0..p {
                tx[[i, r]] = tk_i * x[[i, r]];
            }
        }
        let b_k = x.t().dot(&tx);
        c_per_k.push(b_k.dot(&a_inv));
    }

    // --- Per-penalty: S_j · A⁻¹ (p × p) ---
    // Used by the asymmetric "term 4" = -λ_j · tr(C_k · S_j · A⁻¹) which
    // captures the variation of lev_uw[i] through the penalty piece of
    // ∂A/∂ρ_j = λ_j · S_j.
    let mut s_ainv_per_j: Vec<Array2<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let mut sa = Array2::<f64>::zeros((p, p));
        for col in 0..p {
            let a_inv_col: Array1<f64> = a_inv.column(col).to_owned();
            let s_a_col = penalties_blocks[j].dot_vec(&a_inv_col);
            for r in 0..p {
                sa[[r, col]] = s_a_col[r];
            }
        }
        s_ainv_per_j.push(sa);
    }

    // --- Final assembly ---
    // The (k, j) entry computes the W-dependent pieces of mgcv's det2[k,j]
    // (gdi.c:919-932): P1 (Tkm·diagKKt), P2 (-tr(C_k·C_j)), P4
    // (-λ_j·tr(C_k·S_j·A⁻¹)), and P5 (-λ_k·tr(C_j·S_k·A⁻¹)). The penalty-
    // only pieces P3 (δ_kj·λ_k·tr(A⁻¹·S_k)) and P6 (-λ_k·λ_j·
    // tr(A⁻¹·S_k·A⁻¹·S_j)) are assembled by the caller. P4 + P5 together
    // are symmetric in (k,j), matching mgcv's det2[mk] = det2[km] at
    // gdi.c:938.
    let mut hess_tk = Array2::<f64>::zeros((m, m));
    for k in 0..m {
        for j in 0..m {
            // P2: -tr(C_k · C_j)   (gdi.c:922-923)
            let mut trace_piece = 0.0;
            {
                let c_k = &c_per_k[k];
                let c_j = &c_per_k[j];
                for pp in 0..p {
                    for qq in 0..p {
                        trace_piece -= c_k[[pp, qq]] * c_j[[qq, pp]];
                    }
                }
            }
            // P4: -λ_j · tr(C_k · S_j · A⁻¹)   (gdi.c:928-929)
            let mut term4 = 0.0;
            {
                let c_k = &c_per_k[k];
                let sa_j = &s_ainv_per_j[j];
                for pp in 0..p {
                    for qq in 0..p {
                        term4 -= c_k[[pp, qq]] * sa_j[[qq, pp]];
                    }
                }
                term4 *= lambdas[j];
            }
            // P5: -λ_k · tr(C_j · S_k · A⁻¹)   (gdi.c:931-932). Symmetric
            // counterpart of P4; together they make hess_tk[k,j] symmetric.
            let mut term5 = 0.0;
            {
                let c_j = &c_per_k[j];
                let sa_k = &s_ainv_per_j[k];
                for pp in 0..p {
                    for qq in 0..p {
                        term5 -= c_j[[pp, qq]] * sa_k[[qq, pp]];
                    }
                }
                term5 *= lambdas[k];
            }
            // P1: Σᵢ Tkm[i,k,j] · sign(w_i) · lev_uw[i]   (gdi.c:919-920)
            // pair_idx requires k ≤ j; b2 is symmetric in (k,j) so use
            // sorted indices.
            let (lo, hi) = if k <= j { (k, j) } else { (j, k) };
            let eta2_kj = &eta2_cols[pair_idx(lo, hi)];
            let mut tkm_piece = 0.0;
            for i in 0..n {
                let tkm_ikj = a2[i] * eta1[[i, k]] * eta1[[i, j]]
                    + a1[i] * eta2_kj[i];
                tkm_piece += tkm_ikj * w[i].signum() * lev_uw[i];
            }
            hess_tk[[k, j]] = trace_piece + term4 + term5 + tkm_piece;
        }
    }
    Ok(hess_tk)
}
