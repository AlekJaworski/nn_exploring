//! Validates each component of our REML gradient (in `reml.rs`) against
//! the brute-force Python reference at
//! `scripts/python/diagnostics/reml_gradient_brute_force.py`.
//!
//! Binomial+logit (canonical link, Fisher PIRLS, σ²=1) — n=30, p=8, M=2, ρ=(0,1).
//!
//! Components verified:
//!   - d1_k                = (∂D/∂β)' b1 + λ_k β'S_kβ + 2(ΣλSβ)' b1   (gam.fit3.r:622 D1)
//!   - tk_kkt[k]           = Σᵢ a1[i]·η₁[i,k]·sign(w)·lev_uw[i]       (gdi.c:857 Tk·KK')
//!   - λ_k·tr(A⁻¹·S_k)
//!   - rank_k              (= rp$det1[k] for rank-deficient 2nd-diff penalties scaled by λ_k)
//!   - full gradient       = d1/(2·σ²) + (tk_kkt + λ_k·trAinvS)/2 - rank/2
//!
//! If each component matches the Python reference but the parity battery still
//! regresses on binomial when we add tk_kkt to the gradient, the bug is *not*
//! in the formula — it must be in how mgcv-via-fixture picks up a different
//! β̂ than we do.
#![cfg(feature = "blas")]

use mgcv_rust::pirls::Family;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;

include!("fixtures/binomial_gradient_reference.rs.fragment");

fn build() -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<Array2<f64>>, Array2<f64>) {
    let x = Array2::from_shape_vec((N_BIN, P_BIN), X_BIN_FLAT.to_vec()).unwrap();
    let y = Array1::from(Y_BIN.to_vec());
    let beta = Array1::from(BETA_BIN.to_vec());
    let s_blocks = vec![
        Array2::from_shape_vec((P_BIN, P_BIN), S1_BIN_FLAT.to_vec()).unwrap(),
        Array2::from_shape_vec((P_BIN, P_BIN), S2_BIN_FLAT.to_vec()).unwrap(),
    ];
    let s_lambda = &s_blocks[0] * LAMBDA_BIN[0] + &s_blocks[1] * LAMBDA_BIN[1];
    (x, y, beta, s_blocks, s_lambda)
}

#[test]
fn binomial_gradient_components_match_reference() {
    let (x, y, beta, s_blocks, s_lambda) = build();
    let family = Family::Binomial;
    let n = N_BIN;
    let p = P_BIN;
    let m = 2;

    // 1. PIRLS-converged μ, w, α (canonical → Fisher → α≡1).
    let eta = x.dot(&beta);
    let mu: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let dmu_deta = family.d_inverse_link(eta[i]);
        let v = family.variance(mu[i]);
        w[i] = (dmu_deta * dmu_deta) / v.max(1e-300); // Fisher (α=1 for canonical)
    }

    // 2. A = X'WX + Sλ; A_inv.
    let mut xtwx = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for r in 0..p {
            for c in 0..p {
                xtwx[[r, c]] += x[[i, r]] * w[i] * x[[i, c]];
            }
        }
    }
    let a = &xtwx + &s_lambda;
    let a_inv = a.inv().unwrap();

    // 3. Per-component computations matching reml.rs:
    //    lev_uw, a1 (Fisher), eta1 from b1.
    let xa = x.dot(&a_inv);
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    let mut a1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let dmu_deta = family.d_inverse_link(eta[i]);
        let g1 = 1.0 / dmu_deta;
        let v = family.variance(mu[i]);
        let v1n = family.dvar(mu[i]) / v;
        let g2n = family.d2link(mu[i]) * dmu_deta;
        a1[i] = -w[i] * (v1n + 2.0 * g2n) / g1;
    }

    let mut b1 = Array2::<f64>::zeros((p, m));
    for k in 0..m {
        let s_k_beta = s_blocks[k].dot(&beta);
        let ainv_sk_beta = a_inv.dot(&s_k_beta);
        for r in 0..p {
            b1[[r, k]] = -LAMBDA_BIN[k] * ainv_sk_beta[r];
        }
    }
    let eta1 = x.dot(&b1);

    // 4. d1_k via the same formula as reml.rs:1069.
    //    dev_grad = -2 X' (y-μ) / (V·g')   (compute_dev_grad_beta non-Gaussian branch)
    let mut v1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let dmu_deta = family.d_inverse_link(eta[i]);
        let v = family.variance(mu[i]);
        let g_prime = 1.0 / dmu_deta;
        v1[i] = -2.0 * (y[i] - mu[i]) / (v * g_prime);
    }
    let dev_grad_beta = x.t().dot(&v1);

    let sum_lambda_s_beta: Array1<f64> = (0..m)
        .map(|j| s_blocks[j].dot(&beta) * LAMBDA_BIN[j])
        .fold(Array1::<f64>::zeros(p), |acc, x| acc + x);

    // 5. Per-k checks.
    let mut all_passed = true;
    for k in 0..m {
        // d1_k
        let mut dev_dot_b1 = 0.0;
        for r in 0..p {
            dev_dot_b1 += dev_grad_beta[r] * b1[[r, k]];
        }
        let bsb_k: f64 = (0..p)
            .map(|r| (0..p).map(|c| beta[r] * s_blocks[k][[r, c]] * beta[c]).sum::<f64>())
            .sum();
        let mut sls_dot_b1 = 0.0;
        for r in 0..p {
            sls_dot_b1 += sum_lambda_s_beta[r] * b1[[r, k]];
        }
        let d1_k = dev_dot_b1 + LAMBDA_BIN[k] * bsb_k + 2.0 * sls_dot_b1;

        // tk_kkt
        let mut tk_kkt = 0.0;
        for i in 0..n {
            tk_kkt += a1[i] * eta1[[i, k]] * w[i].signum() * lev_uw[i];
        }

        // lam_k * tr(A_inv * S_k)
        let mut tr_a_inv_s = 0.0;
        for r in 0..p {
            for c in 0..p {
                tr_a_inv_s += a_inv[[r, c]] * s_blocks[k][[c, r]];
            }
        }
        let lam_tr_ainv_sk = LAMBDA_BIN[k] * tr_a_inv_s;

        // Full gradient (with tk_kkt). σ² = 1 for binomial.
        let scale = 1.0;
        let grad_k = d1_k / (2.0 * scale)
            + (tk_kkt + lam_tr_ainv_sk) / 2.0
            - RANK_REF_BIN[k] / 2.0;

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-15);

        let r_d1 = rel(d1_k, D1_REF_BIN[k]);
        let r_tk = rel(tk_kkt, TK_KKT_REF_BIN[k]);
        let r_la = rel(lam_tr_ainv_sk, LAM_TRAINVS_REF_BIN[k]);
        let r_g = rel(grad_k, GRAD_REF_BIN[k]);

        println!(
            "k={}: d1={:+.10e} (rel {:.1e}), tk_kkt={:+.10e} (rel {:.1e}), \
             lam·trAinvS={:+.10e} (rel {:.1e}), grad={:+.10e} (rel {:.1e})",
            k, d1_k, r_d1, tk_kkt, r_tk, lam_tr_ainv_sk, r_la, grad_k, r_g
        );

        for &(name, r) in &[("d1", r_d1), ("tk_kkt", r_tk), ("lam_trAinvS", r_la), ("grad", r_g)] {
            if r > 1e-9 {
                println!("  *** {} mismatch (rel {:.1e})", name, r);
                all_passed = false;
            }
        }
    }
    assert!(all_passed, "one or more gradient components diverged from reference");
}
