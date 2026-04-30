//! Validates our `tr(Tk·KK')` formula (B5 from Plan Option B Newton PIRLS)
//! against the brute-force Python reference at
//! `scripts/python/diagnostics/tk_kk_brute_force.py`.
//!
//! The reference: tiny Gamma+log GLM (n=30, p=8, M=2, ρ=(0,1)), all weights
//! positive. Reference values for Tk·KK' computed analytically and confirmed
//! by central-difference on log|A(ρ)| to ~1e-10 relative agreement. If our
//! Rust code reproduces those numbers, the formula is correct; any remaining
//! parity-battery gap on gamma+log lives elsewhere (likely PIRLS β̂
//! convergence or score-formula).
#![cfg(feature = "blas")]

use mgcv_rust::pirls::Family;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;

include!("fixtures/tk_kkt_reference.rs.fragment");

fn build_x() -> Array2<f64> {
    Array2::from_shape_vec((N, P), X_FLAT.to_vec()).unwrap()
}

fn build_y() -> Array1<f64> {
    Array1::from(Y.to_vec())
}

fn build_beta() -> Array1<f64> {
    Array1::from(BETA.to_vec())
}

fn build_s_lambda() -> Array2<f64> {
    let s1 = Array2::from_shape_vec((P, P), S1_FLAT.to_vec()).unwrap();
    let s2 = Array2::from_shape_vec((P, P), S2_FLAT.to_vec()).unwrap();
    &s1 * LAMBDA[0] + &s2 * LAMBDA[1]
}

fn build_s_blocks() -> Vec<Array2<f64>> {
    vec![
        Array2::from_shape_vec((P, P), S1_FLAT.to_vec()).unwrap(),
        Array2::from_shape_vec((P, P), S2_FLAT.to_vec()).unwrap(),
    ]
}

#[test]
fn tk_kkt_matches_python_reference() {
    let family = Family::GammaLog;
    let x = build_x();
    let y = build_y();
    let beta = build_beta();
    let s_lambda = build_s_lambda();
    let s_blocks = build_s_blocks();

    // 1. Compute μ from β, then Newton α = 1 + (y−μ)·(V'/V + g''·dμ/dη), w = α·(dμ/dη)²/V.
    //    For Gamma+log this gives α = w = y/μ. We compute via the family methods so we exercise the
    //    same code path used in the gradient assembly.
    let eta = x.dot(&beta);
    let mu: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();
    let mut w = Array1::<f64>::zeros(N);
    let mut alpha_vec = Array1::<f64>::zeros(N);
    for i in 0..N {
        let dmu_deta = family.d_inverse_link(eta[i]);
        let v = family.variance(mu[i]);
        let v1n = family.dvar(mu[i]) / v;
        let g2n = family.d2link(mu[i]) * dmu_deta;
        let alpha = 1.0 + (y[i] - mu[i]) * (v1n + g2n);
        alpha_vec[i] = alpha;
        let wf = (dmu_deta * dmu_deta) / v;
        w[i] = wf * alpha;
    }
    // Sanity: Gamma+log has w = y/μ (positive when y>0).
    for i in 0..N {
        let expected = y[i] / mu[i];
        assert!(
            (w[i] - expected).abs() < 1e-10,
            "w[{}]={} vs expected y/mu={}",
            i,
            w[i],
            expected
        );
    }

    // 2. A = X'WX + S_lambda, A_inv.
    let mut xtwx = Array2::<f64>::zeros((P, P));
    for i in 0..N {
        for r in 0..P {
            for c in 0..P {
                xtwx[[r, c]] += x[[i, r]] * w[i] * x[[i, c]];
            }
        }
    }
    let a = &xtwx + &s_lambda;
    let a_inv = a.inv().unwrap();

    // 3. lev_uw[i] = x_i' A_inv x_i; h[i] = w[i]·lev_uw[i] = diag(KK')[i].
    let xa = x.dot(&a_inv);
    let mut lev_uw = Array1::<f64>::zeros(N);
    for i in 0..N {
        let mut s = 0.0;
        for j in 0..P {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    // 4. a1 (= dw/dη) using the same formulas as reml.rs.
    let mut a1 = Array1::<f64>::zeros(N);
    for i in 0..N {
        let dmu_deta = family.d_inverse_link(eta[i]);
        let g1 = 1.0 / dmu_deta;
        let v = family.variance(mu[i]);
        let v1n = family.dvar(mu[i]) / v;
        let v2n = family.d2var(mu[i]) / v;
        let g2n = family.d2link(mu[i]) * dmu_deta;
        let g3n = family.d3link(mu[i]) * dmu_deta;
        let alpha = alpha_vec[i];
        let xx = v2n - v1n * v1n + g3n - g2n * g2n;
        let alpha1 = (-(v1n + g2n) + (y[i] - mu[i]) * xx) / alpha;
        a1[i] = w[i] * (alpha1 - v1n - 2.0 * g2n) / g1;
    }
    // Sanity: Gamma+log has a1 = -y/μ = -w (per Python script comment).
    for i in 0..N {
        let expected = -y[i] / mu[i];
        assert!(
            (a1[i] - expected).abs() < 1e-9,
            "a1[{}]={} vs expected -y/mu={} (diff {})",
            i,
            a1[i],
            expected,
            (a1[i] - expected).abs()
        );
    }

    // 5. b1[:,k] = -λ_k · A_inv · S_k · β; eta1 = X·b1.
    let mut b1 = Array2::<f64>::zeros((P, M));
    for k in 0..M {
        let s_k_beta = s_blocks[k].dot(&beta);
        let ainv_sk_beta = a_inv.dot(&s_k_beta);
        for r in 0..P {
            b1[[r, k]] = -LAMBDA[k] * ainv_sk_beta[r];
        }
    }
    let eta1 = x.dot(&b1);

    // 6. tk_kkt[k] = Σᵢ a1[i] · eta1[i,k] · sign(w[i]) · lev_uw[i].
    for k in 0..M {
        let mut tk_kkt = 0.0;
        for i in 0..N {
            tk_kkt += a1[i] * eta1[[i, k]] * w[i].signum() * lev_uw[i];
        }
        let diff = (tk_kkt - TK_KKT_REF[k]).abs();
        let rel = diff / TK_KKT_REF[k].abs();
        println!(
            "k={}: rust_tk_kkt={:+.15e}, ref={:+.15e}, |diff|={:.3e}, rel={:.3e}",
            k, tk_kkt, TK_KKT_REF[k], diff, rel
        );
        assert!(
            rel < 1e-9,
            "tk_kkt[{}] mismatch: rust={}, ref={}, rel={}",
            k,
            tk_kkt,
            TK_KKT_REF[k],
            rel
        );
    }
}
