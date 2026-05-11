//! Validates `tk_kkt_hessian_analytical` against `tk_kkt_hessian_fd` on the
//! two-smooth Gamma+log fixture from `tests/test_tk_kkt_hessian.rs`.
//!
//! The FD path central-differences the analytical Tk vector with h=1e-4
//! (truncation O(h²) ~ 1e-8 plus cancellation noise). The analytical path
//! computes the same `det2_tk = Σᵢ Tkm·sign(w)·lev_uw − tr(C_k·C_j)` form
//! from gdi.c:919-923 directly. They should agree to about FD precision —
//! we use a 5e-3 relative tolerance to leave headroom for FD cancellation.
#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::pirls::Family;
use mgcv_rust::reml::{tk_kkt_hessian_analytical, tk_kkt_hessian_fd};
use ndarray::{Array1, Array2};

/// Two-smooth Gamma+log fixture (copy of `build_gamma_log_2smooth` from
/// `tests/test_tk_kkt_hessian.rs`) so we get a 2×2 matrix.
fn build_gamma_log_2smooth() -> (
    Array1<f64>,
    Array2<f64>,
    Vec<BlockPenalty>,
    Array1<f64>,
) {
    let n = 120usize;
    let k1 = 5usize;
    let k2 = 5usize;
    let p = 1 + k1 + k2;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let zs: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.79_f64).fract())
        .collect();
    let mut y_vec: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let lmu = 0.5
            + 0.8 * (2.0 * std::f64::consts::PI * xs[i]).sin()
            + 0.4 * zs[i];
        let noise = 0.12 * (std::f64::consts::PI * (2 * i + 1) as f64 / n as f64).cos();
        y_vec.push((lmu + noise).exp().max(1e-6));
    }
    let y = Array1::from(y_vec);

    let kn1: Vec<f64> = (1..k1 - 1)
        .map(|i| i as f64 / (k1 as f64 - 1.0))
        .collect();
    let kn2: Vec<f64> = (1..k2 - 1)
        .map(|i| i as f64 / (k2 as f64 - 1.0))
        .collect();
    let mut x_mat = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x_mat[[i, 0]] = 1.0;
        let xi = xs[i];
        for j in 0..k1 {
            if j < 4 {
                x_mat[[i, 1 + j]] = xi.powi(j as i32);
            } else {
                let t = kn1[j - 4];
                let d = xi - t;
                x_mat[[i, 1 + j]] = if d > 0.0 { d.powi(3) } else { 0.0 };
            }
        }
        let zi = zs[i];
        for j in 0..k2 {
            if j < 4 {
                x_mat[[i, 1 + k1 + j]] = zi.powi(j as i32);
            } else {
                let t = kn2[j - 4];
                let d = zi - t;
                x_mat[[i, 1 + k1 + j]] = if d > 0.0 { d.powi(3) } else { 0.0 };
            }
        }
    }
    let mut s1 = Array2::<f64>::zeros((k1, k1));
    for i in 0..(k1 - 2) {
        s1[[i, i]] += 1.0;
        s1[[i, i + 1]] -= 2.0;
        s1[[i, i + 2]] += 1.0;
        s1[[i + 1, i]] -= 2.0;
        s1[[i + 1, i + 1]] += 4.0;
        s1[[i + 1, i + 2]] -= 2.0;
        s1[[i + 2, i]] += 1.0;
        s1[[i + 2, i + 1]] -= 2.0;
        s1[[i + 2, i + 2]] += 1.0;
    }
    let mut s2 = Array2::<f64>::zeros((k2, k2));
    for i in 0..(k2 - 2) {
        s2[[i, i]] += 1.0;
        s2[[i, i + 1]] -= 2.0;
        s2[[i, i + 2]] += 1.0;
        s2[[i + 1, i]] -= 2.0;
        s2[[i + 1, i + 1]] += 4.0;
        s2[[i + 1, i + 2]] -= 2.0;
        s2[[i + 2, i]] += 1.0;
        s2[[i + 2, i + 1]] -= 2.0;
        s2[[i + 2, i + 2]] += 1.0;
    }
    let penalty1 = BlockPenalty::new(s1, 1, p);
    let penalty2 = BlockPenalty::new(s2, 1 + k1, p);
    let w = Array1::<f64>::ones(n);
    (y, x_mat, vec![penalty1, penalty2], w)
}

#[test]
fn tk_kkt_hessian_analytical_matches_fd_gamma_log_2d() {
    let (y, x, penalties, w) = build_gamma_log_2smooth();
    let lambdas = vec![1.5_f64, 0.7_f64];
    let y_orig = y.clone();

    let h_fd = tk_kkt_hessian_fd(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        Family::GammaLog,
        Some(&y_orig),
    )
    .unwrap();
    let h_an = tk_kkt_hessian_analytical(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        Family::GammaLog,
        Some(&y_orig),
    )
    .unwrap();

    let m = lambdas.len();
    let mut max_mag = 0.0_f64;
    let mut max_diff = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for k in 0..m {
        for j in 0..m {
            max_mag = max_mag
                .max(h_fd[[k, j]].abs())
                .max(h_an[[k, j]].abs());
        }
    }
    println!("--- tk_kkt_hessian: FD vs analytical (GammaLog 2-smooth) ---");
    for k in 0..m {
        for j in 0..m {
            let diff = (h_fd[[k, j]] - h_an[[k, j]]).abs();
            max_diff = max_diff.max(diff);
            let denom = h_fd[[k, j]].abs().max(h_an[[k, j]].abs());
            if denom > 1e-9 * max_mag {
                max_rel = max_rel.max(diff / denom);
            }
            println!(
                "[gamma_log_2d] ({},{}) FD={:+.6e}, analytical={:+.6e}, |diff|={:.3e}",
                k,
                j,
                h_fd[[k, j]],
                h_an[[k, j]],
                diff
            );
        }
    }
    println!(
        "[gamma_log_2d] max|diff|={:.3e}, max rel={:.3e}, max|H|={:.3e}",
        max_diff, max_rel, max_mag
    );
    assert!(
        max_rel < 5e-3,
        "tk_kkt_hessian_analytical does not match FD: max rel={}, abs={}",
        max_rel,
        max_diff,
    );
}
