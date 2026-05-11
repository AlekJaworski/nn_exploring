//! Validates `tk_kkt_hessian_fd` against an independent brute-force FD of the
//! analytical T_k = Σᵢ a1·η₁[:,k]·sign(w)·lev_uw quantity.
//!
//! Strategy: T_k is the analytical Tk·KK' gradient term that we already know
//! is correct (see `tests/test_tk_kkt_reference.rs` validating it against a
//! Python brute force). The IFT gradient code path adds `tk_kkt[k]/2` to
//! grad[k] iff a family flag is hit OR `MGCV_TK_GRAD` is set. So for a
//! non-default-on family (e.g. `GammaLog`):
//!
//!   tk_kkt_at(λ)[k] = 2 · (grad_with_tk_env_set[k] − grad_without[k])
//!
//! Brute-force central-FD this against ρ_j gives `∂T_k/∂ρ_j`, which is what
//! `tk_kkt_hessian_fd(...)` computes. We assert rel < 1e-3 across all (k,j).
//!
//! Both Gaussian and pure-canonical Binomial give very small T_k near a fitted
//! β so we use GammaLog where the term is comfortably above FD noise.
#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::pirls::Family;
use mgcv_rust::reml::{reml_gradient_mgcv_exact_ift_inner, tk_kkt_hessian_fd};
use ndarray::{Array1, Array2};
use std::sync::Mutex;

/// Serializes the env-var dance in `tk_kkt_at_lambdas` so concurrent tests
/// don't race over `MGCV_TK_GRAD`. Cargo's default test threading would
/// otherwise produce nondeterministic gradient values.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Gamma+log fixture (non-default-Tk family) with one smooth.
fn build_gamma_log_1smooth() -> (
    Array1<f64>,
    Array2<f64>,
    Vec<BlockPenalty>,
    Array1<f64>,
    usize,
) {
    let n = 100usize;
    let k = 6usize;
    let p = 1 + k;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let true_log_mu: Vec<f64> = xs
        .iter()
        .map(|&xi| 1.0 + 2.0 * (2.0 * std::f64::consts::PI * xi).sin())
        .collect();
    let mut y_vec: Vec<f64> = Vec::with_capacity(n);
    for (i, &lmu) in true_log_mu.iter().enumerate() {
        let noise = 0.15 * (std::f64::consts::PI * i as f64 / n as f64).cos();
        y_vec.push((lmu + noise).exp().max(1e-6));
    }
    let y = Array1::from(y_vec);

    let knots_interior: Vec<f64> = (1..k - 1)
        .map(|i| i as f64 / (k as f64 - 1.0))
        .collect();
    let mut x_mat = Array2::<f64>::zeros((n, p));
    for (i, &xi) in xs.iter().enumerate() {
        x_mat[[i, 0]] = 1.0;
        for j in 0..k {
            if j < 4 {
                x_mat[[i, 1 + j]] = xi.powi(j as i32);
            } else {
                let t = knots_interior[j - 4];
                let d = xi - t;
                x_mat[[i, 1 + j]] = if d > 0.0 { d.powi(3) } else { 0.0 };
            }
        }
    }
    let mut s_block = Array2::<f64>::zeros((k, k));
    for i in 0..(k - 2) {
        s_block[[i, i]] += 1.0;
        s_block[[i, i + 1]] -= 2.0;
        s_block[[i, i + 2]] += 1.0;
        s_block[[i + 1, i]] -= 2.0;
        s_block[[i + 1, i + 1]] += 4.0;
        s_block[[i + 1, i + 2]] -= 2.0;
        s_block[[i + 2, i]] += 1.0;
        s_block[[i + 2, i + 1]] -= 2.0;
        s_block[[i + 2, i + 2]] += 1.0;
    }
    let penalty = BlockPenalty::new(s_block, 1, p);
    let w = Array1::<f64>::ones(n);
    let mp = 1 + 2;
    (y, x_mat, vec![penalty], w, mp)
}

/// Two-smooth Gamma+log fixture so we get a 2×2 cross-derivative matrix.
fn build_gamma_log_2smooth() -> (
    Array1<f64>,
    Array2<f64>,
    Vec<BlockPenalty>,
    Array1<f64>,
    usize,
) {
    let n = 120usize;
    let k1 = 5usize;
    let k2 = 5usize;
    let p = 1 + k1 + k2;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
    let zs: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 0.79_f64).fract()))
        .collect();
    let mut y_vec: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        // Keep η in roughly [-1, 2] so μ stays in [0.4, 7] — well away from
        // saturation. Big-η blows up the Gamma deviance gradient and trashes
        // the FD signal-to-noise ratio.
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
    let mp = 1 + 2 + 2;
    (y, x_mat, vec![penalty1, penalty2], w, mp)
}

/// Computes `tk_kkt[k]` for a given λ by taking
///     2·(grad_with_TK_GRAD_env − grad_without)
/// in the IFT inner gradient. GammaLog is not in the default-on list, so the
/// env toggle exactly swings the Tk contribution on/off without side effects.
fn tk_kkt_at_lambdas(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    y_original: Option<&Array1<f64>>,
    mp: usize,
) -> Array1<f64> {
    // Serialize the env-var dance and capture+restore so tests don't race
    // over MGCV_TK_GRAD (Cargo runs tests in parallel threads).
    let _guard = ENV_MUTEX.lock().unwrap();
    let saved = std::env::var("MGCV_TK_GRAD").ok();
    std::env::set_var("MGCV_TK_GRAD", "1");
    let g_with = reml_gradient_mgcv_exact_ift_inner(
        y, x, w, lambdas, penalties, None, family, y_original, false, mp,
    )
    .unwrap();
    std::env::remove_var("MGCV_TK_GRAD");
    let g_without = reml_gradient_mgcv_exact_ift_inner(
        y, x, w, lambdas, penalties, None, family, y_original, false, mp,
    )
    .unwrap();
    // restore
    match saved {
        Some(v) => std::env::set_var("MGCV_TK_GRAD", v),
        None => std::env::remove_var("MGCV_TK_GRAD"),
    }
    drop(_guard);
    let m = lambdas.len();
    let mut tk = Array1::<f64>::zeros(m);
    for k in 0..m {
        tk[k] = 2.0 * (g_with[k] - g_without[k]);
    }
    tk
}

/// Brute-force central FD: `∂T_k/∂ρ_j` via `tk_kkt_at_lambdas` at ρ ± h.
fn fd_tk_kkt_brute(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    y_original: Option<&Array1<f64>>,
    mp: usize,
    h: f64,
) -> Array2<f64> {
    let m = lambdas.len();
    let mut out = Array2::<f64>::zeros((m, m));
    let log_lam: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
    for j in 0..m {
        let mut lp = log_lam.clone();
        lp[j] += h;
        let mut lm = log_lam.clone();
        lm[j] -= h;
        let lam_plus: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
        let lam_minus: Vec<f64> = lm.iter().map(|l| l.exp()).collect();
        let tk_plus =
            tk_kkt_at_lambdas(y, x, w, &lam_plus, penalties, family, y_original, mp);
        let tk_minus =
            tk_kkt_at_lambdas(y, x, w, &lam_minus, penalties, family, y_original, mp);
        for k in 0..m {
            out[[k, j]] = (tk_plus[k] - tk_minus[k]) / (2.0 * h);
        }
    }
    out
}

fn run_test_case(
    name: &str,
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    family: Family,
    y_original: Option<&Array1<f64>>,
    mp: usize,
) {
    // Helper under test (uses h=1e-4 internally)
    let h_helper = tk_kkt_hessian_fd(
        y,
        x,
        w,
        lambdas,
        penalties,
        None,
        family,
        y_original,
    )
    .unwrap();
    // Brute-force reference uses h_ref=1e-6 — two orders below the helper's
    // h=1e-4. Same-h reference only proves implementation consistency, not
    // FD-convergence rigor. With h_ref << h_helper, the helper's O(h²)
    // truncation dominates the disagreement, so a small max-rel proves the
    // helper's h is small enough.
    let h_ref = fd_tk_kkt_brute(
        y,
        x,
        w,
        lambdas,
        penalties,
        family,
        y_original,
        mp,
        1.0e-6_f64,
    );

    let m = lambdas.len();
    let mut max_diff = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut max_mag = 0.0_f64;
    for k in 0..m {
        for j in 0..m {
            max_mag = max_mag
                .max(h_helper[[k, j]].abs())
                .max(h_ref[[k, j]].abs());
        }
    }
    for k in 0..m {
        for j in 0..m {
            let diff = (h_helper[[k, j]] - h_ref[[k, j]]).abs();
            max_diff = max_diff.max(diff);
            let denom = h_helper[[k, j]].abs().max(h_ref[[k, j]].abs());
            if denom > 1e-9 * max_mag {
                max_rel = max_rel.max(diff / denom);
            }
            println!(
                "[{}] ({},{}) helper={:+.6e}, ref={:+.6e}, |diff|={:.3e}",
                name,
                k,
                j,
                h_helper[[k, j]],
                h_ref[[k, j]],
                diff
            );
        }
    }
    println!(
        "[{}] max|diff|={:.3e}, max rel={:.3e}, max|H|={:.3e}",
        name, max_diff, max_rel, max_mag
    );
    assert!(
        max_rel < 1e-3,
        "[{}] tk_kkt_hessian_fd does not match brute-force FD of T_k: max rel={}, abs={}",
        name,
        max_rel,
        max_diff
    );
}

#[test]
fn tk_kkt_hessian_fd_matches_brute_force_gamma_log_1d() {
    let (y, x, penalties, w, mp) = build_gamma_log_1smooth();
    let lambdas = vec![1.0_f64];
    let y_orig = y.clone();
    run_test_case(
        "gamma_log_1d",
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Family::GammaLog,
        Some(&y_orig),
        mp,
    );
}

#[test]
fn tk_kkt_hessian_fd_matches_brute_force_gamma_log_2d() {
    let (y, x, penalties, w, mp) = build_gamma_log_2smooth();
    let lambdas = vec![1.5_f64, 0.7_f64];
    let y_orig = y.clone();
    run_test_case(
        "gamma_log_2d",
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Family::GammaLog,
        Some(&y_orig),
        mp,
    );
}
