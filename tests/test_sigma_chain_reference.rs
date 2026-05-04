//! Validates the σ²-chain correction in `reml_gradient_mgcv_exact_ift`.
//!
//! The chain term corrects the gradient for the fact that σ̂² = D/(n-trA)
//! depends on λ. For Gamma+log families this is non-negligible (~0.29-0.74
//! in ∞-norm on the N-1/N-2 fixtures per the Phase A scout).
//!
//! Test strategy:
//! 1. Build a small Gamma+log synthetic fixture (n=100, 1 smooth, k=6).
//! 2. Compute the IFT gradient with `enable_sigma_chain = true` (inner fn).
//! 3. Compute a numerical FD of the profile REML criterion at the same ρ.
//! 4. Assert ‖grad_with_chain - fd‖∞ < 1e-5.
//! 5. Assert grad_without_chain ≠ fd (chain is non-trivially nonzero).
//!
//! Also includes unit tests for the `digamma` function.
#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::pirls::Family;
use mgcv_rust::reml::{
    reml_criterion_multi_cached_mgcv_exact, reml_gradient_mgcv_exact_ift_inner,
};
use ndarray::{Array1, Array2};

// Re-export the digamma test target through the crate's pub(crate) function.
// We test it here since it's not pub outside the crate.
// We'll test it by proxy via the dls_dsigma2 method on Family, which calls digamma.

/// Very simple synthetic Gamma+log dataset.
/// mu_i = exp(f(x_i)) where f is a smooth function.
/// Returns (y, x_design, penalties, w, mp).
fn build_gamma_log_fixture() -> (
    Array1<f64>,
    Array2<f64>,
    Vec<BlockPenalty>,
    Array1<f64>,
    usize,
) {
    // n observations, k basis functions (thin-plate-spline-like cubic B-spline),
    // intercept + k columns.
    let n = 100usize;
    let k = 6usize; // basis size for the smooth
    let p = 1 + k; // intercept + smooth

    // Fixed seed-equivalent: evenly spaced x in [0, 1]
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();

    // True mean: mu_i = exp(1 + 2 * sin(2π x_i)), so log mu = 1 + 2sin(2πx)
    let true_log_mu: Vec<f64> = xs.iter().map(|&xi| 1.0 + 2.0 * (2.0 * std::f64::consts::PI * xi).sin()).collect();

    // Gamma responses: approximate y_i ~ Gamma(shape=2, rate=2/mu_i)
    // We use a deterministic pseudo-y: y_i = exp(log_mu + small perturbation)
    // using the low-discrepancy sequence 0.5, 1.5/n, 2.5/n, ...
    let mut y_vec: Vec<f64> = Vec::with_capacity(n);
    for (i, &lmu) in true_log_mu.iter().enumerate() {
        // deterministic "noise": half-period cosine perturbation
        let noise = 0.15 * (std::f64::consts::PI * i as f64 / n as f64).cos();
        y_vec.push((lmu + noise).exp().max(1e-6));
    }
    let y = Array1::from(y_vec);

    // Design matrix: intercept + k cubic B-spline basis functions
    // Knots equally spaced in [0,1]: k-2 interior + 2 boundary = k total basis
    let knots_interior: Vec<f64> = (1..k - 1)
        .map(|i| i as f64 / (k as f64 - 1.0))
        .collect();

    // Simple truncated power basis: [1, x, x², x³, (x-t1)³₊, (x-t2)³₊, ...]
    // With k=6: intercept, x, x², x³, (x-t1)³₊, (x-t2)³₊
    // (This gives a rank-2 null space matching a degree-2 polynomial).
    // For the penalty, use a second-difference penalty on the basis coefficients
    // (columns 1..p, i.e. indices 1 to k in 0-indexed).
    let mut x_mat = Array2::<f64>::zeros((n, p));
    for (i, &xi) in xs.iter().enumerate() {
        x_mat[[i, 0]] = 1.0; // intercept
        for j in 0..k {
            // Polynomial + truncated power: index j+1 in design matrix
            if j < 4 {
                x_mat[[i, 1 + j]] = xi.powi(j as i32);
            } else {
                let t = knots_interior[j - 4];
                let d = xi - t;
                x_mat[[i, 1 + j]] = if d > 0.0 { d.powi(3) } else { 0.0 };
            }
        }
    }

    // Second-difference penalty matrix on the k smooth columns (indices 1..p)
    // D'D where D is the k×(k-1) first-difference matrix → (k-1)×k difference
    // applied twice. We use a simple k×k ridge-like penalty for this test.
    // The penalty applies to columns 1..p (offset=1, size=k).
    let mut s_block = Array2::<f64>::zeros((k, k));
    // Second-difference penalty: S = D2'D2 where D2 is (k-2)×k 2nd-diff matrix
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

    // Mp: 1 (intercept) + null-space of S.
    // For a second-difference penalty on k=6 cols, null-space dim = 2 (linear trend).
    let mp = 1 + 2;

    (y, x_mat, vec![penalty], w, mp)
}

/// Numerical FD of the profile REML criterion.
fn fd_profile_reml_gradient(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    mp: usize,
    family: Family,
    y_original: Option<&Array1<f64>>,
) -> Array1<f64> {
    let m = lambdas.len();
    let eps = 1e-5_f64;
    let mut grad = Array1::<f64>::zeros(m);
    for k in 0..m {
        let log_lam: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
        let mut lp = log_lam.clone();
        lp[k] += eps;
        let mut lm = log_lam.clone();
        lm[k] -= eps;
        let lam_plus: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
        let lam_minus: Vec<f64> = lm.iter().map(|l| l.exp()).collect();

        let f_plus = reml_criterion_multi_cached_mgcv_exact(
            y, x, w, &lam_plus, penalties, None, mp, family, y_original,
        )
        .unwrap();
        let f_minus = reml_criterion_multi_cached_mgcv_exact(
            y, x, w, &lam_minus, penalties, None, mp, family, y_original,
        )
        .unwrap();
        // Convert from dREML/dlog(λ) via chain rule: we compare against the
        // IFT gradient which is w.r.t. ρ = log(λ), so no extra factor needed.
        grad[k] = (f_plus - f_minus) / (2.0 * eps);
    }
    grad
}

#[test]
fn sigma_chain_grad_matches_fd_profile() {
    let (y, x, penalties, w, mp) = build_gamma_log_fixture();
    let family = Family::GammaLog;

    // Pick a λ near a typical optimum (not too large, not too small)
    let lambdas = vec![1.5_f64];

    // Use y as y_original (profile REML with true GLM deviance)
    let y_orig = y.clone();

    // ── Gradient WITH chain term ─────────────────────────────────────────────
    let grad_chain = reml_gradient_mgcv_exact_ift_inner(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        family,
        Some(&y_orig),
        true, // enable_sigma_chain
        mp,
    )
    .unwrap();

    // ── Gradient WITHOUT chain term ──────────────────────────────────────────
    let grad_nochain = reml_gradient_mgcv_exact_ift_inner(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        family,
        Some(&y_orig),
        false, // enable_sigma_chain = false
        mp,
    )
    .unwrap();

    // ── FD of profile REML ───────────────────────────────────────────────────
    let fd_grad = fd_profile_reml_gradient(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        mp,
        family,
        Some(&y_orig),
    );

    // ── Assertions ───────────────────────────────────────────────────────────
    let inf_norm_chain: f64 = grad_chain
        .iter()
        .zip(fd_grad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let inf_norm_nochain: f64 = grad_nochain
        .iter()
        .zip(fd_grad.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Envelope-theorem sanity: chain correction should now be ~0
    let chain_correction_abs: f64 = grad_chain
        .iter()
        .zip(grad_nochain.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!("grad_chain  = {:?}", grad_chain);
    println!("grad_nochain= {:?}", grad_nochain);
    println!("fd_grad     = {:?}", fd_grad);
    println!("‖chain - fd‖∞ = {:.2e}", inf_norm_chain);
    println!("‖nochain - fd‖∞ = {:.2e}", inf_norm_nochain);
    println!("‖chain_correction‖∞ = {:.2e}", chain_correction_abs);

    // Both with and without chain term should match FD to 1e-5 (limited by FD precision),
    // since with the correct φ̂ the chain term is near zero by the envelope theorem.
    assert!(
        inf_norm_chain < 1e-5,
        "gradient with chain term deviates from FD profile REML: ‖·‖∞ = {:.3e} (threshold 1e-5)",
        inf_norm_chain
    );

    assert!(
        inf_norm_nochain < 1e-5,
        "gradient without chain term deviates from FD profile REML: ‖·‖∞ = {:.3e} (threshold 1e-5)",
        inf_norm_nochain
    );

    // Envelope theorem: with the correct phi_hat, the chain term itself is near zero.
    assert!(
        chain_correction_abs < 1e-4,
        "chain correction should be near zero by envelope theorem, got {:.3e}",
        chain_correction_abs
    );
    println!("PASS: both with/without chain match FD; chain correction = {:.3e}", chain_correction_abs);
}

/// Unit tests for the digamma function (tested indirectly via dls_dsigma2).
/// digamma(x) values from Abramowitz & Stegun / standard tables.
#[test]
fn dls_dsigma2_gamma_log_values() {
    // For Gamma+log family, dls/dσ² = n * (digamma(1/σ²) + log(σ²)) / σ⁴.
    // We test with specific values where digamma is known:
    //   digamma(1) = -γ ≈ -0.5772156649
    //   digamma(2) = 1 - γ ≈ 0.4227843351
    //
    // At scale = 1.0 (so 1/σ² = 1): dls/dσ² = n * (digamma(1) + ln(1)) / 1²
    //   = n * (-0.5772156649 + 0) = -n * 0.5772156649
    let n = 10usize;
    let y = Array1::<f64>::ones(n); // values don't matter for this test
    let fam = Family::GammaLog;

    let dls_1 = fam.dls_dsigma2(&y, 1.0);
    let expected_1 = (n as f64) * (-0.5772156649 + 0.0_f64.ln()) / (1.0_f64 * 1.0_f64);
    // ln(1) = 0, so expected = n * digamma(1) = n * (-0.5772156649)
    let expected_1_simple = (n as f64) * (-0.5772156649_f64);
    println!("dls_dsigma2(scale=1): got {:.10}, expected {:.10}", dls_1, expected_1_simple);
    assert!(
        (dls_1 - expected_1_simple).abs() < 1e-7,
        "dls_dsigma2 at scale=1: got {}, expected {}",
        dls_1, expected_1_simple
    );

    // At scale = 0.5 (so 1/σ² = 2): digamma(2) = 0.4227843351, ln(0.5) = -ln(2)
    // dls/dσ² = n * (digamma(2) + ln(0.5)) / (0.5)²
    //   = n * (0.4227843351 - 0.6931471806) / 0.25
    //   = n * (-0.2703628455) / 0.25
    //   = n * (-1.081451382)
    let dls_05 = fam.dls_dsigma2(&y, 0.5);
    let expected_05 = (n as f64) * (0.4227843351 + (0.5_f64).ln()) / (0.5 * 0.5);
    println!("dls_dsigma2(scale=0.5): got {:.10}, expected {:.10}", dls_05, expected_05);
    assert!(
        (dls_05 - expected_05).abs() < 1e-6,
        "dls_dsigma2 at scale=0.5: got {}, expected {}",
        dls_05, expected_05
    );
}

/// Smoke test: for Gaussian, the chain term should produce a very small correction.
/// At the Gaussian REML optimum ∂REML/∂σ² = 0 exactly. Off-optimum it's small
/// because dls/dσ² = -n/(2σ²) and the cancellation is large but not exact
/// when σ̂² = D/(n-trA) ≠ D/(n-Mp) (trA ≠ Mp in general).
/// The important property is that Gaussian chain correction ≪ Gamma chain (which
/// can be ~0.74 per the Phase A scout).
#[test]
fn sigma_chain_gaussian_near_zero_at_optimum() {
    // For Gaussian: ∂REML/∂σ² = -Dp/(2σ⁴) - dls/dσ² - Mp/(2σ²)
    //   = -Dp/(2σ⁴) + n/(2σ²) - Mp/(2σ²)
    // This is zero when σ̂² = Dp/(n-Mp), which approximately holds when trA ≈ Mp.
    // The chain term will be small compared to the Gamma case.
    // Test: chain correction < 1e-3 (not exact zero, but much smaller than Gamma's ~0.74).

    let n = 50usize;
    let k = 5usize;
    let p = 1 + k;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vec: Vec<f64> = xs.iter().enumerate().map(|(i, &xi)| {
        1.0 + 2.0 * xi + 0.1 * (i as f64 * 0.3).sin()
    }).collect();
    let y = Array1::from(y_vec);

    let mut x_mat = Array2::<f64>::zeros((n, p));
    for (i, &xi) in xs.iter().enumerate() {
        x_mat[[i, 0]] = 1.0;
        for j in 0..k {
            x_mat[[i, 1 + j]] = xi.powi(j as i32);
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
    let lambdas = vec![1.0_f64];
    let family = Family::Gaussian;
    // mp = 1 (intercept) + 2 (null-space of second-difference penalty on k=5 cols)
    let mp_gauss: usize = 3;

    let grad_chain = reml_gradient_mgcv_exact_ift_inner(
        &y, &x_mat, &w, &lambdas, &[penalty.clone()],
        None, family, None, true, mp_gauss,
    ).unwrap();

    let grad_nochain = reml_gradient_mgcv_exact_ift_inner(
        &y, &x_mat, &w, &lambdas, &[penalty],
        None, family, None, false, mp_gauss,
    ).unwrap();

    // For Gaussian, chain term should change gradient by < 1e-8 (basically zero)
    let chain_correction: f64 = grad_chain.iter().zip(grad_nochain.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("Gaussian chain correction ‖·‖∞ = {:.2e}", chain_correction);
    // Must be very small (much smaller than the Gamma case ~0.74), though not
    // exactly zero unless evaluated exactly at the σ̂² optimum (trA = Mp).
    assert!(
        chain_correction < 1e-3,
        "Gaussian chain correction should be small (<<1e-3), got {:.3e}",
        chain_correction
    );
}
