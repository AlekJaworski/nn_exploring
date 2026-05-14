//! End-to-end Rust-level test that proves the D4 discrete-binning wiring
//! works. Fits a Gaussian GAM on a fixture twice — once with the un-binned
//! BLAS path (`discrete_enabled=false`, the default), once with the discrete
//! fast path forced on (`discrete_enabled=true`) — and asserts the
//! coefficients agree within the design-doc tolerance (5e-3 absolute /
//! 1e-3 relative on coefficients).
//!
//! This is the load-bearing wiring test: if it fails, the discrete kernel
//! is producing different numerics than the full GEMM. Tolerances are loose
//! per `docs/DISCRETE_BINNING_DESIGN.md` §6 because the discrete path is an
//! admitted approximation (it inherits mgcv's published bias from
//! `bam(discrete=TRUE)` vs `gam`).
//!
//! With the fixture below (n=2400, all-unique covariates with k=10 cr
//! smooth), the compress.df path runs in pure-dedup mode (zero
//! approximation error) so we expect agreement to ~1e-8 or better in
//! practice. The 5e-3 tolerance is the design floor.

#![cfg(feature = "blas")]

use mgcv_rust::gam::{SmoothTerm, GAM};
use mgcv_rust::pirls::Family;
use mgcv_rust::smooth::OptimizationMethod;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Build a fresh 1D Gaussian GAM with the same fixture used by both fits.
fn build_gam(x: &Array2<f64>, k: usize) -> GAM {
    let mut gam = GAM::new(Family::Gaussian);
    let col = x.column(0).to_owned();
    let smooth = SmoothTerm::cr_spline_quantile("x".to_string(), k, &col).unwrap();
    gam.add_smooth(smooth);
    gam
}

#[test]
fn discrete_path_matches_undiscretized_gaussian_1d() {
    // Fixture: n=2400 (above the n >= 2000 discretize gate), 1D smooth.
    // The covariate is sampled densely from [-1, 1] — pure-dedup binning
    // applies (every observation has a unique x value, and 2400 > 1000
    // so we'd expect quantile-grid binning to kick in for the smooth).
    let n = 2400usize;
    let k = 10usize;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut x_data = Vec::with_capacity(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = rng.gen_range(-1.0..1.0);
        x_data.push(xi);
        let mu = (2.0 * xi).sin() + 0.3 * xi;
        let noise: f64 = rng.gen_range(-0.5..0.5);
        y[i] = mu + 0.2 * noise;
    }
    let x = Array2::from_shape_vec((n, 1), x_data).unwrap();

    // ── Fit 1: discrete disabled (the default, byte-identical to master). ──
    let mut gam_full = build_gam(&x, k);
    gam_full
        .fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("un-binned fit");
    let beta_full = gam_full.coefficients.clone().expect("coefficients");

    // ── Fit 2: discrete forced on. ──
    let mut gam_disc = build_gam(&x, k);
    gam_disc.discrete_enabled = true;
    gam_disc
        .fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("discrete fit");
    let beta_disc = gam_disc.coefficients.clone().expect("coefficients");

    // Shapes must match exactly — discrete path is a perf opt, not a model change.
    assert_eq!(
        beta_full.len(),
        beta_disc.len(),
        "coefficient vector length must match (full={}, disc={})",
        beta_full.len(),
        beta_disc.len()
    );

    // Coefficient parity within the design tolerance.
    let max_abs_diff: f64 = beta_full
        .iter()
        .zip(beta_disc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let max_abs_beta: f64 = beta_full
        .iter()
        .map(|x: &f64| x.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let max_rel_diff = max_abs_diff / max_abs_beta;

    println!(
        "[discrete e2e] max abs Δβ = {:.3e}, max rel Δβ = {:.3e} (n={}, k={})",
        max_abs_diff, max_rel_diff, n, k
    );
    println!("[discrete e2e] β_full = {:?}", beta_full);
    println!("[discrete e2e] β_disc = {:?}", beta_disc);

    assert!(
        max_abs_diff < 5e-3,
        "max abs Δβ = {} exceeds 5e-3 tolerance (binning bias floor from \
         docs/DISCRETE_BINNING_DESIGN.md §6)",
        max_abs_diff
    );
    assert!(
        max_rel_diff < 1e-3,
        "max rel Δβ = {} exceeds 1e-3 tolerance",
        max_rel_diff
    );
}

/// Sanity test: when `discrete_enabled=false` (the default), the fit
/// must be byte-identical to a fit that doesn't touch the field at all.
/// This guards against accidental enablement of the discrete path in
/// the no-opt-in case.
#[test]
fn discrete_disabled_byte_identical_to_default() {
    let n = 2400usize;
    let k = 8usize;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut x_data = Vec::with_capacity(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi: f64 = rng.gen_range(0.0..1.0);
        x_data.push(xi);
        y[i] = (3.0 * xi).cos() + 0.1 * (i as f64).sin();
    }
    let x = Array2::from_shape_vec((n, 1), x_data).unwrap();

    let mut gam_default = build_gam(&x, k);
    gam_default
        .fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("default fit");
    let beta_default = gam_default.coefficients.clone().unwrap();

    let mut gam_explicit_off = build_gam(&x, k);
    gam_explicit_off.discrete_enabled = false; // explicit, should be a no-op
    gam_explicit_off
        .fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("explicit-off fit");
    let beta_explicit_off = gam_explicit_off.coefficients.clone().unwrap();

    for (a, b) in beta_default.iter().zip(beta_explicit_off.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "discrete_enabled=false must be byte-identical to default: \
             {} vs {}",
            a,
            b
        );
    }
}
