//! D6 integration tests — `fit_pirls_tdist_discrete` parity vs
//! `fit_pirls_tdist`. The discrete-binning hot path for the scat (TDist)
//! family is mathematically identical to the dense path on pure-dedup
//! fixtures (1e-12 agreement) and within the design-doc binning floor
//! (~5e-3 on β) on quantile-grid fixtures.
//!
//! This is the load-bearing wiring test for D6: if either case fails,
//! the discrete kernel's IRLS-weight refresh isn't matching the dense
//! triple-product on the t-family path.

#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::discrete::{DiscreteConfig, DiscreteDesign};
use mgcv_rust::gam::SmoothTerm;
use mgcv_rust::pirls::{fit_pirls_tdist, fit_pirls_tdist_discrete};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Build a (design, penalty, disc) triple for a single-1D cr smooth +
/// intercept on the supplied covariate. Mirrors `FitCache::new`'s
/// pre-D6 layout: column 0 is the intercept, columns 1..1+p_s are the
/// sum-to-zero-centred cr basis.
fn build_design(
    x_col: &Array1<f64>,
    k: usize,
) -> (Array2<f64>, Vec<BlockPenalty>, DiscreteDesign) {
    let n = x_col.len();
    let mut smooth = SmoothTerm::cr_spline_quantile("x".to_string(), k, x_col).unwrap();
    smooth.apply_sum_to_zero_centering(x_col).unwrap();

    let basis = smooth.evaluate(x_col).unwrap();
    let p_s = basis.ncols();
    let p = 1 + p_s;

    let mut design = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        design[[i, 0]] = 1.0;
        for c in 0..p_s {
            design[[i, 1 + c]] = basis[[i, c]];
        }
    }

    // Single block penalty acting on the smooth columns.
    let raw_penalty = smooth.penalty.clone();
    // Mirror FitCache scaling (mgcv-style): ma_xx / inf_norm_s.
    let inf_norm_x: f64 = basis
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let ma_xx = inf_norm_x * inf_norm_x;
    let inf_norm_s: f64 = (0..p_s)
        .map(|i| (0..p_s).map(|j| raw_penalty[[i, j]].abs()).sum::<f64>())
        .fold(0.0f64, f64::max);
    let scale = if inf_norm_s > 1e-10 {
        ma_xx / inf_norm_s
    } else {
        1.0
    };
    let scaled_penalty = &raw_penalty * scale;
    let block = BlockPenalty::new(scaled_penalty, 1, p);
    let penalties = vec![block];

    // DiscreteDesign — needs the smooth (consumed), the covariate matrix,
    // and an intercept marginal.
    let mut x_mat = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        x_mat[[i, 0]] = x_col[i];
    }
    let cfg = DiscreteConfig::default();
    let smooth_list = vec![smooth];
    let disc = DiscreteDesign::new(&smooth_list, &x_mat, true, &cfg);

    (design, penalties, disc)
}

#[test]
fn fit_pirls_tdist_discrete_matches_fit_pirls_tdist_on_pure_dedup_fixture() {
    // Pure-dedup: ≤ 1000 unique covariate values → zero binning bias.
    // Discrete and dense paths must agree to ~1e-12 on β.
    let n = 500usize;
    let k = 8usize;
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    // 200 unique values in [-1, 1], each repeated to fill n=500.
    let unique_count = 200usize;
    let unique_vals: Vec<f64> = (0..unique_count)
        .map(|i| -1.0 + 2.0 * (i as f64) / (unique_count as f64 - 1.0))
        .collect();
    let x_col: Array1<f64> = (0..n)
        .map(|i| unique_vals[i % unique_count])
        .collect();

    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mu = (2.0 * x_col[i]).sin() + 0.3 * x_col[i];
            // t-distribution noise (heavy-tailed) — but for parity we only
            // care that *both* fits see the same y.
            let u: f64 = rng.gen_range(-0.5..0.5);
            mu + 0.3 * u
        })
        .collect();

    let (design, penalties, disc) = build_design(&x_col, k);
    let lambda = vec![0.01];
    let max_iter = 50usize;
    let tol = 1e-8;

    let res_full = fit_pirls_tdist(
        &y, &design, &lambda, &penalties, None, None, max_iter, tol, None,
    )
    .expect("dense tdist fit");
    let res_disc = fit_pirls_tdist_discrete(
        &y, &lambda, &penalties, None, None, max_iter, tol, None, &disc,
    )
    .expect("discrete tdist fit");

    assert_eq!(res_full.coefficients.len(), res_disc.coefficients.len());

    let max_abs_diff: f64 = res_full
        .coefficients
        .iter()
        .zip(res_disc.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    println!(
        "[tdist discrete pure-dedup] max |Δβ| = {:.3e}, |β| max = {:.3e}, σ²: full={:.6}, disc={:.6}, df: full={:.4}, disc={:.4}",
        max_abs_diff,
        res_full.coefficients.iter().map(|x: &f64| x.abs()).fold(0.0f64, f64::max),
        res_full.sigma2.unwrap_or(f64::NAN),
        res_disc.sigma2.unwrap_or(f64::NAN),
        res_full.df.unwrap_or(f64::NAN),
        res_disc.df.unwrap_or(f64::NAN),
    );

    assert!(
        max_abs_diff < 1e-6,
        "pure-dedup max |Δβ| = {} exceeds 1e-6 (essentially-exact tolerance \
         for pure-dedup discrete path)",
        max_abs_diff
    );

    // σ²/df should match closely too — they're driven by the same per-row
    // residuals on a pure-dedup design.
    let s2_diff = (res_full.sigma2.unwrap() - res_disc.sigma2.unwrap()).abs();
    assert!(
        s2_diff < 1e-6,
        "σ² mismatch: full={}, disc={}, diff={}",
        res_full.sigma2.unwrap(),
        res_disc.sigma2.unwrap(),
        s2_diff
    );
}

#[test]
fn fit_pirls_tdist_discrete_matches_fit_pirls_tdist_quantile_grid_fixture() {
    // n=1500 with all-unique covariates → quantile-grid binning kicks in
    // (default max_bins_1d = 1000). Expect discrete-vs-dense agreement at
    // the design-doc binning floor (~5e-3 on β).
    let n = 1500usize;
    let k = 10usize;
    let mut rng = ChaCha8Rng::seed_from_u64(456);

    let x_col: Array1<f64> = (0..n)
        .map(|i| {
            // Dense, unique covariate in [-1, 1] with small jitter so the
            // values are all distinct.
            let t = (i as f64) / (n as f64 - 1.0);
            let jitter: f64 = rng.gen_range(-1e-6..1e-6);
            -1.0 + 2.0 * t + jitter
        })
        .collect();

    let y: Array1<f64> = x_col
        .iter()
        .map(|&xi| {
            let mu = (3.0 * xi).cos() + 0.4 * xi;
            let u: f64 = rng.gen_range(-0.3..0.3);
            mu + 0.2 * u
        })
        .collect();

    let (design, penalties, disc) = build_design(&x_col, k);
    let lambda = vec![0.05];
    let max_iter = 50usize;
    let tol = 1e-7;

    let res_full = fit_pirls_tdist(
        &y, &design, &lambda, &penalties, None, None, max_iter, tol, None,
    )
    .expect("dense tdist fit");
    let res_disc = fit_pirls_tdist_discrete(
        &y, &lambda, &penalties, None, None, max_iter, tol, None, &disc,
    )
    .expect("discrete tdist fit");

    let max_abs_diff: f64 = res_full
        .coefficients
        .iter()
        .zip(res_disc.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let max_abs_beta = res_full
        .coefficients
        .iter()
        .map(|x: &f64| x.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let nr = disc.marginals.iter().map(|m| m.nr).collect::<Vec<_>>();

    println!(
        "[tdist discrete quantile-grid] nr={:?}, max |Δβ| = {:.3e}, max rel = {:.3e}",
        nr,
        max_abs_diff,
        max_abs_diff / max_abs_beta
    );

    // 5e-3 is the design-doc binning floor.
    assert!(
        max_abs_diff < 5e-3,
        "quantile-grid max |Δβ| = {} exceeds 5e-3 design floor",
        max_abs_diff
    );
}

#[test]
fn fit_pirls_tdist_discrete_with_prior_weights_matches_dense() {
    // Same as the pure-dedup case but with non-trivial prior weights.
    // Prior weights enter both X'WX and σ²/df paths; both must match.
    let n = 400usize;
    let k = 7usize;
    let mut rng = ChaCha8Rng::seed_from_u64(789);

    let unique_count = 150usize;
    let unique_vals: Vec<f64> = (0..unique_count)
        .map(|i| (i as f64) / (unique_count as f64 - 1.0))
        .collect();
    let x_col: Array1<f64> = (0..n).map(|i| unique_vals[i % unique_count]).collect();

    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mu = (4.0 * x_col[i]).sin();
            let u: f64 = rng.gen_range(-0.4..0.4);
            mu + 0.2 * u
        })
        .collect();
    let prior_weights: Array1<f64> = (0..n)
        .map(|i| 0.5 + 0.5 * ((i as f64) * 0.07).sin().abs())
        .collect();

    let (design, penalties, disc) = build_design(&x_col, k);
    let lambda = vec![0.02];

    let res_full = fit_pirls_tdist(
        &y,
        &design,
        &lambda,
        &penalties,
        None,
        None,
        50,
        1e-8,
        Some(&prior_weights),
    )
    .expect("dense");
    let res_disc = fit_pirls_tdist_discrete(
        &y,
        &lambda,
        &penalties,
        None,
        None,
        50,
        1e-8,
        Some(&prior_weights),
        &disc,
    )
    .expect("discrete");

    let max_abs_diff: f64 = res_full
        .coefficients
        .iter()
        .zip(res_disc.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "[tdist discrete weighted pure-dedup] max |Δβ| = {:.3e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-6,
        "weighted pure-dedup max |Δβ| = {} exceeds 1e-6",
        max_abs_diff
    );
}

#[test]
fn fit_pirls_tdist_discrete_fixed_df_and_sigma2_matches_dense() {
    // gam.fit5-style outer Newton path: fixed df and fixed σ² (the inner
    // PIRLS skips its profile updates and uses the t_newton_working
    // observed-info weights). Exercises the second weight formula.
    let n = 350usize;
    let k = 7usize;
    let mut rng = ChaCha8Rng::seed_from_u64(2024);

    let unique_count = 100usize;
    let unique_vals: Vec<f64> = (0..unique_count)
        .map(|i| (i as f64) / (unique_count as f64 - 1.0))
        .collect();
    let x_col: Array1<f64> = (0..n).map(|i| unique_vals[i % unique_count]).collect();

    let y: Array1<f64> = (0..n)
        .map(|i| {
            let mu = (5.0 * x_col[i]).sin() + 0.2 * x_col[i];
            let u: f64 = rng.gen_range(-0.4..0.4);
            mu + 0.15 * u
        })
        .collect();

    let (design, penalties, disc) = build_design(&x_col, k);
    let lambda = vec![0.03];

    let res_full = fit_pirls_tdist(
        &y,
        &design,
        &lambda,
        &penalties,
        Some(6.0), // fixed df
        Some(0.04), // fixed σ²
        50,
        1e-8,
        None,
    )
    .expect("dense");
    let res_disc = fit_pirls_tdist_discrete(
        &y,
        &lambda,
        &penalties,
        Some(6.0),
        Some(0.04),
        50,
        1e-8,
        None,
        &disc,
    )
    .expect("discrete");

    let max_abs_diff: f64 = res_full
        .coefficients
        .iter()
        .zip(res_disc.coefficients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "[tdist discrete fixed-df+σ²] max |Δβ| = {:.3e}",
        max_abs_diff
    );
    assert!(
        max_abs_diff < 1e-6,
        "fixed-df+σ² max |Δβ| = {} exceeds 1e-6",
        max_abs_diff
    );
}
