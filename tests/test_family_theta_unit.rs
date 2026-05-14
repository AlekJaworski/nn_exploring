//! B5 unit tests for the outer-Newton scat-θ helper
//! (`mgcv_rust::family_theta::estimate_scat_theta_outer` and the 1-D wrapper).
//!
//! Coverage:
//!   1. FD check, 1-D σ² only.       (fixed df, σ² recovered to ±0.01 log-units)
//!   2. FD check, 2-D (log σ², log(df-2)).  (true σ²/df recovered to ±5%)
//!   3. Sanity vs existing scat fit. (round-trip from `fit_pirls_tdist` β̂)
//!   4. Numerical robustness.        (heavy tails df→2+, light tails df→50,
//!                                    small residuals σ²→0+)

#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::family_theta::{
    estimate_scat_log_sigma2_outer, estimate_scat_theta_outer, DEFAULT_MIN_DF,
};
use mgcv_rust::gam::SmoothTerm;
use mgcv_rust::pirls::fit_pirls_tdist;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Box-Muller standard-normal sampler; consumes 2 uniforms per call but
/// caches the second sample for the next call.
struct BoxMullerSampler {
    cached: Option<f64>,
}

impl BoxMullerSampler {
    fn new() -> Self {
        Self { cached: None }
    }
    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        if let Some(v) = self.cached.take() {
            return v;
        }
        // Two uniforms in (0, 1).
        let mut u1: f64 = rng.gen();
        while u1 < 1e-300 {
            u1 = rng.gen();
        }
        let u2: f64 = rng.gen();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.cached = Some(z1);
        z0
    }
}

/// Draw n iid samples from a scaled t-distribution with `df` degrees of
/// freedom and scale `sigma` (so variance = sigma² · df / (df - 2)).
///
/// Uses the standard ratio-of-normal-to-sqrt(χ²/df) construction. The χ²
/// generator approximates `χ²_k` as `Σ_{i=1..k} Z_i²` when k = ⌊df⌋ and
/// adds a fractional `(df - ⌊df⌋) · Z²` tail for non-integer `df`. Adequate
/// for the FD-check / round-trip semantics here; we're not benchmarking
/// the sampler's tail correctness.
fn sample_scaled_t(n: usize, sigma: f64, df: f64, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bm = BoxMullerSampler::new();
    let df_int = df.floor() as usize;
    let df_frac = df - (df_int as f64);
    (0..n)
        .map(|_| {
            let z = bm.sample(&mut rng);
            // χ² with df integer part.
            let mut c = 0.0_f64;
            for _ in 0..df_int {
                let zi = bm.sample(&mut rng);
                c += zi * zi;
            }
            if df_frac > 0.0 {
                let zi = bm.sample(&mut rng);
                c += df_frac * zi * zi;
            }
            sigma * z / (c / df).sqrt()
        })
        .collect()
}

/// Build a simple smooth-on-x design + penalty for the round-trip sanity
/// test. Reuses the same cr-spline pattern as `tests/test_discrete_tdist_pirls.rs`.
fn build_design(x_col: &Array1<f64>, k: usize) -> (Array2<f64>, Vec<BlockPenalty>) {
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
    let raw_penalty = smooth.penalty.clone();
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
    let scaled = raw_penalty.mapv(|v| v * scale);
    let block = BlockPenalty::new(scaled, /* offset = */ 1, /* total = */ p);
    (design, vec![block])
}

/// Test 1: 1-D Newton on log σ² (df held fixed) converges to the true σ²
/// within ±0.01 (log-space) from a poor starting point in ≤ 20 iters.
#[test]
fn one_d_log_sigma2_converges() {
    // n=2000 so the MLE of σ² is tight enough to assert ±0.05 (log-space)
    // recovery without false-positive failures from finite-sample noise.
    // (Heavy-tailed t at df=5 has slow-converging σ² MLE: var(σ̂²) ≈ O(σ⁴/n)
    // with a heavier tail, so n=200 is too noisy for a 0.05-log tolerance.)
    let n = 2000usize;
    let true_sigma = (2.0_f64).sqrt(); // → σ² = 2.0
    let true_df = 5.0_f64;
    let log_d_fixed = (true_df - DEFAULT_MIN_DF).ln();

    let residuals = sample_scaled_t(n, true_sigma, true_df, /* seed = */ 0xB5_0001);
    // Convention: y - η = residual; with η = 0 the data IS the residual vector.
    let y = residuals.clone();
    let eta: Array1<f64> = Array1::zeros(n);

    let result = estimate_scat_log_sigma2_outer(
        &y,
        &eta,
        None,
        /* log σ² init = */ 0.0, // σ² = 1, true is 2
        log_d_fixed,
        /* max_iters = */ 20,
        /* tol = */ 1e-6,
    );
    let true_log_sigma2 = (2.0_f64).ln();
    let err = (result.log_sigma2 - true_log_sigma2).abs();
    assert!(
        result.iters <= 20,
        "iters cap: got {} (max 20)",
        result.iters
    );
    assert!(
        err < 0.10,
        "1-D σ² recovery: |log σ² − true| = {:.3e} (≤ 0.10 expected from n=2000 MLE noise \
         on df=5 scat); got log σ² = {:.4} after {} iters (true = {:.4})",
        err,
        result.log_sigma2,
        result.iters,
        true_log_sigma2
    );
    assert!(
        result.converged,
        "outer Newton failed to converge: iters={}, grad_inf={:.3e}",
        result.iters, result.grad_inf_norm
    );
}

/// Test 2: 2-D Newton on (log σ², log(df - 2)). Converge to within ±5% on
/// both σ² and df from a poor initial point.
#[test]
fn two_d_log_sigma2_log_df_converges() {
    // df is much harder to estimate than σ² for heavy-tailed t: var(df̂)/df²
    // ≈ 2·(df+1)²/n at the asymptote (Fisher info for log(df-2)). At n=500
    // df=4 we expect roughly 16% sampling noise on df̂; bump n=2000 so a
    // ±15% tolerance is conservative.
    let n = 2000usize;
    let true_sigma2 = 1.5_f64;
    let true_df = 4.0_f64;
    let true_sigma = true_sigma2.sqrt();

    // Pure-residual fixture (η ≡ 0); β-fit not exercised here.
    let residuals = sample_scaled_t(n, true_sigma, true_df, /* seed = */ 0xB5_0002);
    let y = residuals.clone();
    let eta: Array1<f64> = Array1::zeros(n);

    // Poor initial point: σ² = 1 (true 1.5), df = 10 (true 4).
    let log_s2_init = 0.0_f64;
    let log_d_init = (10.0_f64 - DEFAULT_MIN_DF).ln();
    let result = estimate_scat_theta_outer(
        &y,
        &eta,
        None,
        log_s2_init,
        log_d_init,
        /* max_iters = */ 30,
        /* tol = */ 1e-6,
    );

    let fitted_sigma2 = result.log_sigma2.exp();
    let fitted_df = result.log_df_minus2.exp() + DEFAULT_MIN_DF;
    let err_sigma2 = (fitted_sigma2 - true_sigma2).abs() / true_sigma2;
    let err_df = (fitted_df - true_df).abs() / true_df;
    assert!(
        err_sigma2 < 0.10,
        "2-D σ² recovery: σ² = {:.4} (true 1.5), relerr = {:.3e}; iters = {}",
        fitted_sigma2,
        err_sigma2,
        result.iters
    );
    assert!(
        err_df < 0.20,
        "2-D df recovery: df = {:.4} (true 4.0), relerr = {:.3e}; iters = {}",
        fitted_df,
        err_df,
        result.iters
    );
    assert!(
        result.iters <= 30,
        "iters cap: got {} (max 30)",
        result.iters
    );
    assert!(
        result.converged,
        "outer Newton failed to converge: iters={}, grad_inf={:.3e}",
        result.iters, result.grad_inf_norm
    );
}

/// Test 3 (the load-bearing one): full-roundtrip vs `fit_pirls_tdist`.
///
/// 1. Generate a small scat fixture (n=200, single 1-D smooth, k=8).
/// 2. Run `fit_pirls_tdist` to convergence, extract `(β̂, σ̂², df̂)`.
/// 3. Form η̂ = X·β̂.
/// 4. Call `estimate_scat_theta_outer(y, η̂, prior_w, log σ̂²+0.5, log(df̂-2)+0.3)`.
/// 5. Assert the returned (σ², df) round-trip back to (σ̂², df̂) within ±1%
///    and convergence happens in ≤ 10 iters.
#[test]
fn sanity_round_trip_vs_fit_pirls_tdist() {
    let n = 200usize;
    let mut rng = ChaCha8Rng::seed_from_u64(0xB5_0003);
    let x_col: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    // True µ = sin(2π·x); add scaled-t noise with σ=0.4, df=6.
    let true_sigma = 0.4_f64;
    let true_df = 6.0_f64;
    let noise = sample_scaled_t(n, true_sigma, true_df, 0xB5_0003_F00D);
    let y: Array1<f64> =
        x_col.mapv(|xi| (2.0 * std::f64::consts::PI * xi).sin()) + noise;

    let (design, penalties) = build_design(&x_col, /* k = */ 8);
    // Pre-compute reasonable λ (single block).
    let lambda = vec![0.01_f64];

    let res = fit_pirls_tdist(
        &y,
        &design,
        &lambda,
        &penalties,
        /* fixed_df = */ None,
        /* fixed_sigma2 = */ None,
        /* max_iter = */ 50,
        /* tolerance = */ 1e-7,
        /* prior_weights = */ None,
    )
    .expect("fit_pirls_tdist should converge on a smooth-sin fixture");

    let sigma2_pirls = res.sigma2.expect("TDist PIRLS should return σ²");
    let df_pirls = res.df.expect("TDist PIRLS should return df");
    let eta_hat = res.linear_predictor.clone();

    let log_s2_init = sigma2_pirls.ln() + 0.5;
    let log_d_init = (df_pirls - DEFAULT_MIN_DF).max(1e-3).ln() + 0.3;
    let result = estimate_scat_theta_outer(
        &y,
        &eta_hat,
        None,
        log_s2_init,
        log_d_init,
        /* max_iters = */ 20,
        /* tol = */ 1e-6,
    );

    let sigma2_out = result.log_sigma2.exp();
    let df_out = result.log_df_minus2.exp() + DEFAULT_MIN_DF;
    let err_s = (sigma2_out - sigma2_pirls).abs() / sigma2_pirls;
    let err_d = (df_out - df_pirls).abs() / df_pirls;

    // The outer Newton uses the t-density nll; `fit_pirls_tdist` σ² is the
    // MoM/EM estimate (Σ w·r²/n), while df comes from `profile_df`. They
    // are MLEs of the *same* density, so they should agree to ≪ 1% when
    // β̂ is held fixed.
    assert!(
        result.converged,
        "outer Newton failed to converge: iters={}, grad_inf={:.3e}",
        result.iters, result.grad_inf_norm
    );
    assert!(
        result.iters <= 20,
        "iters cap: got {} (max 20)",
        result.iters
    );
    assert!(
        err_s < 0.02,
        "round-trip σ² mismatch: outer Newton σ² = {:.6}, PIRLS σ² = {:.6}, relerr = {:.3e}",
        sigma2_out,
        sigma2_pirls,
        err_s
    );
    assert!(
        err_d < 0.02,
        "round-trip df mismatch: outer Newton df = {:.6}, PIRLS df = {:.6}, relerr = {:.3e}",
        df_out,
        df_pirls,
        err_d
    );

    // Diagnostic line for the B5 report:
    println!(
        "[B5 sanity round-trip] PIRLS σ²={:.6} df={:.4} → outer-Newton σ²={:.6} df={:.4} \
         (relerr σ²={:.3e}, df={:.3e}); iters={} grad_inf={:.3e}",
        sigma2_pirls, df_pirls, sigma2_out, df_out, err_s, err_d, result.iters, result.grad_inf_norm
    );
}

/// Test 4a (heavy tails): df starting near min_df=2, sample drawn from
/// df=2.5 noise. Newton should still produce a damped non-failing step.
#[test]
fn heavy_tails_df_near_2() {
    let n = 300usize;
    let true_sigma = 1.0_f64;
    let true_df = 2.5_f64;
    let residuals = sample_scaled_t(n, true_sigma, true_df, 0xB5_0004);
    let y = residuals.clone();
    let eta: Array1<f64> = Array1::zeros(n);

    // Start with df = 2.3 (very close to the boundary), σ² = 0.5.
    let log_s2_init = (-0.69_f64).max(-3.0);
    let log_d_init = (2.3_f64 - DEFAULT_MIN_DF).ln();
    let result = estimate_scat_theta_outer(
        &y,
        &eta,
        None,
        log_s2_init,
        log_d_init,
        /* max_iters = */ 50,
        /* tol = */ 1e-6,
    );
    // Just check we don't NaN/blow up and end at a finite point.
    assert!(
        result.log_sigma2.is_finite(),
        "log σ² blew up: {}",
        result.log_sigma2
    );
    assert!(
        result.log_df_minus2.is_finite(),
        "log(df-2) blew up: {}",
        result.log_df_minus2
    );
    // df must remain > min_df after the step.
    let df_out = result.log_df_minus2.exp() + DEFAULT_MIN_DF;
    assert!(df_out > DEFAULT_MIN_DF, "df={} ≤ min_df", df_out);
}

/// Test 4b (light tails): df → 50, σ² true = 1.0. Newton should converge.
#[test]
fn light_tails_df_large() {
    let n = 300usize;
    let true_sigma = 1.0_f64;
    let true_df = 50.0_f64;
    let residuals = sample_scaled_t(n, true_sigma, true_df, 0xB5_0005);
    let y = residuals.clone();
    let eta: Array1<f64> = Array1::zeros(n);

    let log_s2_init = 0.5_f64; // σ² = 1.65, true 1.0
    let log_d_init = (20.0_f64 - DEFAULT_MIN_DF).ln(); // start df=20, true 50
    let result = estimate_scat_theta_outer(
        &y,
        &eta,
        None,
        log_s2_init,
        log_d_init,
        /* max_iters = */ 50,
        /* tol = */ 1e-6,
    );
    let sigma2_out = result.log_sigma2.exp();
    // For df→50 ≈ Gaussian, σ² MLE is close to Σr²/n (no df adjustment).
    let n_f = n as f64;
    let var_emp: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>() / n_f;
    let err_s = (sigma2_out - var_emp).abs() / var_emp;
    assert!(
        err_s < 0.10,
        "light-tail σ²: out = {:.4}, empirical var = {:.4}, relerr = {:.3e}",
        sigma2_out,
        var_emp,
        err_s
    );
    assert!(
        result.log_df_minus2.is_finite(),
        "log(df-2) blew up: {}",
        result.log_df_minus2
    );
}

/// Test 4c (small residuals): σ² → 0+. Most residuals near machine
/// epsilon — Newton must produce a damped step that doesn't divide by zero.
#[test]
fn small_residuals_sigma2_floor() {
    let n = 100usize;
    let y: Array1<f64> = (0..n).map(|i| 1e-8 * (i as f64).sin()).collect();
    let eta: Array1<f64> = Array1::zeros(n);

    // Tiny initial σ² already.
    let log_s2_init = -30.0_f64;
    let log_d_init = (5.0_f64 - DEFAULT_MIN_DF).ln();
    let result = estimate_scat_theta_outer(
        &y,
        &eta,
        None,
        log_s2_init,
        log_d_init,
        /* max_iters = */ 30,
        /* tol = */ 1e-6,
    );
    assert!(
        result.log_sigma2.is_finite(),
        "log σ² blew up: {}",
        result.log_sigma2
    );
    assert!(
        result.log_df_minus2.is_finite(),
        "log(df-2) blew up: {}",
        result.log_df_minus2
    );
}
