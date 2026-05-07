//! Pure-Rust integration test for the gaulss + per-obs-σ ELF quantile pipeline.
//!
//! Demonstrates that `fit_pirls_quantile_lss` can be driven without the Python
//! wrapper — relevant for Rust-binary or WASM consumers. Mirrors the Python
//! `fit_quantile_lss` orchestration step-by-step using only the Rust API:
//!
//! 1. Fit a Gaussian GAM on `y` for μ_G(x).
//! 2. Fit a Gaussian GAM on `log|y - μ_G| - E[log|N(0,1)|]` for log σ_G(x).
//! 3. Build the location-side penalty Σ λ_i S_i from the location GAM's
//!    per-smooth blocks and REML-fitted lambdas.
//! 4. Pass design + σ_G(x) to `fit_pirls_quantile_lss`; the σ rescaling
//!    and qgam heuristic for σ_global both run inside the Rust function.
//! 5. Verify in-sample coverage matches τ on a heteroskedastic test case.

#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::gam::{SmoothTerm, GAM};
use mgcv_rust::pirls::{
    fit_pirls_quantile_lss, fit_pirls_quantile_lss_fs_tune, Family,
};
use mgcv_rust::smooth::OptimizationMethod;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

fn build_gaussian_gam(x: &Array2<f64>, k_per_col: &[usize]) -> GAM {
    let mut gam = GAM::new(Family::Gaussian);
    for (i, &k) in k_per_col.iter().enumerate() {
        let col = x.column(i).to_owned();
        let smooth = SmoothTerm::cr_spline_quantile(format!("x{}", i), k, &col).unwrap();
        gam.add_smooth(smooth);
    }
    gam
}

/// Build the full-design penalty Σ λ_i S_i embedded at the intercept-aware
/// offsets used by the GAM design `[1 | s_0 | s_1 | ...]`. Mirrors the
/// Python `_build_total_penalty` helper.
fn build_total_penalty(gam: &GAM) -> Array2<f64> {
    let lambdas = &gam.smoothing_params.as_ref().unwrap().lambda;
    let p = gam.design_matrix.as_ref().unwrap().ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    let mut col = 1usize;
    for (smooth, &lam) in gam.smooth_terms.iter().zip(lambdas.iter()) {
        let nb = smooth.num_basis();
        for i in 0..nb {
            for j in 0..nb {
                s[[col + i, col + j]] += lam * smooth.penalty[[i, j]];
            }
        }
        col += nb;
    }
    s
}

#[test]
fn lss_rust_pipeline_heteroskedastic() {
    // ── Synthetic heteroskedastic data: σ(x) = 0.1 + 0.4|x_0|. ──
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let n = 800;
    let mut x_data = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x_data.push(rng.gen_range(-1.0..1.0));
        x_data.push(rng.gen_range(-1.0..1.0));
    }
    let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0: f64 = x[[i, 0]];
        let x1: f64 = x[[i, 1]];
        let mu: f64 = (2.0 * x0).sin() + 0.5 * x1;
        let sigma_true: f64 = 0.1 + 0.4 * x0.abs();
        // Box-Muller for one Gaussian sample.
        let u1: f64 = rng.gen_range(1e-9..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        y[i] = mu + sigma_true * z;
    }

    // ── Stage 1a: Gaussian GAM for μ_G(x). ──
    let mut g_loc = build_gaussian_gam(&x, &[10, 10]);
    g_loc.fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("g_loc fit");

    // ── Stage 1b: Gaussian GAM for log σ_G(x), with Euler-Mascheroni
    // bias correction `-(γ/2 + log 2/2) ≈ -0.6351` so E[log|N(0,σ²)|]
    // → log σ exactly. ──
    let mu_g = g_loc.predict(&x).unwrap();
    const E_LOG_ABS_NORMAL: f64 = -0.6351814227307388;
    let y_sd = (y.iter().zip(mu_g.iter())
        .map(|(&yi, &mi)| (yi - mi).powi(2)).sum::<f64>() / n as f64).sqrt();
    let floor = 1e-3 * y_sd;
    let log_abs_r: Array1<f64> = y.iter().zip(mu_g.iter())
        .map(|(&yi, &mi)| (yi - mi).abs().max(floor).ln() - E_LOG_ABS_NORMAL)
        .collect();
    let mut g_scale = build_gaussian_gam(&x, &[5, 5]);
    g_scale.fit_optimized(&x, &log_abs_r, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("g_scale fit");
    let log_sigma_g = g_scale.predict(&x).unwrap();
    let sigma_g: Array1<f64> = log_sigma_g.iter().map(|&v| v.exp()).collect();

    // ── Stage 2: assemble inputs and call the Rust LSS fitter. ──
    let x_loc = g_loc.design_matrix.as_ref().unwrap().clone();
    let s_loc_total = build_total_penalty(&g_loc);

    for &tau in &[0.1_f64, 0.5, 0.9] {
        let (res, sigma_global_used) = fit_pirls_quantile_lss(
            &y, &x_loc, &s_loc_total, &sigma_g,
            None,        // sigma_global = auto qgam heuristic
            tau, 50, 1e-6,
        ).expect("LSS fit");
        assert!(res.converged, "τ={} did not converge", tau);
        assert!(sigma_global_used > 0.0);

        let yhat = x_loc.dot(&res.coefficients_loc);
        let cov = y.iter().zip(yhat.iter())
            .filter(|(yi, yh)| yi < yh).count() as f64 / n as f64;
        let cal_err = (cov - tau).abs();
        // Heteroskedastic gap target: cal_err < 0.05 across τ.
        assert!(
            cal_err < 0.05,
            "τ={}: cal_err={:.4} > 0.05 (cov={:.4})", tau, cal_err, cov
        );

        // σ̂(x) shape recovery (smoke check via correlation with σ_G).
        let sigma_est = &res.sigma;
        let sg_mean = sigma_g.iter().sum::<f64>() / n as f64;
        let se_mean = sigma_est.iter().sum::<f64>() / n as f64;
        let cov_xy: f64 = sigma_est.iter().zip(sigma_g.iter())
            .map(|(&se, &sg)| (se - se_mean) * (sg - sg_mean)).sum::<f64>() / n as f64;
        let var_e: f64 = sigma_est.iter().map(|&s| (s - se_mean).powi(2)).sum::<f64>() / n as f64;
        let var_g: f64 = sigma_g.iter().map(|&s| (s - sg_mean).powi(2)).sum::<f64>() / n as f64;
        let corr = cov_xy / (var_e.sqrt() * var_g.sqrt() + 1e-12);
        // σ̂ should be PROPORTIONAL to σ_G (it's σ_G · σ_global / mean(σ_G)).
        assert!(corr > 0.999, "τ={}: corr(σ̂, σ_G)={:.4} < 0.999", tau, corr);
    }
}

/// Pure-Rust regression test for `fit_pirls_quantile_lss_fs_tune` —
/// the Fellner-Schall λ retuning path. Mirrors `lss_rust_pipeline_heteroskedastic`
/// but supplies per-smooth penalty blocks + initial λs and asserts the
/// FS-tuned fit still hits in-sample coverage targets.
#[test]
fn lss_rust_fs_tune_heteroskedastic() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let n = 800;
    let mut x_data = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x_data.push(rng.gen_range(-1.0..1.0));
        x_data.push(rng.gen_range(-1.0..1.0));
    }
    let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0: f64 = x[[i, 0]];
        let x1: f64 = x[[i, 1]];
        let mu: f64 = (2.0 * x0).sin() + 0.5 * x1;
        let sigma_true: f64 = 0.1 + 0.4 * x0.abs();
        let u1: f64 = rng.gen_range(1e-9..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        y[i] = mu + sigma_true * z;
    }

    let mut g_loc = build_gaussian_gam(&x, &[10, 10]);
    g_loc.fit_optimized(&x, &y, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("g_loc fit");
    let mu_g = g_loc.predict(&x).unwrap();
    const E_LOG_ABS_NORMAL: f64 = -0.6351814227307388;
    let y_sd = (y.iter().zip(mu_g.iter())
        .map(|(&yi, &mi)| (yi - mi).powi(2)).sum::<f64>() / n as f64).sqrt();
    let floor = 1e-3 * y_sd;
    let log_abs_r: Array1<f64> = y.iter().zip(mu_g.iter())
        .map(|(&yi, &mi)| (yi - mi).abs().max(floor).ln() - E_LOG_ABS_NORMAL)
        .collect();
    let mut g_scale = build_gaussian_gam(&x, &[5, 5]);
    g_scale.fit_optimized(&x, &log_abs_r, OptimizationMethod::REML, 10, 100, 1e-6)
        .expect("g_scale fit");
    let log_sigma_g = g_scale.predict(&x).unwrap();
    let sigma_g: Array1<f64> = log_sigma_g.iter().map(|&v| v.exp()).collect();

    let x_loc = g_loc.design_matrix.as_ref().unwrap().clone();
    let p_loc = x_loc.ncols();

    // Build per-smooth BlockPenalty entries at full-design offsets.
    let lambda_init: Vec<f64> = g_loc.smoothing_params.as_ref().unwrap().lambda.clone();
    let mut penalties = Vec::new();
    let mut offset = 1usize;
    for smooth in &g_loc.smooth_terms {
        penalties.push(BlockPenalty::new(smooth.penalty.clone(), offset, p_loc));
        offset += smooth.num_basis();
    }

    for &tau in &[0.1_f64, 0.5, 0.9] {
        let (res, _sg_used) = fit_pirls_quantile_lss_fs_tune(
            &y, &x_loc, &penalties, &lambda_init, &sigma_g,
            None, tau, 20, 50, 1e-6,
        ).expect("FS-tune fit");
        assert!(res.converged, "τ={} did not converge", tau);
        assert_eq!(res.lambda_loc.len(), penalties.len());
        assert!(res.fs_iterations >= 1);

        let yhat = x_loc.dot(&res.coefficients_loc);
        let cov = y.iter().zip(yhat.iter())
            .filter(|(yi, yh)| yi < yh).count() as f64 / n as f64;
        let cal_err = (cov - tau).abs();
        assert!(
            cal_err < 0.05,
            "FS-tune τ={}: cal_err={:.4} > 0.05 (cov={:.4})", tau, cal_err, cov
        );
    }
}
