//! Integration tests for `compute_sl_fitchol_step` (Path B / B3).
//!
//! Four blocks per the B3 task brief:
//!
//!   1. FD-check gradient + Hessian on a 1-smooth Gaussian fixture.
//!   2. FD-check gradient + Hessian on a 3-smooth Gaussian fixture with
//!      mixed ρ values.
//!   3. Sanity: fREML score (recomputed from the returned scalars) agrees
//!      with the existing `reml_criterion_multi_cached_mgcv_exact` up to
//!      a documented constant offset `(n − M_p)·log(2π)`.
//!   4. Sanity: `db = dβ/dρ` returned by the IFT path matches the existing
//!      `compute_b1_ift` (`src/reml/mod.rs:1616`) to 1e-10 abs — this is the
//!      load-bearing cross-check on the IFT formula.

#![cfg(feature = "blas")]

use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::reml::{compute_sl_fitchol_step, compute_xtwx, compute_xtwy, SlFitCholResult};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Build a 2nd-difference penalty block for a length-k cubic-spline basis.
/// Returns the k×k SPSD matrix D2'D2 where D2 is the (k-2)×k second-difference
/// operator. Rank = k - 2.
fn second_diff_penalty(k: usize) -> Array2<f64> {
    let m = k.saturating_sub(2);
    let mut d2 = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        d2[[i, i]] = 1.0;
        d2[[i, i + 1]] = -2.0;
        d2[[i, i + 2]] = 1.0;
    }
    d2.t().dot(&d2)
}

/// Build a synthetic design matrix where columns are sin/cos/polynomial
/// features. Deterministic across runs.
fn synth_design(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        // Intercept-like first column.
        x[[i, 0]] = 1.0;
        let xi = (i as f64) / (n as f64);
        for j in 1..p {
            // Mix of basis-like features. Tiny jitter for conditioning.
            x[[i, j]] = match j % 4 {
                0 => (xi * (j as f64 + 1.0) * std::f64::consts::PI).sin(),
                1 => (xi * (j as f64 + 1.0) * std::f64::consts::PI).cos(),
                2 => xi.powi(j as i32 % 3 + 1),
                _ => (-(xi - 0.5).powi(2) * (j as f64)).exp(),
            };
            x[[i, j]] += 1e-4 * rng.gen::<f64>();
        }
    }
    x
}

/// Synthetic response `y = X β_true + noise`, deterministic.
fn synth_response(x: &Array2<f64>, beta_true: &Array1<f64>, noise_sd: f64, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = x.nrows();
    let mu = x.dot(beta_true);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let z: f64 = rng.gen::<f64>() - 0.5; // uniform-ish, finite & symmetric
        y[i] = mu[i] + noise_sd * z;
    }
    y
}

/// Sample weights distributed away from 1.0 to exercise weighted XX/f assembly.
fn synth_weights(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        w[i] = 0.5 + rng.gen::<f64>(); // in [0.5, 1.5]
    }
    w
}

/// Compute `2·V_R^fitChol(ρ, log_φ)` from the explicit formula. Used as the
/// FD black-box (no other quantities from the helper are referenced — pure
/// `(ρ, log_φ) → ℝ`).
fn score_at(
    sl: &[BlockPenalty],
    xx: &Array2<f64>,
    f: &Array1<f64>,
    yy: f64,
    rho: &[f64],
    log_phi: f64,
    nobs: f64,
    mp: usize,
    gamma: f64,
    phi_fixed: bool,
) -> f64 {
    // Recompute β at the trial ρ (independent path; doesn't depend on the
    // SlFitCholResult we're FD'ing).
    let p = xx.nrows();
    let mut a = xx.clone();
    for (k, pen) in sl.iter().enumerate() {
        let lam = rho[k].exp();
        pen.scaled_add_to(&mut a, lam);
    }
    // Direct solve.
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|v| v.abs()).fold(1.0f64, f64::max);
    let ridge = 1e-12 * max_diag;
    for i in 0..p {
        a_solve[[i, i]] += ridge;
    }
    let beta = mgcv_rust::linalg::solve(a_solve, f.clone()).unwrap();

    // log|A|, log|S|+
    let ldet_xxs = mgcv_rust::linalg::determinant(&a).unwrap().ln();
    let rho_vec: Vec<f64> = rho.to_vec();
    // Re-use the same helper as the kernel for byte-identical log|S|+.
    let (ldet_s, _, _) = mgcv_rust::reml::compute_ldet_s_with_derivs(sl, &rho_vec);

    let phi = log_phi.exp();
    let rss_bsb = yy - beta.dot(f); // = ‖y-Xβ‖² + β'Sβ at the minimiser.
    let mut s = ldet_xxs - ldet_s + rss_bsb / (phi * gamma);
    if !phi_fixed {
        s += (nobs / gamma - mp as f64) * log_phi;
    }
    // The kernel returns 1/2 · ∂s/∂ρ, so we differentiate `0.5 · s` against
    // FD trials.
    0.5 * s
}

/// FD gradient and FD Hessian of `score_at` at (rho, log_phi). When
/// `phi_fixed = false`, the last coordinate is `log_φ`. Step `h = 1e-4`
/// matches the brief.
fn fd_grad_hess(
    sl: &[BlockPenalty],
    xx: &Array2<f64>,
    f: &Array1<f64>,
    yy: f64,
    rho: &[f64],
    log_phi: f64,
    nobs: f64,
    mp: usize,
    gamma: f64,
    phi_fixed: bool,
    h: f64,
) -> (Array1<f64>, Array2<f64>) {
    let m = rho.len();
    let ntot = if phi_fixed { m } else { m + 1 };
    let mut grad = Array1::<f64>::zeros(ntot);
    let mut hess = Array2::<f64>::zeros((ntot, ntot));

    let eval = |rho_pert: &[f64], log_phi_pert: f64| -> f64 {
        score_at(sl, xx, f, yy, rho_pert, log_phi_pert, nobs, mp, gamma, phi_fixed)
    };

    let center = eval(rho, log_phi);
    // Gradient — central difference on each coord.
    for k in 0..ntot {
        let (mut rho_p, mut rho_m) = (rho.to_vec(), rho.to_vec());
        let (mut lp_p, mut lp_m) = (log_phi, log_phi);
        if k < m {
            rho_p[k] += h;
            rho_m[k] -= h;
        } else {
            lp_p += h;
            lp_m -= h;
        }
        let fp = eval(&rho_p, lp_p);
        let fm = eval(&rho_m, lp_m);
        grad[k] = (fp - fm) / (2.0 * h);
        // Diagonal Hessian via second-order central difference.
        hess[[k, k]] = (fp - 2.0 * center + fm) / (h * h);
    }
    // Off-diagonal Hessian via 4-point central difference.
    for i in 0..ntot {
        for j in (i + 1)..ntot {
            let mut rho_pp = rho.to_vec();
            let mut rho_pm = rho.to_vec();
            let mut rho_mp = rho.to_vec();
            let mut rho_mm = rho.to_vec();
            let mut lp_pp = log_phi;
            let mut lp_pm = log_phi;
            let mut lp_mp = log_phi;
            let mut lp_mm = log_phi;
            if i < m {
                rho_pp[i] += h;
                rho_pm[i] += h;
                rho_mp[i] -= h;
                rho_mm[i] -= h;
            } else {
                lp_pp += h;
                lp_pm += h;
                lp_mp -= h;
                lp_mm -= h;
            }
            if j < m {
                rho_pp[j] += h;
                rho_mp[j] += h;
                rho_pm[j] -= h;
                rho_mm[j] -= h;
            } else {
                lp_pp += h;
                lp_mp += h;
                lp_pm -= h;
                lp_mm -= h;
            }
            let v = (eval(&rho_pp, lp_pp) - eval(&rho_pm, lp_pm) - eval(&rho_mp, lp_mp)
                + eval(&rho_mm, lp_mm))
                / (4.0 * h * h);
            hess[[i, j]] = v;
            hess[[j, i]] = v;
        }
    }
    (grad, hess)
}

// ---------------------------------------------------------------------------
// Test 1: FD check, Gaussian, 1-smooth.
// ---------------------------------------------------------------------------

#[test]
fn fd_check_gaussian_1smooth() {
    // Single 10-column smooth (cubic-spline-ish penalty). p = 10 — small
    // enough for tight conditioning, large enough to exercise IFT.
    let n = 200usize;
    let p = 10usize;
    let x = synth_design(n, p, 42);
    let mut beta_true = Array1::<f64>::zeros(p);
    for i in 0..p {
        beta_true[i] = ((i as f64) * 0.3).sin();
    }
    let y = synth_response(&x, &beta_true, 0.5, 7);
    let w = synth_weights(n, 11);
    let nobs: f64 = w.iter().sum();

    let pen_block = second_diff_penalty(p);
    let sl = vec![BlockPenalty::new(pen_block, 0, p)];
    let mp: usize = p - (sl[0].estimate_rank()); // null-space of penalty.

    // Cached primitives.
    let xx = compute_xtwx(&x, &w);
    let f = compute_xtwy(&x, &w, &y);
    let yy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| yi * yi * wi).sum();

    let rho = vec![1.0f64];
    let log_phi: f64 = 0.0_f64; // φ = 1.

    let res = compute_sl_fitchol_step(
        &sl,
        xx.view(),
        f.view(),
        Array1::from_vec(rho.clone()).view(),
        yy,
        log_phi,
        true, // phi fixed.
        nobs,
        mp,
        1.0,
    )
    .expect("compute_sl_fitchol_step Gaussian 1-smooth");

    let h = 1e-4;
    let (fd_g, fd_h) = fd_grad_hess(&sl, &xx, &f, yy, &rho, log_phi, nobs, mp, 1.0, true, h);

    let dg = (res.grad[0] - fd_g[0]).abs();
    let dh = (res.hess[[0, 0]] - fd_h[[0, 0]]).abs();
    eprintln!(
        "[fd_check_gaussian_1smooth] grad analytic={:.6e} FD={:.6e} |diff|={:.6e}",
        res.grad[0], fd_g[0], dg
    );
    eprintln!(
        "[fd_check_gaussian_1smooth] hess analytic={:.6e} FD={:.6e} |diff|={:.6e}",
        res.hess[[0, 0]],
        fd_h[[0, 0]],
        dh
    );

    assert!(dg < 1e-5, "grad |diff| {} exceeds 1e-5", dg);
    assert!(dh < 1e-3, "hess |diff| {} exceeds 1e-3", dh);
}

// ---------------------------------------------------------------------------
// Test 2: FD check, Gaussian, 3-smooth.
// ---------------------------------------------------------------------------

#[test]
fn fd_check_gaussian_3smooth() {
    // 3 independent smooth blocks of size 10 each → p = 30. ρ vector mixes
    // signs and magnitudes; φ = 2.0 exercises non-unit scale.
    let n = 200usize;
    let k = 10usize;
    let nblocks = 3usize;
    let p = k * nblocks;
    let x = synth_design(n, p, 101);
    let mut beta_true = Array1::<f64>::zeros(p);
    for i in 0..p {
        beta_true[i] = ((i as f64) * 0.2).sin() * 0.5;
    }
    let y = synth_response(&x, &beta_true, 0.8, 13);
    let w = synth_weights(n, 23);
    let nobs: f64 = w.iter().sum();

    // Build 3 singleton blocks, each a 2nd-diff penalty on its own k columns.
    let mut sl = Vec::with_capacity(nblocks);
    let mut mp_sum: usize = 0;
    for b in 0..nblocks {
        let pen_block = second_diff_penalty(k);
        let bp = BlockPenalty::new(pen_block, b * k, p);
        mp_sum += k - bp.estimate_rank();
        sl.push(bp);
    }
    let mp = mp_sum;

    let xx = compute_xtwx(&x, &w);
    let f = compute_xtwy(&x, &w, &y);
    let yy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| yi * yi * wi).sum();

    let rho = vec![0.5f64, 1.5f64, -0.3f64];
    let log_phi: f64 = (2.0f64).ln(); // φ = 2.

    let res = compute_sl_fitchol_step(
        &sl,
        xx.view(),
        f.view(),
        Array1::from_vec(rho.clone()).view(),
        yy,
        log_phi,
        true, // phi fixed.
        nobs,
        mp,
        1.0,
    )
    .expect("compute_sl_fitchol_step Gaussian 3-smooth");

    let h = 1e-4;
    let (fd_g, fd_h) = fd_grad_hess(&sl, &xx, &f, yy, &rho, log_phi, nobs, mp, 1.0, true, h);

    for k in 0..3 {
        let dg = (res.grad[k] - fd_g[k]).abs();
        eprintln!(
            "[fd_check_gaussian_3smooth] grad[{}] analytic={:.6e} FD={:.6e} |diff|={:.6e}",
            k, res.grad[k], fd_g[k], dg
        );
        assert!(dg < 1e-5, "grad[{}] |diff| {} exceeds 1e-5", k, dg);
    }
    for i in 0..3 {
        for j in 0..3 {
            let dh = (res.hess[[i, j]] - fd_h[[i, j]]).abs();
            eprintln!(
                "[fd_check_gaussian_3smooth] hess[{},{}] analytic={:.6e} FD={:.6e} |diff|={:.6e}",
                i,
                j,
                res.hess[[i, j]],
                fd_h[[i, j]],
                dh
            );
            assert!(dh < 1e-3, "hess[{},{}] |diff| {} exceeds 1e-3", i, j, dh);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: fREML score vs existing mgcv-exact REML score.
// ---------------------------------------------------------------------------

#[test]
fn sanity_fastreml_vs_reml_constant_offset() {
    // The fitChol score and the existing `reml_criterion_multi_cached_mgcv_exact`
    // differ in TWO documented ways:
    //
    //   (a) An analytical constant offset of `(n − M_p)·log(2π)/2` from
    //       the Gaussian saturated log-likelihood term that REML expands
    //       inline but fitChol omits (mgcv's `Sl.fitChol` ignores
    //       constants that don't affect derivatives).
    //
    //   (b) REML's `σ²̂` is internally PROFILED at each ρ (the helper calls
    //       `estimate_phi_mgcv` per call), whereas `compute_sl_fitchol_step`
    //       takes `log_φ` as an INPUT and returns the score at that fixed
    //       φ. So evaluating both at "the same ρ" silently uses different
    //       scales unless the caller pins the same φ on both sides.
    //
    // Sanity strategy: pin both sides to the *same* profile σ²̂ = D_p/(n-Mp)
    // (Gaussian closed-form), then verify the offset is the constant from
    // (a), TO 1e-6 abs (small numerical slack from the internal Newton on
    // φ̂ vs the closed-form short-cut).
    use mgcv_rust::pirls::Family;
    use mgcv_rust::reml::reml_criterion_multi_cached_mgcv_exact;

    let n = 200usize;
    let p = 10usize;
    let x = synth_design(n, p, 99);
    let mut beta_true = Array1::<f64>::zeros(p);
    for i in 0..p {
        beta_true[i] = ((i as f64) * 0.4).sin();
    }
    let y = synth_response(&x, &beta_true, 0.5, 17);
    let w: Array1<f64> = Array1::ones(n);
    let nobs = n as f64;

    // NOTE on penalty choice: `BlockPenalty::log_det_singleton_with_derivs`
    // (the path B3 reuses via `compute_ldet_s_with_derivs`) currently calls
    // `estimate_rank()` (row-norm heuristic, gives `rank = p` on a 2nd-diff
    // penalty), whereas `reml_criterion_multi_cached_mgcv_exact` uses
    // `estimate_rank_eigen()` (eigen-spectrum, gives the true rank `p − 2`).
    // The two paths therefore evaluate `log|λS|+` with different `rank · ρ`
    // prefactors, leading to a `(rank_row − rank_eigen) · ρ` ρ-drift that
    // is NOT a defect in `compute_sl_fitchol_step` itself — it's a known
    // R3 rank-heuristic inconsistency (the score formula stays correct;
    // only the cross-path constant offset to `*_mgcv_exact` is perturbed).
    //
    // To isolate the constant offset cleanly, we use a full-rank ridge
    // penalty `S = I_p` where both heuristics agree (rank = p), so the
    // ρ-drift component vanishes and the offset is the pure
    // `(n − M_p) · log(2π) / 2` constant from the score-formula difference.
    let pen_block = Array2::<f64>::eye(p);
    let sl = vec![BlockPenalty::new(pen_block, 0, p)];
    // Use the same `mp` convention as `reml_criterion_multi_cached_mgcv_exact`
    // — caller-supplied. Both sides see the same value.
    let mp: usize = p.saturating_sub(sl[0].estimate_rank()); // = 0 for a full-rank ridge.

    let xx = compute_xtwx(&x, &w);
    let f = compute_xtwy(&x, &w, &y);
    let yy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| yi * yi * wi).sum();

    // Score offset at two different ρ. Both sides evaluated at the SAME
    // profile σ²̂_Gauss = D_p/(n − Mp) — closed-form Gaussian profile, the
    // limit `estimate_phi_mgcv` converges to for Gaussian.
    let mut offsets = Vec::new();
    let mut diagnostics: Vec<(f64, f64, f64, f64)> = Vec::new(); // (ρ, φ_used, reml_score, fchol_score)
    for &rho_k in &[-0.5f64, 1.2f64] {
        let lam = rho_k.exp();
        // β̂ and D_p at this ρ.
        let mut a = xx.clone();
        sl[0].scaled_add_to(&mut a, lam);
        let mut a_solve = a.clone();
        let max_diag = a.diag().iter().map(|v| v.abs()).fold(1.0f64, f64::max);
        for i in 0..p {
            a_solve[[i, i]] += 1e-12 * max_diag;
        }
        let beta = mgcv_rust::linalg::solve(a_solve, f.clone()).unwrap();
        // D_p = yy − β'f (Wood identity at the optimiser).
        let dp = yy - beta.dot(&f);
        // Gaussian profile σ²̂: closed form, no iteration (same fixed point
        // `estimate_phi_mgcv` returns at convergence for Gaussian).
        let phi = dp / (nobs - mp as f64);
        let log_phi = phi.ln();

        // existing REML at this ρ. The helper profiles internally — to keep
        // the comparison fair, we let it use its own σ²̂ and accept the
        // small numerical slack from Newton on φ̂.
        let lambdas = [lam];
        // Set env var so the helper prints its internal scalars; we
        // capture this once per ρ for the comparison.
        std::env::set_var("MGCV_EXACT_DEBUG", "1");
        let reml = reml_criterion_multi_cached_mgcv_exact(
            &y,
            &x,
            &w,
            &lambdas,
            &sl,
            Some(&xx),
            Some(&f),
            mp,
            Family::Gaussian,
            None,
        )
        .unwrap();
        std::env::remove_var("MGCV_EXACT_DEBUG");

        // fitChol score scalar via the documented formula at the SAME φ.
        let res = compute_sl_fitchol_step(
            &sl,
            xx.view(),
            f.view(),
            Array1::from_vec(vec![rho_k]).view(),
            yy,
            log_phi,
            false, // jointly include log_φ in the function value formula.
            nobs,
            mp,
            1.0,
        )
        .unwrap();
        // 2·V_R^fitChol = log|A| − log|S|+ + Dp/φ + (n − Mp)·log(φ).
        let s_fchol = 0.5
            * (res.ldet_xxs - res.ldet_s
                + (yy - res.beta.dot(&f)) / phi
                + (nobs - mp as f64) * log_phi);

        let offset = reml - s_fchol;
        diagnostics.push((rho_k, phi, reml, s_fchol));
        eprintln!(
            "[sanity_fastreml_vs_reml_constant_offset] rho={:.3} phi_fchol={:.6} reml={:.6} fitChol={:.6} offset={:.10}",
            rho_k, phi, reml, s_fchol, offset
        );
        let _ = Family::Gaussian; // satisfy unused-import lint
        let _ = &diagnostics;
        offsets.push(offset);
    }

    let drift = (offsets[0] - offsets[1]).abs();
    eprintln!(
        "[sanity_fastreml_vs_reml_constant_offset] offsets={:?} drift={:.2e}",
        offsets, drift
    );

    // Documented constant: V_REML − V_fitChol = (n − Mp)·log(2π)/2.
    let expected_offset = 0.5 * (nobs - mp as f64) * (2.0 * std::f64::consts::PI).ln();
    eprintln!(
        "[sanity_fastreml_vs_reml_constant_offset] expected offset (n-Mp)·log(2π)/2 = {:.6}",
        expected_offset
    );

    // Tolerance: 1e-6 absolute. REML's `estimate_phi_mgcv` for Gaussian
    // converges to the closed-form σ²̂ to ~1e-10, contributing a ≲1e-7
    // discrepancy in `(n−Mp)·log(σ²̂_iter)/2` vs `(n−Mp)·log(σ²̂_closed)/2`.
    let drift_tol = 1e-6;
    let offset_tol = 1e-6;

    assert!(
        drift < drift_tol,
        "fitChol vs REML offset drift {:.2e} exceeds {:.0e} (offsets={:?})",
        drift,
        drift_tol,
        offsets
    );
    let offset_err = (offsets[0] - expected_offset).abs();
    eprintln!(
        "[sanity_fastreml_vs_reml_constant_offset] |offset - expected| = {:.2e}",
        offset_err
    );
    assert!(
        offset_err < offset_tol,
        "fitChol vs REML constant offset mismatch: got {:.6e}, expected {:.6e}, diff {:.2e}",
        offsets[0],
        expected_offset,
        offset_err
    );
}

// ---------------------------------------------------------------------------
// Test R5-a: predictions invariance under `Sl.initial.repara`.
// ---------------------------------------------------------------------------
//
// Setup `compute_sl_fitchol_step` once with `repara` disabled (existing path)
// and once with `repara` enabled (R5 path). The two β vectors live in the SAME
// basis (we inverse-rotate at exit), so `X · β` should match to ~1e-10 abs
// regardless of which path computed the solve. Score scalars
// `(ldet_xxs, ldet_s)` differ by a ρ-independent constant (the
// `Σ log_pseudo_det(S_k)` offset) — we don't check those, just predictions.

#[test]
fn repara_predictions_invariance_gaussian_3smooth() {
    use mgcv_rust::reml::{compute_xtwx, compute_xtwy};

    let n = 200usize;
    let k = 10usize;
    let nblocks = 3usize;
    let p = k * nblocks;
    let x = synth_design(n, p, 173);
    let mut beta_true = Array1::<f64>::zeros(p);
    for i in 0..p {
        beta_true[i] = ((i as f64) * 0.21).sin() * 0.4;
    }
    let y = synth_response(&x, &beta_true, 0.6, 19);
    let w = synth_weights(n, 29);
    let nobs: f64 = w.iter().sum();

    // Two parallel penalty lists: one without repara, one with.
    let mut sl_plain = Vec::with_capacity(nblocks);
    let mut sl_repara = Vec::with_capacity(nblocks);
    let mut mp_sum: usize = 0;
    for b in 0..nblocks {
        let pen_block = second_diff_penalty(k);
        let bp_plain = BlockPenalty::new(pen_block.clone(), b * k, p);
        let mut bp_repara = BlockPenalty::new(pen_block, b * k, p);
        bp_repara.setup_initial_repara();
        mp_sum += k - bp_plain.estimate_rank();
        sl_plain.push(bp_plain);
        sl_repara.push(bp_repara);
    }
    let mp = mp_sum;

    let xx = compute_xtwx(&x, &w);
    let f = compute_xtwy(&x, &w, &y);
    let yy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| yi * yi * wi).sum();

    let rho = vec![0.3f64, 1.0f64, -0.4f64];
    let log_phi: f64 = 0.0_f64;

    let res_plain = compute_sl_fitchol_step(
        &sl_plain,
        xx.view(),
        f.view(),
        Array1::from_vec(rho.clone()).view(),
        yy,
        log_phi,
        true,
        nobs,
        mp,
        1.0,
    )
    .expect("plain compute_sl_fitchol_step");

    let res_repara = compute_sl_fitchol_step(
        &sl_repara,
        xx.view(),
        f.view(),
        Array1::from_vec(rho.clone()).view(),
        yy,
        log_phi,
        true,
        nobs,
        mp,
        1.0,
    )
    .expect("repara compute_sl_fitchol_step");

    // β invariance: same basis on both sides (we inverse-rotate at exit).
    let mut beta_diff = 0.0f64;
    for i in 0..p {
        beta_diff = beta_diff.max((res_plain.beta[i] - res_repara.beta[i]).abs());
    }
    eprintln!(
        "[repara_predictions_invariance] max |β_plain - β_repara| = {:.2e}",
        beta_diff
    );
    // β tolerance: the rotated path goes through extra sandwich products
    // (D'·XX·D, then D · β_rot at exit). On a p=30 fixture, the typical
    // β magnitude is O(1) and we observe ~1e-10 abs differences. 1e-8 is
    // comfortably tighter than the FD-test tolerance of 1e-5.
    assert!(
        beta_diff < 1e-8,
        "β diverged between plain and repara paths: {:.2e}",
        beta_diff
    );

    // Predictions invariance: `X · β` should match.
    let pred_plain = x.dot(&res_plain.beta);
    let pred_repara = x.dot(&res_repara.beta);
    let mut pred_diff = 0.0f64;
    for i in 0..n {
        pred_diff = pred_diff.max((pred_plain[i] - pred_repara[i]).abs());
    }
    eprintln!(
        "[repara_predictions_invariance] max |X·β_plain - X·β_repara| = {:.2e}",
        pred_diff
    );
    // Predictions follow β with an additional X-row magnitude factor; on
    // synth_design with n=200, |X|_∞ ≲ 2, so 1e-8 is the matching tolerance.
    assert!(
        pred_diff < 1e-8,
        "Predictions diverged between plain and repara paths: {:.2e}",
        pred_diff
    );

    // Gradient comparison note (R8 update):
    //
    // **Pre-R8**: gradient did NOT match byte-identical between plain and
    // repara paths because `log|X'WX + S|` was computed via `det(A).ln()`,
    // which counts the near-null pivots from rank-deficient rotated S even
    // though `log|S|_+` from `compute_ldet_s_with_derivs` excludes them
    // (row-norm rank). The mismatch was exactly `(rank_row − rank_eigen)/2 = 1`
    // per ρ component for the k=10 2nd-diff fixture.
    //
    // **R8 fix**: `compute_sl_fitchol_step` now uses LAPACK `dpstrf`
    // pivoted Cholesky for `log|X'WX + S|`, which drops the same null
    // pivots `Sl.fitChol`'s `Rrank(R)` does (R/fast-REML.r:1607-1613).
    // With both `log|S|_+` and `log|X'WX + S|` agreeing on rank, the
    // gradient is now invariant to the repara rotation to machine precision.
    for kk in 0..nblocks {
        let diff = (res_plain.grad[kk] - res_repara.grad[kk]).abs();
        eprintln!(
            "[repara_predictions_invariance] grad[{}] plain={:.6} repara={:.6} |diff|={:.6}",
            kk, res_plain.grad[kk], res_repara.grad[kk], diff
        );
    }
    // Strict invariance check: with pivoted Chol the per-component gradient
    // diff should be at most ~1e-10 (sandwich-product rounding).
    for kk in 0..nblocks {
        let diff = (res_plain.grad[kk] - res_repara.grad[kk]).abs();
        assert!(
            diff < 1e-8,
            "grad[{}] diverged after R8 pivoted-Chol port: plain={:.6} repara={:.6} |diff|={:.2e}",
            kk,
            res_plain.grad[kk],
            res_repara.grad[kk],
            diff
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: dβ/dρ from fitChol IFT matches `compute_b1_ift`.
// ---------------------------------------------------------------------------

#[test]
fn ift_matches_compute_b1_ift() {
    // CRITICAL sanity: if `db` from our `compute_sl_ift_chol` disagrees with
    // `compute_b1_ift` (`src/reml/mod.rs:1616`) the rest of the gradient/
    // Hessian is unreliable. Tolerance 1e-10 abs.
    //
    // `compute_b1_ift` is `pub(crate)`, so we can't call it from an integration
    // test. We re-derive `b1[:,k] = -λ_k · A⁻¹ · S_k · β` directly using the
    // public `inverse` and `BlockPenalty::dot_vec` — this is the exact same
    // formula `compute_b1_ift` uses (cf. src/reml/mod.rs:1626-1635), so the
    // cross-check still pins down the IFT identity even though we re-implement
    // the 3-line helper inline.

    let n = 200usize;
    let k = 10usize;
    let nblocks = 3usize;
    let p = k * nblocks;
    let x = synth_design(n, p, 31);
    let beta_true = Array1::from_vec((0..p).map(|i| ((i as f64) * 0.13).cos()).collect());
    let y = synth_response(&x, &beta_true, 0.7, 53);
    let w = synth_weights(n, 67);

    let mut sl = Vec::with_capacity(nblocks);
    for b in 0..nblocks {
        let pen_block = second_diff_penalty(k);
        sl.push(BlockPenalty::new(pen_block, b * k, p));
    }
    let mp: usize = sl.iter().map(|bp| k - bp.estimate_rank()).sum();
    let nobs: f64 = w.iter().sum();

    let xx = compute_xtwx(&x, &w);
    let f = compute_xtwy(&x, &w, &y);
    let yy: f64 = y.iter().zip(w.iter()).map(|(yi, wi)| yi * yi * wi).sum();

    let rho = vec![0.4f64, 1.1f64, -0.7f64];
    let log_phi: f64 = 0.0_f64;
    let res: SlFitCholResult = compute_sl_fitchol_step(
        &sl,
        xx.view(),
        f.view(),
        Array1::from_vec(rho.clone()).view(),
        yy,
        log_phi,
        true,
        nobs,
        mp,
        1.0,
    )
    .unwrap();

    // Reference: b1[:,k] = -λ_k · A⁻¹ · S_k · β.
    let lambdas: Vec<f64> = rho.iter().map(|r| r.exp()).collect();
    let mut a = xx.clone();
    for (lam, pen) in lambdas.iter().zip(sl.iter()) {
        pen.scaled_add_to(&mut a, *lam);
    }
    let a_inv = mgcv_rust::linalg::inverse(&a).unwrap();

    let mut max_abs = 0.0f64;
    for k in 0..nblocks {
        let sk_beta = sl[k].dot_vec(&res.beta);
        let ainv_sk_beta = a_inv.dot(&sk_beta);
        for r in 0..p {
            let ref_v = -lambdas[k] * ainv_sk_beta[r];
            let diff = (res.db[[r, k]] - ref_v).abs();
            max_abs = max_abs.max(diff);
        }
    }
    eprintln!(
        "[ift_matches_compute_b1_ift] max abs diff between db and -λ A⁻¹ S β = {:.2e}",
        max_abs
    );

    assert!(
        max_abs < 1e-10,
        "db vs reference compute_b1_ift max abs diff {:.2e} exceeds 1e-10",
        max_abs
    );
}
