//! Tests for the edge.correct saturation-detection helper in
//! `src/smooth.rs::detect_saturating_smooths`.
//!
//! Port of the candidate-flat selection from
//! `mgcv_analysis/mgcv/R/gam.fit3.r:1672`:
//!
//! ```r
//! flat <- which(abs(grad2) < abs(grad)*100)   # candidates for reduction
//! ```
//!
//! where `grad2 = diag(hess)`. A smooth is "saturating" when its REML
//! Hessian diagonal has collapsed to two orders of magnitude below
//! its own gradient — the smooth has reached working infinity.
//!
//! These tests stay at the level of the helper, exercising the
//! detection criterion against hand-built (g, H) pairs that mimic the
//! saturating-λ regime seen on the `2d_gamma_log_n200` parity fixture
//! (per `mgcv_rust - Tk·KK' edge.correct regression 2026-05-10.md`).

use mgcv_rust::smooth::detect_saturating_smooths;
use ndarray::{array, Array1, Array2};

#[test]
fn detects_single_saturating_smooth() {
    // Smooth 0: |H_ii| / |g_i| = 1e-4 / 5e-2 = 2e-3  <<  100 → saturating.
    // Smooth 1: |H_ii| / |g_i| = 12 / 0.3 = 40  <<  100 also satisfies condition
    //   per mgcv's `|grad2| < |grad|*100`; we mimic mgcv's exact form here.
    // Smooth 2: |H_ii| / |g_i| = 1500 / 0.3 = 5000 → NOT saturating.
    //
    // The fixture mirrors a 3-smooth Newton iterate where smooth 0 has
    // collapsed to its null space (gradient O(0.05), curvature O(1e-4))
    // while smooth 2 is still healthy.
    let grad: Array1<f64> = array![5.0e-2, 0.3, 0.3];
    let hess: Array2<f64> = array![
        [1.0e-4, 0.0, 0.0],
        [0.0, 12.0, 0.0],
        [0.0, 0.0, 1500.0],
    ];

    let flat = detect_saturating_smooths(&grad, &hess);

    assert_eq!(flat.len(), 3);
    assert!(flat[0], "smooth 0 should be flagged saturating");
    assert!(flat[1], "smooth 1 satisfies |H_ii| < |g_i|*100 too");
    assert!(!flat[2], "smooth 2 has healthy curvature");
}

#[test]
fn ignores_converged_zero_gradient_smooths() {
    // When |g_i| == 0 the smooth is at an interior stationary point — not
    // a saturating-λ candidate. mgcv's `abs(grad2) < abs(grad)*100`
    // evaluates `0 < 0` (false) in that case, which we replicate.
    let grad: Array1<f64> = array![0.0, 1.0];
    let hess: Array2<f64> = array![[0.0, 0.0], [0.0, 50.0]];

    let flat = detect_saturating_smooths(&grad, &hess);

    assert_eq!(flat, vec![false, true]);
}

#[test]
fn returns_all_false_on_dimension_mismatch() {
    // Robustness: if a caller hands us a mis-shaped Hessian we must not
    // panic — return an all-false mask so the Newton loop carries on.
    let grad: Array1<f64> = array![1.0, 1.0, 1.0];
    let hess: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
    assert_eq!(
        detect_saturating_smooths(&grad, &hess),
        vec![false, false, false]
    );
}

#[test]
fn saturating_lambda_synthetic_fit() {
    // Single-smooth fixture chosen so the smooth's gradient at the
    // outer-Newton converged log_λ is small in absolute terms (~5e-4)
    // while its curvature is essentially zero (~5e-8). This matches
    // what the parity battery sees on `2d_gamma_log_n200`'s smooth 2.
    //
    // Specifically the ratio |H_ii|/|g_i| = 1e-4 << 100 so the
    // detector must flag this dimension as saturating.
    let grad: Array1<f64> = array![5.0e-4];
    let hess: Array2<f64> = array![[5.0e-8]];

    let flat = detect_saturating_smooths(&grad, &hess);
    assert_eq!(flat, vec![true]);
}

#[test]
fn healthy_quadratic_well_is_not_saturating() {
    // For a clean quadratic well, |H_ii| dominates |g_i| (curvature
    // bigger than gradient), so the detector returns all-false.
    let grad: Array1<f64> = array![0.2, 0.4];
    let hess: Array2<f64> = array![[100.0, 0.0], [0.0, 50.0]];
    assert_eq!(
        detect_saturating_smooths(&grad, &hess),
        vec![false, false]
    );
}
