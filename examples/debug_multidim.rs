//! Debug multi-dimensional convergence issue
//! Testing d=2 to understand why Newton fails to converge

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm};
use mgcv_rust::basis::{BasisFunction, CubicRegressionSpline};
use mgcv_rust::penalty::compute_penalty;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::distributions::{Distribution, Uniform};
use std::f64::consts::PI;

fn main() {
    println!("=== Debugging Multi-Dimensional Convergence (d=2) ===\n");

    // Generate test data
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let uniform = Uniform::new(0.0, 1.0);

    let n = 500;
    let k = 15;
    let d = 2;

    let x1: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();

    let y: Array1<f64> = {
        let mut y_vec = Vec::new();
        for i in 0..n {
            let signal = (2.0 * PI * x1[i]).sin() + (2.0 * PI * x2[i]).cos();
            let u1 = uniform.sample(&mut rng);
            let u2 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            y_vec.push(signal + noise);
        }
        Array1::from(y_vec)
    };

    println!("Data: n={}, d={}, k={}", n, d, k);
    println!("Signal: sin(2πx₁) + cos(2πx₂)\n");

    // Build design matrices and penalties for each dimension
    let x1_arr = Array1::from(x1);
    let x2_arr = Array1::from(x2);

    let basis1 = CubicRegressionSpline::with_quantile_knots(&x1_arr, k);
    let basis2 = CubicRegressionSpline::with_quantile_knots(&x2_arr, k);

    let design1 = basis1.evaluate(&x1_arr).unwrap();
    let design2 = basis2.evaluate(&x2_arr).unwrap();

    let knots1 = basis1.knots().unwrap();
    let knots2 = basis2.knots().unwrap();

    let mut penalty1 = compute_penalty("cr", k, Some(knots1), 1).unwrap();
    let mut penalty2 = compute_penalty("cr", k, Some(knots2), 1).unwrap();

    // Apply penalty normalization
    for (design, penalty) in [(&design1, &mut penalty1), (&design2, &mut penalty2)] {
        let inf_norm_X = design.rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let maXX = inf_norm_X * inf_norm_X;

        let inf_norm_S = (0..k)
            .map(|i| (0..k).map(|j| penalty[[i, j]].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);

        if inf_norm_S > 1e-10 {
            *penalty *= maXX / inf_norm_S;
        }
    }

    // Combine into full design matrix
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    full_design.slice_mut(ndarray::s![.., 0..k]).assign(&design1);
    full_design.slice_mut(ndarray::s![.., k..2*k]).assign(&design2);

    // Create block-diagonal penalties (one for each smooth)
    let mut penalties = Vec::new();

    let mut penalty_block1 = Array2::zeros((total_basis, total_basis));
    penalty_block1.slice_mut(ndarray::s![0..k, 0..k]).assign(&penalty1);
    penalties.push(penalty_block1);

    let mut penalty_block2 = Array2::zeros((total_basis, total_basis));
    penalty_block2.slice_mut(ndarray::s![k..2*k, k..2*k]).assign(&penalty2);
    penalties.push(penalty_block2);

    println!("Penalty structure:");
    println!("  penalty1: {}×{} block in {}×{} matrix", k, k, total_basis, total_basis);
    println!("  penalty2: {}×{} block in {}×{} matrix\n", k, k, total_basis, total_basis);

    // Test with Newton
    println!("Testing Newton optimization with MGCV_PROFILE=1...\n");
    std::env::set_var("MGCV_PROFILE", "1");
    std::env::set_var("MGCV_GRAD_DEBUG", "1");

    let mut sp = SmoothingParameter::new_with_algorithm(
        d,
        OptimizationMethod::REML,
        REMLAlgorithm::Newton
    );

    let weights = Array1::ones(n);

    match sp.optimize(&y, &full_design, &weights, &penalties, 30, 1e-6) {
        Ok(_) => {
            println!("\n✓ Optimization succeeded!");
            println!("  λ₁ = {:.6}", sp.lambda[0]);
            println!("  λ₂ = {:.6}", sp.lambda[1]);
        }
        Err(e) => {
            println!("\n✗ Optimization failed: {}", e);
            println!("  λ₁ = {:.6}", sp.lambda[0]);
            println!("  λ₂ = {:.6}", sp.lambda[1]);
        }
    }
}
