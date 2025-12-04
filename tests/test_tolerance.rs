//! Test if relaxing tolerance helps multi-D convergence

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm};
use mgcv_rust::basis::{BasisFunction, CubicRegressionSpline};
use mgcv_rust::penalty::compute_penalty;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::distributions::{Distribution, Uniform};
use std::f64::consts::PI;

fn main() {
    // Same setup as debug_multidim.rs
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let uniform = Uniform::new(0.0, 1.0);

    let n = 500;
    let k = 15;

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

    let total_basis = k * 2;
    let mut full_design = Array2::zeros((n, total_basis));
    full_design.slice_mut(ndarray::s![.., 0..k]).assign(&design1);
    full_design.slice_mut(ndarray::s![.., k..2*k]).assign(&design2);

    let mut penalties = Vec::new();
    let mut penalty_block1 = Array2::zeros((total_basis, total_basis));
    penalty_block1.slice_mut(ndarray::s![0..k, 0..k]).assign(&penalty1);
    penalties.push(penalty_block1);

    let mut penalty_block2 = Array2::zeros((total_basis, total_basis));
    penalty_block2.slice_mut(ndarray::s![k..2*k, k..2*k]).assign(&penalty2);
    penalties.push(penalty_block2);

    let weights = Array1::ones(n);

    // Test different tolerances
    for tolerance in [0.05, 0.1, 0.2] {
        println!("\n=== Testing tolerance = {} ===", tolerance);
        std::env::set_var("MGCV_PROFILE", "1");

        let mut sp = SmoothingParameter::new_with_algorithm(
            2,
            OptimizationMethod::REML,
            REMLAlgorithm::Newton
        );

        match sp.optimize(&y, &full_design, &weights, &penalties, 30, tolerance) {
            Ok(_) => {
                println!("✓ Converged!");
                println!("  λ₁ = {:.6}, λ₂ = {:.6}", sp.lambda[0], sp.lambda[1]);
            }
            Err(e) => {
                println!("✗ Failed: {}", e);
                println!("  λ₁ = {:.6}, λ₂ = {:.6}", sp.lambda[0], sp.lambda[1]);
            }
        }

        std::env::remove_var("MGCV_PROFILE");
    }
}
