//! Test d=8, n=5000 with Rust Newton

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm};
use mgcv_rust::basis::{BasisFunction, CubicRegressionSpline};
use mgcv_rust::penalty::compute_penalty;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::distributions::{Distribution, Uniform};
use std::f64::consts::PI;

fn main() {
    println!("=== Testing n=5000, d=8, k=8 with Rust Newton ===\n");
    
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let uniform = Uniform::new(0.0, 1.0);

    let n = 5000;
    let d = 8;
    let k = 8;

    println!("Generating data: n={}, d={}, k={}\n", n, d, k);

    // Generate X matrix and y vector
    let mut x_vecs = Vec::new();
    for _ in 0..d {
        let x: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
        x_vecs.push(Array1::from(x));
    }

    // y = sum of sin(2πxᵢ) + noise
    let y: Array1<f64> = {
        let mut y_vec = Vec::new();
        for i in 0..n {
            let signal: f64 = x_vecs.iter().map(|x| (2.0 * PI * x[i]).sin()).sum();
            let u1 = uniform.sample(&mut rng);
            let u2 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            y_vec.push(signal + noise);
        }
        Array1::from(y_vec)
    };

    // Build design matrices and penalties
    println!("Building design matrices and penalties...");
    let mut design_matrices = Vec::new();
    let mut penalties_vec = Vec::new();

    for x in &x_vecs {
        let basis = CubicRegressionSpline::with_quantile_knots(x, k);
        let design = basis.evaluate(x).unwrap();
        let knots = basis.knots().unwrap();
        let mut penalty = compute_penalty("cr", k, Some(knots), 1).unwrap();

        // Apply penalty normalization
        let inf_norm_X = design.rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let maXX = inf_norm_X * inf_norm_X;

        let inf_norm_S = (0..k)
            .map(|i| (0..k).map(|j| penalty[[i, j]].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);

        if inf_norm_S > 1e-10 {
            penalty *= maXX / inf_norm_S;
        }

        design_matrices.push(design);
        penalties_vec.push(penalty);
    }

    // Combine into full design matrix
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    for (i, mat) in design_matrices.iter().enumerate() {
        full_design.slice_mut(ndarray::s![.., (i*k)..((i+1)*k)]).assign(mat);
    }

    // Create block-diagonal penalties
    let mut penalties = Vec::new();
    for (i, pen) in penalties_vec.iter().enumerate() {
        let mut block = Array2::zeros((total_basis, total_basis));
        block.slice_mut(ndarray::s![i*k..(i+1)*k, i*k..(i+1)*k]).assign(pen);
        penalties.push(block);
    }

    println!("Starting Newton optimization...\n");

    // Test Newton with profiling
    std::env::set_var("MGCV_PROFILE", "1");

    let mut sp = SmoothingParameter::new_with_algorithm(
        d,
        OptimizationMethod::REML,
        REMLAlgorithm::Newton
    );

    let weights = Array1::ones(n);
    let start = std::time::Instant::now();

    match sp.optimize(&y, &full_design, &weights, &penalties, 30, 1e-6) {
        Ok(_) => {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            
            println!("\n✅ Rust Newton CONVERGED!");
            println!("   Time: {:.1} ms", elapsed);
            println!("   Lambda (mean): {:.6}", sp.lambda.iter().sum::<f64>() / sp.lambda.len() as f64);
            println!("   Lambda (min): {:.6}", sp.lambda.iter().cloned().fold(f64::INFINITY, f64::min));
            println!("   Lambda (max): {:.6}", sp.lambda.iter().cloned().fold(0.0f64, f64::max));
            
            println!("\n=== Comparison ===");
            println!("{:<15} {:<10} {:<15} {:<15}", "Method", "Iterations", "Time (ms)", "Lambda (mean)");
            println!("{:<15} {:<10} {:<15.1} {:<15.6}", "gam(REML)", 7, 1162.4, 4.630146);
            println!("{:<15} {:<10} {:<15.1} {:<15.6}", "bam(REML)", 5, 167.7, 4.630127);
            println!("{:<15} {:<10} {:<15.1} {:<15.6}", "Rust Newton", "?", elapsed, 
                     sp.lambda.iter().sum::<f64>() / sp.lambda.len() as f64);
        }
        Err(e) => {
            println!("\n✗ Optimization failed: {}", e);
        }
    }
}
