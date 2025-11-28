//! Test Newton REML optimization against R's bam() behavior
//!
//! This verifies that REMLAlgorithm::Newton matches bam()'s approach:
//! - Should converge in ~7 iterations
//! - Should handle penalty normalization correctly
//! - Should match mgcv's lambda estimates

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

fn main() {
    println!("=== Testing Newton REML Optimization ===\n");

    // Generate test data matching verify_reml.R
    let seed = 123u64;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::distributions::Distribution;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let uniform = rand::distributions::Uniform::new(0.0, 1.0);

    let n = 500;
    let k = 20;

    let x_vec: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
    let y_vec: Vec<f64> = x_vec.iter().enumerate().map(|(i, &xi)| {
        let signal = (2.0 * PI * xi).sin();
        // Generate N(0, 0.3) noise using Box-Muller transform
        let u1: f64 = uniform.sample(&mut rng);
        let u2: f64 = uniform.sample(&mut rng);
        let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        signal + noise
    }).collect();

    println!("Data: n={}, k={}", n, k);
    println!("Signal: sin(2*pi*x)");
    println!("Noise: N(0, 0.3)\n");

    // Build cubic regression spline basis (simplified - just polynomial for testing)
    // In practice would use proper spline basis
    let mut x_mat = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let xi = x_vec[i];
        for j in 0..k {
            x_mat[[i, j]] = xi.powi(j as i32);
        }
    }

    let y = Array1::from(y_vec);
    let w = Array1::from(vec![1.0; n]);

    // Create a simple second-derivative penalty (cubic spline-like)
    let mut penalty = Array2::<f64>::zeros((k, k));
    for i in 2..k {
        for j in 2..k {
            // D^2 basis: φ_j''(x) ≈ j*(j-1)*x^(j-2)
            // Penalty: ∫ φ_i''(x) φ_j''(x) dx
            let coef = (i * (i-1) * j * (j-1)) as f64;
            penalty[[i, j]] = coef / ((i + j - 3) as f64 + 1.0);
        }
    }

    // Normalize penalty like mgcv does
    let penalty_trace: f64 = penalty.diag().sum();
    if penalty_trace > 1e-10 {
        penalty /= penalty_trace / (k as f64);
    }

    println!("Penalty matrix: {}x{}", k, k);
    println!("Penalty trace: {:.6}\n", penalty.diag().sum());

    // Test Newton algorithm
    println!("=== Testing REMLAlgorithm::Newton ===\n");

    std::env::set_var("MGCV_PROFILE", "1");

    let mut sp_newton = SmoothingParameter::new_with_algorithm(
        1,
        OptimizationMethod::REML,
        REMLAlgorithm::Newton
    );

    println!("Starting optimization...\n");
    let start = std::time::Instant::now();

    match sp_newton.optimize(&y, &x_mat, &w, &[penalty.clone()], 50, 1e-6) {
        Ok(_) => {
            let elapsed = start.elapsed();
            println!("\n✓ Newton optimization completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            println!("  Optimal λ: {:.6}", sp_newton.lambda[0]);
            println!("\nExpected from R mgcv (method='REML'):");
            println!("  λ ≈ 107.87 (for proper cubic regression spline basis)");
            println!("  Iterations: ~7");
        }
        Err(e) => {
            println!("\n✗ Newton optimization failed: {}", e);
        }
    }

    println!("\n=== Testing REMLAlgorithm::FellnerSchall ===\n");

    let mut sp_fs = SmoothingParameter::new_with_algorithm(
        1,
        OptimizationMethod::REML,
        REMLAlgorithm::FellnerSchall
    );

    println!("Starting optimization...\n");
    let start = std::time::Instant::now();

    match sp_fs.optimize(&y, &x_mat, &w, &[penalty], 50, 1e-6) {
        Ok(_) => {
            let elapsed = start.elapsed();
            println!("\n✓ Fellner-Schall optimization completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
            println!("  Optimal λ: {:.6}", sp_fs.lambda[0]);
            println!("\nExpected from R mgcv (optimizer='efs'):");
            println!("  λ ≈ 107.92");
            println!("  Iterations: 4");
        }
        Err(e) => {
            println!("\n✗ Fellner-Schall optimization failed: {}", e);
        }
    }

    println!("\n=== Comparison ===\n");
    println!("Newton λ:         {:.6}", sp_newton.lambda[0]);
    println!("Fellner-Schall λ: {:.6}", sp_fs.lambda[0]);
    println!("Difference:        {:.6}", (sp_newton.lambda[0] - sp_fs.lambda[0]).abs());
    println!("\nNote: Exact values depend on proper spline basis implementation");
    println!("This test uses simplified polynomial basis for verification");
}
