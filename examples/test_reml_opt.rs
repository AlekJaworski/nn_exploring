//! Test REML optimization directly

use mgcv_rust::*;
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing REML Optimization ===\n");

    // Generate data
    let n = 300;  // Increased from 30 to 300
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let true_y = (2.0 * std::f64::consts::PI * xi).sin();
            let noise = 0.5 * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data);
    let y = Array1::from_vec(y_data);

    // Create basis
    let smooth = gam::SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
    let basis_matrix = smooth.evaluate(&x)?;
    let penalty = &smooth.penalty;
    let weights = Array1::ones(n);

    println!("Testing REML criterion at different λ values:\n");
    println!("{}", "=".repeat(50));
    println!(" log₁₀(λ) | λ value    | REML score");
    println!("{}", "=".repeat(50));

    for log_lambda in -6..3 {
        let lambda = 10.0_f64.powi(log_lambda);
        let reml_score = reml::reml_criterion(&y, &basis_matrix, &weights, lambda, penalty, None)?;

        println!(" {:8} | {:.6} | {:.6}",
            log_lambda, lambda, reml_score);
    }

    println!("{}", "=".repeat(50));

    println!("\nTesting GCV criterion at different λ values:\n");
    println!("{}", "=".repeat(50));
    println!(" log₁₀(λ) | λ value    | GCV score");
    println!("{}", "=".repeat(50));

    for log_lambda in -6..3 {
        let lambda = 10.0_f64.powi(log_lambda);
        let gcv_score = reml::gcv_criterion(&y, &basis_matrix, &weights, lambda, penalty)?;

        println!(" {:8} | {:.6} | {:.6}",
            log_lambda, lambda, gcv_score);
    }

    println!("{}", "=".repeat(50));

    // Find minimum by grid search
    println!("\nFinding minimum by fine grid search:");
    let mut best_lambda = 0.001;
    let mut best_reml = f64::INFINITY;

    for i in 0..100 {
        let log_lambda = -6.0 + i as f64 * 0.1; // from 10^-6 to 10^4
        let lambda = 10.0_f64.powf(log_lambda);
        let reml_score = reml::reml_criterion(&y, &basis_matrix, &weights, lambda, penalty, None)?;

        if reml_score < best_reml {
            best_reml = reml_score;
            best_lambda = lambda;
        }
    }

    println!("  Best λ: {:.6}", best_lambda);
    println!("  Best REML: {:.6}", best_reml);

    // Compare with GCV
    println!("\nComparing with GCV:");
    let mut best_lambda_gcv = 0.001;
    let mut best_gcv = f64::INFINITY;

    for i in 0..100 {
        let log_lambda = -6.0 + i as f64 * 0.1;
        let lambda = 10.0_f64.powf(log_lambda);
        let gcv_score = reml::gcv_criterion(&y, &basis_matrix, &weights, lambda, penalty)?;

        if gcv_score < best_gcv {
            best_gcv = gcv_score;
            best_lambda_gcv = lambda;
        }
    }

    println!("  Best λ (GCV): {:.6}", best_lambda_gcv);
    println!("  Best GCV: {:.6}", best_gcv);

    Ok(())
}
