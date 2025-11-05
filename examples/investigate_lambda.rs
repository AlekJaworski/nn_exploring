//! Investigate why constant signal doesn't get high lambda

use mgcv_rust::*;
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Investigating Lambda Selection for Constant Signal ===\n");

    let n = 300;
    let noise_level = 0.5;

    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

    // Constant signal with noise
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, _xi)| {
            let true_y = 2.0;  // Constant!
            let noise = noise_level * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data);
    let y = Array1::from_vec(y_data);

    // Test with different number of basis functions
    for num_basis in [10, 15, 20, 25, 30] {
        println!("\n{}", "=".repeat(60));
        println!("Testing with {} basis functions", num_basis);
        println!("{}", "=".repeat(60));

        let smooth = gam::SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
        let basis_matrix = smooth.evaluate(&x)?;
        let penalty = &smooth.penalty;
        let weights = Array1::ones(n);

        // Manually compute GCV for different lambda values
        println!("\nManual GCV scan:");
        println!("{:<12} {:<12}", "lambda", "GCV");
        println!("{}", "-".repeat(24));

        let mut best_lambda = 0.001;
        let mut best_gcv = f64::INFINITY;

        for log_lambda in -40..30 {
            let lambda = 10.0_f64.powf(log_lambda as f64 / 10.0);  // Finer grid
            let gcv = reml::gcv_criterion(&y, &basis_matrix, &weights, lambda, penalty)?;

            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lambda;
            }

            // Print every 5th value
            if log_lambda % 5 == 0 {
                println!("{:<12.6} {:<12.6}", lambda, gcv);
            }
        }

        println!("\nBest lambda: {:.6}", best_lambda);
        println!("Best GCV: {:.6}", best_gcv);
    }

    Ok(())
}
