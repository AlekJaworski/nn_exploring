//! Example demonstrating when HIGHER smoothing parameters are selected
//!
//! NOTE: "High λ" is relative. In practice with GCV/REML:
//! - Complex signals (sine waves): λ ≈ 0.01-0.1
//! - Simple signals (linear/constant): λ ≈ 0.1-0.5
//! - Very simple + high noise: λ ≈ 0.5-2.0
//!
//! This example shows how λ increases with signal simplicity.

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Higher Lambda Example ===\n");
    println!("Comparing lambda selection for different signal complexities\n");

    let n = 300;
    let num_basis = 20;
    let noise_level = 1.5;  // Higher noise to encourage smoothing

    println!("Setup: n={}, k={}, noise={}\n", n, num_basis, noise_level);

    // Test signals of decreasing complexity
    let signals: Vec<(&str, Box<dyn Fn(f64) -> f64>)> = vec![
        ("Complex (double sine)", Box::new(|x: f64| (2.0 * std::f64::consts::PI * x).sin() + 0.5 * (6.0 * std::f64::consts::PI * x).sin())),
        ("Moderate (single sine)", Box::new(|x: f64| (2.0 * std::f64::consts::PI * x).sin())),
        ("Simple (quadratic)", Box::new(|x: f64| (x - 0.5).powi(2))),
        ("Very simple (constant)", Box::new(|x: f64| 2.0)),
    ];

    let mut results = Vec::new();

    for (name, signal_fn) in signals.iter() {
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let true_y = signal_fn(xi);
                let noise = noise_level * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
                true_y + noise
            })
            .collect();

        let x = Array1::from_vec(x_data);
        let y = Array1::from_vec(y_data);
        let x_matrix = x.to_shape((n, 1))?.to_owned();

        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
        gam.add_smooth(smooth);
        gam.fit(&x_matrix, &y, OptimizationMethod::GCV, 10, 100, 1e-6)?;

        if let Some(ref params) = gam.smoothing_params {
            results.push((name, params.lambda[0]));
        }
    }

    // Display results
    println!("{}", "=".repeat(60));
    println!("Results: Lambda Selection by Signal Complexity");
    println!("{}", "=".repeat(60));
    println!("{:<25} {:>12}", "Signal Type", "λ (GCV)");
    println!("{}", "-".repeat(60));

    for (name, lambda) in &results {
        let category = if *lambda < 0.05 {
            "LOW"
        } else if *lambda < 0.15 {
            "MODERATE"
        } else if *lambda < 0.5 {
            "HIGH"
        } else {
            "VERY HIGH"
        };
        println!("{:<25} {:>12.6}  ({})", name, lambda, category);
    }

    println!("\n{}", "=".repeat(60));
    println!("Key Insights:");
    println!("{}", "=".repeat(60));
    println!("1. Simpler signals generally get HIGHER λ");
    println!("2. Higher λ prevents overfitting when flexibility isn't needed");
    println!("3. 'High λ' is relative - typically means λ > 0.2");
    println!("4. Very high λ (> 1.0) is rare with optimal GCV/REML\n");

    Ok(())
}
