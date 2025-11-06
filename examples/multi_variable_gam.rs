//! Test multi-variable GAM with multiple smoothing parameters optimized via REML
//!
//! This example demonstrates:
//! 1. Fitting a GAM with multiple smooth terms (one per predictor)
//! 2. Joint Newton optimization of multiple λ values using REML
//! 3. Following Wood (2011) fast stable REML algorithm

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Variable GAM with REML ===\n");

    // Generate data with 3 predictors
    let n = 300;
    let x1_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let x2_data: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64) * 2.0 - 1.0).collect();
    let x3_data: Vec<f64> = (0..n).map(|i| ((i as f64 / (n - 1) as f64) - 0.5) * 4.0).collect();

    // True model: y = sin(2πx₁) + 0.5·x₂² + 2·x₃ + noise
    let y_data: Vec<f64> = x1_data.iter()
        .zip(x2_data.iter())
        .zip(x3_data.iter())
        .enumerate()
        .map(|(i, ((x1, x2), x3))| {
            let true_y = (2.0 * std::f64::consts::PI * x1).sin()  // Complex: sine wave
                       + 0.5 * x2 * x2                             // Moderate: quadratic
                       + 2.0 * x3;                                 // Simple: linear
            let noise = 0.5 * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    println!("Data:");
    println!("  n = {} observations", n);
    println!("  True model: y = sin(2πx₁) + 0.5·x₂² + 2·x₃ + noise\n");
    println!("Signal complexities:");
    println!("  x₁: Complex (sine wave) → expect LOW λ₁");
    println!("  x₂: Moderate (quadratic) → expect MEDIUM λ₂");
    println!("  x₃: Simple (linear) → expect HIGH λ₃\n");

    // Create GAM with 3 smooth terms
    let mut gam = GAM::new(Family::Gaussian);

    // Add smooth for x1 (sine wave - needs flexibility)
    let smooth1 = SmoothTerm::cubic_spline("x1".to_string(), 15, 0.0, 1.0)?;
    gam.add_smooth(smooth1);

    // Add smooth for x2 (quadratic - moderate complexity)
    let smooth2 = SmoothTerm::cubic_spline("x2".to_string(), 12, -1.0, 1.0)?;
    gam.add_smooth(smooth2);

    // Add smooth for x3 (linear - simple)
    let smooth3 = SmoothTerm::cubic_spline("x3".to_string(), 10, -2.0, 2.0)?;
    gam.add_smooth(smooth3);

    // Combine data into matrix [x1, x2, x3]
    let mut x_matrix = ndarray::Array2::zeros((n, 3));
    for i in 0..n {
        x_matrix[[i, 0]] = x1_data[i];
        x_matrix[[i, 1]] = x2_data[i];
        x_matrix[[i, 2]] = x3_data[i];
    }

    let y = Array1::from_vec(y_data.clone());

    println!("{}", "=".repeat(60));
    println!("Fitting GAM with 3 smooths using REML...");
    println!("{}", "=".repeat(60));
    println!("Using Wood (2011) joint Newton optimization\n");

    // Fit with REML (will use Newton method for multiple λ)
    gam.fit(&x_matrix, &y, OptimizationMethod::REML, 20, 100, 1e-6)?;

    println!("\n{}", "=".repeat(60));
    println!("Results:");
    println!("{}", "=".repeat(60));

    if let Some(ref params) = gam.smoothing_params {
        println!("\nSelected smoothing parameters:");
        println!("  λ₁ (x₁, sine):     {:.6}", params.lambda[0]);
        println!("  λ₂ (x₂, quadratic): {:.6}", params.lambda[1]);
        println!("  λ₃ (x₃, linear):    {:.6}", params.lambda[2]);

        // Check if pattern matches expectations
        println!("\nExpected pattern:");
        if params.lambda[0] < params.lambda[1] && params.lambda[1] < params.lambda[2] {
            println!("  ✓ λ₁ < λ₂ < λ₃ (complex < moderate < simple)");
        } else {
            println!("  Pattern: λ₁={:.3}, λ₂={:.3}, λ₃={:.3}",
                    params.lambda[0], params.lambda[1], params.lambda[2]);
        }
    }

    if let Some(deviance) = gam.deviance {
        let rmse = (deviance / n as f64).sqrt();
        println!("\nModel fit:");
        println!("  Deviance: {:.4}", deviance);
        println!("  RMSE: {:.4}", rmse);
    }

    // Make predictions on test data
    let n_test = 50;
    let x1_test: Vec<f64> = (0..n_test).map(|i| i as f64 / (n_test - 1) as f64).collect();
    let x2_test: Vec<f64> = vec![0.0; n_test];  // Fix x2 = 0
    let x3_test: Vec<f64> = vec![0.0; n_test];  // Fix x3 = 0

    let mut x_test = ndarray::Array2::zeros((n_test, 3));
    for i in 0..n_test {
        x_test[[i, 0]] = x1_test[i];
        x_test[[i, 1]] = x2_test[i];
        x_test[[i, 2]] = x3_test[i];
    }

    let y_pred = gam.predict(&x_test)?;

    // True values (with x2=0, x3=0)
    let y_true: Vec<f64> = x1_test.iter()
        .map(|&x1| (2.0 * std::f64::consts::PI * x1).sin())
        .collect();

    let rmse_test: f64 = y_pred.iter()
        .zip(y_true.iter())
        .map(|(pred, true_val)| (pred - true_val).powi(2))
        .sum::<f64>() / n_test as f64;
    let rmse_test = rmse_test.sqrt();

    println!("\nTest predictions (x₂=0, x₃=0):");
    println!("  RMSE vs true sine: {:.4}", rmse_test);

    println!("\n{}", "=".repeat(60));
    println!("Success! Multi-variable GAM with REML working!");
    println!("{}", "=".repeat(60));

    Ok(())
}
