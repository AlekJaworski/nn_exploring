//! Example demonstrating GAM smoothing on noisy data
//!
//! This shows how REML automatically selects smoothing parameters to balance
//! fit and smoothness when dealing with noisy observations.

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GAM Smoothing Example: Noisy Data ===\n");

    // Generate data with substantial noise
    let n = 300; // Sufficient points for good smoothing parameter selection
    let noise_level = 0.5; // Significant noise

    // True function: y = sin(2πx)
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

    // Add noise using a simple pseudo-random generator
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let true_y = (2.0 * std::f64::consts::PI * xi).sin();
            // Pseudo-random noise based on index
            let noise = noise_level * ((i as f64 * 0.7 + 0.3).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data.clone());
    let y = Array1::from_vec(y_data.clone());
    let x_matrix = x.clone().to_shape((n, 1))?.to_owned();

    println!("Data generated:");
    println!("  - {} observations", n);
    println!("  - X range: [{:.3}, {:.3}]", x[0], x[n - 1]);
    println!("  - Noise level: {:.2}", noise_level);
    println!("  - Y range: [{:.3}, {:.3}]\n",
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    // Show first few data points
    println!("Sample data points:");
    println!("  x     | y_obs   | y_true  | noise");
    println!("--------|---------|---------|--------");
    for i in (0..n.min(10)).step_by(2) {
        let true_y = (2.0 * std::f64::consts::PI * x_data[i]).sin();
        let noise = y_data[i] - true_y;
        println!(" {:.3}  | {:.4}  | {:.4}  | {:.4}",
            x_data[i], y_data[i], true_y, noise);
    }
    println!();

    // Fit GAM with different basis sizes to show smoothing effect
    for &num_basis in &[10, 15, 20] {
        println!("\n{}", "=".repeat(60));
        println!("Fitting GAM with {} basis functions", num_basis);
        println!("{}\n", "=".repeat(60));

        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline(
            "x".to_string(),
            num_basis,
            0.0,
            1.0
        )?;
        gam.add_smooth(smooth);

        // Fit with REML
        println!("Fitting with REML smoothing parameter selection...");
        gam.fit(
            &x_matrix,
            &y,
            OptimizationMethod::REML,
            10,   // more outer iterations
            100,  // more inner iterations
            1e-6  // tighter tolerance
        )?;

        // Print results
        if let Some(ref params) = gam.smoothing_params {
            println!("✓ Converged!");
            println!("\nSmoothing parameters:");
            for (i, &lambda) in params.lambda.iter().enumerate() {
                println!("  λ_{} = {:.6}", i, lambda);
                if lambda < 1e-8 {
                    println!("    → No smoothing (interpolating)");
                } else if lambda < 0.1 {
                    println!("    → Light smoothing");
                } else if lambda < 10.0 {
                    println!("    → Moderate smoothing");
                } else {
                    println!("    → Heavy smoothing");
                }
            }
        }

        if let Some(deviance) = gam.deviance {
            println!("\nModel fit:");
            println!("  Deviance: {:.6}", deviance);
            println!("  RMSE: {:.6}", (deviance / n as f64).sqrt());
        }

        if let Some(edf) = gam.edf() {
            println!("  Effective degrees of freedom: {:.2}", edf);
            println!("  Complexity: {:.1}%", 100.0 * edf / n as f64);
        }

        // Evaluate on fine grid and compute errors
        let n_pred = 200;
        let x_pred_vec: Vec<f64> = (0..n_pred).map(|i| i as f64 / (n_pred - 1) as f64).collect();
        let x_pred = Array1::from_vec(x_pred_vec.clone());
        let x_pred_matrix = x_pred.to_shape((n_pred, 1))?.to_owned();

        let predictions = gam.predict(&x_pred_matrix)?;

        // Compute mean squared error against true function
        let mse_true: f64 = x_pred_vec
            .iter()
            .zip(predictions.iter())
            .map(|(x_val, &y_pred)| {
                let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
                (y_pred - y_true).powi(2)
            })
            .sum::<f64>() / n_pred as f64;

        println!("\nPrediction accuracy (vs true function):");
        println!("  MSE:  {:.6}", mse_true);
        println!("  RMSE: {:.6}", mse_true.sqrt());

        // Show some predictions
        println!("\nSample predictions:");
        println!("  x     | y_pred  | y_true  | error");
        println!("--------|---------|---------|--------");
        for i in (0..n_pred).step_by(40) {
            let x_val = x_pred_vec[i];
            let y_pred = predictions[i];
            let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
            let error = (y_pred - y_true).abs();
            println!(" {:.3}  | {:.4}  | {:.4}  | {:.4}",
                x_val, y_pred, y_true, error);
        }
    }

    // Compare unsmoothed vs smoothed fit
    println!("\n\n{}", "=".repeat(60));
    println!("Comparison: GCV vs REML");
    println!("{}\n", "=".repeat(60));

    for method in &[OptimizationMethod::GCV, OptimizationMethod::REML] {
        let method_name = match method {
            OptimizationMethod::GCV => "GCV",
            OptimizationMethod::REML => "REML",
        };

        println!("\nFitting with {}...", method_name);
        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
        gam.add_smooth(smooth);

        gam.fit(&x_matrix, &y, *method, 10, 100, 1e-6)?;

        if let Some(ref params) = gam.smoothing_params {
            println!("  λ = {:.6}", params.lambda[0]);
        }

        if let Some(deviance) = gam.deviance {
            println!("  RMSE (training): {:.6}", (deviance / n as f64).sqrt());
        }

        // Prediction error
        let n_pred = 200;
        let x_pred_vec: Vec<f64> = (0..n_pred).map(|i| i as f64 / (n_pred - 1) as f64).collect();
        let x_pred = Array1::from_vec(x_pred_vec.clone());
        let x_pred_matrix = x_pred.to_shape((n_pred, 1))?.to_owned();
        let predictions = gam.predict(&x_pred_matrix)?;

        let mse_true: f64 = x_pred_vec
            .iter()
            .zip(predictions.iter())
            .map(|(x_val, &y_pred)| {
                let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
                (y_pred - y_true).powi(2)
            })
            .sum::<f64>() / n_pred as f64;

        println!("  RMSE (true function): {:.6}", mse_true.sqrt());
    }

    println!("\n✓ Smoothing demonstration complete!");
    println!("\nKey insight: The smoothing parameter λ controls the bias-variance tradeoff.");
    println!("- λ → 0: Less smoothing, follows data closely (may overfit)");
    println!("- λ → ∞: More smoothing, smoother curve (may underfit)");
    println!("- REML/GCV automatically find the optimal λ to minimize prediction error");

    Ok(())
}
