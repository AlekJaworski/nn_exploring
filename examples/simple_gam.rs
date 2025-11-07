//! Simple example demonstrating GAM fitting with automatic smoothing parameter selection

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GAM Example: Fitting y = sin(2πx) + noise\n");

    // Generate example data
    let n = 100;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let signal = (2.0 * std::f64::consts::PI * xi).sin();
            let noise = 0.2 * ((i as f64 * 0.1).sin() - 0.5);
            signal + noise
        })
        .collect();

    let x = Array1::from_vec(x_data);
    let y = Array1::from_vec(y_data);

    // Reshape x to be a matrix (n x 1)
    let x_matrix = x.clone().into_shape((n, 1))?;

    println!("Data generated: {} observations", n);
    println!("X range: [{:.3}, {:.3}]", x[0], x[n - 1]);
    println!("Y range: [{:.3}, {:.3}]\n",
        y.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    // Create GAM model
    let mut gam = GAM::new(Family::Gaussian);

    // Add a smooth term with cubic spline basis
    let smooth = SmoothTerm::cubic_spline(
        "x".to_string(),
        20,  // number of basis functions
        0.0,  // min x
        1.0,  // max x
    )?;

    gam.add_smooth(smooth);
    println!("Model created with cubic spline basis (20 basis functions)\n");

    // Fit the model using REML for smoothing parameter selection
    println!("Fitting GAM with REML smoothing parameter selection...");
    gam.fit(
        &x_matrix,
        &y,
        OptimizationMethod::REML,
        5,    // max outer iterations (lambda optimization)
        50,   // max inner iterations (PiRLS)
        1e-4  // convergence tolerance
    )?;

    println!("Model fitted successfully!");

    // Print results
    if let Some(ref params) = gam.smoothing_params {
        println!("\nSmoothing parameters (lambda):");
        for (i, &lambda) in params.lambda.iter().enumerate() {
            println!("  λ_{} = {:.6}", i, lambda);
        }
    }

    if let Some(deviance) = gam.deviance {
        println!("\nDeviance: {:.6}", deviance);
    }

    if let Some(edf) = gam.edf() {
        println!("Effective degrees of freedom: {:.2}", edf);
    }

    // Make predictions on a fine grid
    println!("\nMaking predictions...");
    let n_pred = 200;
    let x_pred_vec: Vec<f64> = (0..n_pred).map(|i| i as f64 / n_pred as f64).collect();
    let x_pred = Array1::from_vec(x_pred_vec.clone());
    let x_pred_matrix = x_pred.into_shape((n_pred, 1))?;

    let predictions = gam.predict(&x_pred_matrix)?;

    // Print some predictions
    println!("\nSample predictions:");
    println!("  x     | y_pred  | y_true  | error");
    println!("--------|---------|---------|--------");
    for i in (0..n_pred).step_by(20) {
        let x_val = x_pred_vec[i];
        let y_pred = predictions[i];
        let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
        let error = (y_pred - y_true).abs();
        println!(" {:.3}  | {:.4}  | {:.4}  | {:.4}",
            x_val, y_pred, y_true, error);
    }

    // Compute mean absolute error
    let mae: f64 = x_pred_vec
        .iter()
        .zip(predictions.iter())
        .map(|(x_val, &y_pred)| {
            let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
            (y_pred - y_true).abs()
        })
        .sum::<f64>() / n_pred as f64;

    println!("\nMean Absolute Error: {:.6}", mae);
    println!("\n✓ GAM fitting complete!");

    Ok(())
}
