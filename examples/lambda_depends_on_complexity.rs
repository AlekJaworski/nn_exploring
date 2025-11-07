//! Example demonstrating how λ depends on signal complexity
//!
//! With the SAME number of basis functions:
//! - Complex signal (sine wave) → LOW λ (needs flexibility)
//! - Simple signal (linear/constant) → HIGH λ (enforces smoothness)

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Lambda Selection Depends on Signal Complexity ===\n");

    let n = 300;
    let num_basis = 25;  // More basis functions
    let noise_level = 0.5;

    println!("Setup: {} observations, {} basis functions, noise = {}\n",
        n, num_basis, noise_level);

    // Case 1: COMPLEX signal (sine wave)
    println!("{}", "=".repeat(60));
    println!("CASE 1: Complex Signal (sine wave)");
    println!("{}", "=".repeat(60));

    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y_complex: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let true_y = (2.0 * std::f64::consts::PI * xi).sin();
            let noise = noise_level * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data.clone());
    let y_c = Array1::from_vec(y_complex);
    let x_matrix = x.clone().to_shape((n, 1))?.to_owned();

    let mut gam1 = GAM::new(Family::Gaussian);
    let smooth1 = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam1.add_smooth(smooth1);
    gam1.fit(&x_matrix, &y_c, OptimizationMethod::GCV, 10, 100, 1e-6)?;

    if let Some(ref params) = gam1.smoothing_params {
        println!("Selected λ = {:.6}", params.lambda[0]);
        println!("Interpretation: Low λ needed to capture oscillations\n");
    }

    // Case 2: SIMPLE signal (nearly constant with slight trend)
    println!("{}", "=".repeat(60));
    println!("CASE 2: Simple Signal (constant)");
    println!("{}", "=".repeat(60));

    let y_simple: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, _xi)| {
            let true_y = 2.0;  // Constant!
            let noise = noise_level * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let y_s = Array1::from_vec(y_simple);

    let mut gam2 = GAM::new(Family::Gaussian);
    let smooth2 = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam2.add_smooth(smooth2);
    gam2.fit(&x_matrix, &y_s, OptimizationMethod::GCV, 10, 100, 1e-6)?;

    if let Some(ref params) = gam2.smoothing_params {
        println!("Selected λ = {:.6}", params.lambda[0]);
        println!("Interpretation: Higher λ prevents fitting noise\n");
    }

    // Compare
    println!("{}", "=".repeat(60));
    println!("Comparison:");
    println!("{}", "=".repeat(60));

    if let (Some(p1), Some(p2)) = (&gam1.smoothing_params, &gam2.smoothing_params) {
        let ratio = p2.lambda[0] / p1.lambda[0];
        println!("λ (complex) = {:.6}", p1.lambda[0]);
        println!("λ (simple)  = {:.6}", p2.lambda[0]);
        println!("Ratio = {:.2}x", ratio);
        println!("\nKey insight: Simpler signals require HIGHER λ");
        println!("to prevent overfitting with unnecessary complexity.");
    }

    Ok(())
}
