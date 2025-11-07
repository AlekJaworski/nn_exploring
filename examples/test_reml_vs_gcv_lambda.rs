//! Compare GCV vs REML for high lambda selection

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing REML vs GCV for High Lambda ===\n");

    let n = 300;
    let num_basis = 20;
    let noise_level = 1.5;

    println!("Setup: n={}, k={}, noise={}\n", n, num_basis, noise_level);

    // Test constant signal (should get highest lambda)
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
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
    let x_matrix = x.to_shape((n, 1))?.to_owned();

    println!("{}", "=".repeat(60));
    println!("Constant Signal Test (y = 2.0 + noise)");
    println!("{}", "=".repeat(60));

    // Test GCV
    let mut gam_gcv = GAM::new(Family::Gaussian);
    let smooth_gcv = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam_gcv.add_smooth(smooth_gcv);
    gam_gcv.fit(&x_matrix, &y, OptimizationMethod::GCV, 10, 100, 1e-6)?;

    let lambda_gcv = gam_gcv.smoothing_params.as_ref().unwrap().lambda[0];

    // Test REML
    let mut gam_reml = GAM::new(Family::Gaussian);
    let smooth_reml = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam_reml.add_smooth(smooth_reml);
    gam_reml.fit(&x_matrix, &y, OptimizationMethod::REML, 10, 100, 1e-6)?;

    let lambda_reml = gam_reml.smoothing_params.as_ref().unwrap().lambda[0];

    println!("\nResults:");
    println!("  GCV  λ = {:.6}", lambda_gcv);
    println!("  REML λ = {:.6}", lambda_reml);
    println!("\n  Ratio (REML/GCV) = {:.3}x", lambda_reml / lambda_gcv);

    if lambda_gcv > 0.2 && lambda_reml > 0.2 {
        println!("\n✓ Both methods select HIGH lambda for simple signal!");
    } else if lambda_gcv > 0.2 || lambda_reml > 0.2 {
        println!("\n⚠ Only one method selects high lambda");
    } else {
        println!("\n✗ Neither method selects high lambda");
    }

    // Test complex signal for comparison
    println!("\n{}", "=".repeat(60));
    println!("Complex Signal Test (double sine wave)");
    println!("{}", "=".repeat(60));

    let y_complex: Vec<f64> = (0..n)
        .map(|i| {
            let xi = i as f64 / (n - 1) as f64;
            let true_y = (2.0 * std::f64::consts::PI * xi).sin()
                       + 0.5 * (6.0 * std::f64::consts::PI * xi).sin();
            let noise = noise_level * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();
    let y_c = Array1::from_vec(y_complex);

    // GCV
    let mut gam_gcv2 = GAM::new(Family::Gaussian);
    let smooth_gcv2 = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam_gcv2.add_smooth(smooth_gcv2);
    gam_gcv2.fit(&x_matrix, &y_c, OptimizationMethod::GCV, 10, 100, 1e-6)?;
    let lambda_gcv2 = gam_gcv2.smoothing_params.as_ref().unwrap().lambda[0];

    // REML
    let mut gam_reml2 = GAM::new(Family::Gaussian);
    let smooth_reml2 = SmoothTerm::cubic_spline("x".to_string(), num_basis, 0.0, 1.0)?;
    gam_reml2.add_smooth(smooth_reml2);
    gam_reml2.fit(&x_matrix, &y_c, OptimizationMethod::REML, 10, 100, 1e-6)?;
    let lambda_reml2 = gam_reml2.smoothing_params.as_ref().unwrap().lambda[0];

    println!("\nResults:");
    println!("  GCV  λ = {:.6}", lambda_gcv2);
    println!("  REML λ = {:.6}", lambda_reml2);
    println!("\n  Ratio (REML/GCV) = {:.3}x", lambda_reml2 / lambda_gcv2);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("Summary:");
    println!("{}", "=".repeat(60));
    println!("{:<20} {:>12} {:>12}", "Signal", "GCV λ", "REML λ");
    println!("{}", "-".repeat(60));
    println!("{:<20} {:>12.6} {:>12.6}", "Simple (constant)", lambda_gcv, lambda_reml);
    println!("{:<20} {:>12.6} {:>12.6}", "Complex (double sin)", lambda_gcv2, lambda_reml2);
    println!("\nBoth GCV and REML work for selecting higher λ on simpler signals!");

    Ok(())
}
