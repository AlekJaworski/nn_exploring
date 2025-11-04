use mgcv::prelude::*;
use ndarray::Array1;

fn main() -> Result<()> {
    println!("MGCV-RS: Generalized Additive Models in Rust\n");

    // Example 1: Gaussian GAM with cubic spline
    println!("=== Example 1: Fitting a sine wave with cubic splines ===");

    // Generate data: y = sin(x) + noise
    let n = 100;
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n);
    let mut y = Array1::zeros(n);

    for i in 0..n {
        y[i] = x[i].sin() + 0.2 * (rand_normal() - 0.5);
    }

    // Create GAM with REML smoothing parameter selection
    let mut gam = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::REML,
    );

    // Add a cubic spline smooth
    let basis = CubicSpline::new(&x.view(), 15)?;
    gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);

    // Fit the model
    println!("Fitting GAM...");
    gam.fit(&[x.view()], &y.view(), 50, 1e-6)?;

    // Print summary
    if let Some(summary) = gam.summary() {
        println!("Model fitted successfully!");
        println!("  Effective degrees of freedom: {:.2}", summary.edf);
        println!("  Deviance: {:.4}", summary.deviance);
        println!("  Smoothing parameter (lambda): {:.6}", summary.lambdas[0]);
    }

    // Make predictions
    let x_pred = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, 200);
    let y_pred = gam.predict(&[x_pred.view()])?;

    println!("\nFirst 10 predictions:");
    for i in 0..10.min(y_pred.len()) {
        println!("  x={:.3}, y_pred={:.3}", x_pred[i], y_pred[i]);
    }

    // Example 2: P-spline
    println!("\n=== Example 2: Fitting with P-splines ===");

    let mut gam2 = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::GCV { gamma: 1.4 },
    );

    let pspline = PSpline::new(&x.view(), 20, 3)?;
    gam2.add_smooth("s(x)".to_string(), Box::new(pspline), 0.1);

    println!("Fitting GAM with P-splines...");
    gam2.fit(&[x.view()], &y.view(), 50, 1e-6)?;

    if let Some(summary) = gam2.summary() {
        println!("Model fitted successfully!");
        println!("  Effective degrees of freedom: {:.2}", summary.edf);
        println!("  Deviance: {:.4}", summary.deviance);
        println!("  Smoothing parameter (lambda): {:.6}", summary.lambdas[0]);
    }

    println!("\n=== Done! ===");

    Ok(())
}

// Simple random normal generator (for demonstration)
fn rand_normal() -> f64 {
    // Box-Muller transform
    use std::f64::consts::PI;
    let u1: f64 = rand::random();
    let u2: f64 = rand::random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// For random number generation
extern crate rand;
