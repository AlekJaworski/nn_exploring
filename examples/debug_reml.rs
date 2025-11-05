//! Debug REML criterion to understand why it selects λ ≈ 0

use mgcv_rust::*;
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Debugging REML Criterion ===\n");

    // Simple test data
    let n = 30;
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
    let x_matrix = x.to_shape((n, 1))?.to_owned();

    // Create smooth term
    let smooth = gam::SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
    let basis_matrix = smooth.evaluate(&x)?;
    let penalty = &smooth.penalty;

    println!("Penalty matrix info:");
    println!("  Shape: {}x{}", penalty.nrows(), penalty.ncols());

    // Check if penalty is singular
    use mgcv_rust::linalg::determinant;
    let det_s = determinant(penalty)?;
    println!("  det(S): {}", det_s);
    println!("  Is singular: {}", det_s.abs() < 1e-10);

    // Evaluate REML for different lambda values
    println!("\n{}", "=".repeat(60));
    println!("λ value     | REML score  | RSS         | log|X'WX+λS|");
    println!("{}", "=".repeat(60));

    let weights = Array1::ones(n);

    for log_lambda in -6..4 {
        let lambda = 10.0_f64.powi(log_lambda);

        use mgcv_rust::reml::reml_criterion;
        let reml_score = reml_criterion(&y, &basis_matrix, &weights, lambda, penalty, None)?;

        // Compute components manually to see what's happening
        use mgcv_rust::linalg::solve;
        let xtw = basis_matrix.t().to_owned();
        let xtwx = xtw.dot(&basis_matrix);
        let a = &xtwx + &(penalty * lambda);
        let b = xtw.dot(&y);
        let beta = solve(a.clone(), b)?;
        let fitted = basis_matrix.dot(&beta);
        let residuals: Array1<f64> = y.iter().zip(fitted.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        let log_det_a = determinant(&a)?.ln();

        println!("{:.6} | {:.6} | {:.6} | {:.6}",
            lambda, reml_score, rss, log_det_a);
    }

    println!("{}", "=".repeat(60));
    println!("\nObservation: If REML decreases as λ → 0, the criterion is wrong!");

    Ok(())
}
