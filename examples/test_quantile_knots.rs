use ndarray::Array1;
use mgcv_rust::basis::{BasisFunction, CubicSpline, BoundaryCondition};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Comparing Evenly-Spaced vs Quantile-Based Knots");
    println!("{}", "=".repeat(70));

    // Create data with non-uniform distribution
    // More data points in [0.4, 0.6] range
    let mut x_data = Vec::new();

    // Sparse data in [0.0, 0.4]
    for i in 0..10 {
        x_data.push(i as f64 * 0.04);
    }

    // Dense data in [0.4, 0.6]
    for i in 0..40 {
        x_data.push(0.4 + i as f64 * 0.005);
    }

    // Sparse data in [0.6, 1.0]
    for i in 0..10 {
        x_data.push(0.6 + i as f64 * 0.04);
    }

    let x_array = Array1::from_vec(x_data);

    println!("\nData distribution:");
    println!("  [0.0, 0.4]: 10 points (sparse)");
    println!("  [0.4, 0.6]: 40 points (dense)");
    println!("  [0.6, 1.0]: 10 points (sparse)");
    println!("  Total: {} points", x_array.len());

    let num_knots = 5;

    // Evenly-spaced knots
    println!("\n{}", "=".repeat(70));
    println!("Evenly-Spaced Knots");
    println!("{}", "=".repeat(70));

    let spline_even = CubicSpline::with_num_knots(0.0, 1.0, num_knots, BoundaryCondition::Natural);
    let knots_even = spline_even.knots().unwrap();

    println!("Knots:");
    for (i, &knot) in knots_even.iter().enumerate() {
        println!("  knot[{}] = {:.4}", i, knot);
    }

    // Quantile-based knots
    println!("\n{}", "=".repeat(70));
    println!("Quantile-Based Knots (mgcv-style)");
    println!("{}", "=".repeat(70));

    let spline_quantile = CubicSpline::with_quantile_knots(&x_array, num_knots, BoundaryCondition::Natural);
    let knots_quantile = spline_quantile.knots().unwrap();

    println!("Knots:");
    for (i, &knot) in knots_quantile.iter().enumerate() {
        println!("  knot[{}] = {:.4}", i, knot);
    }

    // Analysis
    println!("\n{}", "=".repeat(70));
    println!("Analysis");
    println!("{}", "=".repeat(70));

    println!("\nQuantile knots adapt to data density:");
    println!("  - More knots in dense region [0.4, 0.6]");
    println!("  - Fewer knots in sparse regions [0.0, 0.4] and [0.6, 1.0]");
    println!("\nThis matches mgcv behavior for better model adaptation.");
}
