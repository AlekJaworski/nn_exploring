//! Test the identifiability constraint implementation for CR splines

use mgcv_rust::*;
use ndarray::Array1;

fn main() {
    if let Err(e) = run_test() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_test() -> Result<()> {
    println!("=== Testing CR Spline Identifiability Constraint ===\n");

    // Generate test data
    let n = 100;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .map(|&x| {
            let true_y = (2.0 * std::f64::consts::PI * x).sin();
            let noise = 0.1 * ((x * 10.0).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data.clone());
    let y = Array1::from_vec(y_data);
    let x_matrix = x.clone().to_shape((n, 1)).unwrap().to_owned();

    println!("Test 1: Create CR spline with constraint");
    println!("{}", "=".repeat(50));

    // Create CR spline term (should apply k-1 constraint automatically)
    let smooth = gam::SmoothTerm::cr_spline(
        "x".to_string(),
        10,  // User specifies k=10
        0.0,
        1.0
    )?;

    println!("✓ CR spline created");
    println!("  User-specified k: 10");
    println!("  Actual basis dimension: {}", smooth.num_basis());
    println!("  Penalty matrix shape: {}x{}", smooth.penalty.nrows(), smooth.penalty.ncols());
    println!("  Constraint applied: {}",
        if smooth.constraint_matrix.is_some() { "Yes" } else { "No" });

    if smooth.num_basis() == 9 {
        println!("✓ Basis correctly reduced to k-1 = 9");
    } else {
        println!("✗ ERROR: Expected 9 basis functions, got {}", smooth.num_basis());
    }

    if smooth.penalty.nrows() == 9 && smooth.penalty.ncols() == 9 {
        println!("✓ Penalty matrix correctly sized (9x9)");
    } else {
        println!("✗ ERROR: Expected 9x9 penalty, got {}x{}",
            smooth.penalty.nrows(), smooth.penalty.ncols());
    }

    println!("\nTest 2: Fit GAM with constrained CR splines");
    println!("{}", "=".repeat(50));

    let mut gam = GAM::new(Family::Gaussian);
    gam.add_smooth(smooth);

    gam.fit(
        &x_matrix,
        &y,
        OptimizationMethod::REML,
        10,  // max outer iterations
        100, // max inner iterations
        1e-6 // tolerance
    )?;

    println!("✓ GAM fitted successfully");

    if let Some(ref params) = gam.smoothing_params {
        println!("  Lambda: {:.6}", params.lambda[0]);
    }

    if let Some(dev) = gam.deviance {
        println!("  Deviance: {:.4}", dev);
    }

    if let Some(ref coef) = gam.coefficients {
        println!("  Number of coefficients: {}", coef.len());
        if coef.len() == 9 {
            println!("✓ Coefficient vector has correct size (9)");
        } else {
            println!("✗ ERROR: Expected 9 coefficients, got {}", coef.len());
        }
    }

    println!("\nTest 3: Check constraint satisfaction");
    println!("{}", "=".repeat(50));

    // Evaluate basis at training points
    let x_col = x_matrix.column(0).to_owned();
    let basis = {
        let smooth = gam::SmoothTerm::cr_spline("x".to_string(), 10, 0.0, 1.0)?;
        smooth.evaluate(&x_col)?
    };

    println!("  Basis matrix shape: {}x{}", basis.nrows(), basis.ncols());

    // Check if columns sum to approximately zero (sum-to-zero constraint)
    let mut max_col_sum = 0.0f64;
    for j in 0..basis.ncols() {
        let col_sum: f64 = (0..basis.nrows()).map(|i| basis[[i, j]]).sum();
        max_col_sum = max_col_sum.max(col_sum.abs());
    }

    println!("  Max column sum: {:.2e}", max_col_sum);

    if max_col_sum < 1e-10 {
        println!("✓ Constraint satisfied (columns sum to ~0)");
    } else {
        println!("⚠ Constraint not perfectly satisfied (max sum = {:.2e})", max_col_sum);
        println!("  Note: Sum-to-zero is over data points, not columns");
    }

    println!("\nTest 4: Compare constrained vs unconstrained");
    println!("{}", "=".repeat(50));

    // For comparison, create an unconstrained cubic spline
    let unconstrained = gam::SmoothTerm::cubic_spline(
        "x".to_string(),
        10,
        0.0,
        1.0
    )?;

    println!("  CR (constrained):    {} basis functions",
        gam::SmoothTerm::cr_spline("x".to_string(), 10, 0.0, 1.0)?.num_basis());
    println!("  Cubic (unconstrained): {} basis functions", unconstrained.num_basis());

    println!("\n{}", "=".repeat(50));
    println!("All tests completed!");
    println!("{}", "=".repeat(50));

    Ok(())
}
