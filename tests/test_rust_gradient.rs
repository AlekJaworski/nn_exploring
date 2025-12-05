///Test the corrected Rust gradient implementation
use ndarray::{Array1, Array2};
use mgcv_rust::reml::reml_gradient_multi_qr;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("TESTING RUST GRADIENT IMPLEMENTATION");
    println!("{}", "=".repeat(80));
    println!();

    // Load data from CSV files
    let df_text = fs::read_to_string("/tmp/fresh_data.csv")?;
    let x_text = fs::read_to_string("/tmp/fresh_X.csv")?;

    // Parse y values
    let y: Vec<f64> = df_text
        .lines()
        .skip(1)  // Skip header
        .filter_map(|line| {
            line.split(',').nth(1)?.parse().ok()
        })
        .collect();

    let n = y.len();
    let y_array = Array1::from_vec(y);

    // Parse X matrix
    let x_lines: Vec<&str> = x_text.lines().skip(1).collect();
    let ncol = x_lines[0].split(',').count();
    let mut x_vec = Vec::new();
    for line in x_lines {
        let row: Vec<f64> = line.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        x_vec.extend(row);
    }
    let x_array = Array2::from_shape_vec((n, ncol), x_vec)?;

    // Load penalty matrices
    let mut penalties = Vec::new();
    for i in 1..=5 {
        let s_text = fs::read_to_string(format!("/tmp/fresh_S{}.csv", i))?;
        let s_lines: Vec<&str> = s_text.lines().skip(1).collect();
        let s_ncol = s_lines[0].split(',').count();
        let mut s_vec = Vec::new();
        for line in s_lines {
            let row: Vec<f64> = line.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            s_vec.extend(row);
        }
        let s_array = Array2::from_shape_vec((s_ncol, s_ncol), s_vec)?;
        penalties.push(s_array);
    }

    // mgcv's solution
    let mgcv_lambda = vec![0.2705535, 9038.71, 150.8265, 400.144, 13747035.0];

    // Weights (all 1.0)
    let w = Array1::from_elem(n, 1.0);

    println!("Computing gradient at mgcv's solution...");
    println!("λ = {:?}", mgcv_lambda);
    println!();

    // Compute gradient
    let gradient = reml_gradient_multi_qr(&y_array, &x_array, &w, &mgcv_lambda, &penalties)?;

    println!("Rust gradient:");
    println!("  {:?}", gradient.as_slice().unwrap());
    println!();

    println!("Expected (from Python numerical):");
    println!("  [-0.07470567, -0.01838225, -0.04935873, -0.04883168, -0.0004793]");
    println!();

    // Compare
    let expected = Array1::from_vec(vec![-0.07470567, -0.01838225, -0.04935873, -0.04883168, -0.0004793]);
    let diff = &gradient - &expected;
    let rel_error = diff.iter().map(|x| x*x).sum::<f64>().sqrt()
                  / expected.iter().map(|x| x*x).sum::<f64>().sqrt();

    println!("Difference:");
    println!("  {:?}", diff.as_slice().unwrap());
    println!("Relative error: {:.6e}", rel_error);

    if rel_error < 0.01 {
        println!();
        println!("✓ SUCCESS: Gradient matches to < 1% error!");
    } else {
        println!();
        println!("✗ FAILURE: Gradient error too large");
    }

    Ok(())
}
