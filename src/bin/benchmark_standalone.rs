// Standalone benchmark binary - no cargo overhead
use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use mgcv_rust::{
    basis::{BasisFunction, CubicRegressionSpline},
    penalty::compute_penalty,
    smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm},
    pirls::{fit_pirls, Family},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <data_file> [algorithm]", args[0]);
        eprintln!("  algorithm: fellner-schall (default) | newton");
        std::process::exit(1);
    }

    let filename = &args[1];
    let algorithm_str = if args.len() > 2 { &args[2] } else { "fellner-schall" };

    // Read data
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines.next().unwrap()?;
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let n: usize = parts[0].parse()?;
    let d: usize = parts[1].parse()?;
    let k: usize = parts[2].parse()?;

    eprintln!("Data: n={}, d={}, k={}", n, d, k);
    eprintln!("Algorithm: {}", algorithm_str);

    let mut x_vec = Vec::new();
    for _ in 0..n {
        let line = lines.next().unwrap()?;
        for val in line.split_whitespace() {
            x_vec.push(val.parse::<f64>()?);
        }
    }
    let x = Array2::from_shape_vec((n, d), x_vec)?;

    let mut y_vec = Vec::new();
    for _ in 0..n {
        let line = lines.next().unwrap()?;
        y_vec.push(line.parse::<f64>()?);
    }
    let y = Array1::from_vec(y_vec);

    // Set up algorithm
    let algorithm = if algorithm_str == "newton" {
        REMLAlgorithm::Newton
    } else {
        REMLAlgorithm::FellnerSchall
    };

    let mut sp = SmoothingParameter::new_with_algorithm(
        d,
        OptimizationMethod::REML,
        algorithm
    );

    // Build design matrix and penalties
    let mut design_matrices = Vec::new();
    let mut individual_penalties = Vec::new();

    for i in 0..d {
        let x_col = x.column(i).to_owned();
        let basis = CubicRegressionSpline::with_quantile_knots(&x_col, k);
        let design = basis.evaluate(&x_col)?;

        let knots = basis.knots().unwrap();
        let mut penalty = compute_penalty("cr", k, Some(knots), 1)?;

        // Apply mgcv's penalty normalization: S_rescaled = S * ||X||_inf^2 / ||S||_inf
        // This makes penalties scale-invariant
        let inf_norm_X = design.rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let maXX = inf_norm_X * inf_norm_X;

        let inf_norm_S = (0..k)
            .map(|i| (0..k).map(|j| penalty[[i, j]].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);

        let scale_factor = if inf_norm_S > 1e-10 {
            maXX / inf_norm_S
        } else {
            1.0
        };

        penalty *= scale_factor;

        design_matrices.push(design);
        individual_penalties.push(penalty);
    }

    // Combine design matrices
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    for (i, mat) in design_matrices.iter().enumerate() {
        full_design.slice_mut(ndarray::s![.., (i*k)..((i+1)*k)]).assign(mat);
    }

    // Create block diagonal penalty matrices (one for each smooth)
    let mut penalties = Vec::new();
    for (i, individual_penalty) in individual_penalties.iter().enumerate() {
        // Debug: print penalty stats
        if std::env::var("MGCV_PROFILE").is_ok() {
            let trace: f64 = (0..individual_penalty.nrows()).map(|j| individual_penalty[[j,j]]).sum();
            let max_elem = individual_penalty.iter().map(|x| x.abs()).fold(0.0, f64::max);
            eprintln!("[PROFILE] Individual penalty {}: dim={}×{}, trace={:.6}, max={:.6}",
                     i, individual_penalty.nrows(), individual_penalty.ncols(), trace, max_elem);
        }

        let mut block_penalty = Array2::zeros((total_basis, total_basis));
        block_penalty.slice_mut(ndarray::s![i*k..(i+1)*k, i*k..(i+1)*k]).assign(individual_penalty);
        penalties.push(block_penalty);
    }

    let w = Array1::ones(n);

    // TIMED SECTION - Optimize smoothing parameters
    let start = Instant::now();
    sp.optimize(&y, &full_design, &w, &penalties, 30, 1e-6)?;
    let elapsed = start.elapsed();

    eprintln!("\nOptimization time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    // Fit final model
    let result = fit_pirls(
        &y,
        &full_design,
        &sp.lambda,
        &penalties,
        Family::Gaussian,
        30,
        1e-6,
    )?;

    // Output results in format expected by R benchmark script
    for (i, lambda) in sp.lambda.iter().enumerate() {
        println!("lambda[{}]={}", i, lambda);
    }

    for (i, val) in result.fitted_values.iter().enumerate() {
        println!("fitted[{}]={}", i, val);
    }

    Ok(())
}
