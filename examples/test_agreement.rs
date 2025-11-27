use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::env;

use mgcv_rust::{
    basis::{BasisFunction, CubicRegressionSpline},
    penalty::compute_penalty,
    smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm},
    pirls::{fit_pirls, Family, PiRLSResult},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <data_file> <algorithm>", args[0]);
        eprintln!("  algorithm: 'newton' or 'fellner-schall'");
        std::process::exit(1);
    }

    let data_file = &args[1];
    let algorithm_str = &args[2];

    // Read data
    let file = File::open(data_file)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let first_line = lines.next().unwrap()?;
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let n: usize = parts[0].parse()?;
    let d: usize = parts[1].parse()?;
    let k: usize = parts[2].parse()?;

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

    // Set up smoothing parameters
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
        design_matrices.push(design);

        let knots = basis.knots().unwrap();
        let penalty = compute_penalty("cr", k, Some(knots), 1)?;
        individual_penalties.push(penalty);
    }

    // Combine design matrices
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    for (i, design) in design_matrices.iter().enumerate() {
        full_design.slice_mut(ndarray::s![.., i*k..(i+1)*k]).assign(design);
    }

    // Create block diagonal penalty matrices (one for each smooth)
    // Each penalty matrix is total_basis x total_basis with the individual penalty in the appropriate block
    let mut penalties = Vec::new();
    for (i, individual_penalty) in individual_penalties.iter().enumerate() {
        let mut block_penalty = Array2::zeros((total_basis, total_basis));
        block_penalty.slice_mut(ndarray::s![i*k..(i+1)*k, i*k..(i+1)*k]).assign(individual_penalty);
        penalties.push(block_penalty);
    }

    // Optimize smoothing parameters
    let weights = Array1::ones(n);
    sp.optimize(&y, &full_design, &weights, &penalties, 30, 1e-6)?;

    // Final fit with optimized parameters
    let result = fit_pirls(
        &y,
        &full_design,
        &sp.lambda,
        &penalties,
        Family::Gaussian,
        20,
        1e-6
    )?;

    // Output results
    println!("SMOOTHING_PARAMETERS");
    for lambda in &sp.lambda {
        println!("{}", lambda);
    }
    println!("END_SMOOTHING_PARAMETERS");

    println!("COEFFICIENTS");
    for coef in result.coefficients.iter() {
        println!("{}", coef);
    }
    println!("END_COEFFICIENTS");

    println!("FITTED_VALUES");
    let fitted = full_design.dot(&result.coefficients);
    for val in fitted.iter() {
        println!("{}", val);
    }
    println!("END_FITTED_VALUES");

    Ok(())
}
