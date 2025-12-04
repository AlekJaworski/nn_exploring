//! Comprehensive Rust Newton performance benchmark
//! Matches configurations from benchmark_performance.R

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod, REMLAlgorithm};
use mgcv_rust::basis::{BasisFunction, CubicRegressionSpline};
use mgcv_rust::penalty::compute_penalty;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::distributions::{Distribution, Uniform};
use std::f64::consts::PI;
use std::time::Instant;

fn generate_data(n: usize, d: usize) -> (Vec<Array1<f64>>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let uniform = Uniform::new(0.0, 1.0);

    let mut x_vecs = Vec::new();
    for _ in 0..d {
        let x: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
        x_vecs.push(Array1::from(x));
    }

    let y: Array1<f64> = if d == 1 {
        x_vecs[0].mapv(|xi| {
            (2.0 * PI * xi).sin() + {
                let u1 = uniform.sample(&mut rng);
                let u2 = uniform.sample(&mut rng);
                0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            }
        })
    } else if d == 2 {
        let mut y_vec = Vec::new();
        for i in 0..n {
            let signal = (2.0 * PI * x_vecs[0][i]).sin() + (2.0 * PI * x_vecs[1][i]).cos();
            let u1 = uniform.sample(&mut rng);
            let u2 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            y_vec.push(signal + noise);
        }
        Array1::from(y_vec)
    } else {
        let mut y_vec = Vec::new();
        for i in 0..n {
            let signal = (2.0 * PI * x_vecs[0][i]).sin()
                       + (2.0 * PI * x_vecs[1][i]).cos()
                       + (4.0 * PI * x_vecs[2][i]).sin();
            let u1 = uniform.sample(&mut rng);
            let u2 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            y_vec.push(signal + noise);
        }
        Array1::from(y_vec)
    };

    (x_vecs, y)
}

fn benchmark_config(n: usize, d: usize, k: usize) -> Result<(usize, f64, f64), Box<dyn std::error::Error>> {
    // Generate data
    let (x_vecs, y) = generate_data(n, d);

    // Build design matrix and penalties
    let mut design_matrices = Vec::new();
    let mut penalties = Vec::new();

    for x in &x_vecs {
        let basis = CubicRegressionSpline::with_quantile_knots(x, k);
        let design = basis.evaluate(x)?;
        let knots = basis.knots().unwrap();
        let mut penalty = compute_penalty("cr", k, Some(knots), 1)?;

        // Apply penalty normalization (like mgcv)
        let inf_norm_X = design.rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let maXX = inf_norm_X * inf_norm_X;

        let inf_norm_S = (0..k)
            .map(|i| (0..k).map(|j| penalty[[i, j]].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);

        if inf_norm_S > 1e-10 {
            penalty *= maXX / inf_norm_S;
        }

        design_matrices.push(design);
        penalties.push(penalty);
    }

    // Combine design matrices
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    for (i, mat) in design_matrices.iter().enumerate() {
        full_design.slice_mut(ndarray::s![.., (i*k)..((i+1)*k)]).assign(mat);
    }

    // Create block diagonal penalties
    let mut block_penalties = Vec::new();
    for (i, pen) in penalties.iter().enumerate() {
        let mut block = Array2::zeros((total_basis, total_basis));
        block.slice_mut(ndarray::s![i*k..(i+1)*k, i*k..(i+1)*k]).assign(pen);
        block_penalties.push(block);
    }

    // Count iterations by enabling profiling temporarily
    std::env::set_var("MGCV_PROFILE", "1");

    // Benchmark Newton optimization
    let mut sp = SmoothingParameter::new_with_algorithm(
        d,
        OptimizationMethod::REML,
        REMLAlgorithm::Newton
    );

    let weights = Array1::ones(n);
    let start = Instant::now();
    sp.optimize(&y, &full_design, &weights, &block_penalties, 30, 1e-6)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    std::env::remove_var("MGCV_PROFILE");

    let lambda_mean = sp.lambda.iter().sum::<f64>() / sp.lambda.len() as f64;

    // Parse iteration count from stderr (hacky but works)
    // For now, we'll return 0 and manually count
    Ok((0, lambda_mean, elapsed_ms))
}

fn main() {
    println!("=== Rust Newton Performance Benchmark ===\n");
    println!("{:<8} {:<6} {:<6} {:<20} {:<10} {:<15} {:<10}",
             "n", "d", "k", "Method", "Iterations", "Lambda", "Time(ms)");
    println!("{}", "-".repeat(85));

    let configs = vec![
        (100, 1, 10),
        (500, 1, 20),
        (1000, 1, 20),
        (2000, 1, 30),
        (500, 2, 15),
        (1000, 2, 15),
        (500, 3, 12),
        (5000, 1, 30),
        (10000, 1, 30),
    ];

    for (n, d, k) in configs {
        match benchmark_config(n, d, k) {
            Ok((iters, lambda, time_ms)) => {
                println!("{:<8} {:<6} {:<6} {:<20} {:<10} {:<15.4} {:<10.1}",
                         n, d, k, "Rust Newton", "?", lambda, time_ms);
            }
            Err(e) => {
                println!("{:<8} {:<6} {:<6} {:<20} {:<10} {:<15} {:<10}",
                         n, d, k, "Rust Newton", "FAILED", "-", "-");
                eprintln!("Error: {}", e);
            }
        }
    }
}
