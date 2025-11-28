/// Comprehensive Rust Newton benchmark matching R's setup
/// Uses cubic regression splines like the R benchmark

use mgcv_rust::smooth::{SmoothingParameter, OptimizationMethod};
use mgcv_rust::basis::{CubicRegressionSpline, BasisFunction};
use mgcv_rust::penalty::compute_penalty;
use ndarray::{Array1, Array2};
use rand::{SeedableRng, distributions::{Distribution, Uniform}};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;
use std::time::Instant;

fn run_benchmark(n: usize, d: usize, k: usize) -> (f64, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let uniform = Uniform::new(0.0, 1.0);

    // Generate data
    let mut x_vecs = Vec::new();
    for _ in 0..d {
        let x: Vec<f64> = (0..n).map(|_| uniform.sample(&mut rng)).collect();
        x_vecs.push(Array1::from(x));
    }

    let y: Array1<f64> = {
        let mut y_vec = Vec::new();
        for i in 0..n {
            let signal: f64 = x_vecs.iter().map(|x| (2.0 * PI * x[i]).sin()).sum();
            let u1 = uniform.sample(&mut rng);
            let u2 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            y_vec.push(signal + noise);
        }
        Array1::from(y_vec)
    };

    // Construct design matrices and penalties
    let mut design_matrices = Vec::new();
    let mut penalties_vec = Vec::new();

    for x in &x_vecs {
        let basis = CubicRegressionSpline::with_quantile_knots(x, k);
        let design = basis.evaluate(x).unwrap();
        let knots = basis.knots().unwrap();
        let mut penalty = compute_penalty("cr", k, Some(knots), 1).unwrap();

        // Apply penalty normalization
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
        penalties_vec.push(penalty);
    }

    // Assemble full design matrix
    let total_basis = k * d;
    let mut full_design = Array2::zeros((n, total_basis));
    for (i, mat) in design_matrices.iter().enumerate() {
        full_design.slice_mut(ndarray::s![.., (i*k)..((i+1)*k)]).assign(mat);
    }

    // Assemble block-diagonal penalty
    let mut full_penalty_list = Vec::new();
    for penalty in penalties_vec {
        let mut block_penalty = Array2::zeros((total_basis, total_basis));
        let start_idx = full_penalty_list.len() * k;
        for i in 0..k {
            for j in 0..k {
                block_penalty[[start_idx + i, start_idx + j]] = penalty[[i, j]];
            }
        }
        full_penalty_list.push(block_penalty);
    }

    // Weights (all ones)
    let w = Array1::ones(n);

    // Run optimization
    let mut smoother = SmoothingParameter::new(d, OptimizationMethod::REML);

    let start = Instant::now();
    smoother.optimize(&y, &full_design, &w, &full_penalty_list, 30, 0.05).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    (elapsed, smoother.lambda.clone())
}

fn main() {
    println!("\n=== Comprehensive Rust Newton Benchmark ===\n");
    println!("Testing Newton REML across various problem sizes");
    println!("(Matching R benchmark setup: CR splines, REML)\n");

    // Test configurations (matching R benchmark)
    let configs = vec![
        (1000, 1, 10),
        (1000, 2, 10),
        (1000, 4, 10),

        (2000, 1, 10),
        (2000, 2, 10),
        (2000, 4, 10),
        (2000, 8, 8),

        (5000, 1, 10),
        (5000, 2, 10),
        (5000, 4, 8),
        (5000, 8, 8),

        (10000, 1, 10),
        (10000, 2, 10),
        (10000, 4, 8),
    ];

    println!("┌─────────┬─────┬─────┬────────┬──────────────┬─────────────┐");
    println!("│    n    │  d  │  k  │  p=d×k │ Rust (ms)    │  λ (mean)   │");
    println!("├─────────┼─────┼─────┼────────┼──────────────┼─────────────┤");

    for (n, d, k) in configs {
        let p = d * k;
        print!("│ {:7} │ {:3} │ {:3} │ {:6} │", n, d, k, p);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (time_ms, lambdas) = run_benchmark(n, d, k);
        let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;

        println!(" {:12.1} │ {:11.4} │", time_ms, lambda_mean);
    }

    println!("└─────────┴─────┴─────┴────────┴──────────────┴─────────────┘");
    println!("\nNotes:");
    println!("- All optimizations enabled: zero-step, X'WX caching, REML convergence, Cholesky");
    println!("- Cubic regression splines matching R benchmark");
    println!("- Single-run measurements (not averaged)");
}
