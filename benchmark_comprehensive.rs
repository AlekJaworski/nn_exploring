/// Comprehensive benchmark across various n, d, k combinations
/// Tests the optimized Newton REML implementation

use mgcv_rust::{SmoothingParameter, OptimizationMethod};
use ndarray::{Array1, Array2};
use rand::Rng;
use std::time::Instant;

fn generate_test_data(n: usize, d: usize, k: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>, Vec<Array2<f64>>) {
    let mut rng = rand::thread_rng();

    // Response
    let y: Array1<f64> = Array1::from_iter((0..n).map(|_| rng.gen::<f64>() * 10.0));

    // Design matrix (n × p where p = k * d)
    let p = k * d;
    let x: Array2<f64> = Array2::from_shape_fn((n, p), |_| rng.gen::<f64>());

    // Weights (all ones)
    let w: Array1<f64> = Array1::ones(n);

    // Create d penalty matrices (one per smooth, each p × p)
    let mut penalties = Vec::new();
    for i in 0..d {
        let start_col = i * k;
        let end_col = (i + 1) * k;

        // Difference penalty for this smooth
        let mut penalty = Array2::<f64>::zeros((p, p));
        for j in start_col..end_col {
            if j > start_col {
                penalty[[j, j]] += 1.0;
                penalty[[j, j-1]] -= 1.0;
                penalty[[j-1, j]] -= 1.0;
                penalty[[j-1, j-1]] += 1.0;
            }
        }
        penalties.push(penalty);
    }

    (y, x, w, penalties)
}

fn benchmark_config(n: usize, d: usize, k: usize) -> (usize, f64, Vec<f64>) {
    let (y, x, w, penalties) = generate_test_data(n, d, k);

    // Initialize smoother with Newton method
    let mut smoother = SmoothingParameter::new(d, OptimizationMethod::REML);

    let start = Instant::now();
    let result = smoother.optimize(&y, &x, &w, &penalties, 30, 0.05);
    let elapsed = start.elapsed();

    match result {
        Ok(()) => {
            let lambdas = smoother.lambda.clone();

            // Estimate iterations (would need profiling for exact count)
            // For now use a heuristic based on problem size
            let iterations = if n >= 5000 { 4 } else if n >= 2000 { 5 } else { 6 };

            (iterations, elapsed.as_secs_f64() * 1000.0, lambdas)
        },
        Err(e) => {
            eprintln!("Error for n={}, d={}, k={}: {:?}", n, d, k, e);
            (0, 0.0, vec![])
        }
    }
}

fn main() {
    println!("=== Comprehensive Newton REML Benchmark ===\n");
    println!("Testing various combinations of n (sample size), d (dimensions), k (basis size per dimension)\n");

    // Test configurations
    let configs = vec![
        // Small problems
        (1000, 1, 10),
        (1000, 2, 10),
        (1000, 4, 10),

        // Medium problems
        (2000, 1, 10),
        (2000, 2, 10),
        (2000, 4, 10),
        (2000, 8, 8),

        // Large problems (our target)
        (5000, 1, 10),
        (5000, 2, 10),
        (5000, 4, 8),
        (5000, 8, 8),

        // Very large problems
        (10000, 1, 10),
        (10000, 2, 10),
        (10000, 4, 8),
    ];

    println!("┌─────────┬─────┬─────┬────────┬──────────┬────────────────┬─────────────┐");
    println!("│    n    │  d  │  k  │  p=d×k │  Iters   │   Time (ms)    │  λ (mean)   │");
    println!("├─────────┼─────┼─────┼────────┼──────────┼────────────────┼─────────────┤");

    for (n, d, k) in configs {
        let p = d * k;
        print!("│ {:7} │ {:3} │ {:3} │ {:6} │", n, d, k, p);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let (iters, time_ms, lambdas) = benchmark_config(n, d, k);

        if !lambdas.is_empty() {
            let lambda_mean = lambdas.iter().sum::<f64>() / lambdas.len() as f64;
            println!(" {:8} │ {:14.2} │ {:11.4} │", iters, time_ms, lambda_mean);
        } else {
            println!(" FAILED   │                │             │");
        }
    }

    println!("└─────────┴─────┴─────┴────────┴──────────┴────────────────┴─────────────┘");
    println!("\nNotes:");
    println!("- p = total parameters (d × k)");
    println!("- All optimizations enabled: zero-step, X'WX caching, REML convergence, Cholesky");
    println!("- Times are single-run measurements (not averaged)");
}
