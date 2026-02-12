/// End-to-end GAM benchmark exercising the full pipeline:
/// basis evaluation -> discretized design -> PiRLS -> REML -> final fit
///
/// This tests the actual performance a user would see via the Python bindings.
use mgcv_rust::basis::CubicRegressionSpline;
use mgcv_rust::gam::{SmoothTerm, GAM};
use mgcv_rust::pirls::Family;
use mgcv_rust::smooth::OptimizationMethod;
use ndarray::{Array1, Array2};
use rand::{
    distributions::{Distribution, Uniform},
    SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;
use std::time::Instant;

fn run_gam_benchmark(n: usize, d: usize, k: usize, algorithm: &str) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let uniform = Uniform::new(0.0, 1.0);

    // Generate covariates
    let mut x_data = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            x_data[[i, j]] = uniform.sample(&mut rng);
        }
    }

    // Generate response
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let signal: f64 = (0..d).map(|j| (2.0 * PI * x_data[[i, j]]).sin()).sum();
            let u1: f64 = uniform.sample(&mut rng);
            let u2: f64 = uniform.sample(&mut rng);
            let noise = 0.3 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            signal + noise
        })
        .collect();

    // Build GAM model
    let mut gam = GAM::new(Family::Gaussian);
    for j in 0..d {
        let x_col = x_data.column(j).to_owned();
        let x_min = x_col.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let smooth = SmoothTerm::cr_spline_quantile(format!("x{}", j), k, &x_col).unwrap();
        gam.add_smooth(smooth);
    }

    // Fit
    let algo = match algorithm {
        "newton" => Some(mgcv_rust::smooth::REMLAlgorithm::Newton),
        "fs" => Some(mgcv_rust::smooth::REMLAlgorithm::FellnerSchall),
        _ => None,
    };

    let start = Instant::now();
    gam.fit_optimized_full(
        &x_data,
        &y,
        OptimizationMethod::REML,
        20,  // max_outer_iter (for FS)
        100, // max_inner_iter (PiRLS)
        0.05,
        mgcv_rust::reml::ScaleParameterMethod::EDF,
        algo,
    )
    .unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    elapsed
}

#[cfg(feature = "blas")]
fn main() {
    println!("\n=== End-to-End GAM Benchmark (with discretized design) ===\n");
    println!("Tests full pipeline: basis eval -> discretize -> PiRLS -> REML -> final fit");
    println!("Seed=42, Gaussian family, CR splines\n");

    let configs = vec![
        (500, 1, 10),
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

    // Warm up BLAS
    let _ = run_gam_benchmark(100, 1, 5, "newton");

    println!("┌─────────┬─────┬─────┬────────┬──────────────┬──────────────┐");
    println!("│    n    │  d  │  k  │  p=d×k │ Newton (ms)  │  FS (ms)     │");
    println!("├─────────┼─────┼─────┼────────┼──────────────┼──────────────┤");

    for &(n, d, k) in &configs {
        let p = d * k;
        print!("│ {:7} │ {:3} │ {:3} │ {:6} │", n, d, k, p);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Run Newton (best of 3)
        let newton_times: Vec<f64> = (0..3)
            .map(|_| run_gam_benchmark(n, d, k, "newton"))
            .collect();
        let newton_best = newton_times.iter().cloned().fold(f64::INFINITY, f64::min);

        // Run FS (best of 3)
        let fs_times: Vec<f64> = (0..3).map(|_| run_gam_benchmark(n, d, k, "fs")).collect();
        let fs_best = fs_times.iter().cloned().fold(f64::INFINITY, f64::min);

        println!(" {:12.1} │ {:12.1} │", newton_best, fs_best);
    }

    println!("└─────────┴─────┴─────┴────────┴──────────────┴──────────────┘");
    println!();
    println!("R bam() reference (from OPTIMIZATION_PLAN.md):");
    println!("  n=1000,d=2: 17ms | n=2000,d=4: 45ms | n=5000,d=8: 134ms | n=10000,d=4: 86ms");
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("This benchmark requires the 'blas' feature");
}
