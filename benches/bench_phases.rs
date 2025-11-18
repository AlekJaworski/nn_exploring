// Benchmark different phases of GAM fitting to find hotspots
use mgcv_rust::gam::GAM;
use mgcv_rust::basis::CubicRegressionSpline;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn generate_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 4));
    let mut y = Array1::<f64>::zeros(n);

    // Simple random number generator
    let mut seed = 12345u64;
    for i in 0..n {
        for j in 0..4 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            x[[i, j]] = ((seed >> 16) & 0xFFFF) as f64 / 65536.0 * 4.0 - 2.0;
        }
    }

    // y = sin(x0) + 0.5*x1^2 + cos(x2) + 0.3*x3
    for i in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((seed >> 16) & 0xFFFF) as f64 / 65536.0 * 0.2 - 0.1;

        y[i] = x[[i, 0]].sin()
            + 0.5 * x[[i, 1]] * x[[i, 1]]
            + x[[i, 2]].cos()
            + 0.3 * x[[i, 3]]
            + noise;
    }

    (x, y)
}

fn main() {
    let sizes = vec![500, 1500, 2500];
    let k = 16;

    println!("GAM Fitting Phase Benchmark");
    println!("===========================\n");

    for &n in &sizes {
        println!("n={}", n);
        println!("-" . repeat(40));

        let (x, y) = generate_data(n);

        // Create GAM
        let mut gam = GAM::new();
        for i in 0..4 {
            let col = x.column(i).to_owned();
            let spline = CubicRegressionSpline::new(&col, k).unwrap();
            gam.add_smooth(spline);
        }

        // Warmup
        for _ in 0..2 {
            let _ = gam.fit_auto_optimized(&x, &y, 50, 10, 1e-6);
        }

        // Timed run with breakdown
        let total_start = Instant::now();

        // This is a black box from here, but we can time the total
        let fit_start = Instant::now();
        let result = gam.fit_auto_optimized(&x, &y, 50, 10, 1e-6);
        let fit_time = fit_start.elapsed();

        let total_time = total_start.elapsed();

        match result {
            Ok(_) => {
                println!("  Total fit time:     {:7.2} ms", fit_time.as_secs_f64() * 1000.0);
                println!("  Total elapsed:      {:7.2} ms", total_time.as_secs_f64() * 1000.0);
            },
            Err(e) => {
                println!("  Error: {:?}", e);
            }
        }

        println!();
    }

    println!("\nNote: To get detailed phase timing, we need to instrument the Rust code.");
    println!("Main phases:");
    println!("  1. Basis evaluation (compute design matrix)");
    println!("  2. Penalty matrix computation");
    println!("  3. REML optimization (smoothing parameter estimation)");
    println!("  4. PiRLS iterations (coefficient estimation)");
}

fn repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}
