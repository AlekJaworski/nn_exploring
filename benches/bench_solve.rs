// Benchmark solve() performance
use mgcv_rust::linalg::solve;
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    let sizes = vec![50, 100, 200, 400, 800, 1600];
    let iterations = 20;

    #[cfg(feature = "blas")]
    println!("=== WITH BLAS ===\n");

    #[cfg(not(feature = "blas"))]
    println!("=== WITHOUT BLAS ===\n");

    for &n in &sizes {
        // Generate random SPD matrix
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let val = (i * 7 + j * 13) as f64 / 100.0;
                a[[i, j]] = val;
                a[[j, i]] = val;
            }
            a[[i, i]] += n as f64;
        }

        let b = Array1::<f64>::from_vec((0..n).map(|i| i as f64 / 10.0).collect());

        // Warmup
        for _ in 0..5 {
            let _ = solve(a.clone(), b.clone());
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = solve(a.clone(), b.clone()).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("n={:4}: {:8.3} ms/solve", n, avg_ms);
    }
}
