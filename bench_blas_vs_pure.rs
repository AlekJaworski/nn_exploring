// Micro-benchmark to test BLAS vs pure Rust solve() performance
use std::time::Instant;
use ndarray::{Array1, Array2};

#[path = "src/linalg.rs"]
mod linalg;
use linalg::solve;

fn main() {
    let sizes = vec![50, 100, 200, 400];

    println!("BLAS vs Pure Rust solve() Performance");
    println!("=====================================\n");

    for &n in &sizes {
        // Generate random SPD matrix
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let val: f64 = rand::random::<f64>() - 0.5;
                a[[i, j]] = val;
                a[[j, i]] = val;
            }
            a[[i, i]] += n as f64;  // Make diagonally dominant
        }

        let b = Array1::<f64>::from_vec((0..n).map(|_| rand::random::<f64>()).collect());

        // Warmup
        for _ in 0..5 {
            let _ = solve(a.clone(), b.clone());
        }

        // Benchmark
        let iterations = if n <= 100 { 100 } else { 20 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = solve(a.clone(), b.clone()).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        println!("n={:4}: {:8.3} ms/solve", n, avg_ms);
    }
}

mod rand {
    static mut SEED: u64 = 123456789;

    pub fn random<T>() -> T
    where
        T: From<f64>
    {
        unsafe {
            SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
            let val = ((SEED >> 16) & 0x7FFF) as f64 / 32768.0;
            T::from(val)
        }
    }
}
