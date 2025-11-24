// Profile gradient computation to identify bottlenecks
#[cfg(feature = "blas")]
fn main() {
    use mgcv_rust::reml::reml_gradient_multi_qr;
    use ndarray::{Array1, Array2};
    use rand::{SeedableRng, Rng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Test case: n=6000, 8D similar to benchmark
    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("Profiling gradient computation:");
    println!("  n = {}, dims = {}, k = {}, p = {}\n", n, n_dims, k, p);

    // Generate data
    println!("[1/4] Generating data...");
    let start = Instant::now();
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen::<f64>();
        }
    }
    let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let w = Array1::ones(n);
    println!("  Generated in {:.3}s\n", start.elapsed().as_secs_f64());

    // Create penalties
    println!("[2/4] Creating penalties...");
    let start = Instant::now();
    let mut penalties = Vec::new();
    for dim in 0..n_dims {
        let mut penalty = Array2::zeros((p, p));
        let start_col = dim * k;
        let end_col = start_col + k;

        for i in start_col..end_col {
            for j in start_col..end_col {
                if i == j {
                    penalty[[i, j]] = 2.0;
                } else if (i as i32 - j as i32).abs() == 1 {
                    penalty[[i, j]] = -1.0;
                }
            }
        }
        penalties.push(penalty);
    }
    println!("  Created in {:.3}s\n", start.elapsed().as_secs_f64());

    let lambdas: Vec<f64> = (0..n_dims).map(|i| 1.0 + (i as f64) * 10.0).collect();

    // Warm-up call
    println!("[3/4] Warm-up call...");
    let _ = reml_gradient_multi_qr(&y, &x, &w, &lambdas, &penalties);

    // Profile multiple iterations
    println!("[4/4] Profiling {} gradient calls...\n", 10);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_qr(&y, &x, &w, &lambdas, &penalties);
    }
    let total_time = start.elapsed().as_secs_f64();

    println!("Results:");
    println!("  Total time: {:.3}s", total_time);
    println!("  Per call: {:.3}s", total_time / 10.0);
    println!("  Per iteration (5 calls): {:.3}s", total_time / 2.0);

    // Estimate breakdown (based on known operations)
    let per_call = total_time / 10.0;
    println!("\nEstimated breakdown (per gradient call):");
    println!("  QR decomposition: ~{:.0}%", 30.0);
    println!("  Trace solves (64 calls): ~{:.0}%", 40.0);
    println!("  Beta derivatives (8 calls): ~{:.0}%", 20.0);
    println!("  Other computations: ~{:.0}%", 10.0);

    println!("\nBottleneck candidates:");
    println!("  1. Trace computation: {} solve() calls", n_dims * k);
    println!("     → Batch to {} calls with solve_triangular()", n_dims);
    println!("  2. Beta derivatives: {} solve() calls (no caching)", n_dims);
    println!("     → Cache Cholesky factorization, reuse {} times", n_dims);
    println!("  3. QR decomposition: Large matrix {}×{}", n + n_dims * k, p);
    println!("     → Consider blockwise approach for n > 2000");
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin profile_gradient --features blas --release");
}
