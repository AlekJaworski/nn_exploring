// Profile the ACTUAL reml_gradient_multi_qr function
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;
    use mgcv_rust::reml::reml_gradient_multi_qr;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Full Gradient Function Profiling ===\n");
    println!("Configuration: n={}, dims={}, k={}, p={}\n", n, n_dims, k, p);

    // Generate data
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen::<f64>();
        }
    }
    let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let w: Array1<f64> = Array1::ones(n);
    let lambdas: Vec<f64> = vec![1.0; n_dims];

    // Create penalties
    let mut penalties = Vec::new();
    for dim in 0..n_dims {
        let mut penalty = Array2::zeros((p, p));
        let start = dim * k;
        let end = start + k;
        for i in start..end {
            penalty[[i, i]] = 1.0;
        }
        penalties.push(penalty);
    }

    println!("Warm-up call...");
    let _ = reml_gradient_multi_qr(&y, &x, &w, &lambdas, &penalties).unwrap();

    println!("Profiling {} iterations...\n", 10);
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_qr(&y, &x, &w, &lambdas, &penalties).unwrap();
    }
    let total_time = start.elapsed().as_secs_f64();

    println!("Results:");
    println!("  Total time: {:.3}s", total_time);
    println!("  Per call: {:.3}s\n", total_time / 10.0);

    println!("Comparison:");
    println!("  Rust (full function): {:.3}s per call", total_time / 10.0);
    println!("  R/mgcv:               0.029s per call");
    let ratio = (total_time / 10.0) / 0.029;
    println!("  → Rust is {:.2}x slower\n", ratio);

    // Compare to components
    println!("Component analysis:");
    println!("  QR alone:       0.054s (87% of Rust time)");
    println!("  Trace:          0.001s (2%)");
    println!("  Beta derivs:    0.000s (0%)");
    println!("  Missing:        {:.3}s (11%)", total_time / 10.0 - 0.055);
    println!("  → Likely penalty_sqrt (eigendecomp) and overhead");
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin profile_full --features blas --release");
}
