// Benchmark individual components
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;
    use mgcv_rust::reml::reml_gradient_multi_cholesky_cached;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Component Benchmarking ===\n");

    // Setup
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen::<f64>();
        }
    }
    let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let w: Array1<f64> = Array1::ones(n);
    let lambdas: Vec<f64> = vec![1.0; n_dims];

    let mut penalties = Vec::new();
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    
    for dim in 0..n_dims {
        let mut penalty = Array2::zeros((p, p));
        let start_idx = dim * k;
        let end_idx = start_idx + k;
        for i in start_idx..end_idx {
            penalty[[i, i]] = 1.0;
        }
        
        // Dummy sqrt (identity blocks)
        let mut sqrt_pen = Array2::zeros((p, k));
        for i in 0..k {
            sqrt_pen[[start_idx + i, i]] = 1.0;
        }
        
        penalties.push(penalty);
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(k);
    }

    println!("Profiling gradient call (10 iterations)...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_cholesky_cached(
            &y, &x, &w, &lambdas, &penalties, &sqrt_penalties, &penalty_ranks
        ).unwrap();
    }
    let total = start.elapsed().as_secs_f64();
    
    println!("  Total: {:.3}s", total);
    println!("  Per call: {:.3}s\n", total / 10.0);

    println!("Comparison:");
    println!("  R/mgcv:         0.029s per call");
    println!("  Rust (cached):  {:.3}s per call", total / 10.0);
    let ratio = (total / 10.0) / 0.029;
    if ratio < 1.0 {
        println!("  → Rust is {:.2}x FASTER!", 1.0 / ratio);
    } else {
        println!("  → Rust is {:.2}x slower", ratio);
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin benchmark_components --features blas --release");
}
