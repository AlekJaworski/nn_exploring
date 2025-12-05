// Test cached sqrt_penalties version
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;
    use mgcv_rust::reml::{reml_gradient_multi_cholesky, reml_gradient_multi_cholesky_cached};

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Testing Cached sqrt_penalties ===\n");
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

    println!("[1/3] Pre-compute sqrt_penalties once...");
    let start = Instant::now();
    use ndarray_linalg::Eigh;
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    for penalty in &penalties {
        let (eigenvalues, eigenvectors) = penalty.eigh(ndarray_linalg::UPLO::Upper).unwrap();
        let max_eig = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let threshold = 1e-10 * max_eig.max(1.0);
        
        let non_zero: Vec<_> = eigenvalues.iter().enumerate()
            .filter(|(_, &e)| e > threshold)
            .collect();
        let rank = non_zero.len();
        
        let mut sqrt_pen = Array2::zeros((p, rank));
        for (out_j, &(in_j, &eig)) in non_zero.iter().enumerate() {
            let sqrt_eval = eig.sqrt();
            for i in 0..p {
                sqrt_pen[[i, out_j]] = eigenvectors[[i, in_j]] * sqrt_eval;
            }
        }
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }
    let precomp_time = start.elapsed().as_secs_f64();
    println!("  Eigendecomp time: {:.3}s\n", precomp_time);

    println!("[2/3] Benchmark WITHOUT caching (10 calls)...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_cholesky(&y, &x, &w, &lambdas, &penalties).unwrap();
    }
    let time_uncached = start.elapsed().as_secs_f64();
    println!("  Total: {:.3}s", time_uncached);
    println!("  Per call: {:.3}s\n", time_uncached / 10.0);

    println!("[3/3] Benchmark WITH caching (10 calls)...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_cholesky_cached(&y, &x, &w, &lambdas, &penalties, &sqrt_penalties, &penalty_ranks).unwrap();
    }
    let time_cached = start.elapsed().as_secs_f64();
    println!("  Total: {:.3}s", time_cached);
    println!("  Per call: {:.3}s\n", time_cached / 10.0);

    println!("Performance:");
    println!("  Uncached (with eigendecomp): {:.3}s per call", time_uncached / 10.0);
    println!("  Cached (no eigendecomp):     {:.3}s per call", time_cached / 10.0);
    let speedup = (time_uncached / 10.0) / (time_cached / 10.0);
    println!("  Speedup:                     {:.2}x\n", speedup);

    println!("Amortized cost over 10 calls:");
    let amortized = (precomp_time + time_cached) / 10.0;
    println!("  Precomp once + 10 cached:    {:.3}s per call (amortized)", amortized);
    println!("  vs Uncached:                 {:.3}s per call", time_uncached / 10.0);
    println!("  Savings:                     {:.3}s per call\n", time_uncached / 10.0 - amortized);

    println!("Comparison to R:");
    println!("  R/mgcv:                      0.029s per call");
    println!("  Rust cached (amortized):     {:.3}s per call", amortized);
    let vs_r = amortized / 0.029;
    if vs_r < 1.0 {
        println!("  → Rust is {:.2}x FASTER than R!", 1.0 / vs_r);
    } else {
        println!("  → Rust is {:.2}x slower than R", vs_r);
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_cached_gradient --features blas --release");
}
