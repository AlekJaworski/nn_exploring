// Test fully cached gradient (sqrt_penalties + X'WX + X'Wy)
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use ndarray_linalg::Eigh;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;
    use mgcv_rust::reml::{reml_gradient_multi_cholesky, reml_gradient_multi_cholesky_fully_cached};

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Testing Fully Cached Gradient ===\n");
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

    println!("[1/3] Pre-compute ALL cacheable values...");
    let start = Instant::now();
    
    // Sqrt penalties
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
    
    // X'WX
    let mut x_weighted = Array2::zeros((n, p));
    for i in 0..n {
        let sqrt_w_val = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] = x[[i, j]] * sqrt_w_val;
        }
    }
    let xtwx = x_weighted.t().dot(&x_weighted);
    
    // X'Wy
    let mut xtwy = Array1::zeros(p);
    for i in 0..p {
        let mut sum = 0.0;
        for j in 0..n {
            sum += x[[j, i]] * w[j] * y[j];
        }
        xtwy[i] = sum;
    }
    
    let precomp_time = start.elapsed().as_secs_f64();
    println!("  Precomputation time: {:.3}s\n", precomp_time);

    println!("[2/3] Benchmark standard cached (10 calls)...");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_cholesky(&y, &x, &w, &lambdas, &penalties).unwrap();
    }
    let time_standard = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.3}s per call)\n", time_standard, time_standard / 10.0);

    println!("[3/3] Benchmark fully cached (10 calls)...");
    let y_res_data = (y.clone(), w.clone());
    let start = Instant::now();
    for _ in 0..10 {
        let _ = reml_gradient_multi_cholesky_fully_cached(
            &x, &lambdas, &penalties, &sqrt_penalties, &penalty_ranks,
            &xtwx, &xtwy, &y_res_data
        ).unwrap();
    }
    let time_fully_cached = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s ({:.3}s per call)\n", time_fully_cached, time_fully_cached / 10.0);

    println!("Performance:");
    println!("  Standard Cholesky:  {:.3}s per call", time_standard / 10.0);
    println!("  Fully cached:       {:.3}s per call", time_fully_cached / 10.0);
    let speedup = (time_standard / 10.0) / (time_fully_cached / 10.0);
    println!("  Speedup:            {:.2}x\n", speedup);

    println!("Amortized (1 precomp + 10 calls):");
    let amortized = (precomp_time + time_fully_cached) / 10.0;
    println!("  Per call: {:.3}s\n", amortized);

    println!("vs R/mgcv:");
    println!("  R:              0.029s per call");
    println!("  Rust (amortized): {:.3}s per call", amortized);
    let vs_r = amortized / 0.029;
    if vs_r < 1.0 {
        println!("  → Rust is {:.2}x FASTER!", 1.0 / vs_r);
    } else {
        println!("  → Rust is {:.2}x slower", vs_r);
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_fully_cached --features blas --release");
}
