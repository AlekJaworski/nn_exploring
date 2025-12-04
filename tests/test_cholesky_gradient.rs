// Test the new Cholesky-based gradient against QR version
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;
    use mgcv_rust::reml::{reml_gradient_multi_qr, reml_gradient_multi_cholesky};

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Testing Cholesky vs QR Gradient ===\n");
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

    println!("[1/3] Computing gradient with QR method...");
    let start = Instant::now();
    let grad_qr = reml_gradient_multi_qr(&y, &x, &w, &lambdas, &penalties).unwrap();
    let time_qr = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s", time_qr);
    println!("  Gradient: {:?}\n", grad_qr);

    println!("[2/3] Computing gradient with Cholesky method...");
    let start = Instant::now();
    let grad_chol = reml_gradient_multi_cholesky(&y, &x, &w, &lambdas, &penalties).unwrap();
    let time_chol = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s", time_chol);
    println!("  Gradient: {:?}\n", grad_chol);

    println!("[3/3] Comparing results...");
    let max_diff = grad_qr.iter().zip(grad_chol.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    
    println!("  Max difference: {:.2e}", max_diff);
    
    if max_diff < 1e-10 {
        println!("  ✓ Results match!\n");
    } else {
        println!("  ✗ Results DIFFER!\n");
    }

    println!("Performance:");
    println!("  QR method:       {:.3}s", time_qr);
    println!("  Cholesky method: {:.3}s", time_chol);
    let speedup = time_qr / time_chol;
    println!("  Speedup:         {:.2}x\n", speedup);

    println!("Comparison to R:");
    println!("  R/mgcv:          0.029s");
    println!("  Rust Cholesky:   {:.3}s", time_chol);
    let vs_r = time_chol / 0.029;
    if vs_r < 1.0 {
        println!("  → Rust is {:.2}x FASTER than R!", 1.0 / vs_r);
    } else {
        println!("  → Rust is {:.2}x slower than R", vs_r);
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_cholesky_gradient --features blas --release");
}
