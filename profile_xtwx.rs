// Profile X'WX computation
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let p = 80;

    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen::<f64>();
        }
    }
    let w: Array1<f64> = Array1::ones(n);

    println!("=== X'WX Computation Profiling ===\n");
    println!("Matrix: {}×{}\n", n, p);

    println!("[1/2] Creating weighted matrix sqrt(W)·X...");
    let start = Instant::now();
    let mut x_weighted = Array2::zeros((n, p));
    for i in 0..n {
        let sqrt_w = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] = x[[i, j]] * sqrt_w;
        }
    }
    let weight_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", weight_time);

    println!("[2/2] BLAS matrix multiply X_w' * X_w...");
    let start = Instant::now();
    let xtwx = x_weighted.t().dot(&x_weighted);
    let gemm_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", gemm_time);

    println!("Breakdown:");
    println!("  Weighting loop: {:.3}s ({:.1}%)", weight_time, 100.0 * weight_time / (weight_time + gemm_time));
    println!("  BLAS GEMM:      {:.3}s ({:.1}%)", gemm_time, 100.0 * gemm_time / (weight_time + gemm_time));
    println!("  Total:          {:.3}s\n", weight_time + gemm_time);

    println!("Target: Reduce from 0.050s to match R's total gradient time");
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin profile_xtwx --features blas --release");
}
