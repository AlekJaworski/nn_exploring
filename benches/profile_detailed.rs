// Detailed profiling of gradient computation components
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use ndarray_linalg::{QR, SolveTriangular, UPLO, Diag};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Detailed Rust Gradient Profiling ===\n");
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

    println!("[1/6] Computing sqrt(W)·X...");
    let start = Instant::now();
    let mut sqrt_w_x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] = x[[i, j]] * weight_sqrt;
        }
    }
    println!("  Time: {:.3}s\n", start.elapsed().as_secs_f64());

    println!("[2/6] Computing penalty square roots...");
    let start = Instant::now();
    let mut sqrt_penalties = Vec::new();
    for penalty in &penalties {
        use ndarray_linalg::Eigh;
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
    }
    println!("  Time: {:.3}s\n", start.elapsed().as_secs_f64());

    println!("[3/6] Building augmented matrix Z...");
    let start = Instant::now();
    let mut total_rows = n;
    for sqrt_pen in &sqrt_penalties {
        total_rows += sqrt_pen.ncols();
    }
    
    let mut z = Array2::<f64>::zeros((total_rows, p));
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = sqrt_w_x[[i, j]];
        }
    }
    
    let mut row_offset = n;
    for (sqrt_pen, &lambda) in sqrt_penalties.iter().zip(lambdas.iter()) {
        let sqrt_lambda = lambda.sqrt();
        let rank = sqrt_pen.ncols();
        for i in 0..rank {
            for j in 0..p {
                z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];
            }
        }
        row_offset += rank;
    }
    println!("  Time: {:.3}s\n", start.elapsed().as_secs_f64());

    println!("[4/6] QR decomposition...");
    let start = Instant::now();
    let (_, r) = z.qr().unwrap();
    let mut r_upper = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            r_upper[[i, j]] = r[[i, j]];
        }
    }
    let qr_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", qr_time);

    println!("[5/6] Trace computations (batch solve)...");
    let start = Instant::now();
    let r_t = r_upper.t().to_owned();
    let mut total_trace = 0.0;
    for sqrt_pen in &sqrt_penalties {
        let x_batch = r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_pen).unwrap();
        total_trace += x_batch.iter().map(|xi| xi * xi).sum::<f64>();
    }
    let trace_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", trace_time);

    println!("[6/6] Beta derivatives (cached R)...");
    let start = Instant::now();
    let beta = Array1::zeros(p);  // Dummy beta
    for (i, penalty) in penalties.iter().enumerate() {
        let s_beta = penalty.dot(&beta);
        let lambda_s_beta = s_beta.mapv(|x| lambdas[i] * x);
        
        // Two triangular solves using cached R
        let y = r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta).unwrap();
        let _dbeta = r_upper.solve_triangular(UPLO::Upper, Diag::NonUnit, &y).unwrap();
    }
    let beta_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", beta_time);

    let total_time = qr_time + trace_time + beta_time;
    
    println!("=== Breakdown ===");
    println!("  QR:             {:.3}s ({:.1}%)", qr_time, 100.0 * qr_time / total_time);
    println!("  Trace solves:   {:.3}s ({:.1}%)", trace_time, 100.0 * trace_time / total_time);
    println!("  Beta derivs:    {:.3}s ({:.1}%)", beta_time, 100.0 * beta_time / total_time);
    println!("  ─────────────────────────");
    println!("  Total:          {:.3}s\n", total_time);
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin profile_detailed --features blas --release");
}
