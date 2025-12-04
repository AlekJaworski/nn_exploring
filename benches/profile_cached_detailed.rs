// Detailed profiling of cached gradient
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use ndarray_linalg::{Cholesky, UPLO, SolveTriangular, Diag, Eigh};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use std::time::Instant;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 6000;
    let n_dims = 8;
    let k = 10;
    let p = n_dims * k;

    println!("=== Detailed Cached Gradient Profiling ===\n");
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

    // Create penalties and pre-compute sqrt
    let mut penalties = Vec::new();
    let mut sqrt_penalties = Vec::new();
    for dim in 0..n_dims {
        let mut penalty = Array2::zeros((p, p));
        let start = dim * k;
        let end = start + k;
        for i in start..end {
            penalty[[i, i]] = 1.0;
        }
        
        // Precompute sqrt
        let (eigenvalues, eigenvectors) = penalty.eigh(UPLO::Upper).unwrap();
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
        
        penalties.push(penalty);
        sqrt_penalties.push(sqrt_pen);
    }

    // Now profile each component of the cached gradient

    println!("[1/7] X'WX formation...");
    let start = Instant::now();
    let mut xtwx = Array2::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[[k, i]] * w[k] * x[[k, j]];
            }
            xtwx[[i, j]] = sum;
            if i != j {
                xtwx[[j, i]] = sum;
            }
        }
    }
    let xtwx_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", xtwx_time);

    println!("[2/7] Form A = X'WX + λS...");
    let start = Instant::now();
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        a.scaled_add(*lambda, penalty);
    }
    let form_a_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", form_a_time);

    println!("[3/7] Cholesky factorization...");
    let start = Instant::now();
    let r_upper = a.cholesky(UPLO::Upper).unwrap();
    let chol_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", chol_time);

    println!("[4/7] Compute beta...");
    let start = Instant::now();
    let mut xtwy = Array1::zeros(p);
    for i in 0..p {
        let mut sum = 0.0;
        for j in 0..n {
            sum += x[[j, i]] * w[j] * y[j];
        }
        xtwy[i] = sum;
    }
    
    let r_t = r_upper.t().to_owned();
    let y_temp = r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &xtwy).unwrap();
    let beta = r_upper.solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp).unwrap();
    let beta_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", beta_time);

    println!("[5/7] Compute residuals and phi...");
    let start = Instant::now();
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = y.iter().zip(y_hat.iter())
        .map(|(yi, yhati)| yi - yhati)
        .collect();
    let rss: f64 = residuals.iter().zip(w.iter())
        .map(|(ri, wi)| ri * ri * wi)
        .sum();
    let n_minus_r = n as f64 - (n_dims * k) as f64;
    let phi = rss / n_minus_r;
    let resid_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", resid_time);

    println!("[6/7] Trace computations (batch)...");
    let start = Instant::now();
    for sqrt_pen in &sqrt_penalties {
        let x_batch = r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_pen).unwrap();
        let _trace: f64 = x_batch.iter().map(|xi| xi * xi).sum();
    }
    let trace_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", trace_time);

    println!("[7/7] Beta derivatives...");
    let start = Instant::now();
    for (i, penalty) in penalties.iter().enumerate() {
        let s_beta = penalty.dot(&beta);
        let lambda_s_beta = s_beta.mapv(|x| lambdas[i] * x);
        let y_temp = r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta).unwrap();
        let _dbeta = r_upper.solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp).unwrap();
    }
    let beta_deriv_time = start.elapsed().as_secs_f64();
    println!("  Time: {:.3}s\n", beta_deriv_time);

    let total = xtwx_time + form_a_time + chol_time + beta_time + resid_time + trace_time + beta_deriv_time;

    println!("=== Breakdown ===");
    println!("  X'WX:           {:.3}s ({:.1}%)", xtwx_time, 100.0 * xtwx_time / total);
    println!("  Form A:         {:.3}s ({:.1}%)", form_a_time, 100.0 * form_a_time / total);
    println!("  Cholesky:       {:.3}s ({:.1}%)", chol_time, 100.0 * chol_time / total);
    println!("  Compute beta:   {:.3}s ({:.1}%)", beta_time, 100.0 * beta_time / total);
    println!("  Residuals/phi:  {:.3}s ({:.1}%)", resid_time, 100.0 * resid_time / total);
    println!("  Trace solves:   {:.3}s ({:.1}%)", trace_time, 100.0 * trace_time / total);
    println!("  Beta derivs:    {:.3}s ({:.1}%)", beta_deriv_time, 100.0 * beta_deriv_time / total);
    println!("  ─────────────────────────");
    println!("  Total:          {:.3}s", total);
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin profile_cached_detailed --features blas --release");
}
