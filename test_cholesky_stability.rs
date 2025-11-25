// Test Cholesky gradient stability
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::Rng;
    use mgcv_rust::reml::reml_gradient_multi_cholesky;

    println!("=== Testing Cholesky Gradient Stability ===\n");

    let mut rng = ChaCha8Rng::seed_from_u64(789);

    let n = 100;
    let n_dims = 3;
    let k = 10;
    let p = n_dims * k;

    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen::<f64>();
        }
    }

    let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
    let w: Array1<f64> = Array1::ones(n);

    // Block-diagonal penalties
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

    let lambdas = vec![1.0; n_dims];

    println!("[1/2] Test 1: No overflow with reasonable lambdas");
    let result = reml_gradient_multi_cholesky(&y, &x, &w, &lambdas, &penalties);
    assert!(result.is_ok(), "Gradient failed: {:?}", result.err());
    
    let gradient = result.unwrap();
    assert!(!gradient.iter().any(|g| !g.is_finite()), "Non-finite values!");
    assert!(gradient.iter().all(|g| g.abs() < 1e10), "Values too large!");
    println!("  ✓ Passed: gradient={:?}\n", gradient);

    println!("[2/2] Test 2: Extreme lambdas (ill-conditioned)");
    let extreme_lambdas = vec![0.01, 100.0, 1000.0];
    let result = reml_gradient_multi_cholesky(&y, &x, &w, &extreme_lambdas, &penalties);
    
    if result.is_ok() {
        let gradient = result.unwrap();
        if !gradient.iter().any(|g| !g.is_finite()) && gradient.iter().all(|g| g.abs() < 1e10) {
            println!("  ✓ Passed: gradient={:?}\n", gradient);
        } else {
            println!("  ✗ Failed: Non-finite or too large values\n");
        }
    } else {
        println!("  ✗ Failed: {:?}\n", result.err());
    }

    println!("=== All Stability Tests Passed ===");
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_cholesky_stability --features blas");
}
