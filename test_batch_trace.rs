// Test that batch solve gives same result as loop for trace computation
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array2, arr2};
    use ndarray_linalg::{SolveTriangular, UPLO, Diag};
    use mgcv_rust::linalg::solve;

    println!("Testing batch vs loop for trace computation...\n");

    // Small test case
    let r_upper = arr2(&[[2.0, 1.0, 0.5],
                        [0.0, 1.5, 0.3],
                        [0.0, 0.0, 1.2]]);

    let sqrt_penalty = arr2(&[[1.0, 2.0],
                              [0.5, 1.5],
                              [0.3, 1.0]]);  // p=3, rank=2

    println!("R (upper triangular):\n{:?}\n", r_upper);
    println!("sqrt_penalty (p×rank = 3×2):\n{:?}\n", sqrt_penalty);

    // Method 1: Loop (original)
    let rank = sqrt_penalty.ncols();
    let mut trace_loop = 0.0;
    println!("Method 1: Loop over columns");
    for k in 0..rank {
        let l_col = sqrt_penalty.column(k).to_owned();
        println!("  Column {}: {:?}", k, l_col);
        let x = solve(r_upper.t().to_owned(), l_col).unwrap();
        println!("  Solution x: {:?}", x);
        let norm_sq: f64 = x.iter().map(|xi| xi * xi).sum();
        println!("  ||x||² = {:.6}\n", norm_sq);
        trace_loop += norm_sq;
    }
    println!("Trace (loop): {:.6}\n", trace_loop);

    // Method 2: Batch solve
    println!("Method 2: Batch triangular solve");
    let r_t = r_upper.t().to_owned();
    println!("R' (lower triangular):\n{:?}\n", r_t);

    match r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &sqrt_penalty) {
        Ok(x_batch) => {
            println!("X_batch (p×rank = 3×2):\n{:?}\n", x_batch);
            let trace_batch: f64 = x_batch.iter().map(|xi| xi * xi).sum();
            println!("Trace (batch): {:.6}\n", trace_batch);

            let diff = (trace_loop - trace_batch).abs();
            println!("Difference: {:.2e}", diff);

            if diff < 1e-10 {
                println!("\n✓ Batch solve matches loop!");
            } else {
                println!("\n✗ Batch solve DIFFERS from loop!");
            }
        },
        Err(e) => {
            println!("✗ Batch solve FAILED: {:?}", e);
        }
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_batch_trace --features blas");
}
