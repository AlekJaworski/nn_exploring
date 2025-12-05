// Test if ndarray-linalg supports solving A·X = B where B is a matrix
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array2, arr2};
    use ndarray_linalg::{SolveTriangular, UPLO, Diag};

    println!("Testing triangular solve R'·X = B with matrix RHS...\n");

    // Test: solve R'·X = B where B is a matrix (multiple RHS)
    // R is upper triangular
    let r = arr2(&[[2.0, 1.0],
                   [0.0, 3.0]]);

    let b = arr2(&[[4.0, 2.0],  // Two RHS vectors as columns
                   [6.0, 3.0]]);

    println!("R = {:?}", r);
    println!("B = {:?}", b);

    // Solve R'·X = B (transpose of upper triangular)
    // R' is lower triangular, so use UPLO::Lower
    let r_t = r.t().to_owned();
    println!("\nR' (transposed, lower triangular) = {:?}", r_t);

    match r_t.solve_triangular(UPLO::Lower, Diag::NonUnit, &b) {
        Ok(x) => {
            println!("\n✓ Triangular matrix solve WORKS!");
            println!("X = {:?}", x);

            // Verify: R'·X should equal B
            let rtx = r_t.dot(&x);
            println!("\nVerification: R'·X = {:?}", rtx);
            println!("Expected B = {:?}", b);

            // Check if close
            let diff = (&rtx - &b).mapv(|v: f64| v.abs()).sum();
            println!("Difference: {:.2e}", diff);

            if diff < 1e-10 {
                println!("\n✓ Verification PASSED");
                println!("\n🎉 BATCH TRIANGULAR SOLVE IS SUPPORTED!");
                println!("   We can solve R'·X = L for ALL columns at once");
            }
        },
        Err(e) => {
            println!("\n✗ Triangular solve FAILED: {:?}", e);
        }
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin test_batch_solve --features blas");
}
