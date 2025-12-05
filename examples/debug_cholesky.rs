// Debug why Cholesky-based optimization gives different gradients
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2, ShapeBuilder};
    use ndarray_linalg::{Cholesky, UPLO, Solve};

    println!("=== Debugging Cholesky vs Direct Solve ===\n");

    // Simple 3x3 system to debug
    // Try BOTH row-major and column-major layouts
    let a_row = Array2::from_shape_vec((3, 3), vec![
        4.0, 1.0, 1.0,
        1.0, 3.0, 1.0,
        1.0, 1.0, 2.0,
    ]).unwrap();

    // Create column-major version (Fortran layout for LAPACK)
    use ndarray::Array;
    let mut a = Array::zeros((3, 3).f());
    a[[0, 0]] = 4.0; a[[0, 1]] = 1.0; a[[0, 2]] = 1.0;
    a[[1, 0]] = 1.0; a[[1, 1]] = 3.0; a[[1, 2]] = 1.0;
    a[[2, 0]] = 1.0; a[[2, 1]] = 1.0; a[[2, 2]] = 2.0;

    // Check memory layout
    println!("Matrix A layout: {:?}", a.strides());
    println!("Is standard layout: {}\n", a.is_standard_layout());

    let b1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let b2 = Array1::from_vec(vec![2.0, 1.0, 1.0]);

    println!("Matrix A:\n{:?}\n", a);
    println!("RHS b1: {:?}", b1);
    println!("RHS b2: {:?}\n", b2);

    // Method 1: Direct solve (baseline)
    println!("Method 1: Direct solve (using mgcv_rust::linalg::solve)");
    let x1_direct = mgcv_rust::linalg::solve(a.clone(), b1.clone()).unwrap();
    let x2_direct = mgcv_rust::linalg::solve(a.clone(), b2.clone()).unwrap();
    println!("  x1 = {:?}", x1_direct);
    println!("  x2 = {:?}\n", x2_direct);

    // Method 2: SolveC trait on original matrix (does Cholesky internally)
    println!("Method 2: solvec on original matrix");
    {
        use ndarray_linalg::SolveC;
        let x1_chol = a.solvec(&b1).unwrap();
        let x2_chol = a.solvec(&b2).unwrap();
        println!("  x1 = {:?}", x1_chol);
        println!("  x2 = {:?}", x2_chol);

        // Verify: Does A·x = b?
        let ax1 = a.dot(&x1_chol);
        let ax2 = a.dot(&x2_chol);
        println!("\n  Verification: A·x1 = {:?}", ax1);
        println!("  Expected b1 = {:?}", b1);
        println!("  Verification: A·x2 = {:?}", ax2);
        println!("  Expected b2 = {:?}\n", b2);

        // Compare
        let diff1 = (&x1_direct - &x1_chol).mapv(|v: f64| v.abs()).sum();
        let diff2 = (&x2_direct - &x2_chol).mapv(|v: f64| v.abs()).sum();

        println!("Difference:");
        println!("  ||x1_direct - x1_chol|| = {:.2e}", diff1);
        println!("  ||x2_direct - x2_chol|| = {:.2e}\n", diff2);

        if diff1 < 1e-10 && diff2 < 1e-10 {
            println!("✓ SolveC method matches direct solve!");
            println!("  Cholesky-based solving works correctly!\n");
        } else {
            println!("✗ SolveC method DIFFERS from direct solve!");
            println!("  Difference is significant\n");
        }
    }

    // Method 3: Test with ndarray-linalg's Solve trait directly
    println!("Method 3: ndarray-linalg Solve trait");
    let x1_trait = a.solve(&b1).unwrap();
    let x2_trait = a.solve(&b2).unwrap();
    println!("  x1 = {:?}", x1_trait);
    println!("  x2 = {:?}\n", x2_trait);

    let diff1 = (&x1_direct - &x1_trait).mapv(|v: f64| v.abs()).sum();
    let diff2 = (&x2_direct - &x2_trait).mapv(|v: f64| v.abs()).sum();
    println!("Difference from direct:");
    println!("  ||x1_direct - x1_trait|| = {:.2e}", diff1);
    println!("  ||x2_direct - x2_trait|| = {:.2e}\n", diff2);

    println!("=== Diagnosis ===");
    println!("If Cholesky matches: Issue is in gradient formula");
    println!("If Cholesky differs: Issue is in factorization or matrix conditioning");

    // Deep dive: Check if factorization is correct
    println!("\n=== Deep Dive: Verify Cholesky Factorization ===\n");
    match a.cholesky(UPLO::Lower) {
        Ok(l) => {
            println!("Lower triangular L:");
            println!("{:?}\n", l);

            // Reconstruct A = L·L'
            let l_t = l.t();
            let a_reconstructed = l.dot(&l_t);
            println!("Reconstructed A = L·L':");
            println!("{:?}\n", a_reconstructed);

            let diff_a = (&a - &a_reconstructed).mapv(|v: f64| v.abs()).sum();
            println!("||A - L·L'|| = {:.2e}", diff_a);

            if diff_a < 1e-10 {
                println!("✓ Factorization is correct!");
                println!("  BUT solveh() gives wrong answers!");
                println!("  This is a BUG in how solveh() is being called\n");
            } else {
                println!("✗ Factorization is WRONG!");
                println!("  Cholesky decomposition failed silently\n");
            }
        },
        Err(e) => {
            println!("Cholesky failed: {:?}", e);
        }
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("Run with: cargo run --bin debug_cholesky --features blas");
}
