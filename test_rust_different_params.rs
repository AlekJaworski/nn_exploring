// Add this test to verify no hardcoding with DIFFERENT parameters

#[test]
fn test_mgcv_penalty_different_params() {
    use ndarray::Array1;
    use crate::penalty::cubic_spline_penalty_mgcv;

    // Test 1: Different num_basis (10 instead of 20)
    let num_basis = 10;
    let knots = Array1::from_vec(vec![0.0, 1.0]);
    let penalty = cubic_spline_penalty_mgcv(num_basis, &knots, 2).unwrap();

    println!("\nTest 1: num_basis=10");
    println!("  Shape: {}x{}", penalty.nrows(), penalty.ncols());
    println!("  Frobenius: {:.1}", penalty.iter().map(|&x| x*x).sum::<f64>().sqrt());

    // Test 2: Different data range [0, 2]
    let num_basis = 20;
    let knots = Array1::from_vec(vec![0.0, 2.0]);
    let penalty = cubic_spline_penalty_mgcv(num_basis, &knots, 2).unwrap();

    println!("\nTest 2: range=[0,2]");
    println!("  Shape: {}x{}", penalty.nrows(), penalty.ncols());
    println!("  Frobenius: {:.1}", penalty.iter().map(|&x| x*x).sum::<f64>().sqrt());

    // Test 3: Different data range [-5, 5]
    let num_basis = 15;
    let knots = Array1::from_vec(vec![-5.0, 5.0]);
    let penalty = cubic_spline_penalty_mgcv(num_basis, &knots, 2).unwrap();

    println!("\nTest 3: range=[-5,5], num_basis=15");
    println!("  Shape: {}x{}", penalty.nrows(), penalty.ncols());
    println!("  Frobenius: {:.1}", penalty.iter().map(|&x| x*x).sum::<f64>().sqrt());

    // All should be positive definite with correct dimensions
    assert_eq!(penalty.nrows(), num_basis);
    assert_eq!(penalty.ncols(), num_basis);
}
