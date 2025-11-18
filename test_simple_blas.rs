// Test what actually compiles with ndarray-linalg 0.16
#[cfg(feature = "blas")]
fn main() {
    use ndarray::Array2;
    use ndarray_linalg::Solve;  // Try importing the trait
    
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    let b = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
    
    // Try solve
    match a.solve(&b) {
        Ok(x) => println!("Solution: {:?}", x),
        Err(e) => println!("Error: {:?}", e),
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("BLAS not enabled");
}
