// Test what actually compiles with ndarray-linalg 0.17
#[cfg(feature = "blas")]
fn main() {
    use ndarray::{Array1, Array2};
    use ndarray_linalg::Solve;

    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    let b = Array1::from_vec(vec![1.0, 2.0]);

    match a.solve(&b) {
        Ok(x) => println!("Solution: {:?}", x),
        Err(e) => println!("Error: {:?}", e),
    }
}

#[cfg(not(feature = "blas"))]
fn main() {
    println!("BLAS not enabled");
}
