use ndarray::{Array1, Array2};
use ndarray_linalg::*;

fn main() {
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    let b = Array1::from_vec(vec![1.0, 2.0]);

    // Test solve
    let x = a.solve_into(b).unwrap();
    println!("Solution: {:?}", x);

    // Test determinant
    let det = a.det().unwrap();
    println!("Determinant: {}", det);

    // Test inverse
    let inv = a.inv().unwrap();
    println!("Inverse: {:?}", inv);
}
