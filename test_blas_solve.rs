use ndarray::{Array1, Array2};
use ndarray_linalg::*;

fn main() {
    let a = Array2::from_shape_vec((3, 3), vec![
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 2.0,
    ]).unwrap();
    
    let b = Array1::from_vec(vec![4.0, 6.0, 5.0]);
    
    println!("A:\n{}", a);
    println!("b: {}", b);
    
    // Try solve_into
    let x = a.solve_into(b).unwrap();
    println!("x from solve_into: {}", x);
    
    // Verify
    let a2 = Array2::from_shape_vec((3, 3), vec![
        2.0, 1.0, 1.0,
        1.0, 3.0, 2.0,
        1.0, 2.0, 2.0,
    ]).unwrap();
    let ax = a2.dot(&x);
    println!("Ax: {}", ax);
    println!("Should be: [4.0, 6.0, 5.0]");
}
