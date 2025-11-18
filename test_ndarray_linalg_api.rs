// Minimal test to find correct API
use ndarray::Array2;

fn main() {
    let a = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 2.0, 3.0]).unwrap();
    
    // Try to see what methods are available
    println!("Matrix created successfully");
    println!("{:?}", a);
}
