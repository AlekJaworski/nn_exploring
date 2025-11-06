use ndarray::Array1;
use mgcv_rust::basis::{BasisFunction, CubicSpline, BoundaryCondition};

fn main() {
    // Create spline with knots [0.3, 0.7]
    let knots = Array1::linspace(0.3, 0.7, 4); // 4 internal knots
    let spline = CubicSpline::new(knots, BoundaryCondition::Natural);

    println!("Cubic spline with {} basis functions", spline.num_basis());
    println!("Knots: {:?}", spline.knots().unwrap());

    // Test points including extrapolation
    let x_test = Array1::from_vec(vec![0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]);

    println!("\nEvaluating basis at test points:");
    let basis = spline.evaluate(&x_test).unwrap();

    for (i, &xi) in x_test.iter().enumerate() {
        let row_sum: f64 = basis.row(i).iter().sum();
        let all_zero = basis.row(i).iter().all(|&v| v.abs() < 1e-10);

        let status = if xi < 0.3 {
            "LEFT "
        } else if xi > 0.7 {
            "RIGHT"
        } else {
            "IN   "
        };

        println!("  x={:.1} ({}) sum={:.4}, all_zero={}", xi, status, row_sum, all_zero);

        if all_zero {
            println!("    ⚠ ALL BASIS FUNCTIONS ARE ZERO!");
        }
    }
}
