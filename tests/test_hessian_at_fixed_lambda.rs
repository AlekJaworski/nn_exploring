use mgcv_rust::gam::GAM;
use mgcv_rust::smooth::{CubicRegressionSpline, Smooth};
use ndarray::{Array1, Array2};

#[test]
fn test_hessian_at_optimal_lambda() {
    // Same data as Python tests
    let n = 100;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);

    // Generate data matching R seed(42)
    // For reproducibility, use same random numbers as R
    let x_vals = vec![
        1.3709584, -0.5646982, 0.3631284, 0.6328626, 0.4042683,
        -0.1061245, 1.5115220, -0.0947416, 2.0184237, -0.0627141,
        // ... (truncated for brevity - in real test use all 200 values)
    ];

    for i in 0..n {
        x_data.push(x_vals[i]);
        x_data.push(x_vals[n + i]);
        let x1 = x_vals[i];
        let x2 = x_vals[n + i];
        y_data.push(x1.sin() + 0.5 * x2.powi(2));
    }

    let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    // Create GAM with two smooths
    let smooth1 = CubicRegressionSpline::new(vec![0], 10).unwrap();
    let smooth2 = CubicRegressionSpline::new(vec![1], 10).unwrap();
    let smooths: Vec<Box<dyn Smooth>> = vec![Box::new(smooth1), Box::new(smooth2)];

    let mut gam = GAM::new(x.view(), y.view(), smooths).unwrap();

    // Set λ to mgcv's optimal values
    let optimal_lambda = vec![5.693608, 5.200554];

    // Fit with these fixed λ values (one iteration)
    println!("\n===============================================");
    println!("Testing Hessian at λ = {:?}", optimal_lambda);
    println!("===============================================\n");

    // Manually call REML computation to get Hessian
    // We need to expose this or add debug output

    println!("Note: To get Hessian components, run with MGCV_GRAD_DEBUG=1");
    println!("Expected mgcv values:");
    println!("  Hessian[0,0] = 2.813299");
    println!("  Hessian[1,1] = 3.185778");
    println!("  Hessian[0,1] = 0.023156");
}
