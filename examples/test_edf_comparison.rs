//! Test EDF vs Rank-based scale parameter computation
//! 
//! This example demonstrates the difference between using penalty ranks
//! vs effective degrees of freedom for computing the scale parameter φ.
//!
//! For well-conditioned problems (k << n), the difference is small.
//! For ill-conditioned problems (k >> n), EDF provides better convergence.

use mgcv_rust::{GAM, SmoothTerm, SmoothingParameter, OptimizationMethod, Family};
#[cfg(feature = "blas")]
use mgcv_rust::ScaleParameterMethod;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EDF vs Rank-based Scale Parameter Comparison ===\n");
    
    // Generate test data
    let n = 100;
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    
    for i in 0..n {
        let x = i as f64 / n as f64;
        let y = (2.0 * std::f64::consts::PI * x).sin() + 0.1 * (i as f64 % 10.0);
        x_vals.push(x);
        y_vals.push(y);
    }
    
    // Test Case 1: Well-conditioned (k = 10, n = 100, ratio = 0.1)
    println!("Test Case 1: Well-conditioned (k=10, n=100)");
    println!("--------------------------------------------");
    test_case(&x_vals, &y_vals, 10)?;
    
    // Test Case 2: Moderately conditioned (k = 30, n = 100, ratio = 0.3)
    println!("\nTest Case 2: Moderately conditioned (k=30, n=100)");
    println!("--------------------------------------------------");
    test_case(&x_vals, &y_vals, 30)?;
    
    // Test Case 3: Ill-conditioned (k = 50, n = 100, ratio = 0.5)
    println!("\nTest Case 3: Ill-conditioned (k=50, n=100)");
    println!("-------------------------------------------");
    test_case(&x_vals, &y_vals, 50)?;
    
    Ok(())
}

#[cfg(feature = "blas")]
fn test_case(x_vals: &[f64], y_vals: &[f64], k: usize) -> Result<(), Box<dyn std::error::Error>> {
    let x_min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    // Method 1: Rank-based (default, fast)
    let mut gam_rank = GAM::new(Family::Gaussian);
    let smooth = SmoothTerm::cr_spline("x".to_string(), k, x_min, x_max)?;
    gam_rank.add_smooth(smooth);
    
    let x_2d = ndarray::Array2::from_shape_fn((x_vals.len(), 1), |(i, _)| x_vals[i]);
    let y_arr = Array1::from_vec(y_vals.to_vec());
    
    // Create smoothing parameter with Rank method
    let mut sp_rank = SmoothingParameter::new(1, OptimizationMethod::REML);
    sp_rank.scale_method = ScaleParameterMethod::Rank;
    
    let start = std::time::Instant::now();
    sp_rank.optimize(&y_arr, &x_2d, &Array1::ones(y_vals.len()), 
                     &vec![gam_rank.smooth_terms[0].penalty.clone()], 20, 1e-6)?;
    let rank_time = start.elapsed();
    let lambda_rank = sp_rank.lambda[0];
    
    println!("  Rank-based:  λ = {:.6}, time = {:.2}ms", lambda_rank, rank_time.as_secs_f64() * 1000.0);
    
    // Method 2: EDF-based (exact, matches mgcv)
    let mut gam_edf = GAM::new(Family::Gaussian);
    let smooth2 = SmoothTerm::cr_spline("x".to_string(), k, x_min, x_max)?;
    gam_edf.add_smooth(smooth2);
    
    let mut sp_edf = SmoothingParameter::new(1, OptimizationMethod::REML);
    sp_edf.scale_method = ScaleParameterMethod::EDF;
    
    // Enable EDF debug to see the difference
    std::env::set_var("MGCV_EDF_DEBUG", "1");
    
    let start = std::time::Instant::now();
    sp_edf.optimize(&y_arr, &x_2d, &Array1::ones(y_vals.len()), 
                    &vec![gam_edf.smooth_terms[0].penalty.clone()], 20, 1e-6)?;
    let edf_time = start.elapsed();
    let lambda_edf = sp_edf.lambda[0];
    
    std::env::remove_var("MGCV_EDF_DEBUG");
    
    println!("  EDF-based:   λ = {:.6}, time = {:.2}ms", lambda_edf, edf_time.as_secs_f64() * 1000.0);
    println!("  Difference:  λ_ratio = {:.4}, time_ratio = {:.2}x", 
             lambda_edf / lambda_rank, edf_time.as_secs_f64() / rank_time.as_secs_f64());
    
    Ok(())
}

#[cfg(not(feature = "blas"))]
fn test_case(_x_vals: &[f64], _y_vals: &[f64], _k: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("  (EDF requires BLAS feature - skipping comparison)");
    Ok(())
}
