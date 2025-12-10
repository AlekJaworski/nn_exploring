//! Test EDF implementation on extreme k >> n case
//! This reproduces the user's original problem: k=200, n=50

use mgcv_rust::{GAM, SmoothTerm, SmoothingParameter, OptimizationMethod, ScaleParameterMethod};
use mgcv_rust::reml::penalty_sqrt;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TESTING EDF ON EXTREME CASE: k=200, n=50 ===\n");
    
    // Generate data similar to user's case
    let n = 50;
    let k = 200;
    
    // Create synthetic data
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    
    for i in 0..n {
        let x = i as f64 / n as f64;
        // Simple relationship with noise
        let y = (2.0 * std::f64::consts::PI * x).sin() + 0.1 * (i as f64 % 10.0);
        x_data.push(x);
        y_data.push(y);
    }
    
    println!("Data: n={}, k={}", n, k);
    println!("Ratio k/n = {:.1}", k as f64 / n as f64);
    
    // Create design matrix and penalty
    let x_min = x_data.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    // Create GAM with CR spline
    let mut gam = GAM::new(mgcv_rust::Family::Gaussian);
    let smooth = SmoothTerm::cr_spline("x".to_string(), k, x_min, x_max)?;
    gam.add_smooth(smooth);
    
    // Get the design matrix and penalty
    let x_2d = ndarray::Array2::from_shape_fn((n, 1), |(i, _)| x_data[i]);
    let y_arr = Array1::from_vec(y_data);
    let w = Array1::ones(n);
    
    // Extract penalty matrix
    let penalty = &gam.smooth_terms[0].penalty;
    
    println!("\nPenalty matrix: {}×{}", penalty.nrows(), penalty.ncols());
    println!("Penalty rank: {}", penalty_sqrt(penalty)?.ncols());
    
    // Test 1: Rank-based (original method)
    println!("\n{}", "=".repeat(50));
    println!("TEST 1: Rank-based method (original)");
    println!("{}", "=".repeat(50));
    
    let mut sp_rank = SmoothingParameter::new(1, OptimizationMethod::REML);
    sp_rank.scale_method = ScaleParameterMethod::Rank;
    
    std::env::set_var("MGCV_EDF_DEBUG", "1");
    println!("(Enabling EDF debug to show phi computation)");
    
    let start = std::time::Instant::now();
    let result_rank = sp_rank.optimize(&y_arr, &x_2d, &w, &vec![penalty.clone()], 20, 1e-6);
    let time_rank = start.elapsed();
    
    match result_rank {
        Ok(_) => {
            println!("✓ Converged!");
            println!("  λ = {:.6}", sp_rank.lambda[0]);
            println!("  Time: {:.2}ms", time_rank.as_secs_f64() * 1000.0);
        }
        Err(ref e) => {
            println!("✗ Failed: {}", e);
            println!("  This demonstrates the problem with rank-based method");
        }
    }
    
    // Test 2: EDF-based (new method)
    println!("\n{}", "=".repeat(50));
    println!("TEST 2: EDF-based method (NEW)");
    println!("{}", "=".repeat(50));
    
    let mut sp_edf = SmoothingParameter::new(1, OptimizationMethod::REML);
    sp_edf.scale_method = ScaleParameterMethod::EDF;
    
    let start = std::time::Instant::now();
    let result_edf = sp_edf.optimize(&y_arr, &x_2d, &w, &vec![penalty.clone()], 20, 1e-6);
    let time_edf = start.elapsed();
    
    match result_edf {
        Ok(_) => {
            println!("✓ Converged!");
            println!("  λ = {:.6}", sp_edf.lambda[0]);
            println!("  Time: {:.2}ms", time_edf.as_secs_f64() * 1000.0);
            println!("  Overhead: {:.1}x", time_edf.as_secs_f64() / time_rank.as_secs_f64());
        }
        Err(ref e) => {
            println!("✗ Failed: {}", e);
        }
    }
    
    std::env::remove_var("MGCV_EDF_DEBUG");
    
    // Summary
    println!("\n{}", "=".repeat(50));
    println!("SUMMARY");
    println!("{}", "=".repeat(50));
    
    let rank_success = result_rank.is_ok();
    let edf_success = result_edf.is_ok();
    
    println!("Rank-based method: {}", if rank_success { "✓ SUCCESS" } else { "✗ FAILED" });
    println!("EDF-based method:   {}", if edf_success { "✓ SUCCESS" } else { "✗ FAILED" });
    
    if rank_success && edf_success {
        let lambda_ratio = sp_edf.lambda[0] / sp_rank.lambda[0];
        println!("λ ratio (EDF/Rank): {:.3}", lambda_ratio);
        println!("Time ratio (EDF/Rank): {:.2}", time_edf.as_secs_f64() / time_rank.as_secs_f64());
    }
    
    if edf_success && !rank_success {
        println!("\n🎉 EDF method successfully handles k=200, n=50 case!");
        println!("   This fixes the original convergence issue.");
    } else if edf_success && rank_success {
        println!("\n✓ Both methods work, but EDF is more mathematically correct");
    } else {
        println!("\n⚠️  Both methods failed - may need different approach for k=200, n=50");
    }
    
    Ok(())
}

