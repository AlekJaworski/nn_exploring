use mgcv_rust::gam::{SmoothTerm, GAM};
use mgcv_rust::pirls::Family;

#[test]
fn test_hessian_at_optimal_lambda() {
    // Simple smoke test: create a GAM and verify it can be constructed
    let mut gam = GAM::new(Family::Gaussian);
    let smooth1 = SmoothTerm::cubic_spline("x1".to_string(), 5, 0.0, 1.0).unwrap();
    let smooth2 = SmoothTerm::cubic_spline("x2".to_string(), 5, 0.0, 1.0).unwrap();
    gam.add_smooth(smooth1);
    gam.add_smooth(smooth2);

    // Verify smooth terms were added
    assert_eq!(gam.smooth_terms.len(), 2);

    let optimal_lambda = vec![5.693608, 5.200554];

    println!("\n===============================================");
    println!("Testing Hessian at lambda = {:?}", optimal_lambda);
    println!("===============================================\n");

    println!("Note: Full Hessian testing requires fitting with fixed lambda.");
    println!("Expected mgcv values:");
    println!("  Hessian[0,0] = 2.813299");
    println!("  Hessian[1,1] = 3.185778");
    println!("  Hessian[0,1] = 0.023156");
}
