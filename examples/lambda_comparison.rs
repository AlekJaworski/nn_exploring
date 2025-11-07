//! Demonstrate the effect of different smoothing parameter values
//! by manually setting λ and comparing the results

use mgcv_rust::{GAM, Family, SmoothTerm, OptimizationMethod};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Effect of Smoothing Parameter λ ===\n");

    // Generate noisy data
    let n = 300;  // Increased from 30 to 300 for better n/p ratio
    let noise_level = 0.6;

    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let true_y = (2.0 * std::f64::consts::PI * xi).sin();
            let noise = noise_level * ((i as f64 * 0.7 + 0.3).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data.clone());
    let y = Array1::from_vec(y_data.clone());
    let x_matrix = x.clone().to_shape((n, 1))?.to_owned();

    println!("Data: {} points with noise level {:.2}\n", n, noise_level);

    // Test different lambda values manually by modifying the smooth term
    println!("{}", "=".repeat(70));
    println!(" λ value    | Training RMSE | Test RMSE  | Interpretation");
    println!("{}", "=".repeat(70));

    // We'll use GCV to find optimal, then compare with other values
    let mut gam_auto = GAM::new(Family::Gaussian);
    let smooth_auto = SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
    gam_auto.add_smooth(smooth_auto);
    gam_auto.fit(&x_matrix, &y, OptimizationMethod::GCV, 10, 100, 1e-6)?;

    let optimal_lambda = if let Some(ref params) = gam_auto.smoothing_params {
        params.lambda[0]
    } else {
        0.1
    };

    println!("Automatic selection (GCV):");
    println!(" {:.6} | {:.6}      | {:.6}   | ← Optimal by GCV",
        optimal_lambda,
        compute_training_rmse(&gam_auto, n),
        compute_test_rmse(&gam_auto)?
    );

    println!("\nManual comparisons:");

    // Try various lambda values
    for &lambda_val in &[0.0001, 0.01, 0.1, 1.0, 10.0, 100.0] {
        let mut gam = GAM::new(Family::Gaussian);
        let mut smooth = SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;

        // Set the lambda manually
        smooth.lambda = lambda_val;
        gam.add_smooth(smooth);

        // Fit WITHOUT optimizing lambda (just run PiRLS once)
        // We do this by running just inner loop
        use mgcv_rust::pirls::fit_pirls;
        use ndarray::Array2;

        // Evaluate basis
        let basis_matrix = {
            let mut gam_temp = GAM::new(Family::Gaussian);
            let smooth_temp = SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
            gam_temp.add_smooth(smooth_temp);

            // Get design matrix (this is a hack - ideally we'd expose this)
            let mut total_basis = 0;
            for smooth in &gam_temp.smooth_terms {
                total_basis += smooth.num_basis();
            }

            let mut full_design = Array2::zeros((n, total_basis));
            let x_col = x.clone();
            let basis = gam_temp.smooth_terms[0].evaluate(&x_col)?;

            for i in 0..n {
                for j in 0..basis.ncols() {
                    full_design[[i, j]] = basis[[i, j]];
                }
            }
            full_design
        };

        // Get penalty matrix
        let penalty = gam.smooth_terms[0].penalty.clone();
        let penalties = vec![penalty];

        // Fit with fixed lambda
        let pirls_result = fit_pirls(
            &y,
            &basis_matrix,
            &[lambda_val],
            &penalties,
            Family::Gaussian,
            100,
            1e-6
        )?;

        // Store results in gam
        gam.coefficients = Some(pirls_result.coefficients.clone());
        gam.fitted_values = Some(pirls_result.fitted_values);
        gam.deviance = Some(pirls_result.deviance);
        gam.fitted = true;

        let train_rmse = (pirls_result.deviance / n as f64).sqrt();
        let test_rmse = compute_test_rmse(&gam)?;

        let interpretation = if lambda_val < 0.001 {
            "Severe overfitting"
        } else if lambda_val < 0.05 {
            "Likely overfitting"
        } else if lambda_val < 0.5 {
            "Balanced fit"
        } else if lambda_val < 5.0 {
            "Some undersmoothing"
        } else {
            "Heavy smoothing"
        };

        println!(" {:.6} | {:.6}      | {:.6}   | {}",
            lambda_val, train_rmse, test_rmse, interpretation);
    }

    println!("{}", "=".repeat(70));

    println!("\n\nKey observations:");
    println!("1. As λ increases, training error increases (less flexible fit)");
    println!("2. Test error is minimized at intermediate λ (bias-variance tradeoff)");
    println!("3. Very small λ → overfits training data, poor generalization");
    println!("4. Very large λ → undersmooths, misses signal");
    println!("5. GCV/REML automatically find near-optimal λ");

    Ok(())
}

fn compute_training_rmse(gam: &GAM, n: usize) -> f64 {
    if let Some(dev) = gam.deviance {
        (dev / n as f64).sqrt()
    } else {
        0.0
    }
}

fn compute_test_rmse(gam: &GAM) -> Result<f64, Box<dyn std::error::Error>> {
    let n_pred = 200;
    let x_pred_vec: Vec<f64> = (0..n_pred).map(|i| i as f64 / (n_pred - 1) as f64).collect();
    let x_pred = Array1::from_vec(x_pred_vec.clone());
    let x_pred_matrix = x_pred.to_shape((n_pred, 1))?.to_owned();

    let predictions = gam.predict(&x_pred_matrix)?;

    let mse: f64 = x_pred_vec
        .iter()
        .zip(predictions.iter())
        .map(|(x_val, &y_pred)| {
            let y_true = (2.0 * std::f64::consts::PI * x_val).sin();
            (y_pred - y_true).powi(2)
        })
        .sum::<f64>() / n_pred as f64;

    Ok(mse.sqrt())
}
