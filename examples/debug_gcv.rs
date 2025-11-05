//! Debug GCV to understand RSS and EDF behavior

use mgcv_rust::*;
use ndarray::Array1;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Debugging GCV Components ===\n");

    let n = 30;
    let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let true_y = (2.0 * std::f64::consts::PI * xi).sin();
            let noise = 0.5 * ((i as f64 * 0.7).sin() * 2.0 - 1.0);
            true_y + noise
        })
        .collect();

    let x = Array1::from_vec(x_data);
    let y = Array1::from_vec(y_data);

    let smooth = gam::SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0)?;
    let basis_matrix = smooth.evaluate(&x)?;
    let penalty = &smooth.penalty;
    let p = basis_matrix.ncols();

    println!("n = {}, p = {}\n", n, p);
    println!("{}", "=".repeat(70));
    println!(" λ value    | RSS      | EDF    | n-EDF  | GCV");
    println!("{}", "=".repeat(70));

    use mgcv_rust::linalg::{solve, inverse};
    use ndarray::Array2;

    for log_lambda in -6..3 {
        let lambda = 10.0_f64.powi(log_lambda);

        // Compute manually to get components
        let xtx = basis_matrix.t().dot(&basis_matrix);
        let a = &xtx + &(penalty * lambda);
        let xty = basis_matrix.t().dot(&y);

        let beta = solve(a.clone(), xty)?;
        let fitted = basis_matrix.dot(&beta);
        let residuals: Array1<f64> = y.iter().zip(fitted.iter())
            .map(|(yi, fi)| yi - fi)
            .collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();

        // Compute EDF
        let a_inv = inverse(&a)?;
        let mut xtwfull = Array2::zeros((p, n));
        for i in 0..n {
            for j in 0..p {
                xtwfull[[j, i]] = basis_matrix[[i, j]];
            }
        }

        let h_temp = basis_matrix.dot(&a_inv);
        let influence = h_temp.dot(&xtwfull);

        let mut edf = 0.0;
        for i in 0..n {
            edf += influence[[i, i]];
        }

        let gcv = (n as f64) * rss / ((n as f64) - edf).powi(2);

        println!(" {:.6} | {:.4} | {:.2} | {:.2}  | {:.4}",
            lambda, rss, edf, n as f64 - edf, gcv);
    }

    println!("{}", "=".repeat(70));
    println!("\nExpected behavior:");
    println!("- As λ → 0: EDF → p ({}), RSS → 0", p);
    println!("- As λ → ∞: EDF → 0, RSS → large");
    println!("- GCV should have minimum at intermediate λ");

    Ok(())
}
