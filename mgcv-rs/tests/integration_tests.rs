use mgcv::prelude::*;
use ndarray::Array1;
use approx::assert_abs_diff_eq;

#[test]
fn test_cubic_spline_basis() {
    let x = Array1::linspace(0.0, 10.0, 100);
    let spline = CubicSpline::new(&x.view(), 10).unwrap();

    assert_eq!(spline.n_basis(), 10);

    let X = spline.basis_matrix(&x.view()).unwrap();
    assert_eq!(X.nrows(), 100);
    assert_eq!(X.ncols(), 10);

    let S = spline.penalty_matrix().unwrap();
    assert_eq!(S.nrows(), 10);
    assert_eq!(S.ncols(), 10);
}

#[test]
fn test_pspline_basis() {
    let x = Array1::linspace(0.0, 10.0, 100);
    let pspline = PSpline::new(&x.view(), 15, 3).unwrap();

    assert_eq!(pspline.n_basis(), 15);

    let X = pspline.basis_matrix(&x.view()).unwrap();
    assert_eq!(X.nrows(), 100);
    assert_eq!(X.ncols(), 15);
}

#[test]
fn test_gaussian_family() {
    let gaussian = Gaussian;

    let mu = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let eta = gaussian.link(&mu.view());

    // For Gaussian with identity link, eta = mu
    assert_abs_diff_eq!(eta[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(eta[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(eta[2], 3.0, epsilon = 1e-10);

    let mu_back = gaussian.linkinv(&eta.view());
    assert_abs_diff_eq!(mu_back[0], mu[0], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[1], mu[1], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[2], mu[2], epsilon = 1e-10);
}

#[test]
fn test_binomial_family() {
    let binomial = Binomial;

    let mu = Array1::from_vec(vec![0.1, 0.5, 0.9]);
    let eta = binomial.link(&mu.view());

    let mu_back = binomial.linkinv(&eta.view());
    assert_abs_diff_eq!(mu_back[0], mu[0], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[1], mu[1], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[2], mu[2], epsilon = 1e-10);

    assert!(binomial.validmu(&mu.view()));
}

#[test]
fn test_poisson_family() {
    let poisson = Poisson;

    let mu = Array1::from_vec(vec![1.0, 5.0, 10.0]);
    let eta = poisson.link(&mu.view());

    // For Poisson with log link, eta = log(mu)
    assert_abs_diff_eq!(eta[0], 1.0_f64.ln(), epsilon = 1e-10);
    assert_abs_diff_eq!(eta[1], 5.0_f64.ln(), epsilon = 1e-10);
    assert_abs_diff_eq!(eta[2], 10.0_f64.ln(), epsilon = 1e-10);

    let mu_back = poisson.linkinv(&eta.view());
    assert_abs_diff_eq!(mu_back[0], mu[0], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[1], mu[1], epsilon = 1e-10);
    assert_abs_diff_eq!(mu_back[2], mu[2], epsilon = 1e-10);
}

#[test]
fn test_smooth_fit() {
    // Test fitting a simple smooth
    let x = Array1::linspace(0.0, 10.0, 50);
    let y: Array1<f64> = x.mapv(|xi| (xi / 2.0).sin());

    let basis = CubicSpline::new(&x.view(), 10).unwrap();
    let mut smooth = Smooth::new(Box::new(basis));
    smooth.set_lambda(0.01);

    smooth.fit(&x.view(), &y.view(), None).unwrap();

    assert!(smooth.coefficients().is_some());
    assert!(smooth.fitted_values().is_some());

    let fitted = smooth.fitted_values().unwrap();
    assert_eq!(fitted.len(), y.len());
}

#[test]
fn test_gam_gaussian_fit() {
    // Test fitting a GAM with Gaussian family
    let n = 100;
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n);
    let y: Array1<f64> = x.mapv(|xi| xi.sin());

    let mut gam = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::Manual,
    );

    let basis = CubicSpline::new(&x.view(), 15).unwrap();
    gam.add_smooth("s(x)".to_string(), Box::new(basis), 0.1);

    gam.fit(&[x.view()], &y.view(), 50, 1e-6).unwrap();

    assert!(gam.coefficients().is_some());
    assert!(gam.fitted_values().is_some());

    let fitted = gam.fitted_values().unwrap();
    assert_eq!(fitted.len(), n);

    // Check that fit is reasonably close to true function
    let mut max_error = 0.0;
    for i in 0..n {
        let error = (fitted[i] - y[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    assert!(max_error < 0.2, "Max error: {}", max_error);
}

#[test]
fn test_gam_prediction() {
    let n = 50;
    let x = Array1::linspace(0.0, 10.0, n);
    let y: Array1<f64> = x.mapv(|xi| xi * 0.5);

    let mut gam = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::Manual,
    );

    let basis = CubicSpline::new(&x.view(), 10).unwrap();
    gam.add_smooth("s(x)".to_string(), Box::new(basis), 0.01);

    gam.fit(&[x.view()], &y.view(), 50, 1e-6).unwrap();

    // Predict on new data
    let x_new = Array1::linspace(0.0, 10.0, 100);
    let y_pred = gam.predict(&[x_new.view()]).unwrap();

    assert_eq!(y_pred.len(), 100);
}

#[test]
fn test_reml_score() {
    let x = Array1::linspace(0.0, 10.0, 50);
    let y: Array1<f64> = x.mapv(|xi| (xi / 2.0).sin());

    let spline = CubicSpline::new(&x.view(), 10).unwrap();
    let X = spline.basis_matrix(&x.view()).unwrap();
    let S = spline.penalty_matrix().unwrap();

    let score1 = REML::score(&X, &S, &y.view(), 0.01).unwrap();
    let score2 = REML::score(&X, &S, &y.view(), 1.0).unwrap();

    // Scores should be different for different lambdas
    assert!((score1 - score2).abs() > 1e-6);
}

#[test]
fn test_gcv_score() {
    let x = Array1::linspace(0.0, 10.0, 50);
    let y: Array1<f64> = x.mapv(|xi| (xi / 2.0).sin());

    let spline = CubicSpline::new(&x.view(), 10).unwrap();
    let X = spline.basis_matrix(&x.view()).unwrap();
    let S = spline.penalty_matrix().unwrap();

    let score1 = GCV::score(&X, &S, &y.view(), 0.01, 1.0).unwrap();
    let score2 = GCV::score(&X, &S, &y.view(), 1.0, 1.0).unwrap();

    // Scores should be different for different lambdas
    assert!((score1 - score2).abs() > 1e-6);
}

#[test]
fn test_gam_with_reml() {
    // Test GAM with automatic REML smoothing parameter selection
    let n = 100;
    let x = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, n);
    let y: Array1<f64> = x.mapv(|xi| xi.sin());

    let mut gam = GAM::new(
        Box::new(Gaussian),
        SmoothingMethod::REML,
    );

    let basis = CubicSpline::new(&x.view(), 15).unwrap();
    gam.add_smooth("s(x)".to_string(), Box::new(basis), 1.0);

    // This will optimize lambda automatically
    gam.fit(&[x.view()], &y.view(), 50, 1e-6).unwrap();

    let summary = gam.summary().unwrap();
    println!("REML optimized lambda: {}", summary.lambdas[0]);
    println!("EDF: {}", summary.edf);

    assert!(summary.lambdas[0] > 0.0);
    assert!(summary.edf > 0.0);
}
