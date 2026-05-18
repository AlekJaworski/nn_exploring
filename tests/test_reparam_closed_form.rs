use mgcv_rust::block_penalty::BlockPenalty;
use mgcv_rust::pirls::Family;
use mgcv_rust::reml::{
    compute_xtwx_dispatch, compute_xtwy_dispatch, reml_gradient_mgcv_exact_closed_form,
    reml_gradient_mgcv_exact_ift, reml_hessian_mgcv_exact_closed_form,
    reml_hessian_mgcv_exact_ift,
};
use ndarray::{Array1, Array2};

fn synthetic_gaussian_system() -> (Array1<f64>, Array2<f64>, Array1<f64>, Vec<f64>, Vec<BlockPenalty>) {
    let n = 80;
    let d = 2;
    let k = 5;
    let p = d * k;

    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t0 = (i as f64 + 0.5) / n as f64;
        let t1 = ((i * 7 % n) as f64 + 0.25) / n as f64;
        for kk in 0..k {
            x[[i, kk]] = ((kk + 1) as f64 * std::f64::consts::PI * t0).sin();
            x[[i, k + kk]] = ((kk + 1) as f64 * std::f64::consts::PI * t1).cos();
        }
        y[i] = (2.0 * std::f64::consts::PI * t0).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * t1).cos();
    }

    let mut penalties = Vec::new();
    for dim in 0..d {
        let mut s = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            s[[r, r]] = 2.0;
            if r > 0 {
                s[[r, r - 1]] = -1.0;
                s[[r - 1, r]] = -1.0;
            }
        }
        penalties.push(BlockPenalty::new(s, dim * k, p));
    }

    (y, x, Array1::ones(n), vec![0.7, 12.0], penalties)
}

#[test]
fn closed_form_gradient_hessian_use_reparam_basis() {
    let (y, x, w, lambdas, penalties) = synthetic_gaussian_system();
    let xtwx = compute_xtwx_dispatch(None, &x, &w);
    let xtwy = compute_xtwy_dispatch(None, &x, &w, &y);

    std::env::set_var("MGCV_REPARAM", "1");

    let grad_closed = reml_gradient_mgcv_exact_closed_form(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Some(&xtwx),
        Some(&xtwy),
        Family::Gaussian,
    )
    .expect("closed-form gradient failed");
    let grad_closed_no_cache = reml_gradient_mgcv_exact_closed_form(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        None,
        Family::Gaussian,
    )
    .expect("closed-form no-cache gradient failed");
    let grad_ift = reml_gradient_mgcv_exact_ift(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Some(&xtwx),
        Family::Gaussian,
        None,
        None,
    )
    .expect("IFT gradient failed");

    let hess_closed = reml_hessian_mgcv_exact_closed_form(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Some(&xtwx),
        Some(&xtwy),
        Family::Gaussian,
    )
    .expect("closed-form Hessian failed");
    let hess_closed_no_cache = reml_hessian_mgcv_exact_closed_form(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        None,
        None,
        Family::Gaussian,
    )
    .expect("closed-form no-cache Hessian failed");
    let _hess_ift = reml_hessian_mgcv_exact_ift(
        &y,
        &x,
        &w,
        &lambdas,
        &penalties,
        Some(&xtwx),
        Family::Gaussian,
        None,
    )
    .expect("IFT Hessian failed");

    std::env::remove_var("MGCV_REPARAM");

    for (got, want) in grad_closed.iter().zip(grad_ift.iter()) {
        assert!((got - want).abs() < 1e-6, "gradient mismatch: {got} vs {want}");
    }
    for (got, want) in grad_closed.iter().zip(grad_closed_no_cache.iter()) {
        assert!((got - want).abs() < 1e-12, "cached gradient mismatch: {got} vs {want}");
    }
    for (got, want) in hess_closed.iter().zip(hess_closed_no_cache.iter()) {
        assert!((got - want).abs() < 1e-12, "cached hessian mismatch: {got} vs {want}");
    }
    for value in hess_closed.iter() {
        assert!(value.is_finite(), "non-finite hessian value: {value}");
    }
    for i in 0..hess_closed.nrows() {
        for j in 0..hess_closed.ncols() {
            assert!(
                (hess_closed[[i, j]] - hess_closed[[j, i]]).abs() < 1e-10,
                "hessian is not symmetric at ({i}, {j})"
            );
        }
    }
}
