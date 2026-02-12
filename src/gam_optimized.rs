//! Optimized GAM fitting with caching and improved matrix operations

#[cfg(feature = "blas")]
use crate::reml::ScaleParameterMethod;
use crate::{
    gam::{SmoothTerm, GAM},
    pirls::fit_pirls,
    smooth::{OptimizationMethod, SmoothingParameter},
    GAMError, Result,
};
use ndarray::{s, Array1, Array2};
use std::time::Instant;

/// Helper struct to cache computations during GAM fitting
struct FitCache {
    /// Full design matrix (n x p)
    design_matrix: Array2<f64>,
    /// Penalty matrices (one per smooth, each total_basis x total_basis)
    penalties: Vec<Array2<f64>>,
    /// Penalty scale factors (one per smooth)
    penalty_scales: Vec<f64>,
    /// X'X matrix (cached for reuse)
    xtx: Option<Array2<f64>>,
}

impl FitCache {
    /// Build cache from data and smooth terms
    fn new(x: &Array2<f64>, smooth_terms: &[SmoothTerm]) -> Result<Self> {
        let cache_start = Instant::now();
        let n = x.nrows();

        // Evaluate all basis functions (this is expensive, so cache it)
        let basis_start = Instant::now();
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
        }
        let basis_time = basis_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Basis evaluation: {:.2}ms",
                basis_time.as_secs_f64() * 1000.0
            );
        }

        // Build full design matrix using efficient slicing (not loops!)
        let mut full_design = Array2::zeros((n, total_basis));
        let mut col_offset = 0;

        for design in &design_matrices {
            let num_cols = design.ncols();
            // Use ndarray slicing - much faster than element-by-element
            full_design
                .slice_mut(s![.., col_offset..col_offset + num_cols])
                .assign(design);
            col_offset += num_cols;
        }

        // Compute penalty normalizations (cache these!)
        let penalty_start = Instant::now();
        let mut penalties = Vec::new();
        let mut penalty_scales = Vec::new();
        col_offset = 0;

        for (idx, smooth) in smooth_terms.iter().enumerate() {
            let num_basis = smooth.num_basis();
            let design = &design_matrices[idx];

            // Compute infinity norms (for mgcv-style normalization)
            let inf_norm_x = design
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let ma_xx = inf_norm_x * inf_norm_x;

            let inf_norm_s = (0..num_basis)
                .map(|i| {
                    (0..num_basis)
                        .map(|j| smooth.penalty[[i, j]].abs())
                        .sum::<f64>()
                })
                .fold(0.0f64, f64::max);

            let scale_factor = if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0
            };

            penalty_scales.push(scale_factor);

            // Build penalty matrix with normalization
            let mut penalty_full = Array2::zeros((total_basis, total_basis));

            // Use slicing for penalty block (faster than loops)
            let mut penalty_block = penalty_full.slice_mut(s![
                col_offset..col_offset + num_basis,
                col_offset..col_offset + num_basis
            ]);
            penalty_block.assign(&(&smooth.penalty * scale_factor));

            penalties.push(penalty_full);
            col_offset += num_basis;
        }
        let penalty_time = penalty_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Penalty computation: {:.2}ms",
                penalty_time.as_secs_f64() * 1000.0
            );
        }

        let cache_time = cache_start.elapsed();
        if std::env::var("MGCV_PROFILE").is_ok() {
            eprintln!(
                "[PROFILE] Total cache build: {:.2}ms",
                cache_time.as_secs_f64() * 1000.0
            );
        }

        Ok(FitCache {
            design_matrix: full_design,
            penalties,
            penalty_scales,
            xtx: None,
        })
    }

    /// Get or compute X'X (cached for efficiency)
    fn get_xtx(&mut self) -> &Array2<f64> {
        if self.xtx.is_none() {
            let xt = self.design_matrix.t().to_owned();
            self.xtx = Some(xt.dot(&self.design_matrix));
        }
        self.xtx.as_ref().unwrap()
    }
}

/// Initialize lambda with smart heuristic based on data
fn initialize_lambda_smart(y: &Array1<f64>, x: &Array2<f64>, penalty: &Array2<f64>) -> f64 {
    // Use a heuristic based on the ratio of signal variance to penalty norm
    let y_var = {
        let y_mean = y.sum() / y.len() as f64;
        y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / y.len() as f64
    };

    // Penalty norm (Frobenius)
    let penalty_norm = penalty.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Design matrix norm
    let x_norm = x.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Heuristic: lambda ~ y_var / (x_norm^2 / n) * penalty_norm
    let n = y.len() as f64;
    let lambda_init = (y_var * penalty_norm * n) / (x_norm * x_norm + 1e-10);

    // Clamp to reasonable range
    lambda_init.max(1e-6).min(1e6)
}

impl GAM {
    /// Optimized GAM fitting with caching and improved convergence
    ///
    /// Improvements over standard fit():
    /// - Caches design matrix and penalty computations
    /// - Uses ndarray slicing instead of loops for matrix construction
    /// - Better lambda initialization
    /// - Adaptive tolerance for early stopping
    /// - Caches X'X computation
    #[cfg(feature = "blas")]
    pub fn fit_optimized(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        self.fit_optimized_with_scale_method(
            x,
            y,
            opt_method,
            max_outer_iter,
            max_inner_iter,
            tolerance,
            crate::reml::ScaleParameterMethod::EDF,
        )
    }

    #[cfg(feature = "blas")]
    pub fn fit_optimized_with_scale_method(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
        scale_method: crate::reml::ScaleParameterMethod,
    ) -> Result<()> {
        self.fit_optimized_full(
            x,
            y,
            opt_method,
            max_outer_iter,
            max_inner_iter,
            tolerance,
            scale_method,
            None,
        )
    }

    #[cfg(feature = "blas")]
    pub fn fit_optimized_full(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
        scale_method: crate::reml::ScaleParameterMethod,
        algorithm: Option<crate::smooth::REMLAlgorithm>,
    ) -> Result<()> {
        let n = y.len();

        if x.nrows() != n {
            return Err(GAMError::DimensionMismatch(format!(
                "X has {} rows but y has {} elements",
                x.nrows(),
                n
            )));
        }

        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(format!(
                "X has {} columns but model has {} smooth terms",
                x.ncols(),
                self.smooth_terms.len()
            )));
        }

        // Build cache (design matrix, penalties, normalizations)
        let mut cache = FitCache::new(x, &self.smooth_terms)?;

        // Initialize smoothing parameters with chosen algorithm
        let mut smoothing_params = if let Some(algo) = algorithm {
            SmoothingParameter::new_with_algorithm(self.smooth_terms.len(), opt_method, algo)
        } else {
            SmoothingParameter::new(self.smooth_terms.len(), opt_method)
        };
        smoothing_params.scale_method = scale_method;

        // Smart initialization for lambda
        if !cache.penalties.is_empty() {
            let init_lambda = initialize_lambda_smart(y, x, &cache.penalties[0]);
            for lambda in &mut smoothing_params.lambda {
                *lambda = init_lambda;
            }
        }

        let mut weights = Array1::ones(n);

        let mut total_pirls_time = 0.0;
        let mut total_reml_time = 0.0;

        // Adaptive iteration count for Newton: use more iterations for larger/more complex problems
        let num_basis: usize = cache.design_matrix.ncols();
        let newton_max_iter = if num_basis >= n {
            50 // More iterations for overparameterized case
        } else if num_basis >= n / 2 {
            30 // Moderate iterations for k close to n
        } else {
            10 // Standard for well-posed problems
        };

        let is_newton = smoothing_params.reml_algorithm == crate::smooth::REMLAlgorithm::Newton;

        if is_newton {
            // Newton converges fully in a single call — no outer loop needed.
            // For Gaussian family the weights are constant, so PiRLS converges in 1 step
            // and Newton finds optimal lambda internally. Running a second outer iteration
            // would just re-derive the same lambda (wasting ~50% of REML time).
            //
            // Flow: PiRLS(init_lambda) → Newton(converge) → PiRLS(optimal_lambda) → done

            // Step 1: PiRLS with initial lambda
            let pirls_start = Instant::now();
            let pirls_result = fit_pirls(
                y,
                &cache.design_matrix,
                &smoothing_params.lambda,
                &cache.penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;
            weights = pirls_result.weights.clone();

            // Step 2: Newton optimization to convergence
            let reml_start = Instant::now();
            smoothing_params.optimize_with_beta(
                y,
                &cache.design_matrix,
                &weights,
                &cache.penalties,
                newton_max_iter,
                tolerance,
                Some(&pirls_result.coefficients),
            )?;
            total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

            // Step 3: Final PiRLS with optimal lambda
            let pirls_start = Instant::now();
            let final_result = fit_pirls(
                y,
                &cache.design_matrix,
                &smoothing_params.lambda,
                &cache.penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
            }

            self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
        } else {
            // Fellner-Schall: outer loop required — FS does one update step per call,
            // so we iterate PiRLS + FS until lambda converges.
            for outer_iter in 0..max_outer_iter {
                // PiRLS with current smoothing parameters
                let pirls_start = Instant::now();
                let pirls_result = fit_pirls(
                    y,
                    &cache.design_matrix,
                    &smoothing_params.lambda,
                    &cache.penalties,
                    self.family,
                    max_inner_iter,
                    tolerance,
                )?;
                total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                weights = pirls_result.weights.clone();

                // Update smoothing parameters (one FS step)
                let reml_start = Instant::now();
                let old_lambda = smoothing_params.lambda.clone();

                smoothing_params.optimize_with_beta(
                    y,
                    &cache.design_matrix,
                    &weights,
                    &cache.penalties,
                    newton_max_iter,
                    tolerance,
                    Some(&pirls_result.coefficients),
                )?;
                total_reml_time += reml_start.elapsed().as_secs_f64() * 1000.0;

                // Check convergence
                let max_lambda_change = old_lambda
                    .iter()
                    .zip(smoothing_params.lambda.iter())
                    .map(|(old, new)| (old.ln() - new.ln()).abs())
                    .fold(0.0f64, f64::max);

                let adaptive_tol = if outer_iter > 3 {
                    tolerance * 2.0
                } else {
                    tolerance
                };

                if max_lambda_change < adaptive_tol {
                    // Do final fit
                    let pirls_start = Instant::now();
                    let final_result = fit_pirls(
                        y,
                        &cache.design_matrix,
                        &smoothing_params.lambda,
                        &cache.penalties,
                        self.family,
                        max_inner_iter,
                        tolerance,
                    )?;
                    total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

                    if std::env::var("MGCV_PROFILE").is_ok() {
                        eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                        eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
                    }

                    self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
                    return Ok(());
                }
            }

            // Reached max iterations - use current fit
            let pirls_start = Instant::now();
            let final_result = fit_pirls(
                y,
                &cache.design_matrix,
                &smoothing_params.lambda,
                &cache.penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;
            total_pirls_time += pirls_start.elapsed().as_secs_f64() * 1000.0;

            if std::env::var("MGCV_PROFILE").is_ok() {
                eprintln!("[PROFILE] PiRLS iterations: {:.2}ms", total_pirls_time);
                eprintln!("[PROFILE] REML optimization: {:.2}ms", total_reml_time);
            }

            self.store_results(final_result, smoothing_params, y, &cache.design_matrix);
        }

        Ok(())
    }
}
