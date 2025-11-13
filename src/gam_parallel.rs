//! Parallel GAM fitting using rayon for multi-threading
//!
//! This module provides parallelized GAM fitting for multi-dimensional problems.
//! Uses rayon for parallel basis evaluation and penalty construction.

use ndarray::{Array1, Array2, s, Axis};
use rayon::prelude::*;
use crate::{
    Result, GAMError,
    gam::{GAM, SmoothTerm},
    pirls::{fit_pirls, Family},
    smooth::{SmoothingParameter, OptimizationMethod},
};

/// Helper struct to cache computations with parallel construction
struct ParallelFitCache {
    /// Full design matrix (n x p)
    design_matrix: Array2<f64>,
    /// Penalty matrices (one per smooth, each total_basis x total_basis)
    penalties: Vec<Array2<f64>>,
    /// Penalty scale factors (one per smooth)
    penalty_scales: Vec<f64>,
}

impl ParallelFitCache {
    /// Build cache with parallel basis evaluation
    fn new(
        x: &Array2<f64>,
        smooth_terms: &[SmoothTerm],
    ) -> Result<Self> {
        let n = x.nrows();
        let num_terms = smooth_terms.len();

        // Evaluate all basis functions IN PARALLEL
        // This is where we get the biggest speedup for multi-dimensional GAMs
        let design_and_basis: Vec<(Array2<f64>, usize)> = (0..num_terms)
            .into_par_iter()
            .map(|i| {
                let x_col = x.column(i).to_owned();
                let basis_matrix = smooth_terms[i].evaluate(&x_col)?;
                let num_basis = smooth_terms[i].num_basis();
                Ok((basis_matrix, num_basis))
            })
            .collect::<Result<Vec<_>>>()?;

        // Compute total basis size
        let total_basis: usize = design_and_basis.iter().map(|(_, nb)| nb).sum();

        // Build full design matrix using efficient slicing
        let mut full_design = Array2::zeros((n, total_basis));
        let mut col_offset = 0;

        for (design, num_basis) in &design_and_basis {
            full_design.slice_mut(s![.., col_offset..col_offset + num_basis])
                .assign(design);
            col_offset += num_basis;
        }

        // Compute penalty normalizations IN PARALLEL
        let penalty_data: Vec<(Array2<f64>, f64)> = (0..num_terms)
            .into_par_iter()
            .map(|idx| {
                let smooth = &smooth_terms[idx];
                let num_basis = smooth.num_basis();
                let design = &design_and_basis[idx].0;

                // Compute infinity norms for mgcv-style normalization
                let inf_norm_x = design.axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                    .reduce(|| 0.0, f64::max);
                let ma_xx = inf_norm_x * inf_norm_x;

                let inf_norm_s = (0..num_basis)
                    .into_par_iter()
                    .map(|i| (0..num_basis).map(|j| smooth.penalty[[i, j]].abs()).sum::<f64>())
                    .reduce(|| 0.0, f64::max);

                let scale_factor = if inf_norm_s > 1e-10 {
                    ma_xx / inf_norm_s
                } else {
                    1.0
                };

                // Build scaled penalty block
                let penalty_scaled = &smooth.penalty * scale_factor;

                (penalty_scaled, scale_factor)
            })
            .collect();

        // Assemble full penalty matrices
        let mut penalties = Vec::new();
        col_offset = 0;

        for (idx, (penalty_scaled, scale_factor)) in penalty_data.iter().enumerate() {
            let num_basis = smooth_terms[idx].num_basis();
            let mut penalty_full = Array2::zeros((total_basis, total_basis));

            // Use slicing for penalty block
            penalty_full.slice_mut(s![
                col_offset..col_offset + num_basis,
                col_offset..col_offset + num_basis
            ]).assign(penalty_scaled);

            penalties.push(penalty_full);
            col_offset += num_basis;
        }

        let penalty_scales = penalty_data.iter().map(|(_, s)| *s).collect();

        Ok(ParallelFitCache {
            design_matrix: full_design,
            penalties,
            penalty_scales,
        })
    }
}

/// Initialize lambda with smart heuristic
fn initialize_lambda_smart(
    y: &Array1<f64>,
    x: &Array2<f64>,
    penalty: &Array2<f64>,
) -> f64 {
    let y_var = {
        let y_mean = y.sum() / y.len() as f64;
        y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / y.len() as f64
    };

    let penalty_norm = penalty.iter().map(|x| x * x).sum::<f64>().sqrt();
    let x_norm = x.iter().map(|x| x * x).sum::<f64>().sqrt();

    let n = y.len() as f64;
    let lambda_init = (y_var * penalty_norm * n) / (x_norm * x_norm + 1e-10);

    lambda_init.max(1e-6).min(1e6)
}

impl GAM {
    /// Parallel GAM fitting using rayon for multi-threading
    ///
    /// This version parallelizes:
    /// - Basis function evaluation (one thread per smooth term)
    /// - Penalty matrix construction (parallel norm computation)
    /// - Matrix operations where beneficial
    ///
    /// Best for multi-dimensional GAMs (d >= 3) where parallelization overhead is justified.
    pub fn fit_parallel(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        opt_method: OptimizationMethod,
        max_outer_iter: usize,
        max_inner_iter: usize,
        tolerance: f64,
    ) -> Result<()> {
        let n = y.len();

        if x.nrows() != n {
            return Err(GAMError::DimensionMismatch(
                format!("X has {} rows but y has {} elements", x.nrows(), n)
            ));
        }

        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(
                format!("X has {} columns but model has {} smooth terms",
                    x.ncols(), self.smooth_terms.len())
            ));
        }

        // Build cache with PARALLEL basis evaluation and penalty construction
        let cache = ParallelFitCache::new(x, &self.smooth_terms)?;

        // Initialize smoothing parameters with smart heuristic
        let mut smoothing_params = SmoothingParameter::new(
            self.smooth_terms.len(),
            opt_method
        );

        if !cache.penalties.is_empty() {
            let init_lambda = initialize_lambda_smart(y, x, &cache.penalties[0]);
            for lambda in &mut smoothing_params.lambda {
                *lambda = init_lambda;
            }
        }

        let mut weights = Array1::ones(n);

        // Outer loop: optimize smoothing parameters
        for _outer_iter in 0..max_outer_iter {
            // Inner loop: PiRLS with current smoothing parameters
            let pirls_result = fit_pirls(
                y,
                &cache.design_matrix,
                &smoothing_params.lambda,
                &cache.penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;

            weights = pirls_result.weights.clone();

            // Update smoothing parameters
            let old_lambda = smoothing_params.lambda.clone();

            smoothing_params.optimize(
                y,
                &cache.design_matrix,
                &weights,
                &cache.penalties,
                10,
                tolerance,
            )?;

            // Check convergence with adaptive tolerance
            let max_lambda_change = old_lambda.iter()
                .zip(smoothing_params.lambda.iter())
                .map(|(old, new)| ((old.ln() - new.ln()).abs()))
                .fold(0.0f64, f64::max);

            let adaptive_tol = if _outer_iter > 3 {
                tolerance * 2.0
            } else {
                tolerance
            };

            if max_lambda_change < adaptive_tol {
                // Converged - do final fit
                let final_result = fit_pirls(
                    y,
                    &cache.design_matrix,
                    &smoothing_params.lambda,
                    &cache.penalties,
                    self.family,
                    max_inner_iter,
                    tolerance,
                )?;

                self.store_results(final_result, smoothing_params, y);
                return Ok(());
            }
        }

        // Reached max iterations - use current fit
        let final_result = fit_pirls(
            y,
            &cache.design_matrix,
            &smoothing_params.lambda,
            &cache.penalties,
            self.family,
            max_inner_iter,
            tolerance,
        )?;

        self.store_results(final_result, smoothing_params, y);
        Ok(())
    }
}
