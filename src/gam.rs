//! Main GAM model structure and fitting

use crate::{
    basis::{BasisFunction, BoundaryCondition, CubicRegressionSpline, CubicSpline},
    block_penalty::BlockPenalty,
    penalty::compute_penalty,
    pirls::{fit_pirls, Family},
    smooth::{OptimizationMethod, SmoothingParameter},
    GAMError, Result,
};
use ndarray::{Array1, Array2};

/// A smooth term in a GAM
pub struct SmoothTerm {
    /// Name of the covariate
    pub name: String,
    /// Basis function
    pub basis: Box<dyn BasisFunction>,
    /// Penalty matrix (may be constrained)
    pub penalty: Array2<f64>,
    /// Smoothing parameter
    pub lambda: f64,
    /// Constraint matrix Q for identifiability (optional)
    /// If present, transforms unconstrained basis to constrained: X_constrained = X * Q
    pub constraint_matrix: Option<Array2<f64>>,
}

impl SmoothTerm {
    /// Create a new smooth term with cubic spline basis (evenly-spaced knots)
    pub fn cubic_spline(name: String, num_basis: usize, x_min: f64, x_max: f64) -> Result<Self> {
        let basis =
            CubicSpline::with_num_knots(x_min, x_max, num_basis - 2, BoundaryCondition::Natural);

        let knots = basis.knots().unwrap();
        let penalty = compute_penalty("cubic", num_basis, Some(knots), 1)?;

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
            constraint_matrix: None, // No constraint for regular cubic splines
        })
    }

    /// Create a new smooth term with quantile-based knots (like mgcv)
    /// Uses B-spline basis
    /// Cubic B-spline basis matching mgcv's `bs="bs"` (de Boor recursion,
    /// order-4 basis with even-spaced knots over data range, 2nd-derivative
    /// integrated penalty). Replaces the older natural-cubic placeholder
    /// that previously sat under this name — with the de Boor basis the
    /// span and predictions match mgcv to working precision.
    pub fn cubic_spline_quantile(
        name: String,
        num_basis: usize,
        x_data: &Array1<f64>,
    ) -> Result<Self> {
        let basis = crate::basis::BSplineBasis::with_data_range(x_data, num_basis);
        let penalty = basis.second_derivative_penalty();

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
            constraint_matrix: None,
        })
    }

    /// Create a new smooth term with cubic regression splines (cr basis, like mgcv default)
    /// Uses cardinal natural cubic spline basis with quantile-based knots
    /// NOTE: Uses k basis functions (not k-1) to match mgcv's approach
    pub fn cr_spline_quantile(
        name: String,
        num_basis: usize,
        x_data: &Array1<f64>,
    ) -> Result<Self> {
        let basis = CubicRegressionSpline::with_quantile_knots(x_data, num_basis);
        let knots = basis.knots().unwrap();

        // Compute penalty (k x k) - mgcv keeps all k basis functions
        let penalty = compute_penalty("cr", num_basis, Some(knots), 1)?;

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
            constraint_matrix: None, // No pre-transformation (mgcv handles constraints during solving)
        })
    }

    /// Create a new smooth term with cubic regression splines (evenly-spaced knots)
    /// NOTE: Uses k basis functions (not k-1) to match mgcv's approach
    pub fn cr_spline(name: String, num_basis: usize, x_min: f64, x_max: f64) -> Result<Self> {
        let basis = CubicRegressionSpline::with_num_knots(x_min, x_max, num_basis);
        let knots = basis.knots().unwrap();

        // Compute penalty (k x k) - mgcv keeps all k basis functions
        let penalty = compute_penalty("cr", num_basis, Some(knots), 1)?;

        Ok(Self {
            name,
            basis: Box::new(basis),
            penalty,
            lambda: 1.0,
            constraint_matrix: None, // No pre-transformation (mgcv handles constraints during solving)
        })
    }

    /// Evaluate the basis functions for this smooth term (fit-time path).
    /// If constraint matrix is present, applies it to get constrained basis.
    pub fn evaluate(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let basis_unconstrained = self.basis.evaluate(x)?;
        if let Some(ref q_matrix) = self.constraint_matrix {
            Ok(basis_unconstrained.dot(q_matrix))
        } else {
            Ok(basis_unconstrained)
        }
    }

    /// Predict-time evaluation. Uses the basis's `evaluate_for_predict`
    /// which for most bases is identical to `evaluate`, but B-splines
    /// extrapolate linearly past the inner-knot range to match mgcv's
    /// `Predict.matrix.pspline.smooth`.
    pub fn evaluate_for_predict(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let basis_unconstrained = self.basis.evaluate_for_predict(x)?;
        if let Some(ref q_matrix) = self.constraint_matrix {
            Ok(basis_unconstrained.dot(q_matrix))
        } else {
            Ok(basis_unconstrained)
        }
    }

    /// Get number of basis functions (after constraint if applied)
    pub fn num_basis(&self) -> usize {
        // If constraint is applied, return the constrained dimension
        if let Some(ref q_matrix) = self.constraint_matrix {
            q_matrix.ncols() // k-1 for sum-to-zero constraint
        } else {
            self.basis.num_basis() // k for unconstrained
        }
    }

    /// mgcv-exact path: normalise the penalty using mgcv's pre-centring
    /// recipe (smooth.r:3766-3773), then apply the standard sum-to-zero
    /// Z. Order matters: mgcv computes `maS = ||S_raw||_∞ / ||X_raw||_∞²`
    /// on the *uncentred* basis and penalty, then sets `S_norm = S_raw / maS`
    /// before passing to absorb.cons. Our default mode does this *after*
    /// Z, which gives a different scale factor.
    pub fn apply_mgcv_normalisation_then_centring(
        &mut self,
        x_data: &Array1<f64>,
    ) -> Result<()> {
        if self.constraint_matrix.is_some() {
            return Ok(());
        }
        let basis_raw = self.basis.evaluate(x_data)?;
        let k = basis_raw.ncols();
        if k <= 1 {
            return Ok(());
        }
        // Step 1: mgcv-style normalisation on RAW basis and penalty.
        // ma_xx = ||X_raw||_∞² (max row sum of |X_raw|, squared)
        let inf_norm_x: f64 = basis_raw
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let ma_xx = inf_norm_x * inf_norm_x;
        // ||S_raw||_∞ — for symmetric S equals max col abs sum
        let inf_norm_s: f64 = (0..k)
            .map(|i| (0..k).map(|j| self.penalty[[i, j]].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let scale_factor = if inf_norm_s > 1e-10 {
            ma_xx / inf_norm_s
        } else {
            1.0
        };
        // Apply scale to RAW penalty
        self.penalty = &self.penalty * scale_factor;

        // Step 2: standard sum-to-zero Z (Householder of column means).
        let c_row = basis_raw.sum_axis(ndarray::Axis(0)) / (basis_raw.nrows() as f64);
        let c_norm = c_row.dot(&c_row).sqrt();
        let z_kkm1: Array2<f64> = if c_norm < 1e-30 {
            let mut z = Array2::<f64>::zeros((k, k - 1));
            for j in 1..k {
                z[[j, j - 1]] = 1.0;
            }
            z
        } else {
            let mut v = c_row.clone();
            let sign = if c_row[0] >= 0.0 { 1.0 } else { -1.0 };
            v[0] += sign * c_norm;
            let v_norm_sq = v.dot(&v);
            let mut z = Array2::<f64>::zeros((k, k - 1));
            for j in 1..k {
                let coef = 2.0 * v[j] / v_norm_sq;
                for i in 0..k {
                    let e_ij = if i == j { 1.0 } else { 0.0 };
                    z[[i, j - 1]] = e_ij - coef * v[i];
                }
            }
            z
        };
        // Step 3: penalty becomes Z' S_norm_raw Z (S already pre-scaled).
        let new_penalty = z_kkm1.t().dot(&self.penalty).dot(&z_kkm1);
        self.penalty = new_penalty;
        self.constraint_matrix = Some(z_kkm1);
        Ok(())
    }

    /// mgcv-exact path (DEPRECATED, see apply_mgcv_normalisation_then_centring):
    /// sum-to-zero centring **followed by** nat.param (smooth.r:15-128,
    /// type=2, unit.fnorm=FALSE). mgcv only invokes nat.param when
    /// `diagonal.penalty=TRUE` is requested — which gam()/bam() do NOT
    /// use by default. Kept here for future experimentation.
    pub fn apply_sum_to_zero_centering_mgcv_exact(
        &mut self,
        x_data: &Array1<f64>,
    ) -> Result<()> {
        if self.constraint_matrix.is_some() {
            return Ok(());
        }
        let basis_raw = self.basis.evaluate(x_data)?;
        let k = basis_raw.ncols();
        if k <= 1 {
            return Ok(());
        }

        // Step 1: standard sum-to-zero Z (Householder of column means;
        // matches mgcv's qr(t(colMeans(X))) up to column signs, and
        // since the resulting Z'SZ is invariant to those signs, the
        // downstream nat.param gives the same diagonal penalty).
        let c_row = basis_raw.sum_axis(ndarray::Axis(0)) / (basis_raw.nrows() as f64);
        let c_norm = c_row.dot(&c_row).sqrt();
        let z_kkm1: Array2<f64>;
        if c_norm < 1e-30 {
            // Column means already zero — drop a redundant col.
            let mut z = Array2::<f64>::zeros((k, k - 1));
            for j in 1..k {
                z[[j, j - 1]] = 1.0;
            }
            z_kkm1 = z;
        } else {
            let mut v = c_row.clone();
            let sign = if c_row[0] >= 0.0 { 1.0 } else { -1.0 };
            v[0] += sign * c_norm;
            let v_norm_sq = v.dot(&v);
            let mut z = Array2::<f64>::zeros((k, k - 1));
            for j in 1..k {
                let coef = 2.0 * v[j] / v_norm_sq;
                for i in 0..k {
                    let e_ij = if i == j { 1.0 } else { 0.0 };
                    z[[i, j - 1]] = e_ij - coef * v[i];
                }
            }
            z_kkm1 = z;
        }

        // Step 2: nat.param(type=2, unit.fnorm=FALSE) on (X_centred, Z'SZ)
        let s_centred = z_kkm1.t().dot(&self.penalty).dot(&z_kkm1); // (k-1) × (k-1)
        let x_centred = basis_raw.dot(&z_kkm1); // n × (k-1)

        // Eigendecompose S_centred. ndarray-linalg returns eigenvalues
        // in ASCENDING order; mgcv's eigen() returns DESCENDING. We
        // reverse so eigenvalue 0 is the largest.
        use ndarray_linalg::{Eigh, UPLO};
        let (evals_asc, evecs_asc) = s_centred
            .eigh(UPLO::Upper)
            .map_err(|e| GAMError::LinAlgError(format!("S_centred eigh failed: {:?}", e)))?;
        let p = evals_asc.len(); // k-1
        let mut evals = vec![0.0f64; p];
        for i in 0..p {
            evals[i] = evals_asc[p - 1 - i];
        }
        let mut evecs = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            for i in 0..p {
                evecs[[i, j]] = evecs_asc[[i, p - 1 - j]];
            }
        }

        // Determine rank: count eigenvalues > tol * max
        let max_eig = evals[0].max(0.0);
        let tol = f64::EPSILON.powf(0.8);
        let rank: usize = evals.iter().filter(|&&v| v > max_eig * tol).count();

        // Build E vector (length p). Initialised to 1's; first `rank`
        // entries are sqrt(eigenvalue). Null entries (rank+1..p) get
        // assigned later from col norms.
        let mut e_vec = vec![1.0f64; p];
        for i in 0..rank {
            e_vec[i] = evals[i].sqrt().max(1e-300);
        }

        // Rotate basis: X' = X_centred * U
        let x_rot = x_centred.dot(&evecs); // n × p

        // Compute col_norm[j] = sum(X_rot[:,j]^2) / E[j]^2
        let mut col_norms = vec![0.0f64; p];
        for j in 0..p {
            let s: f64 = x_rot.column(j).iter().map(|v| v * v).sum();
            col_norms[j] = s / (e_vec[j] * e_vec[j]);
        }

        // av_norm = mean over the rank penalised columns
        let null_exists = rank < p;
        let av_norm: f64 = if rank > 0 {
            col_norms[..rank].iter().sum::<f64>() / (rank as f64)
        } else {
            1.0
        };
        if null_exists && av_norm > 0.0 {
            for j in rank..p {
                e_vec[j] = (col_norms[j] / av_norm).sqrt().max(1e-300);
            }
        }

        // M = U / E (column-wise division of U by E)
        let mut m_mat = evecs.clone();
        for j in 0..p {
            for i in 0..p {
                m_mat[[i, j]] /= e_vec[j];
            }
        }

        // Combined transformation Z_total = Z * M  (k × p)
        let z_total = z_kkm1.dot(&m_mat);

        // New penalty: diag(1's for rank, 0's for null)
        let mut new_penalty = Array2::<f64>::zeros((p, p));
        for i in 0..rank {
            new_penalty[[i, i]] = 1.0;
        }

        self.penalty = new_penalty;
        self.constraint_matrix = Some(z_total);
        Ok(())
    }

    /// Apply mgcv's sum-to-zero identifiability constraint to the basis.
    /// Computes a (k × k-1) reparameterization Z such that the column sums
    /// of (X_raw @ Z) over the training data are exactly zero. The basis
    /// becomes (k-1)-dimensional and the penalty is transformed
    /// `S_new = Z' S Z`.
    ///
    /// Construction: Householder reflection of the column-sums vector
    /// `c = 1' X_raw`, taking the last (k-1) columns of the reflector.
    /// This is the same `absorb.cons` step mgcv applies to every smooth
    /// by default, so that the global intercept absorbs the mean.
    ///
    /// Idempotent: a no-op if the constraint is already set.
    pub fn apply_sum_to_zero_centering(&mut self, x_data: &Array1<f64>) -> Result<()> {
        if self.constraint_matrix.is_some() {
            return Ok(());
        }
        let basis_raw = self.basis.evaluate(x_data)?;
        let k = basis_raw.ncols();
        if k <= 1 {
            // 1-column basis can't be reduced; leave unconstrained.
            return Ok(());
        }
        let c_row = basis_raw.sum_axis(ndarray::Axis(0)); // length k

        // Householder reflector that maps c → ±||c|| e_0. Its last
        // (k-1) columns span the null space of c, which is exactly Z.
        let c_norm = c_row.dot(&c_row).sqrt();
        if c_norm < 1e-30 {
            // c is already zero; basis happens to be sum-to-zero already.
            // Drop a redundant column by keeping the last (k-1) of I.
            let mut z = Array2::<f64>::zeros((k, k - 1));
            for j in 1..k {
                z[[j, j - 1]] = 1.0;
            }
            let new_penalty = z.t().dot(&self.penalty).dot(&z);
            self.penalty = new_penalty;
            self.constraint_matrix = Some(z);
            return Ok(());
        }
        let mut v = c_row.clone();
        let sign = if c_row[0] >= 0.0 { 1.0 } else { -1.0 };
        v[0] += sign * c_norm;
        let v_norm_sq = v.dot(&v);

        let mut z = Array2::<f64>::zeros((k, k - 1));
        // For j in 1..k, the j-th column of the Householder matrix is
        // e_j - (2 v_j / v'v) v. Stored as the (j-1)-th column of Z.
        for j in 1..k {
            let coef = 2.0 * v[j] / v_norm_sq;
            for i in 0..k {
                let e_ij = if i == j { 1.0 } else { 0.0 };
                z[[i, j - 1]] = e_ij - coef * v[i];
            }
        }

        let new_penalty = z.t().dot(&self.penalty).dot(&z);
        self.penalty = new_penalty;
        self.constraint_matrix = Some(z);
        Ok(())
    }
}

/// Generalized Additive Model
pub struct GAM {
    /// Smooth terms
    pub smooth_terms: Vec<SmoothTerm>,
    /// Distribution family
    pub family: Family,
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Fitted values
    pub fitted_values: Option<Array1<f64>>,
    /// Linear predictor
    pub linear_predictor: Option<Array1<f64>>,
    /// Smoothing parameters
    pub smoothing_params: Option<SmoothingParameter>,
    /// IRLS weights
    pub weights: Option<Array1<f64>>,
    /// Deviance
    pub deviance: Option<f64>,
    /// Design matrix (predictor matrix)
    pub design_matrix: Option<Array2<f64>>,
    /// Whether model has been fitted
    pub fitted: bool,
    /// When true, the fit pipeline uses mgcv-faithful basis, penalty,
    /// and score formulas (no per-smooth penalty normalisation, mgcv's
    /// QR-based sum-to-zero Z, and gam.fit3.r:621 REML). Default false
    /// preserves the current fast/working path.
    pub mgcv_exact: bool,
}

impl GAM {
    /// Create a new GAM with specified family
    pub fn new(family: Family) -> Self {
        Self {
            smooth_terms: Vec::new(),
            family,
            coefficients: None,
            fitted_values: None,
            linear_predictor: None,
            smoothing_params: None,
            weights: None,
            deviance: None,
            design_matrix: None,
            fitted: false,
            mgcv_exact: false,
        }
    }

    /// Add a smooth term to the model
    pub fn add_smooth(&mut self, smooth: SmoothTerm) {
        self.smooth_terms.push(smooth);
    }

    /// Fit the GAM using PiRLS with automatic smoothing parameter selection
    ///
    /// # Arguments
    /// * `x` - Covariate matrix (each column is one covariate)
    /// * `y` - Response vector
    /// * `opt_method` - Optimization method for smoothing parameters (REML or GCV)
    /// * `max_outer_iter` - Maximum iterations for smoothing parameter optimization
    /// * `max_inner_iter` - Maximum iterations for PiRLS
    /// * `tolerance` - Convergence tolerance
    pub fn fit(
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

        // Apply mgcv's sum-to-zero centering before evaluating bases
        // (idempotent — only kicks in on first fit). Layout matches
        // FitCache::new in gam_optimized.rs: [intercept | smooths...].
        for (i, smooth) in self.smooth_terms.iter_mut().enumerate() {
            let x_col = x.column(i).to_owned();
            smooth.apply_sum_to_zero_centering(&x_col)?;
        }

        // Construct design matrix by evaluating all basis functions
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut total_basis = 0;

        for (i, smooth) in self.smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
        }

        // Combine all design matrices: [1 | smooth_1 | smooth_2 | ...]
        let total_cols = total_basis + 1;
        let mut full_design = Array2::zeros((n, total_cols));
        full_design.column_mut(0).fill(1.0);
        let mut col_offset = 1;

        for design in &design_matrices {
            let num_cols = design.ncols();
            full_design
                .slice_mut(ndarray::s![.., col_offset..col_offset + num_cols])
                .assign(design);
            col_offset += num_cols;
        }

        // Construct penalty matrices with mgcv-style normalization
        // (block-diagonal, intercept column unpenalized).
        let mut penalties: Vec<Array2<f64>> = Vec::new();
        col_offset = 1;

        for (idx, smooth) in self.smooth_terms.iter().enumerate() {
            let num_basis = smooth.num_basis();

            // Get the basis matrix for this smooth term
            let design = &design_matrices[idx];

            // Compute mgcv's penalty normalization factor:
            // maXX = ||X||_inf^2 (infinity norm squared = max row sum squared)
            // maS = ||S||_inf / maXX (infinity norm of S divided by maXX)
            // S_rescaled = S / maS = S * maXX / ||S||_inf

            // Compute infinity norm of design matrix (max absolute row sum)
            let inf_norm_x = design
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
                .fold(0.0f64, f64::max);
            let ma_xx = inf_norm_x * inf_norm_x;

            // Compute infinity norm of penalty matrix (max absolute row sum)
            let inf_norm_s = (0..num_basis)
                .map(|i| {
                    (0..num_basis)
                        .map(|j| smooth.penalty[[i, j]].abs())
                        .sum::<f64>()
                })
                .fold(0.0f64, f64::max);

            // Apply normalization: maS = ||S||_inf / maXX, S_new = S / maS
            let scale_factor = if inf_norm_s > 1e-10 {
                ma_xx / inf_norm_s
            } else {
                1.0 // Avoid division by zero for degenerate penalties
            };

            let mut penalty_full = Array2::zeros((total_cols, total_cols));

            // Place this smooth's normalized penalty in the appropriate block using slicing
            penalty_full
                .slice_mut(ndarray::s![
                    col_offset..col_offset + num_basis,
                    col_offset..col_offset + num_basis
                ])
                .assign(&(&smooth.penalty * scale_factor));

            penalties.push(penalty_full);
            col_offset += num_basis;
        }

        // Convert dense p×p penalties to BlockPenalty for the new API
        let block_penalties: Vec<BlockPenalty> = penalties
            .into_iter()
            .map(|p| BlockPenalty::new(p.clone(), 0, p.nrows()))
            .collect();

        // Initialize smoothing parameters
        let mut smoothing_params = SmoothingParameter::new(self.smooth_terms.len(), opt_method);

        // Outer loop: optimize smoothing parameters
        let mut weights;

        for _outer_iter in 0..max_outer_iter {
            // Inner loop: PiRLS with current smoothing parameters
            let pirls_result = fit_pirls(
                y,
                &full_design,
                &smoothing_params.lambda,
                &block_penalties,
                self.family,
                max_inner_iter,
                tolerance,
            )?;

            weights = pirls_result.weights;

            // Update smoothing parameters using REML/GCV
            let old_lambda = smoothing_params.lambda.clone();

            smoothing_params.optimize(
                y,
                &full_design,
                &weights,
                &block_penalties,
                10, // max iterations for lambda optimization
                tolerance,
            )?;

            // Check convergence of smoothing parameters
            let max_lambda_change = old_lambda
                .iter()
                .zip(smoothing_params.lambda.iter())
                .map(|(old, new)| (old.ln() - new.ln()).abs())
                .fold(0.0f64, f64::max);

            if max_lambda_change < tolerance {
                // Converged - do final fit
                let final_result = fit_pirls(
                    y,
                    &full_design,
                    &smoothing_params.lambda,
                    &block_penalties,
                    self.family,
                    max_inner_iter,
                    tolerance,
                )?;

                self.coefficients = Some(final_result.coefficients);
                self.fitted_values = Some(final_result.fitted_values);
                self.linear_predictor = Some(final_result.linear_predictor);
                self.weights = Some(final_result.weights);
                self.design_matrix = Some(full_design.clone());

                // Recompute deviance from fitted values to ensure consistency
                // Note: fit_pirls may return incorrect deviance due to penalty scaling
                let fitted = self.fitted_values.as_ref().unwrap();
                let mut correct_deviance = 0.0;
                for i in 0..y.len() {
                    correct_deviance += (y[i] - fitted[i]).powi(2);
                }

                self.deviance = Some(correct_deviance);
                self.smoothing_params = Some(smoothing_params);
                self.fitted = true;

                return Ok(());
            }
        }

        // Reached max outer iterations - use current fit
        let final_result = fit_pirls(
            y,
            &full_design,
            &smoothing_params.lambda,
            &block_penalties,
            self.family,
            max_inner_iter,
            tolerance,
        )?;

        self.coefficients = Some(final_result.coefficients);
        self.fitted_values = Some(final_result.fitted_values);
        self.linear_predictor = Some(final_result.linear_predictor);
        self.weights = Some(final_result.weights);
        self.design_matrix = Some(full_design.clone());

        // Recompute deviance from fitted values to ensure consistency
        let fitted = self.fitted_values.as_ref().unwrap();
        let mut correct_deviance = 0.0;
        for i in 0..y.len() {
            correct_deviance += (y[i] - fitted[i]).powi(2);
        }

        self.deviance = Some(correct_deviance);
        self.smoothing_params = Some(smoothing_params);
        self.fitted = true;

        Ok(())
    }

    /// Predict response for new data
    /// Build the design matrix `[1 | smooth_1 | smooth_2 | ...]` at the
    /// given `x`, using each smooth's predict-time basis (B-splines get
    /// linear extrapolation past the inner-knot range; cr-splines pass
    /// through unchanged). This is mgcv's `predict.gam(..., type =
    /// "lpmatrix")` and is the shared building block for `predict`,
    /// posterior sampling, and confidence-interval computation.
    pub fn build_lpmatrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(GAMError::InvalidParameter(
                "Model has not been fitted yet".to_string(),
            ));
        }
        let n = x.nrows();
        if x.ncols() != self.smooth_terms.len() {
            return Err(GAMError::DimensionMismatch(format!(
                "X has {} columns but model has {} smooth terms",
                x.ncols(),
                self.smooth_terms.len()
            )));
        }
        let mut design_matrices: Vec<Array2<f64>> = Vec::new();
        let mut total_basis = 0;
        for (i, smooth) in self.smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let basis_matrix = smooth.evaluate_for_predict(&x_col)?;
            total_basis += smooth.num_basis();
            design_matrices.push(basis_matrix);
        }
        let total_cols = total_basis + 1;
        let mut full_design = Array2::zeros((n, total_cols));
        full_design.column_mut(0).fill(1.0);
        let mut col_offset = 1;
        for design in &design_matrices {
            let num_cols = design.ncols();
            full_design
                .slice_mut(ndarray::s![.., col_offset..col_offset + num_cols])
                .assign(design);
            col_offset += num_cols;
        }
        Ok(full_design)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let full_design = self.build_lpmatrix(x)?;
        let coefficients = self.coefficients.as_ref().unwrap();

        // Compute linear predictor
        let eta = full_design.dot(coefficients);

        // Apply inverse link
        let predictions: Array1<f64> = eta.iter().map(|&e| self.family.inverse_link(e)).collect();

        Ok(predictions)
    }

    /// Get effective degrees of freedom
    pub fn edf(&self) -> Option<f64> {
        if !self.fitted {
            return None;
        }

        // Simplified: count non-zero coefficients
        // A proper implementation would compute tr(influence matrix)
        self.coefficients
            .as_ref()
            .map(|coef| coef.iter().filter(|&&c| c.abs() > 1e-10).count() as f64)
    }

    /// Store fit results (used by fit_optimized and fit_parallel)
    #[allow(dead_code)]
    pub(crate) fn store_results(
        &mut self,
        pirls_result: crate::pirls::PiRLSResult,
        smoothing_params: SmoothingParameter,
        y: &Array1<f64>,
        design_matrix: &Array2<f64>,
    ) {
        self.coefficients = Some(pirls_result.coefficients);
        self.fitted_values = Some(pirls_result.fitted_values);
        self.linear_predictor = Some(pirls_result.linear_predictor);
        self.weights = Some(pirls_result.weights);
        self.design_matrix = Some(design_matrix.clone());

        // Recompute deviance for consistency
        let fitted = self.fitted_values.as_ref().unwrap();
        let mut correct_deviance = 0.0;
        for i in 0..y.len() {
            correct_deviance += (y[i] - fitted[i]).powi(2);
        }

        self.deviance = Some(correct_deviance);
        self.smoothing_params = Some(smoothing_params);
        self.fitted = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gam_creation() {
        let gam = GAM::new(Family::Gaussian);
        assert_eq!(gam.smooth_terms.len(), 0);
        assert!(!gam.fitted);
    }

    #[test]
    fn test_gam_add_smooth() {
        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline("x1".to_string(), 10, 0.0, 1.0).unwrap();

        gam.add_smooth(smooth);
        assert_eq!(gam.smooth_terms.len(), 1);
    }

    #[test]
    fn test_gam_fit() {
        let n = 50;

        // Generate test data: y = sin(2*pi*x) + noise
        let x_data: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let y_data: Array1<f64> = x_data
            .iter()
            .enumerate()
            .map(|(i, &xi)| (2.0 * std::f64::consts::PI * xi).sin() + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();

        let x_matrix = x_data.clone().to_shape((n, 1)).unwrap().to_owned();

        let mut gam = GAM::new(Family::Gaussian);
        let smooth = SmoothTerm::cubic_spline("x".to_string(), 15, 0.0, 1.0).unwrap();

        gam.add_smooth(smooth);

        let result = gam.fit(
            &x_matrix,
            &y_data,
            OptimizationMethod::GCV,
            5,  // outer iterations
            50, // inner iterations
            1e-4,
        );

        assert!(result.is_ok());
        assert!(gam.fitted);
        assert!(gam.coefficients.is_some());
        assert!(gam.deviance.is_some());
    }
}
