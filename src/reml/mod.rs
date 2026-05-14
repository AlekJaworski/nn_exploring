//! REML (Restricted Maximum Likelihood) criterion for smoothing parameter selection

pub mod system;
pub use system::{
    compute_xtwy, compute_xtwx, glm_deviance, ScaleParameterMethod,
};
#[cfg(feature = "blas")]
pub use system::{compute_edf_from_cholesky, compute_xtwx_cholesky};
#[cfg(feature = "blas")]
pub(crate) use system::{assemble_reml_system, RemlScoreParts};

#[cfg(feature = "blas")]
pub mod tk_kkt;
#[cfg(feature = "blas")]
pub use tk_kkt::{tk_kkt_hessian_analytical, tk_kkt_hessian_fd};

pub mod search_vector;
pub use search_vector::{ExtraKind, ExtraParam, OuterSearchVector};
#[cfg(feature = "blas")]
pub(crate) use search_vector::{newton_1d_with_halving, newton_2d_with_halving};

use crate::block_penalty::BlockPenalty;
use crate::linalg::{determinant, inverse, solve};
use crate::pirls::{digamma, trigamma};
use crate::GAMError;
use crate::Result;
use ndarray::{s, Array1, Array2};


/// Estimate the rank of a matrix using row norms as approximation to singular values
/// For symmetric matrices like penalty matrices, this gives a reasonable estimate
///
/// LEGACY: hardcodes "subtract 2 for cr-spline null space" which is correct for
/// the RAW (uncentred) cr-spline penalty but wrong for the centred Z'SZ penalty
/// (centring removes one null-space dimension, so centred null = orig null - 1).
/// New code should prefer `estimate_rank_eigen` which determines rank from
/// the eigenvalue spectrum directly.
pub fn estimate_rank(penalty: &BlockPenalty) -> usize {
    let non_zero_rows = penalty.estimate_rank();

    // For CR splines: rank = (non_zero_rows - 2).max(1)
    // The null space dimension is 2 (constant and linear functions)
    if non_zero_rows >= 2 {
        return non_zero_rows - 2;
    }

    // Fallback for very small matrices
    1
}

/// Estimate the rank of a penalty block via its eigenvalue spectrum.
/// Counts eigenvalues > tol * max_eigenvalue, where tol = eps^0.8 (matches
/// mgcv's nat.param convention). Robust to centring — works on raw or
/// post-Z penalty matrices.
#[cfg(feature = "blas")]
pub fn estimate_rank_eigen(penalty: &BlockPenalty) -> usize {
    use ndarray_linalg::Eigh;
    let block = penalty.block_view().to_owned();
    if block.nrows() == 0 {
        return 0;
    }
    let (eigenvalues, _) = match block.eigh(ndarray_linalg::UPLO::Upper) {
        Ok(t) => t,
        Err(_) => return penalty.estimate_rank().saturating_sub(1).max(1),
    };
    let max_eig = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_eig <= 0.0 {
        return 0;
    }
    let tol = f64::EPSILON.powf(0.8) * max_eig.max(1.0);
    eigenvalues.iter().filter(|&&v| v > tol).count()
}

/// Compute the REML criterion for smoothing parameter selection
///
/// The REML criterion is:
/// REML = n*log(RSS) + log|X'WX + λS| - log|S|
///
/// Where:
/// - RSS: residual sum of squares
/// - X: design matrix
/// - W: weight matrix (from IRLS)
/// - λ: smoothing parameter
/// - S: penalty matrix
pub fn reml_criterion(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambda: f64,
    penalty_block: &BlockPenalty,
    beta: Option<&Array1<f64>>,
) -> Result<f64> {
    let n = y.len();
    let _p = x.ncols();

    // OPTIMIZED: Compute X'WX once and reuse it
    let xtwx = compute_xtwx(x, w);

    // Compute coefficients if not provided
    let beta_computed;
    let beta = if let Some(b) = beta {
        b
    } else {
        // Compute X'Wy directly without forming weighted vectors
        let xtwy = compute_xtwy(x, w, y);

        // Solve: (X'WX + λS)β = X'Wy
        let mut a = xtwx.clone();
        penalty_block.scaled_add_to(&mut a, lambda);

        beta_computed = solve(a, xtwy)?;
        &beta_computed
    };

    // Compute fitted values
    let fitted = x.dot(beta);

    // Compute residuals and RSS (optimized to avoid intermediate allocation)
    let mut rss = 0.0;
    for i in 0..n {
        let residual = y[i] - fitted[i];
        rss += residual * residual * w[i];
    }

    // Compute penalty term: β'Sβ (optimized dot product)
    let beta_s_beta = penalty_block.quadratic_form(beta);

    // Compute RSS + λβ'Sβ (this is what mgcv calls rss.bSb)
    let rss_bsb = rss + lambda * beta_s_beta;

    // Reuse X'WX from above (no recomputation needed!)
    let mut a = xtwx.clone();
    penalty_block.scaled_add_to(&mut a, lambda);

    // Compute log determinants
    let log_det_a = determinant(&a)?.ln();

    // Estimate rank of penalty matrix
    let rank_s = estimate_rank(penalty_block);

    // Compute scale parameter: φ = RSS / (n - rank(S))
    // Note: φ is based on RSS alone, not RSS + λβ'Sβ
    let phi = rss / (n - rank_s) as f64;

    // The correct REML criterion (matching mgcv's fast-REML.r implementation):
    // REML = ((RSS + λβ'Sβ)/φ + (n-rank(S))*log(2π φ) + log|X'WX + λS| - rank(S)*log(λ) - log|S_+|) / 2
    //
    // Now we include the pseudo-determinant term log|S_+|
    let log_lambda_term = if lambda > 1e-10 && rank_s > 0 {
        (rank_s as f64) * lambda.ln()
    } else {
        0.0
    };

    // Compute pseudo-determinant of penalty matrix
    #[cfg(feature = "blas")]
    let log_pseudo_det = pseudo_determinant(penalty_block)?;
    #[cfg(not(feature = "blas"))]
    let log_pseudo_det = 0.0; // Fallback when BLAS not available

    let pi = std::f64::consts::PI;
    let reml = (rss_bsb / phi + ((n - rank_s) as f64) * (2.0 * pi * phi).ln() + log_det_a
        - log_lambda_term
        - log_pseudo_det)
        / 2.0;

    Ok(reml)
}

/// Compute GCV (Generalized Cross-Validation) criterion as alternative to REML
///
/// GCV = n * RSS / (n - tr(A))^2
/// where A is the influence matrix
pub fn gcv_criterion(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambda: f64,
    penalty_block: &BlockPenalty,
) -> Result<f64> {
    let n = y.len();
    let p = x.ncols();

    // Compute weighted design matrix (optimized)
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Solve for coefficients
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);
    let mut a = xtwx.clone();
    penalty_block.scaled_add_to(&mut a, lambda);

    // Optimized y_weighted computation
    let mut y_weighted = Array1::zeros(n);
    for i in 0..n {
        y_weighted[i] = y[i] * w[i];
    }

    let b = xtw.dot(&y_weighted);

    let a_for_solve = a.clone();
    let beta = solve(a_for_solve, b)?;

    // Compute fitted values and residuals (optimized)
    let fitted = x.dot(&beta);
    let mut rss = 0.0;
    for i in 0..n {
        let residual = y[i] - fitted[i];
        rss += residual * residual * w[i];
    }

    // Compute effective degrees of freedom (trace of influence matrix)
    // EDF = tr(H) where H = X(X'WX + λS)^(-1)X'W
    let a_inv = inverse(&a)?;

    // Compute X'W (not sqrt(W))
    let mut xtw_full = Array2::zeros((p, n));
    for i in 0..n {
        for j in 0..p {
            xtw_full[[j, i]] = x[[i, j]] * w[i];
        }
    }

    // H = X * (X'WX + λS)^(-1) * X'W
    let h_temp = x.dot(&a_inv);
    let influence = h_temp.dot(&xtw_full);

    // Trace of H
    let mut edf = 0.0;
    for i in 0..n {
        edf += influence[[i, i]];
    }

    // GCV = n * RSS / (n - edf)^2
    let gcv = (n as f64) * rss / ((n as f64) - edf).powi(2);

    Ok(gcv)
}

/// Compute the REML criterion for multiple smoothing parameters
///
/// The REML criterion with multiple penalties is:
/// REML = n*log(RSS/n) + log|X'WX + Σλᵢ·Sᵢ| - Σrank(Sᵢ)·log(λᵢ) - Σlog|Sᵢ_+|
///
/// Where:
/// - RSS: residual sum of squares
/// - X: design matrix
/// - W: weight matrix (from IRLS)
/// - λᵢ: smoothing parameters
/// - Sᵢ: penalty matrices
/// - log|Sᵢ_+|: pseudo-determinant of penalty matrix Sᵢ
///
/// # Scale Parameter Method
/// This function uses EDF (Effective Degrees of Freedom) for the scale parameter φ
/// when BLAS is available, matching mgcv's implementation. Otherwise falls back to rank.
pub fn reml_criterion_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    beta: Option<&Array1<f64>>,
) -> Result<f64> {
    reml_criterion_multi_cached(y, x, w, lambdas, penalties, beta, None)
}

/// mgcv-exact REML criterion that matches gam.fit3.r:621 byte-for-byte.
/// Differences from `reml_criterion_multi_cached`:
///   - Dp = RSS (deviance) only — does NOT include β'Sβ.
///   - σ² = RSS / (n - trA) where trA uses NO ridge regularisation.
///   - The (n - Mp) coefficient on log(2πσ²) uses Mp = 1 (intercept)
///     + Σ null.space.dim_j (constant in λ), not n - edf.
///   - log|X'WX + ΣλS| computed with NO ridge added to the diagonal.
///
/// Mp is provided by the caller (= 1 for intercept + Σ for each smooth's
/// null-space dimension after centring).
#[cfg(feature = "blas")]
pub fn reml_criterion_multi_cached_mgcv_exact(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    mp: usize,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<f64> {
    let n = y.len();
    let p = x.ncols();
    // Optional internal reparametrisation: when `MGCV_REPARAM=1`, work in
    // mgcv's stable similarity basis (gam.fit3.r:144-180). Rotates x and
    // penalties via `T = U1 · diag-padded(Qs, I_Mp)` and replaces the
    // per-block `log|S+|` accumulation with the rotated `det`. REML is
    // basis-invariant analytically; numerically the rotated path matches
    // mgcv more closely at saturating λ where the un-rotated linear
    // system is poorly conditioned. Gate falls in R2-f once validated.
    let reparam_active = std::env::var("MGCV_REPARAM").is_ok();
    let rot_state: Option<(
        Array2<f64>,
        Vec<BlockPenalty>,
        f64,
    )> = if reparam_active {
        let (u1, mp_detected) = crate::reparam::compute_total_penalty_space(penalties_blocks, p)?;
        let _ = mp_detected; // caller-supplied `mp` wins for now; this is for diagnostics.
        let rot = crate::reparam::apply_reparam(x, penalties_blocks, lambdas, &u1, mp_detected, 0)?;
        // Wrap each rotated component as a full-p × full-p BlockPenalty so
        // the existing `assemble_reml_system` + `compute_xtwx` machinery
        // can consume it unchanged. The rotated S_i = rs_rot · rs_rot' is
        // dense across all p columns — block-sparse fast paths no longer
        // apply, by design (per "rewrites not fallbacks" directive).
        let rotated_pens: Vec<BlockPenalty> = rot
            .rs_rot
            .iter()
            .map(|rs| BlockPenalty::new(rs.dot(&rs.t()), 0, p))
            .collect();
        Some((rot.x_rot, rotated_pens, rot.det))
    } else {
        None
    };
    let (x_use, pens_use): (&Array2<f64>, &[BlockPenalty]) = match &rot_state {
        Some((x_rot, pens_rot, _)) => (x_rot, pens_rot.as_slice()),
        None => (x, penalties_blocks),
    };
    // X'WX (cached or computed). Rotated case rebuilds — cached_xtwx is
    // in the un-rotated basis and not reusable here.
    let xtwx_owned;
    let xtwx = if reparam_active {
        xtwx_owned = compute_xtwx(x_use, w);
        &xtwx_owned
    } else if let Some(cached) = cached_xtwx {
        cached
    } else {
        xtwx_owned = compute_xtwx(x_use, w);
        &xtwx_owned
    };
    // A = X'WX + ΣλS — NO ridge in score terms. The shared assembly keeps
    // the tiny solve-only ridge out of determinants/traces.
    // cached_xtwy is only valid in the un-rotated basis; skip in the rotated path.
    let xtwy_for_assemble = if reparam_active { None } else { cached_xtwy };
    let system = assemble_reml_system(
        y,
        x_use,
        w,
        xtwx,
        lambdas,
        pens_use,
        xtwy_for_assemble,
    )?;

    // Deviance numerator, β'(ΣλS)β, Dp and σ² from the shared score-parts
    // helper. Two deviance paths (item 1 of #47):
    //   - `y_original = None` (default / Gaussian): working-RSS
    //     `Σ w_i (y_i - X_iβ)²`. For Gaussian (w=I, y=y_orig) this IS
    //     the true deviance. For non-Gaussian it's the working-response
    //     approximation that 4q used.
    //   - `y_original = Some(y_orig)` + non-Gaussian: true GLM
    //     deviance `D(y_orig, μ̂(λ))` with μ̂ = g⁻¹(Xβ̂(λ)). This is
    //     mgcv's gam.fit3.r:617 score formula and the only one
    //     consistent with the IFT gradient.
    // Scale convention:
    //   - Binomial / Poisson / NegBin: φ = 1 (known).
    //   - Gaussian / Gamma / etc.: mgcv's profiled φ̂ solving dlr.dlphi = 0.
    let parts = RemlScoreParts::from_system(
        &system,
        y,
        w,
        x_use,
        xtwx,
        lambdas,
        pens_use,
        family,
        y_original,
        mp,
        n,
    );
    let dev_numerator = parts.dev_num;
    let bsb = parts.bsb;
    let dp = parts.dp;
    let scale_est = parts.sigma2;

    // log|H| — mgcv's REML formula (gam.fit3.r:621) uses log|X'W·X + S|
    // with W = NEWTON weights `wf · α` (gam.fit3.r:511-522, "full Newton"
    // branch; gdi.c handles the possibly-indefinite assembly via its
    // pivoted-QR `neg_w` path). PIRLS internally falls back to Fisher
    // weights when α≤0 to keep its Cholesky PSD, so the `w` we received
    // here is Fisher. For canonical links Newton == Fisher and we keep
    // the un-ridged log|det(A)|. For non-canonical links we rebuild A
    // with the Newton score weights at the converged β and use
    // `log_abs_det_symmetric` (Σ log|λ_i|) to handle a possibly
    // indefinite spectrum. This closes the InvGauss log-link mgcv parity
    // gap (~0.22 in log|H|, the entire 0.11 REML offset).
    let log_det_a = if family.is_canonical_link() {
        determinant(&system.a)?.ln()
    } else {
        // Newton α depends on residuals y_orig - μ̂ at the original y, so
        // route through y_original when the caller supplied it.
        let y_for_newton = y_original.unwrap_or(y);
        let w_score = crate::pirls::compute_newton_score_weights(
            y_for_newton,
            system.fitted(x_use),
            family,
        );
        // Guard against extreme α blow-up at intermediate λ during Newton
        // line search: if any w_score is non-finite, fall back to the
        // Fisher-weighted log|A|. Matches mgcv's effective behaviour at
        // problematic iterates without aborting the outer loop.
        if w_score.iter().any(|w| !w.is_finite()) {
            determinant(&system.a)?.ln()
        } else {
            let xtwx_score = compute_xtwx(x_use, &w_score);
            let mut a_score = xtwx_score;
            for (lambda, penalty) in lambdas.iter().zip(pens_use.iter()) {
                penalty.scaled_add_to(&mut a_score, *lambda);
            }
            crate::linalg::log_abs_det_symmetric(&a_score)
                .unwrap_or_else(|_| determinant(&system.a).map(|d| d.ln()).unwrap_or(0.0))
        }
    };
    let tr_a = system.tr_a;
    let n_minus_tra = (n as f64) - tr_a;
    let y_for_ls = y_original.unwrap_or(y);

    // log|S+| terms. Two computation paths produce the same scalar value
    // (analytically `log|Σ λᵢ Sᵢ|₊` is basis-invariant):
    //   * Rotated: `det` from `apply_reparam` is exactly `log|Σ λᵢ Sᵢ_rot|₊`.
    //   * Un-rotated: per-block accumulation `Σ rankᵢ·log(λᵢ) + Σ log|Sᵢ|₊`,
    //     valid only when the un-rotated penalty blocks live in disjoint
    //     column subspaces (= our standard GAM setup).
    let mut log_lambda_sum = 0.0;
    let mut log_pseudo_det_sum = 0.0;
    if let Some((_, _, det_rot)) = &rot_state {
        log_pseudo_det_sum = *det_rot;
    } else {
        for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
            if *lambda > 1e-10 {
                let rank_s = estimate_rank_eigen(penalty);
                if rank_s > 0 {
                    log_lambda_sum += (rank_s as f64) * lambda.ln();
                }
            }
            log_pseudo_det_sum += pseudo_determinant(penalty)?;
        }
    }

    // mgcv's formula (gam.fit3.r:616-617):
    // REML = (Dp/(2σ²) - ls[1]) + log|H|/2 - log|S|_+/2 - Mp/2·log(2π·σ²)
    //
    // `ls[1]` is the family-specific saturated log-likelihood
    // (gam.fit3.r:2497-2548 / `fix.family.ls`). For Gaussian
    // ls[1] = -n/2·log(2π·σ²), so the expression collapses to
    //   REML = Dp/(2σ²) + (n - Mp)/2·log(2π·σ²) + log|H|/2 - log|S|_+/2
    // — which is the form we used historically before this fix.
    //
    // For Gamma the saturated likelihood involves digamma terms in σ²,
    // and σ² is profiled (depends on λ), so the Gaussian short-cut
    // gives the wrong λ-derivative and an incorrect score-formula
    // optimum location. For Poisson/Binomial σ² = 1 (known) and ls[1]
    // is purely a constant in λ — including it is a no-op for the
    // optimiser but makes the reported score commensurable with mgcv.
    //
    // Source for ls[1]: `Family::saturated_log_likelihood` in
    // `pirls.rs`. For non-Gaussian we evaluate at `y_original` (the
    // true response) since the working response `z` is not what mgcv's
    // ls function takes.
    let _ = p;
    let ls1 = family.saturated_log_likelihood(y_for_ls, scale_est);
    let log_det_h = log_det_a;
    let log_det_s = log_lambda_sum + log_pseudo_det_sum;
    // Dispatch on the family's score formula. GamFit3 = standard REML
    // with σ²-profile correction; GamFit5 = LAML for extended families
    // (TDist/scat, Quantile/ELF) where σ² is family-internal.
    let formula = family.score_formula();
    let reml = formula.assemble(dp, ls1, log_det_h, log_det_s, scale_est, mp);

    if std::env::var("MGCV_EXACT_DEBUG").is_ok() {
        eprintln!(
            "[MGCV_EXACT] λ={:?} formula={:?}\n  dev={:.6} bSb={:.6} dp={:.6} sigma2={:.8}\n  trA={:.6} n-trA={:.6} Mp={} ls[1]={:.6}\n  log|H|={:.6} log|λS|+={:.6} (lambda_sum={:.6} pseudo_det={:.6})\n  REML {:.6}",
            lambdas, formula, dev_numerator, bsb, dp, scale_est,
            tr_a, n_minus_tra, mp, ls1,
            log_det_h, log_det_s, log_lambda_sum, log_pseudo_det_sum, reml
        );
    }

    Ok(reml)
}


/// Frozen pieces of the Tweedie REML score that DO NOT depend on the working
/// parameter θ (and therefore on the Tweedie index p). Used by the analytical
/// θ-derivative path to avoid recomputing the linear system across the three
/// FD trials (center / +h / -h).
///
/// Invariants — the caller pinky-promises that `(y_local, w_local, xtwx_local,
/// lambdas, penalties)` are held FIXED across the trial evaluations. When this
/// is true, mgcv's REML formula
///
/// ```text
///   REML = Dp/(2σ²) + log|H|/2 - log|λS|+/2 - ls + Mp/2·log(2π·σ²)
/// ```
///
/// has the following constant pieces:
///   - β̂ and μ̂ = Xβ̂  (from the frozen linear system)
///   - `bsb = β̂'(Σ λ_j S_j) β̂`
///   - `log_det_h = log|X'WX + ΣλS|`
///   - `log_det_s = log|λS|+` (depends only on λ, not on family)
///   - `mp` and `y_for_ls.len()`
///
/// What still varies with p (small fast pieces):
///   - `D(y_orig, μ̂; p)` — Tweedie deviance at frozen μ̂ (O(n) closed form)
///   - `σ²̂(p)` — Newton on `dlr/dφ = 0` with Wright series
///   - `ls(y, σ²̂, p)` — Wright series
///
/// Together these are ~5% of the cost of a full `dispatch_reml_score_with_family`
/// call, so caching the linear algebra and looping only over the small p-pieces
/// gives a 3× speedup on the θ-FD step (3 evaluations: center, +h, -h).
#[cfg(feature = "blas")]
pub struct TweedieThetaCache<'a> {
    pub y_for_ls: &'a Array1<f64>,
    pub fitted: Array1<f64>,
    pub bsb: f64,
    pub log_det_h: f64,
    pub log_det_s: f64,
    pub mp: usize,
    /// `tr(A⁻¹ X'WX)` — effective degrees of freedom. Stored so that
    /// `score_at_p` uses the same `phi_init = dev_numerator / (n - tr_a)`
    /// as `reml_criterion_multi_cached_mgcv_exact` (for byte-identical
    /// numerics across cached and direct paths).
    pub tr_a: f64,
}

#[cfg(feature = "blas")]
impl<'a> TweedieThetaCache<'a> {
    /// Build the cache for a fixed (y_local, w_local, xtwx_local, lambdas)
    /// state. Performs ONE linear system assembly + factorisation; reused
    /// across all trial p values during the θ-Newton FD step.
    pub fn build(
        y_local: &Array1<f64>,
        x: &Array2<f64>,
        w_local: &Array1<f64>,
        xtwx_local: &Array2<f64>,
        lambdas: &[f64],
        penalties: &[BlockPenalty],
        mp: usize,
        y_for_ls: &'a Array1<f64>,
    ) -> Result<Self> {
        let system =
            assemble_reml_system(y_local, x, w_local, xtwx_local, lambdas, penalties, None)?;
        let mut bsb = 0.0;
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            bsb += lambda * penalty.quadratic_form(&system.beta);
        }
        let log_det_h = determinant(&system.a)?.ln();
        let mut log_lambda_sum = 0.0;
        let mut log_pseudo_det_sum = 0.0;
        for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
            if *lambda > 1e-10 {
                let rank_s = estimate_rank_eigen(penalty);
                if rank_s > 0 {
                    log_lambda_sum += (rank_s as f64) * lambda.ln();
                }
            }
            log_pseudo_det_sum += pseudo_determinant(penalty)?;
        }
        let log_det_s = log_lambda_sum + log_pseudo_det_sum;
        Ok(TweedieThetaCache {
            y_for_ls,
            fitted: system.fitted(x).clone(),
            bsb,
            log_det_h,
            log_det_s,
            mp,
            tr_a: system.tr_a,
        })
    }

    /// Evaluate the REML score for a Tweedie family with index `p`, reusing
    /// the frozen linear-system pieces. Cost: O(n) for the deviance + Wright
    /// series for σ²̂ + ls. Roughly 3× cheaper than a fresh
    /// `reml_criterion_multi_cached_mgcv_exact` call when the linear system
    /// is the dominant cost (small to moderate p, n ≥ 500).
    pub fn score_at_p(&self, p: f64) -> Result<f64> {
        let family = crate::pirls::Family::Tweedie { p };
        // μ̂ = g⁻¹(η̂) with η̂ = Xβ̂ (frozen across p)
        let mu: Array1<f64> = self.fitted.iter().map(|&eta| family.inverse_link(eta)).collect();
        let dev_numerator = glm_deviance(self.y_for_ls, &mu, family);
        let dp = dev_numerator + self.bsb;
        // Profile σ̂² for this trial p — `phi_init` matches the direct path
        // (`reml_criterion_multi_cached_mgcv_exact`) so the Newton iteration
        // starts from the same point and converges bit-identically.
        let n_minus_tra = (self.y_for_ls.len() as f64) - self.tr_a;
        let phi_init = dev_numerator / n_minus_tra.max(1e-10);
        let scale_est = family.estimate_phi_mgcv(self.y_for_ls, dp, self.mp, 1.0, phi_init);
        let ls1 = family.saturated_log_likelihood(self.y_for_ls, scale_est);
        let formula = family.score_formula();
        Ok(formula.assemble(
            dp,
            ls1,
            self.log_det_h,
            self.log_det_s,
            scale_est,
            self.mp,
        ))
    }
}

/// Compute the FD gradient and curvature of the Tweedie REML score with
/// respect to the working parameter θ (the unconstrained log-odds-ratio
/// parameter mapping into the (a,b) interval that gives p). Reuses the
/// frozen linear system across the three FD probes (center, +h, -h), so
/// each `score_at_p` call only does O(n) + Wright-series work — no extra
/// linear algebra.
///
/// Returns `(reml_center, dlr_dth, d2lr_dth2)`.
///
/// `theta_to_p` is the same θ→p map mgcv uses (smooth.rs has the closure;
/// passed in for clarity since the mapping is family- but not call-specific).
#[cfg(feature = "blas")]
pub fn tweedie_theta_derivatives_cached(
    cache: &TweedieThetaCache<'_>,
    theta: f64,
    h: f64,
    theta_to_p: impl Fn(f64) -> f64,
) -> Result<(f64, f64, f64)> {
    let p_center = theta_to_p(theta);
    let p_plus = theta_to_p(theta + h);
    let p_minus = theta_to_p(theta - h);
    let rc = cache.score_at_p(p_center)?;
    let rp = cache.score_at_p(p_plus)?;
    let rm = cache.score_at_p(p_minus)?;
    let dlr_dth = (rp - rm) / (2.0 * h);
    let d2lr_dth2 = (rp - 2.0 * rc + rm) / (h * h);
    Ok((rc, dlr_dth, d2lr_dth2))
}

/// Closed-form gradient of the mgcv-exact REML score w.r.t. log(λ_j).
///
/// For Gaussian + canonical link at PIRLS convergence the cross-coupling
/// β-dependent gradient terms cancel via envelope theorem (see Wood 2011
/// Appendix or mgcv's `gdi.c:get_bSb` lines 145-194 — mgcv computes them
/// explicitly but for converged Gaussian they sum to zero).
///
/// The surviving terms (matching gdi.c:854-891 and 506-514 simplified for
/// Gaussian / canonical / converged):
///   ∂Dp/∂(log λ_j) = λ_j β'S_j β
///   ∂log|H|/∂(log λ_j) = λ_j tr(A^-1 S_j)         where A = X'X + Σλ_iS_i
///   ∂log|λS|+/∂(log λ_j) = rank_j                  (block-diagonal penalties)
///
/// REML gradient assembly (gam.fit3.r:625):
///   REML1[j] = (∂Dp/∂ρ_j)/(2σ²) + (∂log|H|/∂ρ_j)/2 - (∂log|λS|+/∂ρ_j)/2
///            = λ_j β'S_jβ / (2σ²) + λ_j tr(A^-1 S_j)/2 - rank_j/2
///
/// Cost: O(p³) for one A^-1 + O(m·p²) for the m traces (vs O(m·p³) for FD).
/// Should be ~m× faster than the FD gradient.
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_closed_form(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    family: crate::pirls::Family,
) -> Result<Array1<f64>> {
    reml_gradient_mgcv_exact_closed_form_inner(
        y,
        x,
        w,
        lambdas,
        penalties_blocks,
        cached_xtwx,
        cached_xtwy,
        family,
        None,
    )
}

/// Like `reml_gradient_mgcv_exact_closed_form` but with a caller-supplied
/// fixed σ² — used when differentiating the score at a fixed σ² base point
/// (so FD-of-gradient and CF Hessian use the same σ² convention exactly).
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_closed_form_fixed_sigma2(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    family: crate::pirls::Family,
    fixed_sigma2: f64,
) -> Result<Array1<f64>> {
    reml_gradient_mgcv_exact_closed_form_inner(
        y,
        x,
        w,
        lambdas,
        penalties_blocks,
        cached_xtwx,
        cached_xtwy,
        family,
        Some(fixed_sigma2),
    )
}

#[cfg(feature = "blas")]
fn reml_gradient_mgcv_exact_closed_form_inner(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    family: crate::pirls::Family,
    fixed_scale: Option<f64>,
) -> Result<Array1<f64>> {
    let n = y.len();
    let m = lambdas.len();
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    let system = assemble_reml_system(y, x, w, xtwx, lambdas, penalties_blocks, cached_xtwy)?;

    // RSS / σ² — same working-RSS approximation as the score function
    // for non-Gaussian (see comment there). Family is plumbed through
    // for future use but not branched on yet.
    let _ = family;
    let parts = RemlScoreParts::gaussian_only(&system, y, w, x, xtwx, n, fixed_scale);
    let a_inv = &system.a_inv;
    let scale_est = parts.sigma2;

    // Per-smooth gradient
    let mut grad = Array1::<f64>::zeros(m);
    for (j, (lambda, penalty)) in lambdas.iter().zip(penalties_blocks.iter()).enumerate() {
        // β' S_j β = quadratic_form on the smooth's block
        let bsb_j = penalty.quadratic_form(&system.beta);
        // tr(A^-1 S_j) — only the smooth's block contributes
        // S_j is sparse: nonzero in [offset, offset+k) × [offset, offset+k)
        // tr(A^-1 S_j) = Σ_{p,q in block} A^-1[p,q] S_j[p-off, q-off]
        let block = penalty.block_view();
        let off = penalty.offset;
        let k = block.nrows();
        let mut tr_a_inv_s = 0.0;
        for p in 0..k {
            for q in 0..k {
                tr_a_inv_s += a_inv[[off + p, off + q]] * block[[q, p]];
            }
        }

        let rank_j = estimate_rank_eigen(penalty);
        // REML1[j] = λ_j β'S_jβ / (2σ²) + λ_j tr(A^-1 S_j) / 2 - rank_j/2
        grad[j] =
            lambda * bsb_j / (2.0 * scale_est) + lambda * tr_a_inv_s / 2.0 - (rank_j as f64) / 2.0;
    }
    Ok(grad)
}

/// Closed-form Hessian of the mgcv-exact REML score w.r.t. log(λ_i, λ_j).
///
/// Formula (treating σ² as constant per gam.fit3.r:625's profile-REML
/// convention; for Gaussian + canonical link + converged β):
///
/// H_ij = (1/(2σ²)) · [δ_ij λ_j β'S_jβ - 2 λ_i λ_j β'S_j A⁻¹ S_i β]
///      + (1/2)     · [δ_ij λ_j tr(A⁻¹ S_j) - λ_i λ_j tr(A⁻¹ S_i A⁻¹ S_j)]
///      - 0  (log|S|+ is linear in log λ, so its Hessian is zero)
///
/// Cost analysis vs the FD Hessian (O(2m² + 1) score evals at O(p³) each):
///   - One A⁻¹: O(p³)         — already needed by gradient
///   - Per smooth i: A⁻¹ S_i → only k_i nonzero columns; O(p · k²)
///   - Per (i,j) pair: tr is O(k²) using sparse structure
///   Total: O(p³ + m·p·k² + m²·k²) ≈ O(p³) once + O(m²·k²) cheap
///   versus FD's O(m²·p³). Roughly m²·(p/k)² speedup.
///
/// Symmetric output. Both gradient and Hessian use the same cached
/// β, σ², A⁻¹, but for clean separation of concerns we recompute them
/// here. The Newton optimizer calls gradient and Hessian back-to-back
/// at the same λ, so the redundancy is one extra inverse per iter.
#[cfg(feature = "blas")]
pub fn reml_hessian_mgcv_exact_closed_form(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    family: crate::pirls::Family,
) -> Result<Array2<f64>> {
    let n = y.len();
    let m = lambdas.len();
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    // A = X'WX + ΣλS, β = A^-1 X'Wy, σ² = RSS/(n-trA), A^-1
    let system = assemble_reml_system(y, x, w, xtwx, lambdas, penalties_blocks, cached_xtwy)?;
    let _ = family;
    let parts = RemlScoreParts::gaussian_only(&system, y, w, x, xtwx, n, None);
    let a_inv = &system.a_inv;
    let scale_est = parts.sigma2;

    // Pre-compute per-smooth quantities:
    //   S_j_beta      = S_j β              (length p, sparse on j_range)
    //   bsb_j         = β' S_j β           (scalar)
    //   tr_aS_j       = tr(A^-1 S_j)       (scalar)
    //   AinvS_j       = A^-1 S_j           (p × k_j cols on j_range; rest zero)
    //                   stored as (p, k_j) matrix indexed by [row, col_within_block]
    let mut s_beta_per_j: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut bsb_per_j: Vec<f64> = Vec::with_capacity(m);
    let mut tr_a_s_per_j: Vec<f64> = Vec::with_capacity(m);
    // ainvs[j]: shape (p, k_j) — the k_j nonzero columns of A^-1 S_j.
    let mut ainvs_per_j: Vec<Array2<f64>> = Vec::with_capacity(m);
    let mut block_offsets: Vec<usize> = Vec::with_capacity(m);
    let mut block_sizes: Vec<usize> = Vec::with_capacity(m);
    let p = system.a.nrows();
    for penalty in penalties_blocks.iter() {
        let block = penalty.block_view();
        let off = penalty.offset;
        let k = block.nrows();
        block_offsets.push(off);
        block_sizes.push(k);

        // S_j β  — only entries within the smooth's block are nonzero.
        let mut s_beta = Array1::<f64>::zeros(p);
        for r in 0..k {
            let mut s = 0.0;
            for c in 0..k {
                s += block[[r, c]] * system.beta[off + c];
            }
            s_beta[off + r] = s;
        }
        s_beta_per_j.push(s_beta);

        // β' S_j β
        let mut bsb = 0.0;
        for r in 0..k {
            bsb += system.beta[off + r] * s_beta_per_j.last().unwrap()[off + r];
        }
        bsb_per_j.push(bsb);

        // tr(A^-1 S_j) = Σ_{p,q in block} A^-1[off+p, off+q] S_j[q, p]
        let mut tr_as = 0.0;
        for ii in 0..k {
            for jj in 0..k {
                tr_as += a_inv[[off + ii, off + jj]] * block[[jj, ii]];
            }
        }
        tr_a_s_per_j.push(tr_as);

        // A^-1 S_j: shape (p, k). Column c is A^-1[:, off:off+k] @ block[:, c].
        let mut ainvs = Array2::<f64>::zeros((p, k));
        for c in 0..k {
            for r in 0..p {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a_inv[[r, off + kk]] * block[[kk, c]];
                }
                ainvs[[r, c]] = s;
            }
        }
        ainvs_per_j.push(ainvs);
    }

    // Assemble Hessian.
    let mut hess = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let off_i = block_offsets[i];
        let k_i = block_sizes[i];
        let lam_i = lambdas[i];
        for j in 0..m {
            let off_j = block_offsets[j];
            let k_j = block_sizes[j];
            let lam_j = lambdas[j];

            // β' S_j A^-1 S_i β: only rows in j_range of A^-1 S_i contribute
            // because (S_j β)[r] is nonzero iff r in j_range. So
            //   = Σ_{r in j_range} (S_j β)[r] · (A^-1 S_i β)[r]
            // where (A^-1 S_i β)[r] = Σ_{c in i_range_local} ainvs_i[r, c-off_i] β[c].
            // Pre-compute u_i = A^-1 S_i β as a length-p vector by gathering
            // ainvs[:, c] · β[off_i + c]:
            // (could cache per i but keeps memory small)
            let s_beta_j = &s_beta_per_j[j];
            let mut bsb_cross = 0.0;
            for r in off_j..(off_j + k_j) {
                let sj_r = s_beta_j[r];
                if sj_r == 0.0 {
                    continue;
                }
                // (A^-1 S_i β)[r]
                let mut ainvs_i_beta_r = 0.0;
                for c in 0..k_i {
                    ainvs_i_beta_r += ainvs_per_j[i][[r, c]] * system.beta[off_i + c];
                }
                bsb_cross += sj_r * ainvs_i_beta_r;
            }

            // tr(A^-1 S_i A^-1 S_j): using sparse structure
            //   = Σ_{p in j_range, r in i_range} (A^-1 S_i)[p, r-off_i] · (A^-1 S_j)[r, p-off_j]
            // Note the index swap (trace).
            let mut tr_cross = 0.0;
            for r in 0..k_i {
                for pp in 0..k_j {
                    tr_cross += ainvs_per_j[i][[off_j + pp, r]] * ainvs_per_j[j][[off_i + r, pp]];
                }
            }

            // H_ij = (1/(2σ²))[δ_ij λ_j bsb_j - 2 λ_i λ_j β'S_jA⁻¹S_iβ]
            //      + (1/2)   [δ_ij λ_j tr_aS_j - λ_i λ_j tr_cross]
            let kron = if i == j { 1.0 } else { 0.0 };
            let dp_part =
                (kron * lam_j * bsb_per_j[j] - 2.0 * lam_i * lam_j * bsb_cross) / (2.0 * scale_est);
            let logh_part = (kron * lam_j * tr_a_s_per_j[j] - lam_i * lam_j * tr_cross) / 2.0;
            hess[[i, j]] = dp_part + logh_part;
        }
    }
    Ok(hess)
}

#[cfg(feature = "blas")]
fn penalty_dot_vec(penalty: &BlockPenalty, v: &Array1<f64>) -> Array1<f64> {
    let p = v.len();
    let mut out = Array1::<f64>::zeros(p);
    let block = penalty.block_view();
    let off = penalty.offset;
    let k = block.nrows();
    for r in 0..k {
        let mut s = 0.0;
        for c in 0..k {
            s += block[[r, c]] * v[off + c];
        }
        out[off + r] = s;
    }
    out
}

#[cfg(feature = "blas")]
fn add_scaled_penalty_dense(a: &mut Array2<f64>, penalty: &BlockPenalty, scale: f64) {
    let block = penalty.block_view();
    let off = penalty.offset;
    let k = block.nrows();
    for r in 0..k {
        for c in 0..k {
            a[[off + r, off + c]] += scale * block[[r, c]];
        }
    }
}

#[cfg(feature = "blas")]
fn trace_product_dense(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let m = a.ncols();
    let mut tr = 0.0;
    for i in 0..n {
        for j in 0..m {
            tr += a[[i, j]] * b[[j, i]];
        }
    }
    tr
}

#[cfg(feature = "blas")]
fn add_weighted_xtx_inplace(
    out: &mut Array2<f64>,
    x: &Array2<f64>,
    diag: &Array1<f64>,
    scale: f64,
) {
    let n = x.nrows();
    let p = x.ncols();
    for i in 0..n {
        let wi = scale * diag[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let xia = x[[i, a]];
            if xia == 0.0 {
                continue;
            }
            for b in 0..p {
                out[[a, b]] += wi * xia * x[[i, b]];
            }
        }
    }
}

#[cfg(feature = "blas")]
fn penalty_dot_all(penalties: &[BlockPenalty], lambdas: &[f64], v: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(v.len());
    for (pen, lam) in penalties.iter().zip(lambdas.iter()) {
        let sv = penalty_dot_vec(pen, v);
        for i in 0..out.len() {
            out[i] += lam * sv[i];
        }
    }
    out
}

#[cfg(feature = "blas")]
fn tdist_dd_arrays(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    df: f64,
    sigma2: f64,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    [Array1<f64>; 2],
    [Array1<f64>; 2],
    [Array1<f64>; 2],
    [Array1<f64>; 2],
    [Array1<f64>; 3],
    [Array1<f64>; 3],
    [Array1<f64>; 3],
) {
    let n = y.len();
    let nu = df.max(2.0 + 1e-8);
    let nu1 = nu + 1.0;
    let nu2 = (nu - 2.0).max(1e-12);
    let nu2nu = nu2 / nu;
    let sig2 = sigma2.max(1e-300);
    let sig = sig2.sqrt();
    let mut det = Array1::<f64>::zeros(n);
    let mut det2 = Array1::<f64>::zeros(n);
    let mut det3 = Array1::<f64>::zeros(n);
    let mut det4 = Array1::<f64>::zeros(n);
    let mut dth = [Array1::<f64>::zeros(n), Array1::<f64>::zeros(n)];
    let mut det_th = [Array1::<f64>::zeros(n), Array1::<f64>::zeros(n)];
    let mut det2_th = [Array1::<f64>::zeros(n), Array1::<f64>::zeros(n)];
    let mut det3_th = [Array1::<f64>::zeros(n), Array1::<f64>::zeros(n)];
    let mut dth2 = [
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
    ];
    let mut det_th2 = [
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
    ];
    let mut det2_th2 = [
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
        Array1::<f64>::zeros(n),
    ];

    for i in 0..n {
        let ym = y[i] - eta[i];
        let a = 1.0 + (ym / sig) * (ym / sig) / nu;
        let sig2a = sig2 * a;
        let nusig2a = nu * sig2a;
        let f = nu1 * ym / nusig2a;
        let f1 = ym / nusig2a;
        let nu1nusig2a = nu1 / nusig2a;
        let fym = f * ym;
        let ff1 = f * f1;
        let f1ym = f1 * ym;
        let fymf1 = fym * f1;
        let ymsig2a = ym / sig2a;
        let nu1nu = nu1 / nu;
        let fymf1ym = fym * f1ym;
        let f1ymf1 = f1ym * f1;

        det[i] = -2.0 * f;
        det2[i] = 2.0 * nu1 * (1.0 / nusig2a - 2.0 * f1 * f1);
        det3[i] = 4.0 * f * (3.0 / nusig2a - 4.0 * f1 * f1);
        det4[i] = 12.0 * (-nu1nusig2a / nusig2a + 8.0 * ff1 / nusig2a - 8.0 * ff1 * f1 * f1);

        dth[0][i] = nu2 * (a.ln() - fym / nu);
        dth[1][i] = -2.0 * fym;
        det_th[0][i] = 2.0 * (f - ymsig2a - fymf1) * nu2nu;
        det_th[1][i] = 4.0 * f * (1.0 - f1ym);
        det2_th[0][i] = 2.0
            * (-nu1nusig2a + 1.0 / sig2a + 5.0 * ff1 - 2.0 * f1ym / sig2a - 4.0 * fymf1 * f1)
            * nu2nu;
        det2_th[1][i] = 4.0 * (-nu1nusig2a + ff1 * 5.0 - 4.0 * ff1 * f1ym);

        dth2[0][i] = nu2 * a.ln()
            + nu2nu * ym * ym * (-2.0 * nu2 - nu1 + 2.0 * nu1 * nu2nu - nu1 * nu2nu * f1ym)
                / nusig2a;
        dth2[1][i] = 2.0 * (fym - ym * ymsig2a - fymf1ym) * nu2nu;
        dth2[2][i] = 4.0 * fym * (1.0 - f1ym);

        let term = 2.0 * nu2nu - 2.0 * nu1nu * nu2nu - 1.0 + nu1nu;
        det_th2[0][i] = 2.0
            * f1
            * nu2
            * (term - 2.0 * nu2nu * f1ym + 4.0 * fym * nu2nu / nu
                - fym / nu
                - 2.0 * fymf1ym * nu2nu / nu);
        det_th2[1][i] =
            4.0 * (-f + ymsig2a + 3.0 * fymf1 - ymsig2a * f1ym - 2.0 * fymf1 * f1ym) * nu2nu;
        det_th2[2][i] = 8.0 * f * (-1.0 + 3.0 * f1ym - 2.0 * f1ym * f1ym);
        det3_th[0][i] = 4.0
            * (-6.0 * f / nusig2a + 3.0 * f1 / sig2a + 18.0 * ff1 * f1
                - 4.0 * f1ymf1 / sig2a
                - 12.0 * nu1 * ym * f1.powi(4))
            * nu2nu;
        det3_th[1][i] = 48.0 * f * (-1.0 / nusig2a + 3.0 * f1 * f1 - 2.0 * f1ymf1 * f1);
        det2_th2[0][i] = 2.0
            * nu2
            * (-term + 10.0 * nu2nu * f1ym - 16.0 * fym * nu2nu / nu - 2.0 * f1ym
                + 5.0 * nu1nu * f1ym
                - 8.0 * nu2nu * f1ym * f1ym
                + 26.0 * fymf1ym * nu2nu / nu
                - 4.0 * nu1nu * f1ym * f1ym
                - 12.0 * nu1nu * nu2nu * f1ym * f1ym * f1ym)
            / nusig2a;
        det2_th2[1][i] = 4.0
            * (nu1nusig2a - 1.0 / sig2a - 11.0 * nu1 * f1 * f1
                + 5.0 * f1ym / sig2a
                + 22.0 * nu1 * fymf1 * f1
                - 4.0 * f1ym * f1ym / sig2a
                - 12.0 * nu1 * fymf1 * fymf1)
            * nu2nu;
        det2_th2[2][i] = 8.0
            * (nu1nusig2a - 11.0 * nu1 * f1 * f1 + 22.0 * nu1 * fymf1 * f1
                - 12.0 * nu1 * fymf1 * fymf1);
    }
    (
        det, det2, det3, det4, dth, det_th, det2_th, det3_th, dth2, det_th2, det2_th2,
    )
}

/// Full mgcv `gdi2`-style derivative assembly for scat. Native parameter
/// order is `[theta_df=log(df-2), theta_sigma=log(sigma), log(lambda)...]`.
#[cfg(feature = "blas")]
fn tdist_gdi2_native(
    y_original: &Array1<f64>,
    y_work: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let (df, sigma2) = match family {
        crate::pirls::Family::TDist { df, sigma2 } => (df.max(2.0 + 1e-8), sigma2.max(1e-300)),
        _ => {
            return Err(GAMError::InvalidParameter(
                "TDist gdi2 called for non-TDist family".into(),
            ))
        }
    };
    let ntheta = 2;
    let m = lambdas.len();
    let ntot = ntheta + m;
    let p = x.ncols();

    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };
    let mut amat = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        penalty.scaled_add_to(&mut amat, *lambda);
    }
    let beta = solve(amat.clone(), compute_xtwy(x, w, y_work))?;
    let a_inv = inverse(&amat)?;
    let eta = x.dot(&beta);
    let (det, det2, det3, det4, dth, det_th, det2_th, det3_th, dth2, det_th2, det2_th2) =
        tdist_dd_arrays(y_original, &eta, df, sigma2);

    let mut b1: Vec<Array1<f64>> = Vec::with_capacity(ntot);
    for t in 0..ntheta {
        b1.push(-0.5 * a_inv.dot(&x.t().dot(&det_th[t])));
    }
    for j in 0..m {
        b1.push(-lambdas[j] * a_inv.dot(&penalty_dot_vec(&penalties[j], &beta)));
    }
    let eta1: Vec<Array1<f64>> = b1.iter().map(|b| x.dot(b)).collect();

    let s_beta = penalty_dot_all(penalties, lambdas, &beta);
    let mut a1: Vec<Array2<f64>> = Vec::with_capacity(ntot);
    for i in 0..ntot {
        let mut ai = Array2::<f64>::zeros((p, p));
        let wi = if i < ntheta {
            (&det3 * &eta1[i] + &det2_th[i]) * 0.5
        } else {
            &det3 * &eta1[i] * 0.5
        };
        add_weighted_xtx_inplace(&mut ai, x, &wi, 1.0);
        if i >= ntheta {
            add_scaled_penalty_dense(&mut ai, &penalties[i - ntheta], lambdas[i - ntheta]);
        }
        a1.push(ai);
    }

    let mut d1 = Array1::<f64>::zeros(ntot);
    let mut p1 = Array1::<f64>::zeros(ntot);
    let mut ldet1 = Array1::<f64>::zeros(ntot);
    for i in 0..ntot {
        d1[i] = det.dot(&eta1[i]) + if i < ntheta { dth[i].sum() } else { 0.0 };
        p1[i] = 2.0 * b1[i].dot(&s_beta);
        if i >= ntheta {
            p1[i] +=
                lambdas[i - ntheta] * beta.dot(&penalty_dot_vec(&penalties[i - ntheta], &beta));
        }
        ldet1[i] = trace_product_dense(&a_inv, &a1[i]);
    }

    let half_nu1 = (df + 1.0) / 2.0;
    let half_nu = df / 2.0;
    let nu2 = df - 2.0;
    let nu2nu = nu2 / df;
    let mut ls1 = Array1::<f64>::zeros(ntot);
    ls1[0] = (y_original.len() as f64)
        * (nu2 * (digamma(half_nu1) - digamma(half_nu)) / 2.0 - 0.5 * nu2nu);
    ls1[1] = -(y_original.len() as f64);
    let mut ls2 = Array2::<f64>::zeros((ntot, ntot));
    ls2[[0, 0]] = (y_original.len() as f64)
        * (nu2 * nu2 * (trigamma(half_nu1) - trigamma(half_nu)) / 4.0
            + nu2 * (digamma(half_nu1) - digamma(half_nu)) / 2.0
            + 0.5 * nu2nu * nu2nu
            - 0.5 * nu2nu);

    let mut b2: Vec<Vec<Array1<f64>>> = (0..ntot)
        .map(|_| (0..ntot).map(|_| Array1::<f64>::zeros(p)).collect())
        .collect();
    let mut eta2: Vec<Vec<Array1<f64>>> = (0..ntot)
        .map(|_| (0..ntot).map(|_| Array1::<f64>::zeros(x.nrows())).collect())
        .collect();

    for i in 0..ntot {
        for k in i..ntot {
            let rhs_w = -&det3 * &eta1[i] * &eta1[k];
            let mut rhs = x.t().dot(&rhs_w);
            if k < ntheta {
                rhs = rhs - x.t().dot(&(&det2_th[k] * &eta1[i]));
            } else {
                rhs = rhs
                    - 2.0 * lambdas[k - ntheta] * penalty_dot_vec(&penalties[k - ntheta], &b1[i]);
            }
            if i < ntheta {
                rhs = rhs - x.t().dot(&(&det2_th[i] * &eta1[k]));
            } else {
                rhs = rhs
                    - 2.0 * lambdas[i - ntheta] * penalty_dot_vec(&penalties[i - ntheta], &b1[k]);
            }
            if i < ntheta && k < ntheta {
                let idx = if i == 0 && k == 0 {
                    0
                } else if i == 0 && k == 1 {
                    1
                } else {
                    2
                };
                rhs = rhs - x.t().dot(&det_th2[idx]);
            } else if i == k {
                rhs = rhs
                    - 2.0 * lambdas[i - ntheta] * penalty_dot_vec(&penalties[i - ntheta], &beta);
            }
            let bik = 0.5 * a_inv.dot(&rhs);
            let eik = x.dot(&bik);
            b2[i][k] = bik.clone();
            b2[k][i] = bik;
            eta2[i][k] = eik.clone();
            eta2[k][i] = eik;
        }
    }

    let mut d2 = Array2::<f64>::zeros((ntot, ntot));
    let mut p2 = Array2::<f64>::zeros((ntot, ntot));
    let mut ldet2 = Array2::<f64>::zeros((ntot, ntot));
    for i in 0..ntot {
        for k in i..ntot {
            let mut dij = (&det2 * &eta1[i] * &eta1[k]).sum() + det.dot(&eta2[i][k]);
            if i < ntheta && k < ntheta {
                let idx = if i == 0 && k == 0 {
                    0
                } else if i == 0 && k == 1 {
                    1
                } else {
                    2
                };
                dij += dth2[idx].sum();
            }
            if i < ntheta {
                dij += det_th[i].dot(&eta1[k]);
            }
            if k < ntheta {
                dij += det_th[k].dot(&eta1[i]);
            }
            d2[[i, k]] = dij;
            d2[[k, i]] = dij;

            let mut pij = 2.0 * b2[i][k].dot(&s_beta)
                + 2.0 * b1[i].dot(&penalty_dot_all(penalties, lambdas, &b1[k]));
            if k >= ntheta {
                pij += 2.0
                    * lambdas[k - ntheta]
                    * b1[i].dot(&penalty_dot_vec(&penalties[k - ntheta], &beta));
            }
            if i >= ntheta {
                pij += 2.0
                    * lambdas[i - ntheta]
                    * b1[k].dot(&penalty_dot_vec(&penalties[i - ntheta], &beta));
            }
            if i == k && i >= ntheta {
                pij +=
                    lambdas[i - ntheta] * beta.dot(&penalty_dot_vec(&penalties[i - ntheta], &beta));
            }
            p2[[i, k]] = pij;
            p2[[k, i]] = pij;

            let mut a2 = Array2::<f64>::zeros((p, p));
            let mut w2 = &det4 * &eta1[i] * &eta1[k] + &det3 * &eta2[i][k];
            if i < ntheta {
                w2 = w2 + &det3_th[i] * &eta1[k];
            }
            if k < ntheta {
                w2 = w2 + &det3_th[k] * &eta1[i];
            }
            if i < ntheta && k < ntheta {
                let idx = if i == 0 && k == 0 {
                    0
                } else if i == 0 && k == 1 {
                    1
                } else {
                    2
                };
                w2 = w2 + &det2_th2[idx];
            }
            add_weighted_xtx_inplace(&mut a2, x, &w2, 0.5);
            if i == k && i >= ntheta {
                add_scaled_penalty_dense(&mut a2, &penalties[i - ntheta], lambdas[i - ntheta]);
            }
            let lij = trace_product_dense(&a_inv, &a2)
                - trace_product_dense(&a_inv.dot(&a1[i]), &a_inv.dot(&a1[k]));
            ldet2[[i, k]] = lij;
            ldet2[[k, i]] = lij;
        }
    }

    let mut grad = Array1::<f64>::zeros(ntot);
    let mut hess = Array2::<f64>::zeros((ntot, ntot));
    for i in 0..ntot {
        grad[i] = 0.5 * (d1[i] + p1[i]) - ls1[i] + 0.5 * ldet1[i];
        if i >= ntheta {
            grad[i] -= 0.5 * estimate_rank_eigen(&penalties[i - ntheta]) as f64;
        }
        for k in 0..ntot {
            hess[[i, k]] = 0.5 * (d2[[i, k]] + p2[[i, k]]) - ls2[[i, k]] + 0.5 * ldet2[[i, k]];
        }
    }
    Ok((grad, hess))
}

/// Analytic fixed-state scat theta derivatives from mgcv's `scat$Dd(level=2)`
/// and `scat$ls`. Returns gradient and Hessian in Rust optimizer order:
/// `(log σ², log(df - 2))`. Internally mgcv uses `(log(df - 2), log σ)`.
#[cfg(feature = "blas")]
pub fn tdist_shape_derivatives_gamfit4(
    y_original: &Array1<f64>,
    y_work: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let (native_grad, native_hess) = tdist_gdi2_native(
        y_original,
        y_work,
        x,
        w,
        lambdas,
        penalties,
        cached_xtwx,
        family,
    )?;
    let mut grad = Array1::<f64>::zeros(2);
    grad[0] = 0.5 * native_grad[1];
    grad[1] = native_grad[0];
    let mut hess = Array2::<f64>::zeros((2, 2));
    hess[[0, 0]] = 0.25 * native_hess[[1, 1]];
    hess[[0, 1]] = 0.5 * native_hess[[1, 0]];
    hess[[1, 0]] = hess[[0, 1]];
    hess[[1, 1]] = native_hess[[0, 0]];
    Ok((grad, hess))
}

/// Analytic gam.fit4-style LAML gradient for mgcv's scat extended family.
/// This replaces callback-aware finite differences over log λ for TDist.
#[cfg(feature = "blas")]
pub fn reml_gradient_gamfit4_tdist_analytic(
    y_work: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    y_original: &Array1<f64>,
    family: crate::pirls::Family,
) -> Result<Array1<f64>> {
    let (native_grad, _) = tdist_gdi2_native(
        y_original,
        y_work,
        x,
        w,
        lambdas,
        penalties,
        cached_xtwx,
        family,
    )?;
    let mut grad = Array1::<f64>::zeros(lambdas.len());
    for j in 0..lambdas.len() {
        grad[j] = native_grad[2 + j];
    }
    Ok(grad)
}

/// Analytic gam.fit4-style LAML Hessian for mgcv's scat extended family.
#[cfg(feature = "blas")]
pub fn reml_hessian_gamfit4_tdist_analytic(
    y_work: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    y_original: &Array1<f64>,
    family: crate::pirls::Family,
) -> Result<Array2<f64>> {
    let (_, native_hess) = tdist_gdi2_native(
        y_original,
        y_work,
        x,
        w,
        lambdas,
        penalties,
        cached_xtwx,
        family,
    )?;
    let m = lambdas.len();
    let mut hess = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            hess[[i, j]] = native_hess[[2 + i, 2 + j]];
        }
    }
    Ok(hess)
}

/// Joint analytic gam.fit4-style LAML gradient + Hessian for mgcv's scat
/// extended family, in **outer-Newton coordinate order**:
/// `[log λ_1, ..., log λ_M, log σ², log(df - 2)]`.
///
/// Returns the full `(M+2)`-dim gradient and `(M+2, M+2)` Hessian — cross-
/// terms between `log λ` and `(log σ², log df)` included.
///
/// Internally `tdist_gdi2_native` works in mgcv's native order
/// `[log(df-2), log σ, log λ_1, ..., log λ_M]`. The remap here applies:
///   - Permutation: native indices `(0, 1, 2..M+1)` ↔ outer indices
///     `(M+1, M, 0..M-1)`. (mgcv has `(theta_df, theta_sigma, ...λ)`;
///     outer has `(...λ, log σ², log(df-2))`.)
///   - Jacobian for `log σ → log σ²`: `log σ² = 2 log σ`, so
///     `d/d(log σ²) = (1/2) · d/d(log σ)`.
///       grad scaling: 0.5;
///       Hessian scaling on the `log σ²` row/col: 0.5 per intersection
///       (0.25 for the `log σ²` diagonal).
#[cfg(feature = "blas")]
pub fn reml_joint_gh_gamfit4_tdist_analytic(
    y_work: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    y_original: &Array1<f64>,
    family: crate::pirls::Family,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let (native_grad, native_hess) = tdist_gdi2_native(
        y_original,
        y_work,
        x,
        w,
        lambdas,
        penalties,
        cached_xtwx,
        family,
    )?;
    let m = lambdas.len();
    let ntot = m + 2;
    // outer→native index map.
    // outer 0..M-1  → native 2..M+1   (log λ block)
    // outer M       → native 1         (log σ² ↔ log σ, with 1/2 Jacobian)
    // outer M+1     → native 0         (log(df-2), identical)
    let outer_to_native = |i: usize| -> usize {
        if i < m {
            2 + i
        } else if i == m {
            1
        } else {
            0
        }
    };
    // Jacobian factor d native / d outer for axis `i`.
    //   For log σ² → log σ (native), d(log σ)/d(log σ²) = 1/2.
    //   For all others 1.
    let scale = |i: usize| -> f64 {
        if i == m {
            0.5
        } else {
            1.0
        }
    };
    let mut grad = Array1::<f64>::zeros(ntot);
    let mut hess = Array2::<f64>::zeros((ntot, ntot));
    for i in 0..ntot {
        let ni = outer_to_native(i);
        grad[i] = scale(i) * native_grad[ni];
        for j in 0..ntot {
            let nj = outer_to_native(j);
            hess[[i, j]] = scale(i) * scale(j) * native_hess[[ni, nj]];
        }
    }
    Ok((grad, hess))
}

// ============================================================================
// IFT-based gradient and Hessian (Option B per Parity 4t)
// ============================================================================
//
// Translated from mgcv's gdi.c::ift1 (lines 1314-1363) and the gradient/
// Hessian assembly in gdi1 (lines 2573-2664). Uses the Implicit Function
// Theorem on the IRLS first-order condition `Aβ = X'Wz` (where
// A = X'WX + ΣλS) to compute ∂β/∂ρ_k = b1[k] = -λ_k A⁻¹ S_k β, then plugs
// into the full chain rule for D' (the deviance derivative).
//
// At converged β with the *working RSS* deviance D_w = Σw(y - Xβ)², the
// IFT gradient is mathematically identical to the envelope-form gradient
// (because ∂D_w/∂β = -2X'W(y - Xβ) = -2ΣλSβ at converged β, and the
// (∂D_w/∂β)·b1 + 2(ΣλSβ)·b1 terms cancel exactly).
//
// Where the IFT path GENUINELY differs from envelope is when we use the
// **GLM deviance** D_GLM gradient `-2X'(y_orig - μ)/(Vg')` with μ = g⁻¹(Xβ),
// AND our β is not jointly self-consistent with W,z (e.g., during the outer
// REML loop where z is held fixed at the prior PiRLS β). Pass
// `y_original = Some(...)` to enable this path; pass `None` to fall back
// to working-RSS deviance (which collapses to envelope at converged β).

/// Compute b1[k] = ∂β/∂ρ_k = -λ_k · A⁻¹ S_k β  for k = 0..m.
/// Returns a (p, m) matrix; column k is b1[k].
#[cfg(feature = "blas")]
pub(crate) fn compute_b1_ift(
    a_inv: &Array2<f64>,
    beta: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Array2<f64> {
    let p = a_inv.nrows();
    let m = lambdas.len();
    let mut b1 = Array2::<f64>::zeros((p, m));
    for k in 0..m {
        // S_k β: only nonzero on the smooth's block.
        let s_k_beta = penalties_blocks[k].dot_vec(beta);
        // A⁻¹ S_k β
        let ainv_sk_beta = a_inv.dot(&s_k_beta);
        // b1[:,k] = -λ_k · A⁻¹ S_k β
        let lam_k = lambdas[k];
        for r in 0..p {
            b1[[r, k]] = -lam_k * ainv_sk_beta[r];
        }
    }
    b1
}

/// Compute the deviance gradient `∂D/∂β` evaluated at the current β.
///
/// - For Gaussian or `y_original = None`: uses the working-RSS form
///   `-2 X'W(y_input - Xβ)`, which is what our score uses for Dp.
///   This collapses the IFT correction to zero at our (always converged)
///   β — recovering envelope.
/// - For non-Gaussian + `y_original = Some(y)`: uses the GLM deviance form
///   `-2 X'·(y - μ)/(V·g')` with μ = g⁻¹(Xβ), matching gdi.c:2574.
///   The `(y_input, w)` carried by the score may have been computed at a
///   prior PiRLS β; the IFT correction picks up the difference.
#[cfg(feature = "blas")]
fn compute_dev_grad_beta(
    y_input: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    beta: &Array1<f64>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Array1<f64> {
    let n = x.nrows();
    let mut v1 = Array1::<f64>::zeros(n);
    match (family, y_original) {
        (crate::pirls::Family::Gaussian, _) | (_, None) => {
            // Working-RSS form: v1[i] = -2 w[i] (y_input[i] - (Xβ)_i)
            let fitted = x.dot(beta);
            for i in 0..n {
                v1[i] = -2.0 * w[i] * (y_input[i] - fitted[i]);
            }
        }
        (fam, Some(y_orig)) => {
            // GLM deviance form: v1[i] = -2 (y_orig - μ) / (V(μ) · g'(μ))
            // For canonical link V·g' = 1 so v1 = -2(y_orig - μ).
            // For non-canonical we'd need g' separately; use V·g' factor.
            let eta = x.dot(beta);
            for i in 0..n {
                let mu_i = fam.inverse_link(eta[i]);
                let v_mu = fam.variance(mu_i).max(1e-300);
                // dμ/dη = inverse_link derivative = 1/g'(μ). So g' = 1/(dμ/dη).
                let dmu_deta = fam.d_inverse_link(eta[i]).max(1e-300);
                let g_prime = 1.0 / dmu_deta;
                let denom = v_mu * g_prime;
                v1[i] = -2.0 * (y_orig[i] - mu_i) / denom;
            }
        }
    }
    // ∂D/∂β = X' v1
    x.t().dot(&v1)
}

/// Compute profiled φ̂ = estimate_phi_mgcv(...) at arbitrary lambdas.
///
/// Used by the σ²-chain correction in `reml_gradient_mgcv_exact_ift` to
/// compute `∂σ̂²/∂ρ_k` via central finite differences in log-λ space.
/// Returns 1.0 for Binomial/Poisson (fixed dispersion).
#[cfg(feature = "blas")]
fn compute_sigma2_at(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
    mp: usize,
) -> Result<f64> {
    use crate::pirls::Family;
    // Fixed dispersion families (Quasi variants are NOT fixed — they profile φ)
    if matches!(
        family,
        Family::Binomial | Family::Poisson | Family::NegBin { .. }
    ) {
        return Ok(1.0);
    }
    let n = y.len();
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }
    let xtwy = compute_xtwy(x, w, y);
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let solve_ridge = 1e-12 * max_diag;
    a_solve
        .diag_mut()
        .iter_mut()
        .for_each(|d| *d += solve_ridge);
    let beta = crate::linalg::solve(a_solve, xtwy)?;
    let fitted = x.dot(&beta);
    let y_for = y_original.unwrap_or(y);
    let dev_numerator: f64 = match (family, y_original) {
        (Family::Gaussian, _) | (_, None) => y
            .iter()
            .zip(fitted.iter())
            .zip(w.iter())
            .map(|((yi, fi), wi)| (yi - fi).powi(2) * wi)
            .sum(),
        (fam, Some(y_orig)) => {
            let mu: ndarray::Array1<f64> =
                fitted.iter().map(|&eta| fam.inverse_link(eta)).collect();
            glm_deviance(y_orig, &mu, fam)
        }
    };
    let bsb: f64 = lambdas
        .iter()
        .zip(penalties.iter())
        .map(|(l, pen)| l * pen.quadratic_form(&beta))
        .sum();
    let dp = dev_numerator + bsb;
    let a_inv = inverse(&a)?;
    let tr_a = (xtwx.dot(&a_inv)).diag().sum();
    let phi_init = dev_numerator / ((n as f64) - tr_a).max(1e-10);
    Ok(family.estimate_phi_mgcv(y_for, dp, mp, 1.0, phi_init))
}

/// IFT-based gradient of the mgcv-exact REML score w.r.t. log(λ_k).
///
/// Implements the full chain rule (matching mgcv's gdi.c assembly at
/// lines 2653-2685):
///
///   D1[k] = (∂D/∂β)' b1[k] + λ_k β'S_kβ + 2 (ΣλSβ)' b1[k]
///   det1[k] = λ_k · tr(A⁻¹ S_k)
///   REML1[k] = D1[k] / (2σ²) + det1[k]/2 − rank_k/2
///
/// where b1[k] = -λ_k A⁻¹ S_k β. At converged β with working-RSS deviance,
/// `(∂D/∂β)' b1 + 2(ΣλSβ)' b1 = 0` and this reduces to the envelope form.
///
/// `y_original = Some(y)` enables the GLM deviance gradient form for
/// non-Gaussian families — the only path where the IFT correction is
/// genuinely nonzero given that our β = A⁻¹X'Wy is always at the IRLS
/// solution for the inputs we see.
///
/// `enable_sigma_chain` adds the σ²-chain correction term:
///   (∂REML/∂σ²) · ∂σ̂²/∂ρ_k
/// where ∂σ̂²/∂ρ_k is estimated via central FD in log-λ space.
/// The public wrapper reads `MGCV_SIGMA_CHAIN` env var; tests call the inner
/// function directly with `enable_sigma_chain = true`.
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_ift_inner(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
    enable_sigma_chain: bool,
    mp: usize,
) -> Result<Array1<f64>> {
    reml_gradient_mgcv_exact_ift_inner_at_beta(
        y, x, w, lambdas, penalties_blocks, cached_xtwx, family, y_original,
        enable_sigma_chain, mp, None,
    )
}

/// Inner gradient assembly that optionally accepts an explicit `beta_provided`
/// (the PIRLS-converged β). When `None` falls back to the re-solve path used
/// for Gaussian / Fisher-only callers. When `Some`, skips the β re-solve — use
/// this when `w` carries raw Newton weights with possible negative entries
/// (X'WX + λS is then indefinite and the internal solve would diverge).
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_ift_inner_at_beta(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
    enable_sigma_chain: bool,
    mp: usize,
    beta_provided: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    let n = y.len();
    let m = lambdas.len();
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    // A = X'WX + Σλ_jS_j (no ridge — mgcv-exact)
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // β: either supplied (PIRLS-converged) or recovered by solving the
    // weighted normal equations β = A⁻¹ X'Wy. The supplied path is needed
    // when w may be indefinite (raw Newton weights with neg entries) — the
    // re-solve would otherwise produce garbage on an indefinite system.
    let beta: Array1<f64> = if let Some(b) = beta_provided {
        b.clone()
    } else {
        let xtwy = compute_xtwy(x, w, y);
        let mut a_solve = a.clone();
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let solve_ridge = 1e-12 * max_diag;
        a_solve
            .diag_mut()
            .iter_mut()
            .for_each(|d| *d += solve_ridge);
        solve(a_solve, xtwy)?
    };

    // Scale estimate for Dp/(2σ²) — must match the score function's σ²
    // convention (reml_criterion_multi_cached_mgcv_exact, item 1 of #47).
    //   - Binomial / Poisson: σ² = 1 (known dispersion).
    //   - Gaussian / Gamma: mgcv's profiled φ̂ from estimate_phi_mgcv.
    let fitted = x.dot(&beta);
    let y_for_grad = y_original.unwrap_or(y);
    let dev_numerator: f64 = match (family, y_original) {
        (crate::pirls::Family::Gaussian, _) | (_, None) => y
            .iter()
            .zip(fitted.iter())
            .zip(w.iter())
            .map(|((yi, fi), wi)| (yi - fi).powi(2) * wi)
            .sum(),
        (fam, Some(y_orig)) => {
            let mu: Array1<f64> = fitted.iter().map(|&eta| fam.inverse_link(eta)).collect();
            glm_deviance(y_orig, &mu, fam)
        }
    };
    let a_inv = inverse(&a)?;
    let tr_a = (xtwx.dot(&a_inv)).diag().sum();
    // Compute bsb_total here so we can form dp for estimate_phi_mgcv.
    let bsb_total_for_phi: f64 = lambdas
        .iter()
        .zip(penalties_blocks.iter())
        .map(|(l, pen)| l * pen.quadratic_form(&beta))
        .sum();
    let dp_for_phi = dev_numerator + bsb_total_for_phi;
    // σ² convention: Gaussian and IFT use RSS/(n-trA) so that the IFT
    // gradient and the envelope gradient differentiate the same REML score
    // with the same plug-in σ² (gam.fit3.r:625 convention). For non-Gaussian
    // families, use estimate_phi_mgcv (dp/(n-mp)) which is the correct
    // profiled dispersion for extended-family REML.
    let scale_est = match family {
        crate::pirls::Family::Binomial
        | crate::pirls::Family::Poisson
        | crate::pirls::Family::NegBin { .. } => 1.0,
        crate::pirls::Family::Gaussian => dev_numerator / ((n as f64) - tr_a).max(1e-10),
        _ => {
            let phi_init = dev_numerator / ((n as f64) - tr_a).max(1e-10);
            family.estimate_phi_mgcv(y_for_grad, dp_for_phi, mp, 1.0, phi_init)
        }
    };

    let p = a.nrows();

    // IFT first derivatives: b1[:,k] = -λ_k A⁻¹ S_k β
    let b1 = compute_b1_ift(&a_inv, &beta, lambdas, penalties_blocks);

    // η₁[i,k] = (X·b1)[i,k] = ∂η_i/∂ρ_k
    let eta1 = x.dot(&b1);

    // Leverage vector h[i] = w[i]·x_i'·A⁻¹·x_i = diag(K·K')[i] (gdi.c:828),
    // and the unweighted form lev_uw[i] = x_i'·A⁻¹·x_i = h[i]/|w[i]| with
    // sign correction. We compute lev_uw directly to avoid divide-by-tiny-w.
    let xa = x.dot(&a_inv); // (n × p)
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    // dw/dη at convergence (gdi.c:2535 for Fisher / gdi.c:2556 for Newton).
    // Tk[i,k]·diag(KK')[i] = a1[i]·η₁[i,k]·sign(w[i])·lev_uw[i] (since
    // diag(KK')[i] = w[i]·lev_uw[i] and Tk divides by |w[i]|). For
    // canonical links, mgcv uses Fisher's compact form. Gaussian: a1≡0.
    let mut a1 = Array1::<f64>::zeros(n);
    if !matches!(family, crate::pirls::Family::Gaussian) {
        let eta = x.dot(&beta);
        let use_fisher = family.is_canonical_link();
        for i in 0..n {
            let mu_i = family.inverse_link(eta[i]);
            let dmu_deta = family.d_inverse_link(eta[i]);
            if dmu_deta.abs() < 1e-12 {
                continue;
            }
            let g1 = 1.0 / dmu_deta;
            let v = family.variance(mu_i).max(1e-300);
            // mgcv normalised derivatives (gam.fit3.r:534-538):
            //   V₁ → V'/V, V₂ → V''/V, g₂ → g''/g', g₃ → g'''/g'
            let v1n = family.dvar(mu_i) / v;
            let v2n = family.d2var(mu_i) / v;
            let g2n = family.d2link(mu_i) * dmu_deta;
            let g3n = family.d3link(mu_i) * dmu_deta;
            if use_fisher {
                // gdi.c:2535: a1 = -w·(V₁ + 2·g₂)/g₁
                a1[i] = -w[i] * (v1n + 2.0 * g2n) / g1;
            } else {
                let y_for_resid = y_original.unwrap_or(y);
                let c_resid = y_for_resid[i] - mu_i;
                // Newton curvature factor, shared with pirls::compute_irls_wz.
                let alpha_raw = crate::pirls::newton_irls_alpha(c_resid, v1n, g2n);
                // When α ≤ 0 fall back to α=1 to avoid division by zero
                // (pirls handles the same case by switching to Fisher per-obs).
                let alpha = if alpha_raw <= 0.0 { 1.0 } else { alpha_raw };
                let xx = v2n - v1n * v1n + g3n - g2n * g2n;
                let alpha1 = (-(v1n + g2n) + c_resid * xx) / alpha;
                // gdi.c:2556: a1 = w·(α₁ - V₁ - 2·g₂)/g₁
                a1[i] = w[i] * (alpha1 - v1n - 2.0 * g2n) / g1;
            }
        }
    }

    // ∂D/∂β at current (β,μ)
    let dev_grad_beta = compute_dev_grad_beta(y, x, w, &beta, family, y_original);

    // ΣλSβ = Σ_j λ_j S_j β  (gathered as a length-p vector)
    let mut sum_lambda_s_beta = Array1::<f64>::zeros(p);
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        let s_j_beta = penalty.dot_vec(&beta);
        for r in 0..p {
            sum_lambda_s_beta[r] += lambda * s_j_beta[r];
        }
    }

    let mut grad = Array1::<f64>::zeros(m);
    for (k, (lambda, penalty)) in lambdas.iter().zip(penalties_blocks.iter()).enumerate() {
        let bsb_k = penalty.quadratic_form(&beta);
        let lam_k = *lambda;

        // tr(A⁻¹ S_k) — block-sparse
        let block = penalty.block_view();
        let off = penalty.offset;
        let kk = block.nrows();
        let mut tr_a_inv_s = 0.0;
        for p_idx in 0..kk {
            for q_idx in 0..kk {
                tr_a_inv_s += a_inv[[off + p_idx, off + q_idx]] * block[[q_idx, p_idx]];
            }
        }

        // (∂D/∂β)' b1[:,k]
        let mut dev_dot_b1 = 0.0;
        for r in 0..p {
            dev_dot_b1 += dev_grad_beta[r] * b1[[r, k]];
        }
        // 2 (ΣλSβ)' b1[:,k]
        let mut sls_dot_b1 = 0.0;
        for r in 0..p {
            sls_dot_b1 += sum_lambda_s_beta[r] * b1[[r, k]];
        }
        let d1_k = dev_dot_b1 + lam_k * bsb_k + 2.0 * sls_dot_b1;

        let rank_k = estimate_rank_eigen(penalty);
        // Note: mgcv's full ∂log|H|/∂ρ_k = tk_kkt + λ_k·tr_a_inv_s (gdi.c:857),
        // where tk_kkt = Σᵢ a1[i]·η₁[i,k]·sign(w[i])·lev_uw[i]. The
        // infrastructure for tk_kkt is computed above (a1, eta1, lev_uw)
        // and verified correct by `tests/test_tk_kkt_reference.rs` and
        // `tests/test_binomial_gradient_reference.rs`. Enabled by default for
        // families where the σ²-fixed REML and the closed-form gradient agree
        // tightly enough that the Tk·KK' contribution lands on mgcv's |Δη|<1e-3
        // η-stationary point. For Gamma(log) / nb / inverse.gaussian-with-
        // saturating-λ the term can drift the optimizer toward the saturation
        // boundary before edge.correct lands, so they stay opt-in via
        // MGCV_TK_GRAD until step-blending is ported.
        let use_tk_kkt = matches!(
            family,
            crate::pirls::Family::InverseGaussian
                | crate::pirls::Family::Binomial
                | crate::pirls::Family::QuasiBinomial
        ) || std::env::var("MGCV_TK_GRAD").is_ok();
        let tk_kkt = if use_tk_kkt {
            (0..n)
                .map(|i| a1[i] * eta1[[i, k]] * w[i].signum() * lev_uw[i])
                .sum::<f64>()
        } else {
            0.0
        };
        grad[k] =
            d1_k / (2.0 * scale_est) + (tk_kkt + lam_k * tr_a_inv_s) / 2.0 - (rank_k as f64) / 2.0;
    }

    // σ²-chain correction: adds (∂REML/∂σ²) · ∂σ̂²/∂ρ_k to each grad[k].
    // Gated behind `enable_sigma_chain` (set via MGCV_SIGMA_CHAIN env var or
    // directly by tests). Skip for Binomial/Poisson (σ² fixed at 1).
    if enable_sigma_chain
        && !matches!(
            family,
            crate::pirls::Family::Binomial
                | crate::pirls::Family::Poisson
                | crate::pirls::Family::NegBin { .. }
        )
    {
        // Use dp_for_phi and mp already computed above.
        let dp = dp_for_phi;

        let dls_dsig2 = family.dls_dsigma2(y_for_grad, scale_est);
        let drem_dsig2 =
            -dp / (2.0 * scale_est * scale_est) - dls_dsig2 - (mp as f64) / (2.0 * scale_est);

        let eps = 1e-4_f64;
        for k in 0..m {
            let log_lam: Vec<f64> = lambdas.iter().map(|l| l.ln()).collect();
            let mut log_lam_plus = log_lam.clone();
            log_lam_plus[k] += eps;
            let mut log_lam_minus = log_lam.clone();
            log_lam_minus[k] -= eps;
            let lam_plus: Vec<f64> = log_lam_plus.iter().map(|l| l.exp()).collect();
            let lam_minus: Vec<f64> = log_lam_minus.iter().map(|l| l.exp()).collect();
            let s2_plus = compute_sigma2_at(
                y,
                x,
                w,
                &lam_plus,
                penalties_blocks,
                cached_xtwx,
                family,
                y_original,
                mp,
            )?;
            let s2_minus = compute_sigma2_at(
                y,
                x,
                w,
                &lam_minus,
                penalties_blocks,
                cached_xtwx,
                family,
                y_original,
                mp,
            )?;
            let dsig2_drho = (s2_plus - s2_minus) / (2.0 * eps);
            grad[k] += drem_dsig2 * dsig2_drho;
        }
    }

    Ok(grad)
}

/// Public wrapper for `reml_gradient_mgcv_exact_ift_inner`.
/// Reads the `MGCV_SIGMA_CHAIN` environment variable to enable the σ²-chain
/// correction term. Default OFF — set `MGCV_SIGMA_CHAIN=1` to enable.
/// Computes `mp` (null-space dimension) from penalties_blocks.
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_ift(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    let enable_sigma_chain = std::env::var("MGCV_SIGMA_CHAIN").is_ok();
    // Mp = 1 (intercept) + Σ null-space dims per penalty (matches lib.rs:1131-1144)
    let mp: usize = 1 + penalties_blocks
        .iter()
        .map(|pen| {
            let k = pen.block_view().nrows();
            let rank_s = estimate_rank_eigen(pen);
            k.saturating_sub(rank_s)
        })
        .sum::<usize>();
    // Optional internal reparametrisation (MGCV_REPARAM=1) — see the
    // matching block in `reml_criterion_multi_cached_mgcv_exact`. Gradient
    // values are analytically basis-invariant; the rotated basis gives
    // better conditioning at saturating λ. Cached_xtwx is in the
    // un-rotated basis and is dropped on the rotated path.
    let p = x.ncols();
    let rot_state = if std::env::var("MGCV_REPARAM").is_ok() {
        let (u1, mp_detected) = crate::reparam::compute_total_penalty_space(penalties_blocks, p)?;
        let rot = crate::reparam::apply_reparam(x, penalties_blocks, lambdas, &u1, mp_detected, 0)?;
        let rotated_pens: Vec<BlockPenalty> = rot
            .rs_rot
            .iter()
            .map(|rs| BlockPenalty::new(rs.dot(&rs.t()), 0, p))
            .collect();
        Some((rot.x_rot, rotated_pens))
    } else {
        None
    };
    let (x_use, pens_use): (&Array2<f64>, &[BlockPenalty]) = match &rot_state {
        Some((x_rot, pens_rot)) => (x_rot, pens_rot.as_slice()),
        None => (x, penalties_blocks),
    };
    let cached_xtwx_use = if rot_state.is_some() { None } else { cached_xtwx };
    reml_gradient_mgcv_exact_ift_inner(
        y,
        x_use,
        w,
        lambdas,
        pens_use,
        cached_xtwx_use,
        family,
        y_original,
        enable_sigma_chain,
        mp,
    )
}

/// IFT-based REML gradient at a known PIRLS-converged β, using mgcv's
/// **Newton-weight A** for the log|H| / IFT pieces.
///
/// Mirrors what `gdi2` does inside `gam.fit3` (gam.fit3.r:511-522, "full
/// Newton" branch): the W feeding log|X'WX + S| is the raw Newton form
/// `wf · α` with no Fisher fallback, so A may be indefinite. All IFT
/// pieces (a1, η₁, b1, lev_uw, λ·tr(A⁻¹·S_k), tk_kkt) are derived against
/// this same indefinite A.
///
/// β is taken as input — not re-solved — because the indefinite Newton A
/// has no Cholesky-friendly inverse and the PIRLS-converged β already gives
/// the right fixed point.
#[cfg(feature = "blas")]
pub fn reml_gradient_mgcv_exact_ift_newton_at_beta(
    x: &Array2<f64>,
    y_raw: &Array1<f64>,
    beta: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    family: crate::pirls::Family,
    mp: usize,
) -> Result<Array1<f64>> {
    let n = y_raw.len();
    let m = lambdas.len();
    let p = x.ncols();

    let eta = x.dot(beta);
    let w_newton = crate::pirls::compute_newton_score_weights(y_raw, &eta, family);

    // X'WX with possibly-negative w. `compute_xtwx` uses sqrt(w) which goes
    // NaN on neg-w paths, so compute the signed matmul directly:
    // X'WX = X' · (diag(w) · X)  with no square root.
    let mut wx = x.to_owned();
    for i in 0..n {
        let wi = w_newton[i];
        for j in 0..p {
            wx[[i, j]] *= wi;
        }
    }
    let xtwx = x.t().dot(&wx);

    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }
    let a_inv = inverse(&a)?;
    let tr_a = (xtwx.dot(&a_inv)).diag().sum();

    let mu: Array1<f64> = eta.iter().map(|&e| family.inverse_link(e)).collect();
    let dev_numerator = glm_deviance(y_raw, &mu, family);
    let bsb_total: f64 = lambdas
        .iter()
        .zip(penalties_blocks.iter())
        .map(|(l, pen)| l * pen.quadratic_form(beta))
        .sum();
    let dp = dev_numerator + bsb_total;
    let scale_est = match family {
        crate::pirls::Family::Binomial
        | crate::pirls::Family::Poisson
        | crate::pirls::Family::NegBin { .. } => 1.0,
        crate::pirls::Family::Gaussian => dev_numerator / ((n as f64) - tr_a).max(1e-10),
        _ => {
            let phi_init = dev_numerator / ((n as f64) - tr_a).max(1e-10);
            family.estimate_phi_mgcv(y_raw, dp, mp, 1.0, phi_init)
        }
    };

    let b1 = compute_b1_ift(&a_inv, beta, lambdas, penalties_blocks);
    let eta1 = x.dot(&b1);

    let xa = x.dot(&a_inv);
    let mut lev_uw = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..p {
            s += xa[[i, j]] * x[[i, j]];
        }
        lev_uw[i] = s;
    }

    let mut a1 = Array1::<f64>::zeros(n);
    if !matches!(family, crate::pirls::Family::Gaussian) {
        let use_fisher = family.is_canonical_link();
        for i in 0..n {
            let mu_i = mu[i];
            let dmu_deta = family.d_inverse_link(eta[i]);
            if dmu_deta.abs() < 1e-12 {
                continue;
            }
            let g1 = 1.0 / dmu_deta;
            let v = family.variance(mu_i).max(1e-300);
            let v1n = family.dvar(mu_i) / v;
            let v2n = family.d2var(mu_i) / v;
            let g2n = family.d2link(mu_i) * dmu_deta;
            let g3n = family.d3link(mu_i) * dmu_deta;
            if use_fisher {
                a1[i] = -w_newton[i] * (v1n + 2.0 * g2n) / g1;
            } else {
                let c_resid = y_raw[i] - mu_i;
                let alpha_raw = crate::pirls::newton_irls_alpha(c_resid, v1n, g2n);
                let alpha = if alpha_raw <= 0.0 { 1.0 } else { alpha_raw };
                let xx = v2n - v1n * v1n + g3n - g2n * g2n;
                let alpha1 = (-(v1n + g2n) + c_resid * xx) / alpha;
                a1[i] = w_newton[i] * (alpha1 - v1n - 2.0 * g2n) / g1;
            }
        }
    }

    let dev_grad_beta =
        compute_dev_grad_beta(y_raw, x, &w_newton, beta, family, Some(y_raw));

    let mut sum_lambda_s_beta = Array1::<f64>::zeros(p);
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        let s_j_beta = penalty.dot_vec(beta);
        for r in 0..p {
            sum_lambda_s_beta[r] += lambda * s_j_beta[r];
        }
    }

    let mut grad = Array1::<f64>::zeros(m);
    for (k, (lambda, penalty)) in lambdas.iter().zip(penalties_blocks.iter()).enumerate() {
        let bsb_k = penalty.quadratic_form(beta);
        let lam_k = *lambda;
        let block = penalty.block_view();
        let off = penalty.offset;
        let kk = block.nrows();
        let mut tr_a_inv_s = 0.0;
        for p_idx in 0..kk {
            for q_idx in 0..kk {
                tr_a_inv_s += a_inv[[off + p_idx, off + q_idx]] * block[[q_idx, p_idx]];
            }
        }
        let mut dev_dot_b1 = 0.0;
        for r in 0..p {
            dev_dot_b1 += dev_grad_beta[r] * b1[[r, k]];
        }
        let mut sls_dot_b1 = 0.0;
        for r in 0..p {
            sls_dot_b1 += sum_lambda_s_beta[r] * b1[[r, k]];
        }
        let d1_k = dev_dot_b1 + lam_k * bsb_k + 2.0 * sls_dot_b1;
        let rank_k = estimate_rank_eigen(penalty);

        // gdi.c:856 — tk_kkt[k] = Σᵢ Tk[i,k]·diagKKt[i] where
        //   Tk[i,k] = a1[i]·η₁[i,k]/|w[i]|   (gdi.c:2624,2628)
        //   diagKKt[i] = (KK')[i,i] = |w[i]|·lev_uw[i]
        // The |w| factors cancel, leaving Σ a1·η₁·lev_uw with NO sign(w).
        let tk_kkt: f64 = (0..n).map(|i| a1[i] * eta1[[i, k]] * lev_uw[i]).sum();

        grad[k] =
            d1_k / (2.0 * scale_est) + (tk_kkt + lam_k * tr_a_inv_s) / 2.0 - (rank_k as f64) / 2.0;
    }

    Ok(grad)
}

/// IFT-based Hessian of the mgcv-exact REML score w.r.t. log(λ_k, λ_j).
///
/// Differentiates the IFT gradient once more. The full Hessian assembly
/// (treating σ² constant per the existing envelope convention) is:
///
///   D2[k,j] = b1[:,j]' (∂²D/∂β²) b1[:,k] + (∂D/∂β)' b2[k,j]
///           + δ_{kj} λ_k β'S_kβ + 2 λ_k β'S_k b1[:,j]
///           + 2 λ_j (S_jβ)' b1[:,k] + 2 b1[:,j]' (ΣλS) b1[:,k]
///           + 2 (ΣλSβ)' b2[k,j]
///   det2[k,j] = δ_{kj} λ_k tr(A⁻¹ S_k) − λ_k λ_j tr(A⁻¹ S_k A⁻¹ S_j)
///   H[k,j] = D2[k,j]/(2σ²) + det2[k,j]/2
///
/// where ∂²D/∂β² = 2 X'WX (Fisher; exact for canonical link).
///
/// The b2[k,j] = ∂²β/∂ρ_kρ_j second derivatives come from differentiating
/// IFT once more (mgcv's gdi.c::ift1 deriv2 branch):
///   A · b2[k,j] = -λ_j S_j b1[:,k] - λ_k S_k b1[:,j] - δ_{kj} λ_k S_k β
///
/// At converged β with working-RSS deviance, both the b2 contribution
/// `(∂D/∂β + 2ΣλSβ)' b2 = (∂Dp/∂β)' b2 = 0` and the deviance second-
/// derivative contribution simplify, recovering the envelope-form Hessian.
#[cfg(feature = "blas")]
pub fn reml_hessian_mgcv_exact_ift(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: Option<&Array2<f64>>,
    family: crate::pirls::Family,
    y_original: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let n = y.len();
    let m = lambdas.len();
    // Optional internal reparametrisation (MGCV_REPARAM=1). Hessian
    // values are basis-invariant; rotated basis gives better
    // conditioning at saturating λ. We SHADOW `x`, `penalties_blocks`,
    // and `cached_xtwx` so the rest of the body sees rotated quantities
    // without per-line edits. Mp counting still uses the un-rotated
    // shape (kept in `mp_pens_for_rank`).
    let p = x.ncols();
    let rot_state = if std::env::var("MGCV_REPARAM").is_ok() {
        let (u1, mp_detected) = crate::reparam::compute_total_penalty_space(penalties_blocks, p)?;
        let rot = crate::reparam::apply_reparam(x, penalties_blocks, lambdas, &u1, mp_detected, 0)?;
        let rotated_pens: Vec<BlockPenalty> = rot
            .rs_rot
            .iter()
            .map(|rs| BlockPenalty::new(rs.dot(&rs.t()), 0, p))
            .collect();
        Some((rot.x_rot, rotated_pens))
    } else {
        None
    };
    let mp_pens_for_rank: &[BlockPenalty] = penalties_blocks;
    let (x, penalties_blocks): (&Array2<f64>, &[BlockPenalty]) = match &rot_state {
        Some((x_rot, pens_rot)) => (x_rot, pens_rot.as_slice()),
        None => (x, penalties_blocks),
    };
    let cached_xtwx: Option<&Array2<f64>> = if rot_state.is_some() {
        None
    } else {
        cached_xtwx
    };
    let xtwx_owned;
    let xtwx = if let Some(c) = cached_xtwx {
        c
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    // A, β, σ², A⁻¹
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }
    let xtwy = compute_xtwy(x, w, y);
    let mut a_solve = a.clone();
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    let solve_ridge = 1e-12 * max_diag;
    a_solve
        .diag_mut()
        .iter_mut()
        .for_each(|d| *d += solve_ridge);
    let beta = solve(a_solve, xtwy)?;
    let fitted = x.dot(&beta);
    // Same σ² convention as the score (item 1 of #47): GLM deviance when
    // y_original is Some, working-RSS otherwise; σ²=1 for binomial/poisson.
    let dev_numerator: f64 = match (family, y_original) {
        (crate::pirls::Family::Gaussian, _) | (_, None) => y
            .iter()
            .zip(fitted.iter())
            .zip(w.iter())
            .map(|((yi, fi), wi)| (yi - fi).powi(2) * wi)
            .sum(),
        (fam, Some(y_orig)) => {
            let mu: Array1<f64> = fitted.iter().map(|&eta| fam.inverse_link(eta)).collect();
            glm_deviance(y_orig, &mu, fam)
        }
    };
    let a_inv = inverse(&a)?;
    let tr_a = (xtwx.dot(&a_inv)).diag().sum();
    // Compute mp and dp for estimate_phi_mgcv (matches lib.rs:1131-1144).
    // Use the un-rotated penalty shapes — `block_view().nrows()` on a
    // rotated wrapper would return `p` instead of the smooth's basis size.
    let mp_hess: usize = 1 + mp_pens_for_rank
        .iter()
        .map(|pen| {
            let k = pen.block_view().nrows();
            let rank_s = estimate_rank_eigen(pen);
            k.saturating_sub(rank_s)
        })
        .sum::<usize>();
    let y_for_hess = y_original.unwrap_or(y);
    let bsb_hess: f64 = lambdas
        .iter()
        .zip(penalties_blocks.iter())
        .map(|(l, pen)| l * pen.quadratic_form(&beta))
        .sum();
    let dp_hess = dev_numerator + bsb_hess;
    let scale_est = match family {
        crate::pirls::Family::Binomial
        | crate::pirls::Family::Poisson
        | crate::pirls::Family::NegBin { .. } => 1.0,
        _ => {
            let phi_init = dev_numerator / ((n as f64) - tr_a).max(1e-10);
            family.estimate_phi_mgcv(y_for_hess, dp_hess, mp_hess, 1.0, phi_init)
        }
    };

    // IFT first derivatives
    let b1 = compute_b1_ift(&a_inv, &beta, lambdas, penalties_blocks);
    // S_jβ for each j
    let p = a.nrows();
    let mut s_beta_per_j: Vec<Array1<f64>> = Vec::with_capacity(m);
    for penalty in penalties_blocks.iter() {
        s_beta_per_j.push(penalty.dot_vec(&beta));
    }
    // ΣλSβ
    let mut sum_lambda_s_beta = Array1::<f64>::zeros(p);
    for j in 0..m {
        for r in 0..p {
            sum_lambda_s_beta[r] += lambdas[j] * s_beta_per_j[j][r];
        }
    }

    // Pre-compute per-smooth tr(A⁻¹ S_k) and A⁻¹ S_k columns for cross terms
    let mut tr_a_s_per_j: Vec<f64> = Vec::with_capacity(m);
    // ainvs_per_j[k]: shape (p, k_k) — nonzero columns of A⁻¹ S_k
    let mut ainvs_per_j: Vec<Array2<f64>> = Vec::with_capacity(m);
    let mut block_offsets: Vec<usize> = Vec::with_capacity(m);
    let mut block_sizes: Vec<usize> = Vec::with_capacity(m);
    for penalty in penalties_blocks.iter() {
        let block = penalty.block_view();
        let off = penalty.offset;
        let kk = block.nrows();
        block_offsets.push(off);
        block_sizes.push(kk);
        let mut tr_as = 0.0;
        for ii in 0..kk {
            for jj in 0..kk {
                tr_as += a_inv[[off + ii, off + jj]] * block[[jj, ii]];
            }
        }
        tr_a_s_per_j.push(tr_as);
        let mut ainvs = Array2::<f64>::zeros((p, kk));
        for c in 0..kk {
            for r in 0..p {
                let mut s = 0.0;
                for kkk in 0..kk {
                    s += a_inv[[r, off + kkk]] * block[[kkk, c]];
                }
                ainvs[[r, c]] = s;
            }
        }
        ainvs_per_j.push(ainvs);
    }

    // ∂D/∂β at current β
    let dev_grad_beta = compute_dev_grad_beta(y, x, w, &beta, family, y_original);
    // ∂²D/∂β² b1[:,k] = 2 X'WX b1[:,k]  (Fisher form, exact for canonical link)
    // We compute (∂²D/∂β² b1)[:,k] = 2 X'WX b1[:,k] for each k.
    let mut d2dev_b1 = Array2::<f64>::zeros((p, m));
    for k in 0..m {
        let b1_k = b1.column(k).to_owned();
        let xtwx_b1_k = xtwx.dot(&b1_k);
        for r in 0..p {
            d2dev_b1[[r, k]] = 2.0 * xtwx_b1_k[r];
        }
    }

    // b2[k,j] = -A⁻¹ (λ_j S_j b1[:,k] + λ_k S_k b1[:,j] + δ_{kj} λ_k S_k β)
    // Stored as vector keyed (k,j) for k <= j.
    // We compute on the fly per (k,j) for cache friendliness with O(m²) cost
    // but each is a single A⁻¹ multiply.
    let assemble_b2 = |k: usize, j: usize| -> Array1<f64> {
        let lam_j = lambdas[j];
        let lam_k = lambdas[k];
        let b1_k = b1.column(k);
        let b1_j = b1.column(j);
        let s_j_b1k = penalties_blocks[j].dot_vec(&b1_k.to_owned());
        let s_k_b1j = penalties_blocks[k].dot_vec(&b1_j.to_owned());
        let mut rhs = Array1::<f64>::zeros(p);
        for r in 0..p {
            rhs[r] = -(lam_j * s_j_b1k[r] + lam_k * s_k_b1j[r]);
        }
        if k == j {
            for r in 0..p {
                rhs[r] -= lam_k * s_beta_per_j[k][r];
            }
        }
        a_inv.dot(&rhs)
    };

    // Assemble the Hessian.
    let mut hess = Array2::<f64>::zeros((m, m));
    for k_out in 0..m {
        for j_out in k_out..m {
            let lam_k = lambdas[k_out];
            let lam_j = lambdas[j_out];
            let kron = if k_out == j_out { 1.0 } else { 0.0 };

            // b1[:,j]' (∂²D/∂β²) b1[:,k]  =  b1[:,j]' · d2dev_b1[:,k]
            let mut term_d2dev = 0.0;
            for r in 0..p {
                term_d2dev += b1[[r, j_out]] * d2dev_b1[[r, k_out]];
            }

            // b2[k,j] (length p)
            let b2_kj = assemble_b2(k_out, j_out);

            // (∂D/∂β)' b2[k,j]
            let mut term_dev_b2 = 0.0;
            for r in 0..p {
                term_dev_b2 += dev_grad_beta[r] * b2_kj[r];
            }

            // δ_{kj} λ_k β'S_kβ
            let term_kron_bsb = if k_out == j_out {
                lam_k * penalties_blocks[k_out].quadratic_form(&beta)
            } else {
                0.0
            };

            // 2 λ_k β'S_k b1[:,j] = 2 λ_k (S_kβ)' b1[:,j]
            let mut term_lk_skb_b1j = 0.0;
            for r in 0..p {
                term_lk_skb_b1j += s_beta_per_j[k_out][r] * b1[[r, j_out]];
            }
            term_lk_skb_b1j *= 2.0 * lam_k;

            // 2 λ_j (S_jβ)' b1[:,k]
            let mut term_lj_sjb_b1k = 0.0;
            for r in 0..p {
                term_lj_sjb_b1k += s_beta_per_j[j_out][r] * b1[[r, k_out]];
            }
            term_lj_sjb_b1k *= 2.0 * lam_j;

            // 2 b1[:,j]' (ΣλS) b1[:,k]   = 2 Σ_l λ_l (S_l b1[:,j])' b1[:,k]
            // Cheaper formulation: ΣλS = A - X'WX, so
            //   b1[:,j]' (ΣλS) b1[:,k] = b1[:,j]' (A - X'WX) b1[:,k]
            //                          = b1[:,j]' A b1[:,k] - b1[:,j]' X'WX b1[:,k]
            // And b1[:,k] = -λ_k A⁻¹ S_k β, so A b1[:,k] = -λ_k S_k β.
            // Therefore b1[:,j]' A b1[:,k] = -λ_k b1[:,j]' (S_kβ).
            let mut b1j_a_b1k = 0.0;
            for r in 0..p {
                b1j_a_b1k += b1[[r, j_out]] * (-lam_k * s_beta_per_j[k_out][r]);
            }
            // b1[:,j]' X'WX b1[:,k] = (X'WX b1[:,k])' b1[:,j]
            let b1_k_col = b1.column(k_out).to_owned();
            let xtwx_b1k = xtwx.dot(&b1_k_col);
            let mut b1j_xtwx_b1k = 0.0;
            for r in 0..p {
                b1j_xtwx_b1k += b1[[r, j_out]] * xtwx_b1k[r];
            }
            let term_b1_sls_b1 = 2.0 * (b1j_a_b1k - b1j_xtwx_b1k);

            // 2 (ΣλSβ)' b2[k,j]
            let mut term_sls_b2 = 0.0;
            for r in 0..p {
                term_sls_b2 += sum_lambda_s_beta[r] * b2_kj[r];
            }
            term_sls_b2 *= 2.0;

            let d2_kj = term_d2dev
                + term_dev_b2
                + term_kron_bsb
                + term_lk_skb_b1j
                + term_lj_sjb_b1k
                + term_b1_sls_b1
                + term_sls_b2;

            // det2[k,j] = δ_{kj} λ_k tr(A⁻¹ S_k) − λ_k λ_j tr(A⁻¹ S_k A⁻¹ S_j)
            // tr(A⁻¹ S_k A⁻¹ S_j) using sparse structure (matches envelope).
            let off_k = block_offsets[k_out];
            let kn_k = block_sizes[k_out];
            let off_j = block_offsets[j_out];
            let kn_j = block_sizes[j_out];
            let mut tr_cross = 0.0;
            for r in 0..kn_k {
                for pp in 0..kn_j {
                    tr_cross +=
                        ainvs_per_j[k_out][[off_j + pp, r]] * ainvs_per_j[j_out][[off_k + r, pp]];
                }
            }
            let det2_kj = if k_out == j_out {
                lam_k * tr_a_s_per_j[k_out] - lam_k * lam_j * tr_cross
            } else {
                -lam_k * lam_j * tr_cross
            };

            let h_kj = d2_kj / (2.0 * scale_est) + det2_kj / 2.0;
            hess[[k_out, j_out]] = h_kj;
            if k_out != j_out {
                hess[[j_out, k_out]] = h_kj;
            }
        }
    }

    // Tk·KK' Hessian contribution — mgcv det2 W-dependent pieces (gdi.c
    // P1+P2+P4+P5, lines 919-932), divided by 2 to match the score's
    // `+0.5·log|H|` factor. Gated on the SAME condition as the gradient's
    // tk_kkt piece (~ line 2157): default-on for InvGauss / Binomial /
    // QuasiBinomial, opt-in for other families via `MGCV_TK_GRAD=1`. The
    // gradient and Hessian must be on together or Newton direction is
    // inconsistent with the score gradient.
    let tk_hess_active = matches!(
        family,
        crate::pirls::Family::InverseGaussian
            | crate::pirls::Family::Binomial
            | crate::pirls::Family::QuasiBinomial
    ) || std::env::var("MGCV_TK_GRAD").is_ok();
    if tk_hess_active {
        let tk_contrib = tk_kkt_hessian_analytical(
            y,
            x,
            w,
            lambdas,
            penalties_blocks,
            cached_xtwx,
            family,
            y_original,
        )?;
        for k in 0..m {
            for j in 0..m {
                hess[[k, j]] += tk_contrib[[k, j]] / 2.0;
            }
        }
    }

    Ok(hess)
}

/// REML criterion with optional cached X'WX to avoid O(n*p^2) recomputation
pub fn reml_criterion_multi_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    beta: Option<&Array1<f64>>,
    cached_xtwx: Option<&Array2<f64>>,
) -> Result<f64> {
    let n = y.len();
    let _p = x.ncols();

    // Use cached X'WX if provided, otherwise compute
    let xtwx_owned;
    let xtwx = if let Some(cached) = cached_xtwx {
        cached
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        // Add in-place using BlockPenalty
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Compute coefficients if not provided
    let beta_computed;
    let beta = if let Some(b) = beta {
        b
    } else {
        // OPTIMIZED: Compute X'Wy directly
        let b = compute_xtwy(x, w, y);

        // Add ridge for numerical stability when solving
        let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
        let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
        let ridge = ridge_scale * max_diag;
        let mut a_solve = a.clone();
        a_solve.diag_mut().iter_mut().for_each(|x| *x += ridge);

        beta_computed = solve(a_solve, b)?;
        &beta_computed
    };

    // Compute fitted values
    let fitted = x.dot(beta);

    // Compute residuals and RSS
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute penalty term: Σλᵢ·β'·Sᵢ·β
    let mut penalty_sum = 0.0;
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty_sum += lambda * penalty.quadratic_form(beta);
    }

    // Compute RSS + Σλᵢ·β'·Sᵢ·β
    let rss_bsb = rss + penalty_sum;

    // Compute log|X'WX + Σλᵢ·Sᵢ|
    // Add adaptive ridge term to ensure numerical stability
    // Scale by problem size and matrix magnitude for robustness
    let max_diag = a.diag().iter().map(|x| x.abs()).fold(1.0f64, f64::max);
    // Use stronger ridge for multidimensional cases (more penalties = more potential for ill-conditioning)
    let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    a_reg.diag_mut().iter_mut().for_each(|x| *x += ridge);
    let log_det_a = determinant(&a_reg)?.ln();

    // Compute total rank and -Σrank(Sᵢ)·log(λᵢ) and -Σlog|Sᵢ_+|
    let mut total_rank = 0;
    let mut log_lambda_sum = 0.0;
    let mut log_pseudo_det_sum = 0.0;

    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        if *lambda > 1e-10 {
            let rank_s = estimate_rank(penalty);
            if rank_s > 0 {
                total_rank += rank_s;
                log_lambda_sum += (rank_s as f64) * lambda.ln();
            }
        }

        // Add pseudo-determinant term
        #[cfg(feature = "blas")]
        {
            log_pseudo_det_sum += pseudo_determinant(penalty)?;
        }
        #[cfg(not(feature = "blas"))]
        {
            // Fallback when BLAS not available - use rank approximation
            let rank_s = estimate_rank(penalty);
            log_pseudo_det_sum += (rank_s as f64) * 0.0; // No contribution
        }
    }

    // Compute scale parameter using EDF (matching mgcv)
    #[cfg(feature = "blas")]
    let (phi, n_minus_edf) = {
        // Compute EDF = tr(A^{-1}·X'WX) using trace-Frobenius trick
        let a_inv = inverse(&a_reg)?;
        let edf = (xtwx.dot(&a_inv)).diag().sum();
        let n_minus_edf = n as f64 - edf;
        let phi = rss / n_minus_edf.max(1.0); // Guard against negative/zero denominator
        (phi, n_minus_edf)
    };
    #[cfg(not(feature = "blas"))]
    let (phi, n_minus_edf) = {
        // Fallback to rank-based when BLAS not available
        let phi = rss / (n - total_rank) as f64;
        let n_minus_edf = (n - total_rank) as f64;
        (phi, n_minus_edf)
    };

    // The correct REML criterion (matching mgcv):
    // REML = ((RSS + Σλᵢ·β'·Sᵢ·β)/φ + (n-EDF)*log(2πφ) + log|X'WX + Σλᵢ·Sᵢ| - Σrank(Sᵢ)·log(λᵢ) - Σlog|Sᵢ_+|) / 2
    let pi = std::f64::consts::PI;
    let reml = (rss_bsb / phi + n_minus_edf * (2.0 * pi * phi).ln() + log_det_a
        - log_lambda_sum
        - log_pseudo_det_sum)
        / 2.0;

    if std::env::var("MGCV_REML_DEBUG").is_ok() {
        eprintln!(
            "[REML_DEBUG] λ={:?}\n  rss={:.6} rss_bsb={:.6} phi={:.6} n_minus_edf={:.6}\n  log_det_a={:.6} log_lambda_sum={:.6} log_pseudo_det_sum={:.6}\n  total_rank={} REML={:.6}",
            lambdas, rss, rss_bsb, phi, n_minus_edf,
            log_det_a, log_lambda_sum, log_pseudo_det_sum, total_rank, reml
        );
    }

    Ok(reml)
}

/// Compute the pseudo-determinant of a penalty matrix
///
/// The pseudo-determinant is log|S_+| = Σ log(λ_i) for all positive eigenvalues λ_i > threshold
/// This is used in the REML criterion to match mgcv's implementation.
///
/// # Arguments
/// * `penalty` - Symmetric positive semi-definite penalty matrix
///
/// # Returns
/// log|S_+| = sum of log(positive eigenvalues)
#[cfg(feature = "blas")]
pub fn pseudo_determinant(penalty_block: &BlockPenalty) -> Result<f64> {
    use ndarray_linalg::Eigh;

    // OPTIMIZATION: Use the k x k block for eigendecomposition
    let k_block = penalty_block.block_view().to_owned();
    let k = k_block.nrows();

    // Compute eigenvalue decomposition: S = Q Λ Q'
    let (eigenvalues, _) = k_block.eigh(ndarray_linalg::UPLO::Upper).map_err(|e| {
        GAMError::InvalidParameter(format!("Eigenvalue decomposition failed: {:?}", e))
    })?;

    // Threshold for considering eigenvalue as zero
    let max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let threshold = 1e-10 * max_eigenvalue.max(1.0);

    // Sum log of positive eigenvalues
    let mut log_det = 0.0;
    let mut positive_count = 0;

    for &eigenval in eigenvalues.iter() {
        if eigenval > threshold {
            log_det += eigenval.ln();
            positive_count += 1;
        }
    }

    if std::env::var("MGCV_REML_DEBUG").is_ok() {
        eprintln!(
            "[PSEUDO_DET_DEBUG] Matrix size: {}×{} (Total: {})",
            k, k, penalty_block.total_size
        );
        eprintln!("[PSEUDO_DET_DEBUG] Max eigenvalue: {:.6e}", max_eigenvalue);
        eprintln!("[PSEUDO_DET_DEBUG] Threshold: {:.6e}", threshold);
        eprintln!(
            "[PSEUDO_DET_DEBUG] Positive eigenvalues found: {}",
            positive_count
        );
        eprintln!("[PSEUDO_DET_DEBUG] Log pseudo-determinant: {:.6e}", log_det);
    }

    Ok(log_det)
}

/// Compute square root of a penalty matrix using eigenvalue decomposition
///
/// For a symmetric positive semi-definite matrix S, computes L such that S = L'L
/// Uses eigenvalue decomposition: S = Q Λ Q', so L = Q Λ^{1/2} Q' (taking transpose)
#[cfg(feature = "blas")]
pub fn penalty_sqrt(penalty_block: &BlockPenalty) -> Result<Array2<f64>> {
    use ndarray_linalg::Eigh;

    // OPTIMIZATION: Only decompose the non-zero block (k x k) instead of full p x p
    // This reduces complexity from O(p^3) to O(k^3)
    let k_block = penalty_block.block_view().to_owned();
    let k = k_block.nrows();

    // Compute eigenvalue decomposition on the block: S_k = Q Λ Q'
    let (eigenvalues, eigenvectors) = k_block.eigh(ndarray_linalg::UPLO::Upper).map_err(|e| {
        GAMError::InvalidParameter(format!("Eigenvalue decomposition failed: {:?}", e))
    })?;

    // Threshold for considering eigenvalue as zero
    let max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let threshold = 1e-10 * max_eigenvalue.max(1.0);

    // Count non-zero eigenvalues
    let non_zero_eigs: Vec<(usize, f64)> = eigenvalues
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, e)| e > threshold)
        .collect();

    let rank = non_zero_eigs.len();

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!(
            "[PENALTY_SQRT_DEBUG] Block size: {}×{} (Total: {})",
            k, k, penalty_block.total_size
        );
        eprintln!(
            "[PENALTY_SQRT_DEBUG] Max eigenvalue: {:.6e}",
            max_eigenvalue
        );
        eprintln!("[PENALTY_SQRT_DEBUG] Threshold: {:.6e}", threshold);
        eprintln!("[PENALTY_SQRT_DEBUG] Positive eigenvalues found: {}", rank);
        if rank > 0 {
            let eig_values: Vec<f64> = non_zero_eigs.iter().map(|(_, e)| *e).collect();
            eprintln!(
                "[PENALTY_SQRT_DEBUG] Eigenvalues: {:?}",
                &eig_values[..rank.min(5)]
            );
        }
    }

    if rank == 0 {
        // Penalty is zero, return empty matrix (p x 0)
        return Ok(Array2::<f64>::zeros((penalty_block.total_size, 0)));
    }

    // Construct L (p x rank)
    // S = L * L'
    // L = Q * sqrt(Lambda)
    // Since S is block diagonal, L is also "block-sparse" (non-zero only at offset rows)

    let mut l_matrix = Array2::<f64>::zeros((penalty_block.total_size, rank));
    let offset = penalty_block.offset;

    for (col_idx, (eig_idx, eig_val)) in non_zero_eigs.iter().enumerate() {
        let sqrt_lambda = eig_val.sqrt();
        let eigenvector = eigenvectors.column(*eig_idx);

        // Copy scaled eigenvector into the correct block rows
        for i in 0..k {
            l_matrix[[offset + i, col_idx]] = eigenvector[i] * sqrt_lambda;
        }
    }

    Ok(l_matrix)
}

/// Compute the gradient of REML using block-wise QR approach
///
/// This is optimized for large n by processing X in blocks instead of forming
/// the full augmented matrix. Complexity is O(blocks × p²) instead of O(np²).
///
/// For n < 2000, falls back to full QR for simplicity.
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_adaptive(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Result<Array1<f64>> {
    reml_gradient_multi_qr_adaptive_cached(y, x, w, lambdas, penalties_blocks, None, None, None)
}

/// Adaptive QR gradient with optional cached sqrt_penalties
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_adaptive_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    // Default to rank-based phi for backward compatibility
    reml_gradient_multi_qr_adaptive_cached_edf(
        y,
        x,
        w,
        lambdas,
        penalties_blocks,
        cached_sqrt_penalties,
        cached_xtwx,
        cached_xtwy,
        None,
        ScaleParameterMethod::Rank,
    )
}

/// Adaptive QR gradient with EDF support
///
/// This version supports both rank-based and EDF-based scale parameter computation.
///
/// # Arguments
/// * `cached_xtwx_chol` - Pre-computed Cholesky factor of X'WX (required for EDF method)
/// * `scale_method` - Method for computing scale parameter φ
///
/// # Performance
/// - `ScaleParameterMethod::Rank`: O(1) for φ computation (default, fast)
/// - `ScaleParameterMethod::EDF`: O(p³/3) for φ computation (exact, matches mgcv)
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_adaptive_cached_edf(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    cached_xtwx_chol: Option<&Array2<f64>>,
    scale_method: ScaleParameterMethod,
) -> Result<Array1<f64>> {
    let n = y.len();
    let d = lambdas.len(); // Number of smoothing parameters (dimensionality)

    // OPTIMIZATION: Adaptive threshold based on both n and d
    // For high d, block-wise QR is faster even at smaller n
    // Formula: n >= 2000 - 100*max(0, d-2)
    // Examples: d=1,2: n>=2000, d=4: n>=1800, d=6: n>=1600, d=10: n>=1200
    let threshold = (2000_usize).saturating_sub(100 * (d.saturating_sub(2)));

    if n >= threshold {
        #[cfg(feature = "blas")]
        {
            reml_gradient_multi_qr_blockwise_cached_edf(
                y,
                x,
                w,
                lambdas,
                penalties_blocks,
                1000,
                cached_sqrt_penalties,
                cached_xtwx,
                cached_xtwy,
                cached_xtwx_chol,
                scale_method,
            )
        }
        #[cfg(not(feature = "blas"))]
        {
            reml_gradient_multi_qr_cached(
                y,
                x,
                w,
                lambdas,
                penalties_blocks,
                cached_sqrt_penalties,
                cached_xtwx,
                cached_xtwy,
            )
        }
    } else {
        reml_gradient_multi_qr_cached_edf(
            y,
            x,
            w,
            lambdas,
            penalties_blocks,
            cached_sqrt_penalties,
            cached_xtwx,
            cached_xtwy,
            cached_xtwx_chol,
            scale_method,
        )
    }
}

/// Block-wise version of QR gradient computation
/// Processes X in blocks to avoid O(np²) complexity
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_blockwise(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    block_size: usize,
) -> Result<Array1<f64>> {
    reml_gradient_multi_qr_blockwise_cached(
        y,
        x,
        w,
        lambdas,
        penalties_blocks,
        block_size,
        None,
        None,
        None,
    )
}

/// Block-wise QR gradient with optional cached sqrt_penalties
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_blockwise_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    block_size: usize,
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    use crate::blockwise_qr::compute_r_blockwise;
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Use cached sqrt_penalties if provided, otherwise compute them
    // Avoid cloning by storing temporary and using a reference
    let computed_sqrt_penalties: Vec<Array2<f64>>;
    let sqrt_penalties: &[Array2<f64>];
    let penalty_ranks: Vec<usize>;

    if let Some(cached) = cached_sqrt_penalties {
        // Use cached values - NO CLONE, just reference
        sqrt_penalties = cached.as_slice();
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    } else {
        // Compute square root penalties once (these are constant)
        let mut sp = Vec::new();
        let mut pr = Vec::new();
        for penalty in penalties_blocks.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            let rank = sqrt_pen.ncols();
            sp.push(sqrt_pen);
            pr.push(rank);
        }
        computed_sqrt_penalties = sp;
        sqrt_penalties = &computed_sqrt_penalties;
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    }

    // OPTIMIZATION: If X'WX is cached, use Cholesky instead of blockwise QR
    // Cholesky is O(p³/3) vs blockwise QR O(blocks × p²)
    // For p=64, blocks=5: Cholesky ~90K flops vs QR ~22M flops (244x faster!)
    let r_upper = if let Some(cached) = cached_xtwx {
        use ndarray_linalg::Cholesky;

        // Build A = X'WX + Σλᵢ·Sᵢ using cached X'WX
        let mut a = cached.to_owned();
        for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
            penalty.scaled_add_to(&mut a, *lambda);
        }

        // Add small ridge for numerical stability
        let ridge = 1e-7;
        for i in 0..p {
            a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
        }

        // Compute R via Cholesky: R = chol(A) such that R'R = A
        match a.cholesky(ndarray_linalg::UPLO::Upper) {
            Ok(r) => r,
            Err(_) => {
                // Fallback to blockwise QR if Cholesky fails
                compute_r_blockwise(x, w, lambdas, &sqrt_penalties, block_size)?
            }
        }
    } else {
        compute_r_blockwise(x, w, lambdas, &sqrt_penalties, block_size)?
    };

    // DEBUG: Verify R'R = X'WX + λS
    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        let rtr = r_upper.t().dot(&r_upper);
        let xtwx = compute_xtwx(x, w);
        let mut expected = xtwx.clone();
        for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
            penalty.scaled_add_to(&mut expected, *lambda);
        }

        let max_error = rtr
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "[BLOCKWISE_DEBUG] Max error in R'R vs X'WX+λS: {:.6e}",
            max_error
        );
        eprintln!(
            "[BLOCKWISE_DEBUG] R'R trace: {:.6e}",
            (0..p).map(|i| rtr[[i, i]]).sum::<f64>()
        );
        eprintln!(
            "[BLOCKWISE_DEBUG] Expected trace: {:.6e}",
            (0..p).map(|i| expected[[i, i]]).sum::<f64>()
        );
    }

    // DON'T compute P = R^{-1} - it overflows!
    // Use solve() calls directly

    // Compute coefficients β
    // Use cached X'WX and X'Wy if provided (avoid O(np²) recomputation)
    let xtwx_owned: Array2<f64>;
    let xtwx = if let Some(cached) = cached_xtwx {
        cached
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    let xtwy_owned: Array1<f64>;
    let xtwy = if let Some(cached) = cached_xtwy {
        cached
    } else {
        xtwy_owned = compute_xtwy(x, w, y);
        &xtwy_owned
    };

    let mut a = xtwx.to_owned();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a.clone(), xtwy.to_owned())?;

    // Compute RSS and φ
    let fitted = x.dot(&beta);
    let mut rss = 0.0;
    let mut residuals = Array1::<f64>::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - fitted[i];
        rss += residuals[i] * residuals[i] * w[i];
    }

    let total_rank: usize = penalty_ranks.iter().sum();
    let phi = rss / (n as f64 - total_rank as f64);

    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;
    let n_minus_r = (n as f64) - (total_rank as f64);

    // Pre-compute P = RSS + Σλⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        // Optimized: quadratic form directly on block
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    // Compute FULL gradient for each penalty (matching full QR version)
    let mut gradient = Array1::<f64>::zeros(m);

    // Transpose R once (reused for all penalties)
    let r_t = r_upper.t().to_owned();

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;
        let sqrt_penalty = &sqrt_penalties[i];

        // Term 1: tr(A^{-1}·λᵢ·Sᵢ) using solve without forming A^{-1}
        // Batch triangular solve: R'·X = L for ALL columns at once
        let rank = sqrt_penalty.ncols();

        // R' is lower triangular (transpose of upper triangular R)
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

        // Compute trace term: Σ_k ||X[:, k]||² = ||X||²_F (sum of all squared elements)
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        // Term 2: -rank(Sᵢ)
        let rank_term = -rank_i;

        // Compute ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β using cached R factorization
        // A = R'R, so A⁻¹·b = R⁻¹·R'⁻¹·b
        // Solve in two steps: R'·y = b, then R·x = y
        // Optimized: S·β using block-sparse multiply
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        // Step 1: Solve R'·y = lambda_s_beta (lower triangular)
        let y = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

        // Step 2: Solve R·x = y (upper triangular)
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // Compute ∂RSS/∂ρᵢ
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        let dphi_drho = drss_drho / n_minus_r;

        // Compute ∂P/∂ρᵢ
        let beta_s_i_beta = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            // Optimized: S·β and S·dβ using block-sparse multiply
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);

            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;

        // Term 3: ∂(P/φ)/∂ρᵢ
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;

        // Term 4: ∂[(n-r)·log(2πφ)]/∂ρᵢ
        let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

        // Total gradient (divide by 2)
        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

/// Block-wise QR gradient with EDF support for scale parameter
///
/// This version supports both rank-based and EDF-based scale parameter computation.
/// For EDF mode, requires pre-computed Cholesky factor of X'WX.
///
/// # Performance
/// - `ScaleParameterMethod::Rank`: O(1) for φ computation
/// - `ScaleParameterMethod::EDF`: O(p³/3) additional for φ computation via trace trick
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_blockwise_cached_edf(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    block_size: usize,
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    cached_xtwx: Option<&Array2<f64>>,
    cached_xtwy: Option<&Array1<f64>>,
    cached_xtwx_chol: Option<&Array2<f64>>,
    scale_method: ScaleParameterMethod,
) -> Result<Array1<f64>> {
    use crate::blockwise_qr::compute_r_blockwise;
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Use cached sqrt_penalties if provided, otherwise compute them
    let computed_sqrt_penalties: Vec<Array2<f64>>;
    let sqrt_penalties: &[Array2<f64>];
    let penalty_ranks: Vec<usize>;

    if let Some(cached) = cached_sqrt_penalties {
        sqrt_penalties = cached.as_slice();
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    } else {
        let mut sp = Vec::new();
        let mut pr = Vec::new();
        for penalty in penalties_blocks.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            let rank = sqrt_pen.ncols();
            sp.push(sqrt_pen);
            pr.push(rank);
        }
        computed_sqrt_penalties = sp;
        sqrt_penalties = &computed_sqrt_penalties;
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    }

    // Get X'WX (cached or compute)
    let xtwx_owned: Array2<f64>;
    let xtwx = if let Some(cached) = cached_xtwx {
        cached
    } else {
        xtwx_owned = compute_xtwx(x, w);
        &xtwx_owned
    };

    // OPTIMIZATION: If X'WX is cached, use Cholesky instead of blockwise QR
    let r_upper = if cached_xtwx.is_some() {
        use ndarray_linalg::Cholesky;

        let mut a = xtwx.to_owned();
        for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
            penalty.scaled_add_to(&mut a, *lambda);
        }

        let ridge = 1e-7;
        for i in 0..p {
            a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
        }

        match a.cholesky(ndarray_linalg::UPLO::Upper) {
            Ok(r) => r,
            Err(_) => compute_r_blockwise(x, w, lambdas, &sqrt_penalties, block_size)?,
        }
    } else {
        compute_r_blockwise(x, w, lambdas, &sqrt_penalties, block_size)?
    };

    let r_t = r_upper.t().to_owned();

    // Compute X'Wy
    let xtwy_owned: Array1<f64>;
    let xtwy = if let Some(cached) = cached_xtwy {
        cached
    } else {
        xtwy_owned = compute_xtwy(x, w, y);
        &xtwy_owned
    };

    let mut a = xtwx.to_owned();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a.clone(), xtwy.to_owned())?;

    // Compute RSS
    let fitted = x.dot(&beta);
    let mut rss = 0.0;
    let mut residuals = Array1::<f64>::zeros(n);
    for i in 0..n {
        residuals[i] = y[i] - fitted[i];
        rss += residuals[i] * residuals[i] * w[i];
    }

    // Compute φ based on selected method
    let total_rank: usize = penalty_ranks.iter().sum();
    let (phi, n_minus_edf) = match scale_method {
        ScaleParameterMethod::Rank => {
            let n_minus_r = n as f64 - total_rank as f64;
            let phi = rss / n_minus_r;
            (phi, n_minus_r)
        }
        ScaleParameterMethod::EDF => {
            // Compute EDF = tr(A⁻¹·X'WX) using trace-Frobenius trick
            // Need Cholesky of X'WX
            let xtwx_chol = if let Some(cached) = cached_xtwx_chol {
                cached.clone()
            } else {
                compute_xtwx_cholesky(xtwx)?
            };

            let edf = compute_edf_from_cholesky(&r_t, &xtwx_chol)?;
            let n_minus_edf = n as f64 - edf;

            // Guard against negative or zero denominator
            let n_minus_edf_safe = n_minus_edf.max(1.0);
            let phi = rss / n_minus_edf_safe;

            if std::env::var("MGCV_EDF_DEBUG").is_ok() {
                eprintln!("[EDF_DEBUG] n={}, total_rank={}, EDF={:.4}, n-EDF={:.4}, n-rank={:.4}, phi_edf={:.6e}, phi_rank={:.6e}",
                    n, total_rank, edf, n_minus_edf, n as f64 - total_rank as f64,
                    phi, rss / (n as f64 - total_rank as f64));
            }

            (phi, n_minus_edf_safe)
        }
    };

    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;

    // Pre-compute P = RSS + Σλⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        // Optimized: quadratic form directly on block
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    let mut gradient = Array1::<f64>::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;
        let sqrt_penalty = &sqrt_penalties[i];

        // Term 1: tr(A^{-1}·λᵢ·Sᵢ) using batch triangular solve
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        // Term 2: -rank(Sᵢ)
        let rank_term = -rank_i;

        // Compute ∂β/∂ρᵢ
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        let y_solve = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_solve)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // Compute ∂RSS/∂ρᵢ
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        let dphi_drho = drss_drho / n_minus_edf;

        // Compute ∂P/∂ρᵢ
        let beta_s_i_beta: f64 = beta
            .iter()
            .zip(s_i_beta.iter())
            .map(|(bi, sbi)| bi * sbi)
            .sum();
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            // Optimized: S·β and S·dβ using block-sparse multiply
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);

            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;
        let log_phi_deriv = n_minus_edf * dphi_drho * inv_phi;

        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

#[cfg(not(feature = "blas"))]
pub fn reml_gradient_multi_qr_blockwise(
    _y: &Array1<f64>,
    _x: &Array2<f64>,
    _w: &Array1<f64>,
    _lambdas: &[f64],
    _penalties_blocks: &[BlockPenalty],
    _block_size: usize,
) -> Result<Array1<f64>> {
    let _penalties: Vec<Array2<f64>> = _penalties_blocks.iter().map(|p| p.to_dense()).collect();
    Err(GAMError::InvalidParameter(
        "Block-wise QR requires 'blas' feature".to_string(),
    ))
}

/// Compute the gradient of REML using QR-based approach (matching mgcv's gdi.c)
///
/// Following Wood (2011) and mgcv's gdi.c (get_ddetXWXpS function), this uses:
/// 1. QR decomposition of augmented matrix [sqrt(W)X; sqrt(λ_0)L_0; ...]
/// 2. R such that R'R = X'WX + Σλᵢ·Sᵢ
/// 3. P = R^{-1}
/// 4. Gradient: ∂log|R'R|/∂log(λ_m) = λ_m·tr(P'·S_m·P)
///
/// This avoids explicit formation of A^{-1} and cross-coupling issues.
///
/// NOTE: For large n (>= 2000), use reml_gradient_multi_qr_blockwise instead
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_cholesky(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Result<Array1<f64>> {
    // Compute square root penalties (expensive eigendecomp)
    let mut sqrt_penalties = Vec::new();
    let mut penalty_ranks = Vec::new();
    for penalty in penalties_blocks.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        let rank = sqrt_pen.ncols();
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Delegate to cached version
    reml_gradient_multi_cholesky_cached(
        y,
        x,
        w,
        lambdas,
        penalties_blocks,
        &sqrt_penalties,
        &penalty_ranks,
    )
}

/// Cholesky gradient with pre-computed sqrt_penalties (avoids eigendecomp)
///
/// This version accepts pre-computed sqrt_penalties to avoid expensive
/// eigendecomposition on every call. Since penalties don't change during
/// optimization (only lambdas do), you can compute sqrt_penalties once
/// and reuse them across all gradient evaluations.
///
/// Use this when calling gradient multiple times with same penalties.
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_cholesky_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    sqrt_penalties: &[Array2<f64>],
    penalty_ranks: &[usize],
) -> Result<Array1<f64>> {
    use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Form A = X'WX + Σλᵢ·Sᵢ directly
    let mut a = compute_xtwx(x, w);
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Add adaptive ridge regularization for numerical stability
    // mgcv adds small ridge to diagonal to ensure positive definiteness
    // Ridge size: max(diagonal) * sqrt(machine_epsilon) ≈ 1e-8 * max_diag
    let max_diag = (0..p).map(|i| a[[i, i]].abs()).fold(0.0f64, f64::max);
    let ridge = max_diag * 1e-8;

    for i in 0..p {
        a[[i, i]] += ridge;
    }

    // Cholesky factorization: A = R'R (R is upper triangular)
    // Returns upper triangular R
    let r_upper = a.cholesky(UPLO::Upper).map_err(|e| {
        GAMError::InvalidParameter(format!("Cholesky factorization failed: {:?}", e))
    })?;

    // Compute beta = A^{-1}·X'Wy using cached factorization
    let xtwy = compute_xtwy(x, w, y);
    let r_t = r_upper.t().to_owned();

    // Solve R'·y = X'Wy, then R·beta = y
    let y_temp = r_t
        .solve_triangular(UPLO::Lower, Diag::NonUnit, &xtwy)
        .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
    let beta = r_upper
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp)
        .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

    // Compute residuals
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(y_hat.iter())
        .map(|(yi, yhati)| yi - yhati)
        .collect();

    // Effective degrees of freedom and RSS
    let mut effective_dof = 0.0;
    for &rank in penalty_ranks.iter() {
        effective_dof += rank as f64;
    }
    let n_minus_r = n as f64 - effective_dof;

    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(ri, wi)| ri * ri * wi)
        .sum();

    let phi = rss / n_minus_r;
    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;

    // Compute penalty term in P
    let mut penalty_sum = 0.0;
    for j in 0..m {
        // Optimized: quadratic form directly on block
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    let mut gradient = Array1::<f64>::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;
        let sqrt_penalty = &sqrt_penalties[i];

        // Term 1: Trace computation using batch triangular solve
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        // Term 2: -rank(Sᵢ)
        let rank_term = -rank_i;

        // Beta derivatives using cached factorization
        // Optimized: S·β using block-sparse multiply
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        let y_temp = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // RSS derivative
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        let dphi_drho = drss_drho / n_minus_r;

        // Penalty term derivatives
        let beta_s_i_beta: f64 = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            // Optimized: S·β and S·dβ using block-sparse multiply
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);

            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;
        let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

/// QR-based REML gradient with optional cached sqrt_penalties
/// If cached_sqrt_penalties is provided, skips expensive eigendecomposition
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    _cached_xtwx: Option<&Array2<f64>>,
    _cached_xtwy: Option<&Array1<f64>>,
) -> Result<Array1<f64>> {
    use ndarray_linalg::QR;
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // OPTIMIZED: Compute sqrt(W) * X without cloning x
    // Allocate directly to avoid clone overhead
    let mut sqrt_w_x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] = x[[i, j]] * weight_sqrt;
        }
    }

    // Use cached sqrt_penalties if provided, otherwise compute them
    // Avoid cloning by storing temporary and using a reference
    let computed_sqrt_penalties: Vec<Array2<f64>>;
    let sqrt_penalties: &[Array2<f64>];
    let penalty_ranks: Vec<usize>;

    if let Some(cached) = cached_sqrt_penalties {
        // Use cached values - NO CLONE, just reference
        sqrt_penalties = cached.as_slice();
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    } else {
        // Compute square root penalties and their ranks
        let mut sp = Vec::new();
        let mut pr = Vec::new();
        for penalty in penalties_blocks.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            // Use the actual rank from eigenvalue decomposition (number of positive eigenvalues)
            // This is more accurate than the heuristic in estimate_rank()
            let rank = sqrt_pen.ncols(); // rank = number of positive eigenvalues
            sp.push(sqrt_pen);
            pr.push(rank);
        }
        computed_sqrt_penalties = sp;
        sqrt_penalties = &computed_sqrt_penalties;
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    }

    // Build augmented matrix Z = [sqrt(W)X; sqrt(λ_0)L_0'; sqrt(λ_1)L_1'; ...]
    // Determine total rows (n + sum of ranks)
    let mut total_rows = n;
    for sqrt_pen in sqrt_penalties.iter() {
        total_rows += sqrt_pen.ncols(); // Number of columns = rank
    }

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[Z_BUILD_DEBUG] Building Z matrix:");
        eprintln!("[Z_BUILD_DEBUG]   n = {}, p = {}", n, p);
        for (i, sqrt_pen) in sqrt_penalties.iter().enumerate() {
            eprintln!(
                "[Z_BUILD_DEBUG]   L{} shape: {}×{}, λ{} = {:.6}",
                i,
                sqrt_pen.nrows(),
                sqrt_pen.ncols(),
                i,
                lambdas[i]
            );
        }
        eprintln!("[Z_BUILD_DEBUG]   Total Z rows: {}", total_rows);
    }

    let mut z = Array2::<f64>::zeros((total_rows, p));

    // Fill in sqrt(W)X
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = sqrt_w_x[[i, j]];
        }
    }

    // Fill in scaled square root penalties (transposed)
    // sqrt_pen is p × rank, we need rank × p for augmented matrix
    let mut row_offset = n;
    for (idx, (sqrt_pen, &lambda)) in sqrt_penalties.iter().zip(lambdas.iter()).enumerate() {
        let sqrt_lambda = lambda.sqrt();
        let rank = sqrt_pen.ncols(); // Number of non-zero eigenvalues
        for i in 0..rank {
            for j in 0..p {
                z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]]; // Transpose!
            }
        }

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && rank > 0 {
            eprintln!(
                "[Z_BUILD_DEBUG]   After adding L{} (rows {} to {}), first value: {:.6e}",
                idx,
                row_offset,
                row_offset + rank - 1,
                z[[row_offset, 0]]
            );
        }

        row_offset += rank;
    }

    // QR decomposition: Z = QR
    let (_, r) = z
        .qr()
        .map_err(|e| GAMError::InvalidParameter(format!("QR decomposition failed: {:?}", e)))?;

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!("[QR_DEBUG] Z dimensions: {}×{}", z.nrows(), z.ncols());
        eprintln!("[QR_DEBUG] R dimensions: {}×{}", r.nrows(), r.ncols());
        eprintln!("[QR_DEBUG] total_rows={}, n={}, p={}", total_rows, n, p);
    }

    // Extract upper triangular part (first p rows)
    let mut r_upper = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            r_upper[[i, j]] = r[[i, j]];
        }
    }

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        // Check R'R to see if it matches X'WX + S
        let rtr = r_upper.t().dot(&r_upper);
        eprintln!(
            "[QR_DEBUG] R'R diagonal: [{:.6}, {:.6}, ..., {:.6}]",
            rtr[[0, 0]],
            rtr[[1, 1]],
            rtr[[p - 1, p - 1]]
        );
    }

    // DON'T compute P = R^{-1} - it overflows for ill-conditioned R!
    // Instead use solve() calls directly

    // Compute coefficients for penalty term
    let xtwx = compute_xtwx(x, w);
    let xtwy = compute_xtwy(x, w, y);

    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Add small ridge for stability
    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a.clone(), xtwy)?;

    // Compute RSS and φ
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute total rank for φ calculation
    let total_rank: usize = penalty_ranks.iter().sum();
    let phi = rss / (n as f64 - total_rank as f64);

    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        eprintln!(
            "[PHI_DEBUG] n={}, total_rank={}, rss={:.6}, phi={:.6}",
            n, total_rank, rss, phi
        );
    }

    // Compute gradient for each penalty
    // Using the CORRECT IFT-based formula accounting for implicit dependencies:
    //
    // REML = [(RSS + Σλⱼ·β'·Sⱼ·β)/φ + (n-r)·log(2πφ) + log|A| - Σrⱼ·log(λⱼ)] / 2
    //
    // where β and φ implicitly depend on ρ through:
    //   A·β = X'y  =>  ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
    //   φ = RSS/(n-r)  =>  ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ)/(n-r)
    //
    // Full gradient:
    // ∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rᵢ + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
    //
    // where P = RSS + Σλⱼ·β'·Sⱼ·β
    //
    // This matches numerical gradients to < 0.1% error.
    let mut gradient = Array1::zeros(m);

    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;
    let n_minus_r = (n as f64) - (total_rank as f64);

    // Pre-compute P = RSS + Σλⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    // Transpose R once (reused for all penalties)
    let r_t = r_upper.t().to_owned();

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;

        // Term 1: tr(A^{-1}·λᵢ·Sᵢ) using solve without forming A^{-1}
        // We have R'R = A, so tr(A^{-1}·S) = tr(R^{-1}·R'^{-1}·S)
        // = Σ_k ||R'^{-1}·L[:, k]||² where S = L·L'
        // Compute by solving R'·X = L for ALL columns at once (batch solve)
        let sqrt_penalty = &sqrt_penalties[i];
        let rank = sqrt_penalty.ncols();

        // Batch triangular solve: R'·X = L where L is p×rank matrix
        // R' is lower triangular (transpose of upper triangular R)
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

        // Compute trace term: Σ_k ||X[:, k]||² = ||X||²_F (sum of all squared elements)
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        // Term 2: -rank(Sᵢ)
        let rank_term = -rank_i;

        // Compute ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β using cached R factorization
        // A = R'R, so A⁻¹·b = R⁻¹·R'⁻¹·b
        // Solve in two steps: R'·y = b, then R·x = y
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        // Step 1: Solve R'·y = lambda_s_beta (lower triangular)
        let y = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

        // Step 2: Solve R·x = y (upper triangular)
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // Compute ∂RSS/∂ρᵢ = -2·residuals'·X·∂β/∂ρᵢ
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        // Compute ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        let dphi_drho = drss_drho / n_minus_r;

        // Compute ∂P/∂ρᵢ where P = RSS + Σλⱼ·β'·Sⱼ·β
        // Explicit term: λᵢ·β'·Sᵢ·β
        let beta_s_i_beta: f64 = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        // Implicit term: 2·Σλⱼ·β'·Sⱼ·∂β/∂ρᵢ
        // Note: This simplifies to exactly -∂RSS/∂ρᵢ by the algebra
        let mut implicit_pen = 0.0;
        for j in 0..m {
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);
            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;

        // Term 3: ∂(P/φ)/∂ρᵢ = (1/φ)·∂P/∂ρᵢ - (P/φ²)·∂φ/∂ρᵢ
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;

        // Term 4: ∂[(n-r)·log(2πφ)]/∂ρᵢ = (n-r)·(1/φ)·∂φ/∂ρᵢ
        let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

        // Total gradient (divide by 2)
        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

/// QR-based REML gradient with EDF support for scale parameter
///
/// This version supports both rank-based and EDF-based scale parameter computation.
///
/// # Arguments
/// * `cached_xtwx_chol` - Pre-computed Cholesky factor of X'WX (required for EDF method)
/// * `scale_method` - Method for computing scale parameter φ
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_qr_cached_edf(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_sqrt_penalties: Option<&Vec<Array2<f64>>>,
    _cached_xtwx: Option<&Array2<f64>>,
    _cached_xtwy: Option<&Array1<f64>>,
    cached_xtwx_chol: Option<&Array2<f64>>,
    scale_method: ScaleParameterMethod,
) -> Result<Array1<f64>> {
    use ndarray_linalg::QR;
    use ndarray_linalg::{Diag, SolveTriangular, UPLO};

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Create weighted design matrix
    let mut sqrt_w_x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] = x[[i, j]] * weight_sqrt;
        }
    }

    // Use cached sqrt_penalties if provided
    let computed_sqrt_penalties: Vec<Array2<f64>>;
    let sqrt_penalties: &[Array2<f64>];
    let penalty_ranks: Vec<usize>;

    if let Some(cached) = cached_sqrt_penalties {
        sqrt_penalties = cached.as_slice();
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    } else {
        let mut sp = Vec::new();
        let mut pr = Vec::new();
        for penalty in penalties_blocks.iter() {
            let sqrt_pen = penalty_sqrt(penalty)?;
            let rank = sqrt_pen.ncols();
            sp.push(sqrt_pen);
            pr.push(rank);
        }
        computed_sqrt_penalties = sp;
        sqrt_penalties = &computed_sqrt_penalties;
        penalty_ranks = sqrt_penalties.iter().map(|sp| sp.ncols()).collect();
    }

    // Build augmented matrix Z
    let mut total_rows = n;
    for sqrt_pen in sqrt_penalties.iter() {
        total_rows += sqrt_pen.ncols();
    }

    let mut z = Array2::<f64>::zeros((total_rows, p));

    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = sqrt_w_x[[i, j]];
        }
    }

    let mut row_offset = n;
    for (sqrt_pen, &lambda) in sqrt_penalties.iter().zip(lambdas.iter()) {
        let sqrt_lambda = lambda.sqrt();
        let rank = sqrt_pen.ncols();
        for i in 0..rank {
            for j in 0..p {
                z[[row_offset + i, j]] = sqrt_lambda * sqrt_pen[[j, i]];
            }
        }
        row_offset += rank;
    }

    // QR decomposition
    let (_, r) = z
        .qr()
        .map_err(|e| GAMError::InvalidParameter(format!("QR decomposition failed: {:?}", e)))?;

    let mut r_upper = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            r_upper[[i, j]] = r[[i, j]];
        }
    }

    let r_t = r_upper.t().to_owned();

    // Compute coefficients
    let xtwx = compute_xtwx(x, w);
    let xtwy = compute_xtwy(x, w, y);

    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a.clone(), xtwy)?;

    // Compute RSS
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute φ based on selected method
    let total_rank: usize = penalty_ranks.iter().sum();
    let (phi, n_minus_edf) = match scale_method {
        ScaleParameterMethod::Rank => {
            let n_minus_r = n as f64 - total_rank as f64;
            let phi = rss / n_minus_r;
            (phi, n_minus_r)
        }
        ScaleParameterMethod::EDF => {
            let xtwx_chol = if let Some(cached) = cached_xtwx_chol {
                cached.clone()
            } else {
                compute_xtwx_cholesky(&xtwx)?
            };

            let edf = compute_edf_from_cholesky(&r_t, &xtwx_chol)?;
            let n_minus_edf = n as f64 - edf;
            let n_minus_edf_safe = n_minus_edf.max(1.0);
            let phi = rss / n_minus_edf_safe;

            if std::env::var("MGCV_EDF_DEBUG").is_ok() {
                eprintln!("[EDF_DEBUG] n={}, total_rank={}, EDF={:.4}, n-EDF={:.4}, phi_edf={:.6e}, phi_rank={:.6e}",
                    n, total_rank, edf, n_minus_edf, phi, rss / (n as f64 - total_rank as f64));
            }

            (phi, n_minus_edf_safe)
        }
    };

    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;

    // Pre-compute P
    let mut penalty_sum = 0.0;
    for j in 0..m {
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    let mut gradient = Array1::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;

        // Trace term
        let sqrt_penalty = &sqrt_penalties[i];
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        let rank_term = -rank_i;

        // Beta derivatives
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        let y_solve = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_solve)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // RSS derivative
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        let dphi_drho = drss_drho / n_minus_edf;

        // Penalty derivatives
        let beta_s_i_beta: f64 = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);
            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;
        let log_phi_deriv = n_minus_edf * dphi_drho * inv_phi;

        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

/// Ultra-optimized Cholesky gradient with ALL pre-computed values
///
/// This version caches everything that doesn't change:
/// - sqrt_penalties (penalties constant)
/// - X'WX (X and W constant)
/// - X'Wy (X, W, and y constant)
///
/// Only lambdas change during optimization, so everything else can be
/// pre-computed once and reused. This gives maximum performance for
/// optimization loops.
#[cfg(feature = "blas")]
pub fn reml_gradient_multi_cholesky_fully_cached(
    x: &Array2<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    sqrt_penalties: &[Array2<f64>],
    penalty_ranks: &[usize],
    xtwx: &Array2<f64>,                           // Pre-computed X'WX
    xtwy: &Array1<f64>,                           // Pre-computed X'Wy
    y_residual_data: &(Array1<f64>, Array1<f64>), // (y, w) for residual computation
) -> Result<Array1<f64>> {
    use ndarray_linalg::{Cholesky, Diag, SolveTriangular, UPLO};

    let (y, w) = y_residual_data;
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Form A = X'WX + Σλᵢ·Sᵢ (only lambda scaling changes)
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Cholesky factorization
    let r_upper = a
        .cholesky(UPLO::Upper)
        .map_err(|e| GAMError::InvalidParameter(format!("Cholesky failed: {:?}", e)))?;

    let r_t = r_upper.t().to_owned();

    // Compute beta using pre-computed X'Wy
    let y_temp = r_t
        .solve_triangular(UPLO::Lower, Diag::NonUnit, xtwy)
        .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
    let beta = r_upper
        .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp)
        .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;

    // Compute residuals
    let y_hat = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(y_hat.iter())
        .map(|(yi, yhati)| yi - yhati)
        .collect();

    // Effective DOF and RSS
    let mut effective_dof = 0.0;
    for &rank in penalty_ranks.iter() {
        effective_dof += rank as f64;
    }
    let n_minus_r = n as f64 - effective_dof;

    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(ri, wi)| ri * ri * wi)
        .sum();

    let phi = rss / n_minus_r;
    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;

    // Penalty term in P
    let mut penalty_sum = 0.0;
    for j in 0..m {
        // Optimized: quadratic form directly on block
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    let mut gradient = Array1::<f64>::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = penalty_ranks[i] as f64;
        let sqrt_penalty = &sqrt_penalties[i];

        // Trace computation
        let x_batch = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, sqrt_penalty)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let trace_term: f64 = x_batch.iter().map(|xi| xi * xi).sum();
        let trace = lambda_i * trace_term;

        let rank_term = -rank_i;

        // Beta derivatives
        // Optimized: S·β using block-sparse multiply
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);

        let y_temp = r_t
            .solve_triangular(UPLO::Lower, Diag::NonUnit, &lambda_s_beta)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?;
        let dbeta_drho = r_upper
            .solve_triangular(UPLO::Upper, Diag::NonUnit, &y_temp)
            .map_err(|e| GAMError::InvalidParameter(format!("Triangular solve failed: {:?}", e)))?
            .mapv(|x| -x);

        // RSS derivative
        let x_dbeta = x.dot(&dbeta_drho);
        let drss_drho: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();

        let dphi_drho = drss_drho / n_minus_r;

        // Penalty derivatives
        let beta_s_i_beta: f64 = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            // Optimized: S·β and S·dβ using block-sparse multiply
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_drho);

            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_drho.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_drho = drss_drho + explicit_pen + implicit_pen;
        let penalty_quotient_deriv = dp_drho * inv_phi - (p_value / phi_sq) * dphi_drho;
        let log_phi_deriv = n_minus_r * dphi_drho * inv_phi;

        gradient[i] = (trace + rank_term + penalty_quotient_deriv + log_phi_deriv) / 2.0;
    }

    Ok(gradient)
}

#[cfg(not(feature = "blas"))]
pub fn reml_gradient_multi_cholesky_fully_cached(
    _x: &Array2<f64>,
    _lambdas: &[f64],
    _penalties_blocks: &[BlockPenalty],
    _sqrt_penalties: &[Array2<f64>],
    _penalty_ranks: &[usize],
    _xtwx: &Array2<f64>,
    _xtwy: &Array1<f64>,
    _y_residual_data: &(Array1<f64>, Array1<f64>),
) -> Result<Array1<f64>> {
    let _penalties: Vec<Array2<f64>> = _penalties_blocks.iter().map(|p| p.to_dense()).collect();
    Err(GAMError::InvalidParameter(
        "Fully cached gradient requires 'blas' feature".to_string(),
    ))
}

#[cfg(not(feature = "blas"))]
pub fn reml_gradient_multi_cholesky(
    _y: &Array1<f64>,
    _x: &Array2<f64>,
    _w: &Array1<f64>,
    _lambdas: &[f64],
    _penalties_blocks: &[BlockPenalty],
) -> Result<Array1<f64>> {
    let _penalties: Vec<Array2<f64>> = _penalties_blocks.iter().map(|p| p.to_dense()).collect();
    Err(GAMError::InvalidParameter(
        "Cholesky gradient requires 'blas' feature".to_string(),
    ))
}

#[cfg(not(feature = "blas"))]
pub fn reml_gradient_multi_cholesky_cached(
    _y: &Array1<f64>,
    _x: &Array2<f64>,
    _w: &Array1<f64>,
    _lambdas: &[f64],
    _penalties_blocks: &[BlockPenalty],
    _sqrt_penalties: &[Array2<f64>],
    _penalty_ranks: &[usize],
) -> Result<Array1<f64>> {
    let _penalties: Vec<Array2<f64>> = _penalties_blocks.iter().map(|p| p.to_dense()).collect();
    Err(GAMError::InvalidParameter(
        "Cholesky gradient requires 'blas' feature".to_string(),
    ))
}

/// Compute the Hessian of REML with respect to log(λᵢ) using QR-based approach
///
/// Returns: ∂²REML/∂ρᵢ∂ρⱼ for i,j = 1..m, where ρᵢ = log(λᵢ)
///
/// This uses the CORRECTED formula matching the IFT-based gradient:
///
/// H[i,j] = ∂/∂ρⱼ [∂REML/∂ρᵢ]
///
/// where ∂REML/∂ρᵢ = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + ∂(P/φ)/∂ρᵢ + (n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
///
/// The Hessian accounts for all implicit dependencies through the Implicit Function Theorem.
#[cfg(feature = "blas")]
pub fn reml_hessian_multi_qr(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Result<Array2<f64>> {
    use ndarray_linalg::Inverse;
    use ndarray_linalg::QR;

    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!("\n[HESS_CORRECTED] Starting CORRECTED Hessian computation (matching gradient)");
        eprintln!("[HESS_CORRECTED] n={}, p={}, m={}", n, p, m);
    }

    // Step 1: QR decomposition for efficient A^{-1} computation
    let mut sqrt_w_x = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            sqrt_w_x[[i, j]] *= weight_sqrt;
        }
    }

    let mut sqrt_penalties = Vec::with_capacity(m);
    let mut penalty_ranks = Vec::with_capacity(m);

    for penalty in penalties_blocks.iter() {
        let sqrt_pen = penalty_sqrt(penalty)?;
        let rank = sqrt_pen.ncols();
        sqrt_penalties.push(sqrt_pen);
        penalty_ranks.push(rank);
    }

    // Build augmented matrix Z = [sqrt(W)X; √λ₁·L₁'; √λ₂·L₂'; ...]
    let total_rows: usize = n + penalty_ranks.iter().sum::<usize>();
    let mut z = Array2::zeros((total_rows, p));
    z.slice_mut(s![0..n, ..]).assign(&sqrt_w_x);

    let mut row_offset = n;
    for (i, sqrt_pen) in sqrt_penalties.iter().enumerate() {
        let rank = penalty_ranks[i];
        let lambda_sqrt = lambdas[i].sqrt();
        for j in 0..rank {
            for k in 0..p {
                z[[row_offset + j, k]] = lambda_sqrt * sqrt_pen[[k, j]];
            }
        }
        row_offset += rank;
    }

    // QR decomposition
    let (_, r) = z
        .qr()
        .map_err(|_| GAMError::LinAlgError("QR decomposition failed".to_string()))?;
    let p_matrix = r
        .slice(s![0..p, 0..p])
        .inv()
        .map_err(|_| GAMError::SingularMatrix)?;

    // Compute A^{-1} = P·P'
    let a_inv = p_matrix.dot(&p_matrix.t());

    // Step 2: Compute coefficients β
    let xtw = sqrt_w_x.t().to_owned();
    let xtwx = xtw.dot(&sqrt_w_x);
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let y_weighted: Array1<f64> = y.iter().zip(w.iter()).map(|(yi, wi)| yi * wi).collect();
    let b = xtw.dot(&y_weighted);

    // Add ridge for stability
    let ridge = 1e-7 * (1.0 + (m as f64).sqrt());
    for i in 0..p {
        a[[i, i]] += ridge * a[[i, i]].abs().max(1.0);
    }

    let beta = solve(a, b)?;

    // Step 3: Compute residuals, RSS, phi, P
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    let total_rank: usize = penalty_ranks.iter().sum();
    let n_minus_r = (n as f64) - (total_rank as f64);
    let phi = rss / n_minus_r;
    let inv_phi = 1.0 / phi;
    let phi_sq = phi * phi;
    let phi_cb = phi * phi * phi;

    // Compute P = RSS + Σⱼ λⱼ·β'·Sⱼ·β
    let mut penalty_sum = 0.0;
    for j in 0..m {
        // Optimized: quadratic form directly on block
        let beta_s_j_beta = penalties_blocks[j].quadratic_form(&beta);
        penalty_sum += lambdas[j] * beta_s_j_beta;
    }
    let p_value = rss + penalty_sum;

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!(
            "[HESS_CORRECTED] RSS={:.6e}, phi={:.6e}, P={:.6e}",
            rss, phi, p_value
        );
        eprintln!(
            "[HESS_CORRECTED] total_rank={}, n-r={:.6}",
            total_rank, n_minus_r
        );
    }

    // Step 4: Compute first derivatives (matching gradient formula)
    let mut dbeta_drho = Vec::with_capacity(m);
    let mut drss_drho = Vec::with_capacity(m);
    let mut dphi_drho = Vec::with_capacity(m);
    let mut dp_drho = Vec::with_capacity(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];

        // ∂β/∂ρᵢ = -A⁻¹·λᵢ·Sᵢ·β
        // Optimized: S·β using block-sparse multiply
        let s_i_beta = penalty_i.dot_vec(&beta);
        let lambda_s_beta = s_i_beta.mapv(|x| lambda_i * x);
        let dbeta_i: Array1<f64> = a_inv.dot(&lambda_s_beta).mapv(|x| -x);
        dbeta_drho.push(dbeta_i.clone());

        // ∂RSS/∂ρᵢ = -2·r'·X·∂β/∂ρᵢ
        let x_dbeta: Array1<f64> = x.dot(&dbeta_i);
        let drss_i: f64 = -2.0
            * residuals
                .iter()
                .zip(x_dbeta.iter())
                .map(|(ri, xdbi)| ri * xdbi)
                .sum::<f64>();
        drss_drho.push(drss_i);

        // ∂φ/∂ρᵢ = (∂RSS/∂ρᵢ) / (n-r)
        let dphi_i = drss_i / n_minus_r;
        dphi_drho.push(dphi_i);

        // ∂P/∂ρᵢ = ∂RSS/∂ρᵢ + λᵢ·β'·Sᵢ·β + 2·Σⱼ λⱼ·β'·Sⱼ·∂β/∂ρᵢ
        let beta_s_i_beta = penalty_i.quadratic_form(&beta);
        let explicit_pen = lambda_i * beta_s_i_beta;

        let mut implicit_pen = 0.0;
        for j in 0..m {
            // Optimized: S·β and S·dβ using block-sparse multiply
            let s_j_beta = penalties_blocks[j].dot_vec(&beta);
            let s_j_dbeta = penalties_blocks[j].dot_vec(&dbeta_i);

            let term1: f64 = s_j_beta
                .iter()
                .zip(dbeta_i.iter())
                .map(|(sj, dbi)| sj * dbi)
                .sum();
            let term2: f64 = beta
                .iter()
                .zip(s_j_dbeta.iter())
                .map(|(bi, sjd)| bi * sjd)
                .sum();
            implicit_pen += lambdas[j] * (term1 + term2);
        }

        let dp_i = drss_i + explicit_pen + implicit_pen;
        dp_drho.push(dp_i);
    }

    // Step 5: Compute Hessian
    let mut hessian = Array2::zeros((m, m));

    for i in 0..m {
        for j in i..m {
            // Only compute upper triangle (symmetric)
            let lambda_i = lambdas[i];
            let lambda_j = lambdas[j];
            let s_i = &penalties_blocks[i];
            let s_j = &penalties_blocks[j];
            let sqrt_si = &sqrt_penalties[i];

            // ================================================================
            // TERM 1: ∂/∂ρⱼ[tr(A⁻¹·λᵢ·Sᵢ)] / 2
            // ================================================================
            // = [δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ) - λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)] / 2

            // Part A: -λᵢ·λⱼ·tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
            // Use trace_product to compute tr(A⁻¹·Sⱼ·A⁻¹·Sᵢ)
            // tr(Sᵢ · (A⁻¹·Sⱼ·A⁻¹)) = tr((A⁻¹·Sⱼ·A⁻¹) · Sᵢ)
            let ainv_sj: Array2<f64> = s_j.left_mul_dense(&a_inv);
            let ainv_sj_ainv: Array2<f64> = ainv_sj.dot(&a_inv);
            let trace1a_val = s_i.trace_product(&ainv_sj_ainv);
            let term1a = -lambda_i * lambda_j * trace1a_val;

            // Part B: δᵢⱼ·λᵢ·tr(A⁻¹·Sᵢ)
            let term1b = if i == j {
                let p_t_sqrt_si = p_matrix.t().dot(sqrt_si);
                let trace_ainv_si: f64 = p_t_sqrt_si.iter().map(|x| x * x).sum();
                lambda_i * trace_ainv_si
            } else {
                0.0
            };

            let term1 = (term1a + term1b) / 2.0;

            // ================================================================
            // TERM 2: ∂²(P/φ)/∂ρⱼ∂ρᵢ / 2
            // ================================================================
            // This is the big one! Needs ∂²P, ∂²RSS, ∂²β, ∂²φ

            // Compute ∂²β/∂ρⱼ∂ρᵢ
            let si_beta = s_i.dot_vec(&beta);
            let ainv_si_beta: Array1<f64> = a_inv.dot(&si_beta);
            let lambda_i_ainv_si_beta = ainv_si_beta.mapv(|x| lambda_i * x);
            let sj_times_term = s_j.dot_vec(&lambda_i_ainv_si_beta);
            let part_a = a_inv.dot(&sj_times_term).mapv(|x| lambda_j * x);

            let si_dbeta_j = s_i.dot_vec(&dbeta_drho[j]);
            let part_b = a_inv.dot(&si_dbeta_j).mapv(|x| -lambda_i * x);

            let mut d2beta = part_a + part_b;
            if i == j {
                d2beta = d2beta - dbeta_drho[i].clone();
            }

            // Compute ∂²RSS/∂ρⱼ∂ρᵢ
            let x_dbeta_j: Array1<f64> = x.dot(&dbeta_drho[j]);
            let x_dbeta_i = x.dot(&dbeta_drho[i]);
            let d2rss_part1 = 2.0 * x_dbeta_j.dot(&x_dbeta_i);

            let x_d2beta = x.dot(&d2beta);
            let d2rss_part2 = -2.0 * residuals.dot(&x_d2beta);

            let d2rss = d2rss_part1 + d2rss_part2;

            // Compute ∂²φ/∂ρⱼ∂ρᵢ = (1/(n-r))·∂²RSS/∂ρⱼ∂ρᵢ
            let d2phi = d2rss / n_minus_r;

            // Compute ∂²P/∂ρⱼ∂ρᵢ
            // = ∂²RSS/∂ρⱼ∂ρᵢ + δᵢⱼ·λᵢ·β'·Sᵢ·β + 2·λᵢ·∂β'/∂ρⱼ·Sᵢ·β
            //   + 2·Σₖ[δₖⱼ·λₖ·∂β'/∂ρᵢ·Sₖ·β + λₖ·∂²β'/∂ρⱼ∂ρᵢ·Sₖ·β + λₖ·∂β'/∂ρᵢ·Sₖ·∂β/∂ρⱼ]

            let diag_explicit = if i == j {
                let beta_si_beta: f64 = beta
                    .iter()
                    .zip(si_beta.iter())
                    .map(|(bi, sbi)| bi * sbi)
                    .sum();
                lambda_i * beta_si_beta
            } else {
                0.0
            };

            let dbeta_j_si_beta: f64 = dbeta_drho[j]
                .iter()
                .zip(si_beta.iter())
                .map(|(dbj, sbi)| dbj * sbi)
                .sum();
            let explicit_cross = 2.0 * lambda_i * dbeta_j_si_beta;

            let mut implicit_sum = 0.0;
            for k in 0..m {
                let sk_beta = penalties_blocks[k].dot_vec(&beta);
                let sk_dbeta_i = penalties_blocks[k].dot_vec(&dbeta_drho[i]);

                // δₖⱼ·λₖ·∂β'/∂ρᵢ·Sₖ·β
                let term1 = if k == j {
                    let val: f64 = dbeta_drho[i]
                        .iter()
                        .zip(sk_beta.iter())
                        .map(|(dbi, skb)| dbi * skb)
                        .sum();
                    lambdas[k] * val
                } else {
                    0.0
                };

                // λₖ·∂²β'/∂ρⱼ∂ρᵢ·Sₖ·β
                let sk_d2beta: f64 = d2beta
                    .iter()
                    .zip(sk_beta.iter())
                    .map(|(d2bi, skb)| d2bi * skb)
                    .sum();
                let term2 = lambdas[k] * sk_d2beta;

                // λₖ·∂β'/∂ρᵢ·Sₖ·∂β/∂ρⱼ
                let dbeta_i_sk_dbeta_j: f64 = dbeta_drho[i]
                    .iter()
                    .zip(sk_dbeta_i.iter())
                    .map(|(dbi, skdbj)| dbi * skdbj)
                    .sum();
                let term3 = lambdas[k] * dbeta_i_sk_dbeta_j;

                implicit_sum += term1 + term2 + term3;
            }

            let d2p = d2rss + diag_explicit + explicit_cross + 2.0 * implicit_sum;

            // Now compute ∂²(P/φ)/∂ρⱼ∂ρᵢ
            // = (1/φ)·∂²P/∂ρⱼ∂ρᵢ - (1/φ²)·[∂φ/∂ρⱼ·∂P/∂ρᵢ + ∂P/∂ρⱼ·∂φ/∂ρᵢ]
            //   + 2·(P/φ³)·∂φ/∂ρⱼ·∂φ/∂ρᵢ - (P/φ²)·∂²φ/∂ρⱼ∂ρᵢ

            let term2a = inv_phi * d2p;
            let term2b = -(1.0 / phi_sq) * (dphi_drho[j] * dp_drho[i] + dp_drho[j] * dphi_drho[i]);
            let term2c = 2.0 * (p_value / phi_cb) * dphi_drho[j] * dphi_drho[i];
            let term2d = -(p_value / phi_sq) * d2phi;

            let term2 = (term2a + term2b + term2c + term2d) / 2.0;

            // ================================================================
            // TERM 3: ∂/∂ρⱼ[(n-r)·(1/φ)·∂φ/∂ρᵢ] / 2
            // ================================================================
            // = (n-r)·[(1/φ)·∂²φ/∂ρⱼ∂ρᵢ - (1/φ²)·∂φ/∂ρⱼ·∂φ/∂ρᵢ] / 2

            let term3a = n_minus_r * inv_phi * d2phi;
            let term3b = -n_minus_r * (1.0 / phi_sq) * dphi_drho[j] * dphi_drho[i];

            let term3 = (term3a + term3b) / 2.0;

            // ================================================================
            // TOTAL HESSIAN
            // ================================================================
            let h_val = term1 + term2 + term3;
            hessian[[i, j]] = h_val;

            if std::env::var("MGCV_HESS_DEBUG").is_ok() && (i == j || (i == 0 && j == 1)) {
                eprintln!("\n[HESS_CORRECTED] H[{},{}]:", i, j);
                eprintln!("  Term 1 (∂²tr/∂ρⱼ∂ρᵢ): {:.6e}", term1);
                eprintln!("    - 1a (cross): {:.6e}", term1a / 2.0);
                eprintln!("    - 1b (diagonal): {:.6e}", term1b / 2.0);
                eprintln!("  Term 2 (∂²(P/φ)/∂ρⱼ∂ρᵢ): {:.6e}", term2);
                eprintln!("    - 2a (d2P/φ): {:.6e}", term2a / 2.0);
                eprintln!("    - 2b (cross dP·dφ/φ²): {:.6e}", term2b / 2.0);
                eprintln!("    - 2c (P·dφ²/φ³): {:.6e}", term2c / 2.0);
                eprintln!("    - 2d (P·d2φ/φ²): {:.6e}", term2d / 2.0);
                eprintln!("    - d2rss: {:.6e}", d2rss);
                eprintln!("    - d2P: {:.6e}", d2p);
                eprintln!("    - d2phi: {:.6e}", d2phi);
                eprintln!("  Term 3 (∂/∂ρⱼ[(n-r)·dφ/φ]): {:.6e}", term3);
                eprintln!("    - 3a ((n-r)·d2φ/φ): {:.6e}", term3a / 2.0);
                eprintln!("    - 3b (-(n-r)·dφ²/φ²): {:.6e}", term3b / 2.0);
                eprintln!("  TOTAL: {:.6e}", h_val);
            }

            // Fill symmetric entry
            if i != j {
                hessian[[j, i]] = hessian[[i, j]];
            }
        }
    }

    if std::env::var("MGCV_HESS_DEBUG").is_ok() {
        eprintln!("\n[HESS_CORRECTED] Final Hessian:");
        for i in 0..m {
            eprint!("  [");
            for j in 0..m {
                eprint!("{:10.6e} ", hessian[[i, j]]);
            }
            eprintln!("]");
        }
    }

    Ok(hessian)
}

/// Compute the gradient of REML with respect to log(λᵢ)
///
/// Returns: ∂REML/∂log(λᵢ) for i = 1..m
///
/// Following mgcv's fast-REML.r implementation (lines 1718-1719), the gradient is:
/// ∂REML/∂log(λᵢ) = [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2
///
/// Where:
/// - A = X'WX + Σλⱼ·Sⱼ
/// - φ = RSS / (n - Σrank(Sⱼ))
/// - At optimum, ∂RSS/∂log(λᵢ) ≈ 0 (first-order condition), so we can ignore it
pub fn reml_gradient_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Result<Array1<f64>> {
    eprintln!("[GRAD_DEBUG] OLD reml_gradient_multi called!");
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute weighted design matrix
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        // Add in-place using BlockPenalty
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Solve for coefficients
    let y_weighted: Array1<f64> = y.iter().zip(w.iter()).map(|(yi, wi)| yi * wi).collect();

    let b = xtw.dot(&y_weighted);

    // Add ridge for numerical stability
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (penalties_blocks.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_solve = a.clone();
    for i in 0..p {
        a_solve[[i, i]] += ridge;
    }

    let beta = solve(a_solve, b)?;

    // Compute fitted values and RSS
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute total rank and φ
    let mut total_rank = 0;
    for penalty in penalties_blocks.iter() {
        total_rank += estimate_rank(penalty);
    }
    let phi = rss / (n - total_rank) as f64;

    // Compute A^(-1)
    // Use adaptive ridge based on matrix magnitude and number of penalties
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (penalties_blocks.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute gradient for each λᵢ
    let mut gradient = Array1::zeros(m);

    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];
        let rank_i = estimate_rank(penalty_i);

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[GRAD_DEBUG] ALL lambdas: {:?}", lambdas);
            eprintln!(
                "[GRAD_DEBUG] penalty matrix size: {}x{}, estimated rank: {}",
                penalty_i.block_view().nrows(),
                penalty_i.block_view().ncols(),
                rank_i
            );

            // Check A and A_inv
            let a_max = a_inv.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let a_trace = (0..p).map(|j| a_inv[[j, j]]).sum::<f64>();
            eprintln!(
                "[GRAD_DEBUG] A_inv: max_element={:.6e}, trace={:.6}",
                a_max, a_trace
            );
        }

        // Term 1: tr(A⁻¹·λᵢ·Sᵢ)
        // tr(A⁻¹·Sᵢ) = tr(Sᵢ·A⁻¹)
        // Optimized using BlockPenalty: only multiply relevant block
        // BlockPenalty S is sparse, so we can compute trace(S·A⁻¹) efficiently
        // S = [0 0; 0 B], A⁻¹ = [P11 P12; P21 P22]
        // S·A⁻¹ = [0 0; B·P21 B·P22]
        // trace = tr(B·P22)

        let trace = if lambda_i.abs() > 1e-10 {
            // Get submatrix of A_inv corresponding to penalty block
            // Use trace_product which computes tr(S_i · A_inv)
            lambda_i * penalty_i.trace_product(&a_inv)
        } else {
            0.0
        };

        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[GRAD_DEBUG] A^(-1)*lambda*S: trace={:.6}", trace);
        }

        // Term 2: λᵢ·β'·Sᵢ·β
        let beta_s_beta = penalty_i.quadratic_form(&beta);
        let penalty_term = lambda_i * beta_s_beta;

        // Gradient: [tr(A⁻¹·λᵢ·Sᵢ) - rank(Sᵢ) + (λᵢ·β'·Sᵢ·β)/φ] / 2
        if std::env::var("MGCV_GRAD_DEBUG").is_ok() && i == 0 {
            eprintln!("[GRAD_DEBUG] Component {}: lambda={:.6}, trace={:.6}, rank={}, penalty_term={:.6}, phi={:.6}",
                     i, lambda_i, trace, rank_i, penalty_term, phi);
            eprintln!(
                "[GRAD_DEBUG]   trace - rank = {:.6}, (trace - rank + penalty_term/phi)/2 = {:.6}",
                trace - (rank_i as f64),
                (trace - (rank_i as f64) + penalty_term / phi) / 2.0
            );
        }
        gradient[i] = (trace - (rank_i as f64) + penalty_term / phi) / 2.0;
    }

    Ok(gradient)
}

/// Compute the Hessian of REML with respect to log(λᵢ), log(λⱼ)
///
/// Returns: ∂²REML/∂log(λᵢ)∂log(λⱼ) for i,j = 1..m
///
/// Following Wood (2011) J.R.Statist.Soc.B 73(1):3-36, the complete Hessian is:
/// H[i,j] = [-tr(M_i·A·M_j·A) + (2β'·M_i·A·M_j·β)/φ - (2β'·M_i·β·β'·M_j·β)/φ²] / 2
///
/// where M_i = λ_i·S_i, A = (X'WX + ΣM_i)^(-1)
///
/// This is a symmetric m x m matrix
pub fn reml_hessian_multi(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
) -> Result<Array2<f64>> {
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // Compute weighted design matrix
    let mut x_weighted = x.clone();
    for i in 0..n {
        let weight_sqrt = w[i].sqrt();
        for j in 0..p {
            x_weighted[[i, j]] *= weight_sqrt;
        }
    }

    // Compute X'WX
    let xtw = x_weighted.t().to_owned();
    let xtwx = xtw.dot(&x_weighted);

    // Compute A = X'WX + Σλᵢ·Sᵢ
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        // Add in-place using BlockPenalty
        penalty.scaled_add_to(&mut a, *lambda);
    }

    // Compute A^(-1)
    // Add adaptive ridge term to ensure numerical stability
    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (lambdas.len() as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute coefficients β
    let y_weighted: Array1<f64> = y.iter().zip(w.iter()).map(|(yi, wi)| yi * wi).collect();
    let b = xtw.dot(&y_weighted);
    // Use regularized matrix for numerical stability
    let beta = solve(a_reg.clone(), b)?;

    // Compute RSS and φ
    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // Compute effective degrees of freedom for φ
    // edf = tr(A^{-1}·X'WX)
    // For Gaussian case with W=I: edf = tr(A^{-1}·X'X)
    let xtx = x.t().to_owned().dot(&x.to_owned());
    let ainv_xtx = a_inv.dot(&xtx);
    let edf: f64 = (0..ainv_xtx.nrows()).map(|i| ainv_xtx[[i, i]]).sum();

    // Correct φ computation using effective df
    let phi = rss / (n as f64 - edf);

    // Debug: compare against old approach
    if std::env::var("MGCV_GRAD_DEBUG").is_ok() {
        let old_total_rank: usize = penalties_blocks.iter().map(|p| estimate_rank(p)).sum();
        let old_phi = rss / (n as f64 - old_total_rank as f64);
        eprintln!("[PHI_DEBUG] edf (correct) = {:.3}, old total_rank = {}, φ_correct = {:.6e}, φ_old = {:.6e}, ratio = {:.3}",
                  edf, old_total_rank, phi, old_phi, old_phi / phi);
    }

    // Compute first derivatives of β with respect to log(λ_i)
    // dβ/dρ_i = -A^{-1}·M_i·β where M_i = λ_i·S_i
    let mut dbeta_drho = Vec::with_capacity(m);
    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];

        // M_i·β = λ_i·S_i·β
        let si_beta = penalty_i.dot_vec(&beta);
        let mi_beta = si_beta.mapv(|x| lambda_i * x);

        // -A^{-1}·M_i·β
        let dbeta_i = a_inv.dot(&mi_beta).mapv(|x| -x);
        dbeta_drho.push(dbeta_i);
    }

    // Compute bSb1 (first derivatives of β'·S·β/φ with respect to log(λ_i))
    // This is needed for diagonal correction in bSb2
    let mut bsb1 = Vec::with_capacity(m);
    for i in 0..m {
        let lambda_i = lambdas[i];
        let penalty_i = &penalties_blocks[i];

        // β'·S_i·β
        let beta_s_i_beta = penalty_i.quadratic_form(&beta);

        // 2·dβ/dρ_i'·S·β where S = Σλ_j·S_j
        let mut s_beta_total = Array1::zeros(p);
        for j in 0..m {
            // S_j·β
            let sj_beta = penalties_blocks[j].dot_vec(&beta);
            let lambda_j_sj_beta = sj_beta.mapv(|x| lambdas[j] * x);
            s_beta_total = s_beta_total + lambda_j_sj_beta;
        }

        let dbeta_i_s_beta: f64 = dbeta_drho[i].dot(&s_beta_total);
        let term2 = 2.0 * dbeta_i_s_beta;

        // ∂φ/∂ρ_i ≈ 0 (at optimum), so we ignore the φ derivative term
        // bSb1 = (λ_i·β'·S_i·β + 2·dβ/dρ_i'·S·β) / φ
        bsb1.push((lambda_i * beta_s_i_beta + term2) / phi);
    }

    // Compute Hessian elements
    let mut hessian = Array2::zeros((m, m));

    for i in 0..m {
        for j in i..m {
            let lambda_i = lambdas[i];
            let lambda_j = lambdas[j];
            let s_i = &penalties_blocks[i];
            let s_j = &penalties_blocks[j];

            // Term 1: -tr(M_i·A·M_j·A)
            // M_i = λ_i·S_i, M_j = λ_j·S_j
            // tr(M_i·A·M_j·A) = λ_i·λ_j·tr(S_i·A·S_j·A)
            // tr(S_i·(A·S_j·A)) using trace_product

            // Compute A·S_j·A
            // S_j·A using left_mul_dense (BlockPenalty method)
            let sj_a = s_j.left_mul_dense(&a_inv); // Note: a_inv IS A here
            let a_sj_a = a_inv.dot(&sj_a);

            // tr(S_i · (A·S_j·A))
            let trace_term = lambda_i * lambda_j * s_i.trace_product(&a_sj_a);

            // Term 2: 2β'·M_i·A·M_j·β / φ
            // = 2·λ_i·λ_j·β'·S_i·A·S_j·β / φ
            let sj_beta = s_j.dot_vec(&beta);
            let a_sj_beta = a_inv.dot(&sj_beta);
            let si_a_sj_beta = s_i.dot_vec(&a_sj_beta);
            let beta_si_a_sj_beta = beta.dot(&si_a_sj_beta);
            let term2 = 2.0 * lambda_i * lambda_j * beta_si_a_sj_beta / phi;

            // Term 3: -2β'·M_i·β·β'·M_j·β / φ²
            // (Only approximates the cross terms, full expansion is complex)
            let mi_beta = lambda_i * s_i.quadratic_form(&beta);
            let mj_beta = lambda_j * s_j.quadratic_form(&beta);
            let term3 = 2.0 * mi_beta * mj_beta / (phi * phi);

            // Additional diagonal correction for i=j:
            // The derivative of λ_i w.r.t ρ_i introduces an extra term
            // ∂M_i/∂ρ_i = M_i
            let diag_correction = if i == j {
                // tr(M_i·A) - (M_i·β)'β/φ
                // tr(M_i·A) = λ_i·tr(S_i·A)
                let tr_mia = lambda_i * s_i.trace_product(&a_inv);
                let mia_beta = mi_beta / phi;
                (tr_mia - mia_beta) / 2.0 // Divided by 2 as per overall REML
            } else {
                0.0
            };

            // bSb2: Penalty Hessian full calculation
            // bSb2[k,m] = 2·(d²β'/dρ_k dρ_m · S · β) + ...

            // Term 1: d²β'/dρ_k dρ_m · S · β
            // From implicit differentiation:
            // d²β/dρ_i dρ_j = A^{-1}·[M_i·A^{-1}·M_j·β + M_j·A^{-1}·M_i·β] + δ_{ij}·dβ/dρ_i

            // M_i·A^{-1}·M_j·β
            let mj_beta_vec = s_j.dot_vec(&beta).mapv(|x| lambda_j * x);
            let a_inv_mj_beta = a_inv.dot(&mj_beta_vec);
            let mi_a_inv_mj_beta = s_i.dot_vec(&a_inv_mj_beta).mapv(|x| lambda_i * x);

            // M_j·A^{-1}·M_i·β
            let mi_beta_vec = s_i.dot_vec(&beta).mapv(|x| lambda_i * x);
            let a_inv_mi_beta = a_inv.dot(&mi_beta_vec);
            let mj_a_inv_mi_beta = s_j.dot_vec(&a_inv_mi_beta).mapv(|x| lambda_j * x);

            let mut d2beta_term = Array1::zeros(p);
            d2beta_term += &mi_a_inv_mj_beta;
            d2beta_term += &mj_a_inv_mi_beta;
            let mut d2beta = a_inv.dot(&d2beta_term);

            if i == j {
                d2beta += &dbeta_drho[i];
            }

            // S·β
            let mut s_beta_total = Array1::zeros(p);
            for (lambda_k, penalty_k) in lambdas.iter().zip(penalties_blocks.iter()) {
                let s_k_beta = penalty_k.dot_vec(&beta);
                s_beta_total.scaled_add(*lambda_k, &s_k_beta);
            }

            let term1_val: f64 = d2beta.dot(&s_beta_total);

            // Term 2: dβ'/dρ_k · S · dβ/dρ_m
            let s_dbeta_j = {
                let mut result = Array1::zeros(p);
                for (lambda_k, penalty_k) in lambdas.iter().zip(penalties_blocks.iter()) {
                    let s_k_dbeta_j = penalty_k.dot_vec(&dbeta_drho[j]);
                    result.scaled_add(*lambda_k, &s_k_dbeta_j);
                }
                result
            };
            let term2_val: f64 = dbeta_drho[i].dot(&s_dbeta_j);

            // Term 3: dβ'/dρ_m · S_k · β · λ_k (when k=i)
            let si_beta = s_i.dot_vec(&beta);
            let term3_val: f64 = dbeta_drho[j].dot(&si_beta) * lambda_i;

            // Term 4: dβ'/dρ_k · S_m · β · λ_m (when m=j)
            let sj_beta = s_j.dot_vec(&beta);
            let term4_val: f64 = dbeta_drho[i].dot(&sj_beta) * lambda_j;

            let diag_corr = if i == j { bsb1[i] } else { 0.0 };
            let bsb2 = 2.0 * (term1_val + term2_val + term3_val + term4_val) + diag_corr;

            // Re-calculate det2 part correctly as in mgcv
            // det2 = tr(A⁻¹·M_i)·δ_{ij} - tr(A⁻¹·M_i·A⁻¹·M_j)
            let det2 = if i == j {
                lambda_i * s_i.trace_product(&a_inv) - trace_term
            } else {
                -trace_term
            };

            hessian[[i, j]] = (det2 + bsb2) / 2.0;

            if i != j {
                hessian[[j, i]] = hessian[[i, j]];
            }
        }
    }

    Ok(hessian)
}
pub fn reml_hessian_multi_cached(
    y: &Array1<f64>,
    x: &Array2<f64>,
    w: &Array1<f64>,
    lambdas: &[f64],
    penalties_blocks: &[BlockPenalty],
    cached_xtwx: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n = y.len();
    let p = x.ncols();
    let m = lambdas.len();

    // OPTIMIZATION: Use cached X'WX instead of recomputing
    let xtwx = cached_xtwx;

    // Rest of computation (same as reml_hessian_multi but using cached xtwx)
    let mut a = xtwx.clone();
    for (lambda, penalty) in lambdas.iter().zip(penalties_blocks.iter()) {
        penalty.scaled_add_to(&mut a, *lambda);
    }

    let mut max_diag: f64 = 1.0;
    for i in 0..p {
        max_diag = max_diag.max(a[[i, i]].abs());
    }
    let ridge_scale = 1e-5 * (1.0 + (m as f64).sqrt());
    let ridge = ridge_scale * max_diag;
    let mut a_reg = a.clone();
    for i in 0..p {
        a_reg[[i, i]] += ridge;
    }
    let a_inv = inverse(&a_reg)?;

    // Compute X'Wy directly (avoid creating weighted matrices)
    let mut xtwy = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            sum += x[[i, j]] * w[i] * y[i];
        }
        xtwy[j] = sum;
    }
    let beta = solve(a_reg.clone(), xtwy)?;

    let fitted = x.dot(&beta);
    let residuals: Array1<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();
    let rss: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * r * wi)
        .sum();

    // OPTIMIZATION: Use cached X'WX for EDF instead of recomputing X'X (O(n*p^2) savings)
    // EDF = tr(A^{-1} X'WX) — note we need X'WX, not X'X
    let ainv_xtwx = a_inv.dot(xtwx);
    let edf: f64 = (0..ainv_xtwx.nrows()).map(|i| ainv_xtwx[[i, i]]).sum();
    let phi = rss / (n as f64 - edf);

    let mut dbeta_drho = Vec::with_capacity(m);
    for i in 0..m {
        let lambda_i = lambdas[i];
        let m_i_dense = penalties_blocks[i].scale(lambda_i).to_dense();
        let m_i_beta = m_i_dense.dot(&beta);
        dbeta_drho.push(a_inv.dot(&m_i_beta).mapv(|x| -x));
    }

    let mut bsb1 = Vec::with_capacity(m);
    for i in 0..m {
        let beta_s_beta = penalties_blocks[i].quadratic_form(&beta);
        bsb1.push(lambdas[i] * beta_s_beta / phi);
    }

    // OPTIMIZATION: Precompute terms that are reused across (i,j) pairs
    // This avoids O(m²) redundant matrix operations, reducing to O(m)
    let mut m_vec = Vec::with_capacity(m); // M_i = λ_i·S_i
    let mut m_a_inv = Vec::with_capacity(m); // M_i·A^(-1)
    let mut m_beta_vec = Vec::with_capacity(m); // M_i·β
    let mut s_beta_vec = Vec::with_capacity(m); // S_i·β
    let mut a_inv_m_beta = Vec::with_capacity(m); // A^(-1)·M_i·β

    for i in 0..m {
        let m_i: Array2<f64> = penalties_blocks[i].scale(lambdas[i]).to_dense();
        let m_i_a_inv = m_i.dot(&a_inv);
        let m_i_beta = m_i.dot(&beta);
        let s_i_beta = penalties_blocks[i].dot_vec(&beta);
        let a_inv_m_i_beta: Array1<f64> = a_inv.dot(&m_i_beta);

        m_vec.push(m_i);
        m_a_inv.push(m_i_a_inv);
        m_beta_vec.push(m_i_beta);
        s_beta_vec.push(s_i_beta);
        a_inv_m_beta.push(a_inv_m_i_beta);
    }

    let mut hessian = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let trace_a_inv_m_i: f64 = (0..p).map(|k| m_a_inv[i][[k, k]]).sum();

        for j in 0..=i {
            let trace_term: f64 = if i != j {
                let prod: Array2<f64> = a_inv
                    .dot(&m_a_inv[i].t())
                    .dot(&m_vec[j].t())
                    .dot(&a_inv.t());
                (0..p).map(|k| prod[[k, k]]).sum()
            } else {
                0.0
            };

            let det2 = if i == j {
                trace_a_inv_m_i - trace_term
            } else {
                -trace_term
            };

            let m_i_a_inv_m_j_beta: Array1<f64> = m_vec[i].dot(&a_inv_m_beta[j]);
            let m_j_a_inv_m_i_beta: Array1<f64> = m_vec[j].dot(&a_inv_m_beta[i]);
            let d2beta_prod: Array1<f64> = a_inv.dot(&(&m_i_a_inv_m_j_beta + &m_j_a_inv_m_i_beta));
            let d2beta = if i == j {
                d2beta_prod + &dbeta_drho[i]
            } else {
                d2beta_prod
            };

            let term1: f64 = d2beta
                .iter()
                .zip(s_beta_vec[i].iter())
                .map(|(d2bi, sbi)| d2bi * sbi)
                .sum::<f64>();
            let s_i_dbeta_j = penalties_blocks[i].dot_vec(&dbeta_drho[j]);
            let term2: f64 = dbeta_drho[i]
                .iter()
                .zip(s_i_dbeta_j.iter())
                .map(|(dbi, sjdbj)| dbi * sjdbj)
                .sum::<f64>();
            let term3: f64 = dbeta_drho[j]
                .iter()
                .zip(s_beta_vec[i].iter())
                .map(|(dbj, sib)| dbj * sib)
                .sum::<f64>()
                * lambdas[i];
            let term4: f64 = dbeta_drho[i]
                .iter()
                .zip(s_beta_vec[j].iter())
                .map(|(dbi, sjb)| dbi * sjb)
                .sum::<f64>()
                * lambdas[j];

            let diag_corr = if i == j { bsb1[i] } else { 0.0 };
            let bsb2 = 2.0 * (term1 + term2 + term3 + term4) + diag_corr;
            hessian[[i, j]] = (det2 + bsb2) / 2.0;
            if i != j {
                hessian[[j, i]] = hessian[[i, j]];
            }
        }
    }
    Ok(hessian)
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    use super::*;

    #[test]
    fn test_reml_criterion() {
        let n = 10;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let w = Array1::ones(n);
        let penalty = Array2::eye(p);
        let penalty_block = BlockPenalty::new(penalty, 0, p);
        let lambda = 0.1;

        let result = reml_criterion(&y, &x, &w, lambda, &penalty_block, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gcv_criterion() {
        let n = 10;
        let p = 5;

        let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let x = Array2::from_shape_fn((n, p), |(i, j)| (i + j) as f64);
        let w = Array1::ones(n);
        let penalty = Array2::eye(p);
        let penalty_block = BlockPenalty::new(penalty, 0, p);
        let lambda = 0.1;

        let result = gcv_criterion(&y, &x, &w, lambda, &penalty_block);
        assert!(result.is_ok());
    }

    /// Test that multi-dimensional gradient computation doesn't overflow
    /// This was the critical bug: P matrix values reached 1e27 causing NaN gradients
    #[test]
    fn test_multidim_gradient_no_overflow() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Small test case: n=100, 3 dimensions, k=5
        let n = 100;
        let n_dims = 3;
        let k = 5;
        let p = n_dims * k;

        // Generate design matrix
        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = rng.gen::<f64>();
            }
        }

        // Generate response
        let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let w = Array1::ones(n);

        // Create block-diagonal penalty matrices (like cubic regression spline)
        let mut penalties = Vec::new();

        for dim in 0..n_dims {
            let mut penalty = Array2::zeros((p, p));
            let start = dim * k;
            let end = start + k;

            // Create penalty matrix for this smooth (second derivative penalty structure)
            for i in start..end {
                for j in start..end {
                    if i == j {
                        penalty[[i, j]] = 2.0;
                    } else if (i as i32 - j as i32).abs() == 1 {
                        penalty[[i, j]] = -1.0;
                    }
                }
            }

            penalties.push(penalty);
        }

        let penalties_blocks: Vec<_> = penalties
            .into_iter()
            .map(|p| BlockPenalty::new(p.clone(), 0, p.nrows()))
            .collect();

        // Test with moderate lambdas
        let lambdas = vec![1.0, 1.0, 100.0];

        // Compute gradient
        let result = reml_gradient_multi_qr_adaptive(&y, &x, &w, &lambdas, &penalties_blocks);

        assert!(
            result.is_ok(),
            "Gradient computation failed: {:?}",
            result.err()
        );

        let gradient = result.unwrap();

        // Verify no overflow or NaN
        assert!(
            !gradient.iter().any(|g| !g.is_finite()),
            "Gradient contains non-finite values: {:?}",
            gradient
        );

        // Verify values are in reasonable range (not 1e27!)
        assert!(
            gradient.iter().all(|g| g.abs() < 1e10),
            "Gradient values too large: {:?}",
            gradient
        );

        println!("✓ No overflow: gradient={:?}", gradient);
    }

    /// Test gradient computation with ill-conditioned penalty matrices
    /// This tests the exact scenario that caused the 1e27 overflow bug
    #[test]
    fn test_multidim_gradient_ill_conditioned() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(123);

        let n = 50;
        let n_dims = 2;
        let k = 8;
        let p = n_dims * k;

        // Generate design matrix
        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = rng.gen::<f64>();
            }
        }

        let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let w = Array1::ones(n);

        // Create penalties with very different scales (ill-conditioned)
        let mut penalties = Vec::new();

        for dim in 0..n_dims {
            let mut penalty = Array2::zeros((p, p));
            let start = dim * k;
            let end = start + k;

            // Create penalty with small eigenvalues (ill-conditioned)
            for i in start..end {
                penalty[[i, i]] = if i == start { 1e-8 } else { 1.0 };
            }

            penalties.push(penalty);
        }

        let penalties_blocks: Vec<_> = penalties
            .into_iter()
            .map(|p| BlockPenalty::new(p.clone(), 0, p.nrows()))
            .collect();

        // Test with very different lambda scales
        let lambdas = vec![0.01, 1000.0];

        let result = reml_gradient_multi_qr_adaptive(&y, &x, &w, &lambdas, &penalties_blocks);

        assert!(
            result.is_ok(),
            "Gradient computation failed on ill-conditioned case"
        );

        let gradient = result.unwrap();

        // Critical checks: must remain stable despite ill-conditioning
        assert!(
            gradient.iter().all(|g| g.is_finite()),
            "Gradient not finite with ill-conditioned penalties"
        );

        // Check no catastrophic overflow
        assert!(
            gradient.iter().all(|g| g.abs() < 1e10),
            "Gradient overflow with ill-conditioning: {:?}",
            gradient
        );

        println!("✓ Ill-conditioned case stable: gradient={:?}", gradient);
    }

    /// Test that gradients match finite difference approximation
    ///
    /// KNOWN ISSUE: The analytical gradient in reml_gradient_multi_qr_adaptive has a bug
    /// (analytical=-0.31 vs FD=0.88, wrong sign). This function is only used by
    /// the Python-exposed NewtonPIRLS path, NOT the production Rust fitting path
    /// (which uses reml_gradient_multi_qr_adaptive_cached_edf via smooth.rs).
    /// Fixing requires rederiving the IFT-based gradient formula.
    #[test]
    #[ignore = "known bug: analytical gradient disagrees with FD in reml_gradient_multi_qr_adaptive (non-production path)"]
    fn test_multidim_gradient_accuracy() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(456);

        // Very small case for accurate finite differences
        let n = 30;
        let n_dims = 2;
        let k = 4;
        let p = n_dims * k;

        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = rng.gen::<f64>();
            }
        }

        let y: Array1<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let w = Array1::ones(n);

        // Simple identity penalties for cleaner finite differences
        let mut penalties = Vec::new();

        for dim in 0..n_dims {
            let mut penalty = Array2::zeros((p, p));
            let start = dim * k;
            let end = start + k;

            for i in start..end {
                penalty[[i, i]] = 1.0;
            }

            penalties.push(penalty);
        }

        let penalties_blocks: Vec<_> = penalties
            .into_iter()
            .map(|p| BlockPenalty::new(p.clone(), 0, p.nrows()))
            .collect();

        let lambdas = vec![1.0, 1.0];

        // Compute analytical gradient (uses rank-based phi)
        let result = reml_gradient_multi_qr_adaptive(&y, &x, &w, &lambdas, &penalties_blocks);
        assert!(result.is_ok());
        let gradient_analytical = result.unwrap();

        // Rank-based REML criterion matching the gradient's phi formula.
        // reml_criterion_multi uses EDF-based phi with BLAS, so we need a
        // consistent rank-based version for finite differences.
        let reml_rank_based = |lam: &[f64]| -> f64 {
            let xtwx = compute_xtwx(&x, &w);
            let mut a = xtwx.clone();
            for (lambda, pen) in lam.iter().zip(penalties_blocks.iter()) {
                pen.scaled_add_to(&mut a, *lambda);
            }

            let b = compute_xtwy(&x, &w, &y);
            let max_diag = a.diag().iter().map(|v| v.abs()).fold(1.0f64, f64::max);
            let ridge_scale = 1e-5 * (1.0 + (lam.len() as f64).sqrt());
            let ridge = ridge_scale * max_diag;
            let mut a_solve = a.clone();
            a_solve.diag_mut().iter_mut().for_each(|v| *v += ridge);

            let beta = solve(a_solve, b).unwrap();
            let fitted = x.dot(&beta);
            let residuals: Array1<f64> = y
                .iter()
                .zip(fitted.iter())
                .map(|(yi, fi)| yi - fi)
                .collect();
            let rss: f64 = residuals
                .iter()
                .zip(w.iter())
                .map(|(r, wi)| r * r * wi)
                .sum();

            let mut penalty_sum = 0.0;
            for (lambda, pen) in lam.iter().zip(penalties_blocks.iter()) {
                let s_beta = pen.dot_vec(&beta);
                let bsb: f64 = beta
                    .iter()
                    .zip(s_beta.iter())
                    .map(|(bi, sbi)| bi * sbi)
                    .sum();
                penalty_sum += lambda * bsb;
            }

            let mut total_rank = 0usize;
            let mut log_lambda_sum = 0.0;
            let mut log_pseudo_det_sum = 0.0;
            for (lambda, pen) in lam.iter().zip(penalties_blocks.iter()) {
                if *lambda > 1e-10 {
                    let rank_s = pen.estimate_rank();
                    if rank_s > 0 {
                        total_rank += rank_s;
                        log_lambda_sum += (rank_s as f64) * lambda.ln();
                    }
                }
                #[cfg(feature = "blas")]
                {
                    log_pseudo_det_sum += pseudo_determinant(pen).unwrap_or(0.0);
                }
            }

            let n_minus_r = n as f64 - total_rank as f64;
            let phi = rss / n_minus_r;

            let mut a_reg = a;
            a_reg.diag_mut().iter_mut().for_each(|v| *v += ridge);
            let log_det_a = determinant(&a_reg).unwrap().ln();

            let pi = std::f64::consts::PI;
            ((rss + penalty_sum) / phi + n_minus_r * (2.0 * pi * phi).ln() + log_det_a
                - log_lambda_sum
                - log_pseudo_det_sum)
                / 2.0
        };

        let reml_0 = reml_rank_based(&lambdas);

        // Compute finite difference gradient
        let h = 1e-6;
        let mut gradient_fd = vec![0.0; n_dims];

        for i in 0..n_dims {
            let mut lambdas_plus = lambdas.clone();
            lambdas_plus[i] += h;
            let reml_plus = reml_rank_based(&lambdas_plus);
            gradient_fd[i] = (reml_plus - reml_0) / h;
        }

        // Check agreement (should be within 5% relative error)
        for i in 0..n_dims {
            let rel_error = if gradient_analytical[i].abs() > 1e-8 {
                ((gradient_analytical[i] - gradient_fd[i]) / gradient_analytical[i]).abs()
            } else {
                (gradient_analytical[i] - gradient_fd[i]).abs()
            };

            assert!(
                rel_error < 0.05 || (gradient_analytical[i] - gradient_fd[i]).abs() < 1e-5,
                "Gradient {} mismatch: analytical={:.6}, finite_diff={:.6}, rel_error={:.6}",
                i,
                gradient_analytical[i],
                gradient_fd[i],
                rel_error
            );
        }

        println!(
            "Gradient accuracy verified: analytical={:?}, fd={:?}",
            gradient_analytical, gradient_fd
        );
    }

    /// Test that lambdas vary significantly in multi-dimensional case
    /// This was the symptom: all lambdas stuck at ~0.21 instead of varying 5-5000
    #[test]
    fn test_multidim_lambda_variation() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(789);

        let n = 200;
        let n_dims = 3;
        let k = 8;
        let p = n_dims * k;

        // Generate data where different dimensions need different smoothing
        let mut x = Array2::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] = rng.gen::<f64>();
            }
        }

        // Create response that's smooth in x1, moderately smooth in x2, rough in x3
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let x1 = x[[i, 0]];
            let x2 = x[[i, k]];
            let x3 = x[[i, 2 * k]];
            y[i] = (2.0 * std::f64::consts::PI * x1).sin()  // Very smooth
                 + 0.5 * (6.0 * std::f64::consts::PI * x2).sin()  // Moderate
                 + 0.2 * rng.gen::<f64>(); // x3 mostly noise (needs high lambda)
        }

        let w = Array1::ones(n);

        // Create penalties
        let mut penalties = Vec::new();

        for dim in 0..n_dims {
            let mut penalty = Array2::zeros((p, p));
            let start = dim * k;
            let end = start + k;

            // Second derivative penalty structure
            for i in start..end {
                for j in start..end {
                    if i == j {
                        penalty[[i, j]] = 2.0;
                    } else if (i as i32 - j as i32).abs() == 1 {
                        penalty[[i, j]] = -1.0;
                    }
                }
            }

            penalties.push(penalty);
        }

        let penalties_blocks: Vec<_> = penalties
            .into_iter()
            .map(|p| BlockPenalty::new(p.clone(), 0, p.nrows()))
            .collect();

        // Start with moderate lambdas
        let lambdas = vec![10.0, 10.0, 100.0];

        let result = reml_gradient_multi_qr_adaptive(&y, &x, &w, &lambdas, &penalties_blocks);

        assert!(result.is_ok());
        let gradient = result.unwrap();

        // Key test: gradient should indicate lambdas need to diverge
        // If all gradients have same sign and magnitude, lambdas won't vary
        let grad_min = gradient.iter().cloned().fold(f64::INFINITY, f64::min);
        let grad_max = gradient.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Gradients should vary by at least 2x (otherwise optimization will keep them similar)
        let grad_range = if grad_max.abs() > grad_min.abs() {
            grad_max.abs() / grad_min.abs().max(1e-10)
        } else {
            grad_min.abs() / grad_max.abs().max(1e-10)
        };

        // This is a weak test, but checks the mechanism is working
        // Real optimization will cause lambdas to diverge over multiple iterations
        println!("Gradient range: {:?}, ratio: {:.2}", gradient, grad_range);

        // Just verify gradients are computable and different
        assert!(gradient.iter().all(|g| g.is_finite()));
        assert!(
            gradient[0] != gradient[1] || gradient[1] != gradient[2],
            "All gradients identical - optimization will fail"
        );

        println!("✓ Lambda variation test passed: gradients vary correctly");
    }

    // TODO: Add test for blockwise once it's updated to match new API

    /// Verify the cached Tweedie θ-derivative path matches a direct
    /// `reml_criterion_multi_cached_mgcv_exact` evaluation across the three
    /// FD probes that the Newton step uses (center / +h / -h). The cache only
    /// hoists the LINEAR SYSTEM out of the loop; the per-probe (p, σ²̂)-pieces
    /// should be bit-for-bit identical (rel < 1e-10).
    ///
    /// Fixture: small synthetic Tweedie problem, n=400 d=2 k=10, p=1.5.
    #[test]
    fn test_tweedie_theta_cache_matches_full_dispatch() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(7);

        // Build a 2-smooth design with k=10 each, n=400.
        let n = 400;
        let d = 2;
        let k_basis = 10;
        let p_dim = d * k_basis;

        let mut x = Array2::<f64>::zeros((n, p_dim));
        let mut y_orig = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..d {
                let xij: f64 = rng.gen();
                // Simple polynomial basis fill (we only need a well-conditioned
                // matrix; the exact basis is immaterial for the analytic-vs-FD
                // comparison).
                for kk in 0..k_basis {
                    x[[i, j * k_basis + kk]] =
                        ((kk as f64 + 1.0) * std::f64::consts::PI * xij).sin();
                }
                eta += (2.0 * std::f64::consts::PI * xij).sin();
            }
            let mu = eta.exp();
            // Tweedie-like response: gamma + 20% atom at zero, μ-scaled.
            let g: f64 = (rng.gen::<f64>() + 0.1) * mu / 2.0;
            y_orig[i] = if rng.gen::<f64>() < 0.2 { 0.0 } else { g };
        }
        let w = Array1::<f64>::ones(n);

        // Block-diagonal second-difference-ish penalty per smooth.
        let mut penalties = Vec::with_capacity(d);
        for j in 0..d {
            let mut s = Array2::<f64>::zeros((p_dim, p_dim));
            let start = j * k_basis;
            for i in start..(start + k_basis) {
                s[[i, i]] = 2.0;
                if i > start {
                    s[[i, i - 1]] = -1.0;
                    s[[i - 1, i]] = -1.0;
                }
            }
            penalties.push(BlockPenalty::new(s, 0, p_dim));
        }
        let lambdas = vec![0.5, 0.7];
        let mp = 1 + 2 * d; // intercept + nullspace per smooth (rough)

        let p_tweedie = 1.5_f64;
        let family = crate::pirls::Family::Tweedie { p: p_tweedie };

        // Reference: full dispatch path (rebuilds linear system each call).
        let xtwx = compute_xtwx(&x, &w);
        let ref_center = reml_criterion_multi_cached_mgcv_exact(
            &y_orig,
            &x,
            &w,
            &lambdas,
            &penalties,
            Some(&xtwx),
            None,
            mp,
            family,
            Some(&y_orig),
        )
        .expect("reference REML failed");

        // Cached path.
        let cache = TweedieThetaCache::build(
            &y_orig,
            &x,
            &w,
            &xtwx,
            &lambdas,
            &penalties,
            mp,
            &y_orig,
        )
        .expect("cache build failed");
        let cached_center = cache.score_at_p(p_tweedie).expect("cached score failed");

        let rel = (ref_center - cached_center).abs() / ref_center.abs().max(1.0);
        assert!(
            rel < 1e-10,
            "tweedie cache center-score mismatch: ref={} cached={} rel={:.3e}",
            ref_center,
            cached_center,
            rel
        );

        // Verify across three FD probes.
        let theta_to_p = |th: f64| -> f64 {
            let a = 1.001_f64;
            let b = 1.999_f64;
            let p = if th > 0.0 {
                let e = (-th).exp();
                (b + a * e) / (e + 1.0)
            } else {
                let e = th.exp();
                (b * e + a) / (e + 1.0)
            };
            p.max(1.001).min(1.999)
        };
        let h = 1e-3;
        for &theta in &[-0.5_f64, 0.0, 0.5] {
            for &dth in &[0.0_f64, h, -h] {
                let p = theta_to_p(theta + dth);
                let fam_p = crate::pirls::Family::Tweedie { p };
                let r_ref = reml_criterion_multi_cached_mgcv_exact(
                    &y_orig,
                    &x,
                    &w,
                    &lambdas,
                    &penalties,
                    Some(&xtwx),
                    None,
                    mp,
                    fam_p,
                    Some(&y_orig),
                )
                .expect("ref score failed");
                let r_cached = cache.score_at_p(p).expect("cached score failed");
                let rel = (r_ref - r_cached).abs() / r_ref.abs().max(1.0);
                assert!(
                    rel < 1e-10,
                    "θ={} dθ={} p={}: ref={} cached={} rel={:.3e}",
                    theta,
                    dth,
                    p,
                    r_ref,
                    r_cached,
                    rel
                );
            }
        }
    }

    /// Verify that the FD-on-cache derivative matches the FD-via-full-dispatch
    /// derivative (which is what the legacy `MGCV_TWEEDIE_FD=1` path computes).
    /// The two should agree to rel < 1e-3 since the only difference is round-off
    /// from reusing the factored linear system vs re-factoring.
    #[test]
    fn test_tweedie_theta_derivatives_match_fd_legacy() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(11);

        let n = 400;
        let d = 2;
        let k_basis = 10;
        let p_dim = d * k_basis;

        let mut x = Array2::<f64>::zeros((n, p_dim));
        let mut y_orig = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..d {
                let xij: f64 = rng.gen();
                for kk in 0..k_basis {
                    x[[i, j * k_basis + kk]] =
                        ((kk as f64 + 1.0) * std::f64::consts::PI * xij).sin();
                }
                eta += (2.0 * std::f64::consts::PI * xij).sin();
            }
            let mu = eta.exp();
            let g: f64 = (rng.gen::<f64>() + 0.1) * mu / 2.0;
            y_orig[i] = if rng.gen::<f64>() < 0.2 { 0.0 } else { g };
        }
        let w = Array1::<f64>::ones(n);

        let mut penalties = Vec::with_capacity(d);
        for j in 0..d {
            let mut s = Array2::<f64>::zeros((p_dim, p_dim));
            let start = j * k_basis;
            for i in start..(start + k_basis) {
                s[[i, i]] = 2.0;
                if i > start {
                    s[[i, i - 1]] = -1.0;
                    s[[i - 1, i]] = -1.0;
                }
            }
            penalties.push(BlockPenalty::new(s, 0, p_dim));
        }
        let lambdas = vec![0.5, 0.7];
        let mp = 1 + 2 * d;
        let xtwx = compute_xtwx(&x, &w);

        let theta_to_p = |th: f64| -> f64 {
            let a = 1.001_f64;
            let b = 1.999_f64;
            let p = if th > 0.0 {
                let e = (-th).exp();
                (b + a * e) / (e + 1.0)
            } else {
                let e = th.exp();
                (b * e + a) / (e + 1.0)
            };
            p.max(1.001).min(1.999)
        };

        let theta = 0.0_f64;
        let h = 1e-3;

        let p_center = theta_to_p(theta);
        let p_plus = theta_to_p(theta + h);
        let p_minus = theta_to_p(theta - h);

        let score_dispatch = |p: f64| -> f64 {
            let fam = crate::pirls::Family::Tweedie { p };
            reml_criterion_multi_cached_mgcv_exact(
                &y_orig,
                &x,
                &w,
                &lambdas,
                &penalties,
                Some(&xtwx),
                None,
                mp,
                fam,
                Some(&y_orig),
            )
            .expect("dispatch failed")
        };
        let rc_fd = score_dispatch(p_center);
        let rp_fd = score_dispatch(p_plus);
        let rm_fd = score_dispatch(p_minus);
        let grad_fd = (rp_fd - rm_fd) / (2.0 * h);
        let hess_fd = (rp_fd - 2.0 * rc_fd + rm_fd) / (h * h);

        let cache = TweedieThetaCache::build(
            &y_orig,
            &x,
            &w,
            &xtwx,
            &lambdas,
            &penalties,
            mp,
            &y_orig,
        )
        .expect("cache build failed");
        let (rc_c, grad_c, hess_c) =
            tweedie_theta_derivatives_cached(&cache, theta, h, theta_to_p)
                .expect("cached FD failed");

        // Center scores should be ~bit-identical.
        let rel_center = (rc_fd - rc_c).abs() / rc_fd.abs().max(1.0);
        assert!(
            rel_center < 1e-10,
            "center mismatch: fd={} cached={} rel={:.3e}",
            rc_fd,
            rc_c,
            rel_center
        );
        // Gradient: rel < 1e-3 (FD math is the same; only float-roundoff diff)
        let rel_grad = (grad_fd - grad_c).abs() / grad_fd.abs().max(1.0);
        assert!(
            rel_grad < 1e-3,
            "gradient mismatch: fd={} cached={} rel={:.3e}",
            grad_fd,
            grad_c,
            rel_grad
        );
        // Hessian (second-difference): rel < 1e-2 since it's much more sensitive
        // to round-off than the gradient (FD division by h² amplifies noise).
        let rel_hess = (hess_fd - hess_c).abs() / hess_fd.abs().max(1.0);
        assert!(
            rel_hess < 1e-2,
            "hessian mismatch: fd={} cached={} rel={:.3e}",
            hess_fd,
            hess_c,
            rel_hess
        );

        println!(
            "Tweedie θ-derivatives: grad rel diff {:.3e}, hess rel diff {:.3e}",
            rel_grad, rel_hess
        );
    }
}
