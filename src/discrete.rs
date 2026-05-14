//! Covariate-binning (a.k.a. `discrete=TRUE`) fast path — kernel layer.
//!
//! This is the port of mgcv's `discrete=TRUE` machinery (`compress.df`,
//! `discrete.mf`, `XWXd`, `XWyd`, `Xbd`). See
//! `docs/DISCRETE_BINNING_DESIGN.md` for the math and the rationale for
//! replacing the previous (buggy uniform-grid) implementation.
//!
//! Scope (D1-D3 of the design doc):
//!   1. `compress_1d` — mirrors `mgcv:::compress.df`: exact unique-row
//!      dedup when `nunique ≤ max_bins`, quantile (equal-width-grid) binning
//!      otherwise. The 1-D `max_bins` default is `1000`.
//!   2. `DiscreteMarginal` + `DiscreteDesign::new` — per-smooth / per-
//!      parametric-column compressed marginals plus an intercept marginal.
//!      Mirrors mgcv's `(Xd, kd, ks, nr)` quadruple.
//!   3. `compute_xtwx_discrete`, `compute_xtwy_discrete`,
//!      `compute_eta_discrete` — the scatter-gather kernels. They match
//!      the un-binned BLAS path (`reml::compute_xtwx`, `compute_xtwy`,
//!      `X · β`) to 1e-12 on pure-dedup fixtures.
//!
//! Out of scope here (handled by D4+): REML rewire, PIRLS integration.
//! The downstream callers in `gam_optimized.rs` and `pirls.rs` are
//! stubbed with `todo!()` and a `TODO(d4)` marker — they'll be properly
//! re-wired when D4 lands.

use crate::gam::SmoothTerm;
use ndarray::{s, Array1, Array2};
use std::collections::HashMap;

/// Configuration for the covariate-binning fast path.
///
/// Mirrors mgcv's per-dim defaults: `m = 1000` for 1-D smooths, `100`
/// per dim for 2-D, `25` per dim for 3+. Only 1-D is in scope today.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteConfig {
    /// Per-marginal cap on the compressed-row count. Pure-dedup is
    /// used when `nunique ≤ max_bins_1d`; quantile-grid otherwise.
    pub max_bins_1d: usize,
}

impl Default for DiscreteConfig {
    fn default() -> Self {
        DiscreteConfig { max_bins_1d: 1000 }
    }
}

/// A single smooth's (or parametric column's, or intercept's) compressed
/// marginal.
///
/// Mirrors one column-slab of mgcv's `(Xd, kd, ks, nr)`. The compressed
/// basis `x_d` lives on the bin centres, *not* on the full n observations
/// — that's the key memory and assembly-cost saving.
#[derive(Debug, Clone)]
pub struct DiscreteMarginal {
    /// Compressed basis values: `m × p` (m = number of unique bins,
    /// p = basis size of this term).
    pub x_d: Array2<f64>,
    /// Per-row bin index, 0-based: `indices[i] ∈ {0, ..., m-1}` for
    /// every original observation `i`. Length `n`.
    pub indices: Vec<u32>,
    /// Number of unique bins (= `x_d.nrows()`).
    pub nr: usize,
    /// Column range in the global `β`: `[col_offset, col_offset + num_basis)`.
    pub col_offset: usize,
    /// Number of basis columns `p` (= `x_d.ncols()`).
    pub num_basis: usize,
}

impl DiscreteMarginal {
    #[inline]
    pub fn num_compressed(&self) -> usize {
        self.nr
    }

    #[inline]
    pub fn num_observations(&self) -> usize {
        self.indices.len()
    }
}

/// Full discretized design — one `DiscreteMarginal` per smooth term, one
/// per parametric column, plus an intercept marginal at index 0 when
/// `has_intercept=true`. Mirrors mgcv's `Xd` list.
#[derive(Debug, Clone)]
pub struct DiscreteDesign {
    /// Marginals in column order: intercept (if any), then one per
    /// smooth term, ordered to match the global `β` layout.
    pub marginals: Vec<DiscreteMarginal>,
    /// Total number of basis columns `p` (= Σ marginal.num_basis).
    pub total_basis: usize,
    /// Number of original observations.
    pub n: usize,
}

/// Compress a 1-D covariate column using mgcv's `compress.df` strategy.
///
/// Returns `(per_row_bin_index, bin_centres)`:
///   - `per_row_bin_index[i] ∈ {0, ..., m-1}` for every original row `i`.
///   - `bin_centres` is length `m`. For pure-dedup, these are the unique
///     observed values (in stable first-occurrence order). For
///     quantile-grid, these are the equal-width grid midpoints to which
///     observations were snapped.
///
/// Regime: pure-dedup when `nunique(x_col) ≤ max_bins`; quantile-grid
/// (`kx = round((x - xmin) / dx)`, `dx = range / max_bins`) otherwise.
pub fn compress_1d(x_col: &Array1<f64>, max_bins: usize) -> (Vec<u32>, Vec<f64>) {
    let n = x_col.len();
    assert!(max_bins >= 1, "compress_1d: max_bins must be ≥ 1");

    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Pass 1: count unique values via the bit-pattern hash, stable
    // first-occurrence order.
    let mut value_to_bin: HashMap<u64, u32> = HashMap::with_capacity(n.min(max_bins * 2));
    let mut centres: Vec<f64> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(n);

    for i in 0..n {
        let v = x_col[i];
        let key = f64_bit_key(v);
        match value_to_bin.get(&key) {
            Some(&bin) => indices.push(bin),
            None => {
                let bin = centres.len() as u32;
                value_to_bin.insert(key, bin);
                centres.push(v);
                indices.push(bin);
            }
        }
    }

    if centres.len() <= max_bins {
        // Regime 1: pure dedup. Zero approximation error.
        return (indices, centres);
    }

    // Regime 2: quantile-grid binning. Equal-width snap into max_bins
    // grid cells. mgcv (compress.df, R/bam.r):
    //   xl <- range(x); dx <- diff(xl) / m
    //   kx <- round((x - xl[1]) / dx) + 1
    // We use 0-based; otherwise identical. The resulting kx ∈ {0..m}
    // (inclusive — `m+1` cells), so we have at most `m+1` distinct
    // bin indices. mgcv lives with that off-by-one; so do we.
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for &v in x_col.iter() {
        if v < x_min {
            x_min = v;
        }
        if v > x_max {
            x_max = v;
        }
    }
    let range = x_max - x_min;
    if range <= 0.0 || !range.is_finite() {
        // All values equal (after dedup we wouldn't be here, but be
        // defensive). Fall back to a single bin.
        return (vec![0u32; n], vec![x_min]);
    }
    let dx = range / max_bins as f64;

    // Snap each observation, then dedup the snapped values exactly.
    let mut snap_to_bin: HashMap<u32, u32> = HashMap::new();
    let mut snap_centres: Vec<f64> = Vec::new();
    let mut snap_indices: Vec<u32> = Vec::with_capacity(n);
    for &v in x_col.iter() {
        // `kx = round((v - xmin) / dx)` — kx ∈ {0..max_bins} inclusive.
        let kx_f = ((v - x_min) / dx).round();
        // Clamp to the valid range for safety against floating-point
        // boundary slop. kx ∈ {0..max_bins} is the natural mgcv range.
        let kx = kx_f
            .max(0.0)
            .min(max_bins as f64)
            .min(u32::MAX as f64) as u32;
        match snap_to_bin.get(&kx) {
            Some(&bin) => snap_indices.push(bin),
            None => {
                let bin = snap_centres.len() as u32;
                snap_to_bin.insert(kx, bin);
                let centre = x_min + (kx as f64) * dx;
                snap_centres.push(centre);
                snap_indices.push(bin);
            }
        }
    }

    (snap_indices, snap_centres)
}

/// Stable bit-pattern key for hashing f64s. NaN keys are mapped to the
/// canonical quiet-NaN bit pattern so all NaNs collide; -0.0 and +0.0
/// map to the same key.
#[inline]
fn f64_bit_key(v: f64) -> u64 {
    if v.is_nan() {
        return u64::MAX;
    }
    if v == 0.0 {
        // Both +0.0 and -0.0 → key 0.
        return 0;
    }
    v.to_bits()
}

impl DiscreteDesign {
    /// Build a `DiscreteDesign` from the list of smooth terms and the
    /// raw covariate matrix.
    ///
    /// Layout (matching `gam_optimized::FitCache::new`):
    ///   1. Intercept marginal at index 0 (single 1.0 bin, indices all
    ///      zero) when `has_intercept=true`.
    ///   2. One marginal per smooth term, in input order. For each
    ///      smooth we compress its covariate column via `compress_1d`
    ///      and evaluate the basis on the **bin centres** (not the
    ///      full n original observations) — that's the cheap path.
    ///
    /// Parametric terms (which carry `is_random_effect=true` per the
    /// 0.14.0 convention in `SmoothTerm::parametric`) flow through the
    /// same compress-then-evaluate pipeline. For a 0/1 indicator this
    /// naturally produces a 2-bin marginal; for a level-encoded
    /// integer column it produces one bin per unique level.
    ///
    /// Constraint matrices (sum-to-zero `Z`, pc-anchoring) are honoured
    /// because we go through `SmoothTerm::evaluate`, which applies the
    /// stored `constraint_matrix` after the raw basis eval.
    pub fn new(
        smooth_terms: &[SmoothTerm],
        x: &Array2<f64>,
        has_intercept: bool,
        config: &DiscreteConfig,
    ) -> Self {
        let n = x.nrows();
        let mut marginals: Vec<DiscreteMarginal> = Vec::with_capacity(smooth_terms.len() + 1);
        let mut total_basis: usize = 0;

        if has_intercept {
            marginals.push(DiscreteMarginal {
                x_d: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                indices: vec![0u32; n],
                nr: 1,
                col_offset: 0,
                num_basis: 1,
            });
            total_basis += 1;
        }

        for (i, smooth) in smooth_terms.iter().enumerate() {
            let x_col = x.column(i).to_owned();
            let (indices, centres) = compress_1d(&x_col, config.max_bins_1d);
            let centres_arr = Array1::from(centres);

            // Evaluate the basis on the *bin centres* — this is the
            // m × p compressed basis. SmoothTerm::evaluate applies the
            // constraint matrix (Z or pc-anchor) internally.
            let x_d = smooth
                .evaluate(&centres_arr)
                .expect("DiscreteDesign::new: basis evaluation on bin centres failed");
            let num_basis = x_d.ncols();
            let nr = x_d.nrows();

            marginals.push(DiscreteMarginal {
                x_d,
                indices,
                nr,
                col_offset: total_basis,
                num_basis,
            });
            total_basis += num_basis;
        }

        DiscreteDesign {
            marginals,
            total_basis,
            n,
        }
    }

    /// Materialise the full `n × p` design matrix from the compressed
    /// marginals. **Only for parity / debugging** — defeats the whole
    /// point of binning if called in the hot path.
    pub fn to_full_matrix(&self) -> Array2<f64> {
        let n = self.n;
        let p = self.total_basis;
        let mut full = Array2::zeros((n, p));
        for marg in &self.marginals {
            let off = marg.col_offset;
            let p_j = marg.num_basis;
            for i in 0..n {
                let bin = marg.indices[i] as usize;
                for c in 0..p_j {
                    full[[i, off + c]] = marg.x_d[[bin, c]];
                }
            }
        }
        full
    }
}

/// Compressed `X'WX` assembly (mgcv's `XWXd`).
///
/// Returns the same `p × p` matrix as
/// `reml::compute_xtwx(disc.to_full_matrix(), w)`, matching to 1e-12 on
/// pure-dedup fixtures (no approximation at this layer).
///
/// Two block patterns (see §1.2 of the design doc):
///
///   - **Diagonal block** (a == b): `t[μ] = Σ_{i: k_a[i]=μ} w_i`;
///     result = `X̃_a' · diag(t) · X̃_a` via a single GEMM with `t`
///     applied as a row scale. Cost: `O(n + m_a · p_a²)`.
///   - **Off-diagonal** (a != b): build `T[μ, ν] = Σ_i w_i ·
///     1[k_a[i]=μ ∧ k_b[i]=ν]`; result = `X̃_a' · T · X̃_b`.
///     T is kept dense; for our small m (≤ 1000) this is fine.
///     Cost: `O(n + m_a · m_b · (p_a + p_b))`.
///
/// The full assembly fills both upper and lower triangles (symmetric
/// matrix; matches the canonical `compute_xtwx` output).
pub fn compute_xtwx_discrete(disc: &DiscreteDesign, w: &Array1<f64>) -> Array2<f64> {
    let p = disc.total_basis;
    let n = disc.n;
    assert_eq!(w.len(), n, "compute_xtwx_discrete: w length mismatch");

    let mut xtwx = Array2::<f64>::zeros((p, p));
    let num_marg = disc.marginals.len();

    for a in 0..num_marg {
        let m_a = &disc.marginals[a];
        let off_a = m_a.col_offset;
        let p_a = m_a.num_basis;
        let nr_a = m_a.nr;

        // Diagonal block — row-scaled GEMM.
        // t[μ] = Σ_{i: k_a[i]=μ} w_i
        let mut t = vec![0.0f64; nr_a];
        for i in 0..n {
            let mu = m_a.indices[i] as usize;
            t[mu] += w[i];
        }
        // Form X̃_a' · diag(t) · X̃_a.
        // Implementation: scale a copy of X̃_a row-wise by t, then
        // multiply X̃_a' against it. ndarray .dot uses BLAS GEMM under
        // the `blas` feature.
        let mut scaled = m_a.x_d.clone();
        for mu in 0..nr_a {
            let tmu = t[mu];
            if tmu == 0.0 {
                // Zero out the row to be explicit (clone might keep
                // values that would otherwise contribute zero anyway,
                // but make the contract obvious).
                for c in 0..p_a {
                    scaled[[mu, c]] = 0.0;
                }
            } else {
                for c in 0..p_a {
                    scaled[[mu, c]] *= tmu;
                }
            }
        }
        let block_aa = m_a.x_d.t().dot(&scaled);
        // Write into xtwx[off_a..off_a+p_a, off_a..off_a+p_a].
        for r in 0..p_a {
            for c in 0..p_a {
                xtwx[[off_a + r, off_a + c]] = block_aa[[r, c]];
            }
        }

        for b in (a + 1)..num_marg {
            let m_b = &disc.marginals[b];
            let off_b = m_b.col_offset;
            let p_b = m_b.num_basis;
            let nr_b = m_b.nr;

            // Build T[μ, ν] = Σ_i w_i · 1[k_a[i]=μ ∧ k_b[i]=ν].
            let mut t_mat = Array2::<f64>::zeros((nr_a, nr_b));
            for i in 0..n {
                let mu = m_a.indices[i] as usize;
                let nu = m_b.indices[i] as usize;
                t_mat[[mu, nu]] += w[i];
            }
            // block_ab = X̃_a' · T · X̃_b
            // Compute as (X̃_a' · T) first (p_a × nr_b), then · X̃_b.
            let xa_t_dot_t = m_a.x_d.t().dot(&t_mat);
            let block_ab = xa_t_dot_t.dot(&m_b.x_d);

            // Write upper block and mirror to lower.
            for r in 0..p_a {
                for c in 0..p_b {
                    let val = block_ab[[r, c]];
                    xtwx[[off_a + r, off_b + c]] = val;
                    xtwx[[off_b + c, off_a + r]] = val;
                }
            }
        }
    }

    xtwx
}

/// Compressed `X'Wy` assembly (mgcv's `XWyd`).
///
/// Returns the same length-`p` vector as
/// `reml::compute_xtwy(disc.to_full_matrix(), w, y)`. For each marginal:
///   `t[μ] = Σ_{i: k[i]=μ} w_i · y_i`,
///   `(X'Wy)_j = X̃_j' · t`.
/// Cost: `O(n + Σ_j m_j · p_j)`.
pub fn compute_xtwy_discrete(
    disc: &DiscreteDesign,
    w: &Array1<f64>,
    y: &Array1<f64>,
) -> Array1<f64> {
    let p = disc.total_basis;
    let n = disc.n;
    assert_eq!(w.len(), n, "compute_xtwy_discrete: w length mismatch");
    assert_eq!(y.len(), n, "compute_xtwy_discrete: y length mismatch");

    let mut xtwy = Array1::<f64>::zeros(p);
    for marg in &disc.marginals {
        let off = marg.col_offset;
        let nr = marg.nr;
        let p_j = marg.num_basis;

        // Scatter: t[μ] = Σ_{i: k[i]=μ} w_i · y_i
        let mut t = Array1::<f64>::zeros(nr);
        for i in 0..n {
            let mu = marg.indices[i] as usize;
            t[mu] += w[i] * y[i];
        }
        // Gather: contribution = X̃_j' · t  (length p_j).
        let contrib = marg.x_d.t().dot(&t);
        for c in 0..p_j {
            xtwy[off + c] = contrib[c];
        }
    }
    xtwy
}

/// Compressed `η = X β` (mgcv's `Xbd`).
///
/// For each marginal:
///   `ξ_j[μ] = X̃_j[μ, :] · β_j` (one m_j-long vector per marginal),
///   `η[i]  += ξ_j[k_j[i]]` (gather).
/// Cost: `O(Σ_j m_j · p_j + n · num_marginals)`.
pub fn compute_eta_discrete(disc: &DiscreteDesign, beta: &Array1<f64>) -> Array1<f64> {
    let n = disc.n;
    assert_eq!(
        beta.len(),
        disc.total_basis,
        "compute_eta_discrete: beta length mismatch (expected {}, got {})",
        disc.total_basis,
        beta.len()
    );

    let mut eta = Array1::<f64>::zeros(n);
    for marg in &disc.marginals {
        let off = marg.col_offset;
        let p_j = marg.num_basis;
        let beta_j = beta.slice(s![off..off + p_j]);
        // ξ_j = X̃_j · β_j  (length nr).
        let xi = marg.x_d.dot(&beta_j);
        // Scatter into eta.
        for i in 0..n {
            let mu = marg.indices[i] as usize;
            eta[i] += xi[mu];
        }
    }
    eta
}

// ---------------------------------------------------------------------------
// Back-compat stubs for the pre-D1-D3 callers in `gam_optimized.rs` and
// `pirls.rs`. The old `DiscretizedDesign` / `DiscretizeConfig` /
// `CompressedBasis` types are gone; D4 will rewire the callers properly.
// Until then we expose minimal stubs so the crate compiles. Every call
// site uses `todo!()` so any accidental invocation panics loudly.
//
// TODO(d4): delete the stubs and rewire the REML/PIRLS hot paths to use
// the new `compute_xtwx_discrete` / `compute_xtwy_discrete` /
// `compute_eta_discrete` free functions directly.
// ---------------------------------------------------------------------------

/// **Removed** — use `DiscreteConfig` instead. Kept as a type alias so
/// the deprecated callers still parse until D4 rewires them.
pub type DiscretizeConfig = DiscreteConfig;

impl DiscreteConfig {
    /// Compatibility constructor — accepts the old `max_unique_1d` /
    /// `min_n_for_discretize` field names so the existing
    /// `gam_optimized.rs` literal still parses. The
    /// `min_n_for_discretize` field is dropped (the caller in
    /// `gam_optimized.rs` already gates on `n >= 2000` before
    /// constructing the design, so the field never had teeth).
    #[doc(hidden)]
    #[allow(non_snake_case)]
    pub fn _from_legacy_fields(max_unique_1d: usize, _min_n_for_discretize: usize) -> Self {
        DiscreteConfig {
            max_bins_1d: max_unique_1d,
        }
    }
}

/// **Removed**. Stub alias so the (now-broken) call sites still typecheck.
/// Every method panics — D4 rewires them to use `DiscreteDesign` directly.
pub struct DiscretizedDesign {
    pub terms: Vec<CompressedBasisStub>,
    pub total_basis: usize,
    pub n: usize,
}

/// Minimal stub matching the field-access surface the old callers
/// touched (`.num_compressed()`, `.num_observations()`, `.num_basis`,
/// `.compression_ratio()`). All other methods panic.
pub struct CompressedBasisStub {
    pub num_basis: usize,
}

impl CompressedBasisStub {
    pub fn num_compressed(&self) -> usize {
        unreachable!("CompressedBasisStub: D4 not yet rewired")
    }
    pub fn num_observations(&self) -> usize {
        unreachable!("CompressedBasisStub: D4 not yet rewired")
    }
    pub fn compression_ratio(&self) -> f64 {
        unreachable!("CompressedBasisStub: D4 not yet rewired")
    }
}

impl DiscretizedDesign {
    /// **Stub** — panics. D4 rewires the construction site in
    /// `FitCache::new`. Until then this exists only so the crate parses.
    pub fn new(
        _basis_matrices: &[Array2<f64>],
        _covariates: &[Array1<f64>],
        _config: &DiscreteConfig,
        _has_intercept: bool,
    ) -> Self {
        // TODO(d4): rewire FitCache::new to call DiscreteDesign::new instead.
        todo!("DiscretizedDesign::new is deprecated — D4 rewires to DiscreteDesign::new")
    }

    pub fn compute_xtwx(&self, _w: &Array1<f64>) -> Array2<f64> {
        todo!("DiscretizedDesign::compute_xtwx — D4 dispatches to compute_xtwx_discrete")
    }
    pub fn compute_xtwy(&self, _w: &Array1<f64>, _y: &Array1<f64>) -> Array1<f64> {
        todo!("DiscretizedDesign::compute_xtwy — D4 dispatches to compute_xtwy_discrete")
    }
    pub fn compute_xtwz(&self, _wz: &Array1<f64>) -> Array1<f64> {
        todo!("DiscretizedDesign::compute_xtwz — D4 dispatches to compute_xtwy_discrete (pre-weighted)")
    }
    pub fn compute_xtx(&self) -> Array2<f64> {
        todo!("DiscretizedDesign::compute_xtx — D4 dispatches to compute_xtwx_discrete with w=1")
    }
    pub fn compute_eta(&self, _beta: &Array1<f64>) -> Array1<f64> {
        todo!("DiscretizedDesign::compute_eta — D4 dispatches to compute_eta_discrete")
    }
    pub fn to_full_matrix(&self) -> Array2<f64> {
        todo!("DiscretizedDesign::to_full_matrix — D4 uses DiscreteDesign::to_full_matrix")
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    fn approx_max_abs(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut m = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            m = m.max((x - y).abs());
        }
        m
    }

    fn approx_max_abs_1d(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let mut m = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            m = m.max((x - y).abs());
        }
        m
    }

    // ------------------ D1: compress_1d ------------------

    #[test]
    fn compress_1d_pure_dedup() {
        // 5 unique values repeated; expect 5 bins, indices map back exactly.
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 5.0, 3.0, 2.0, 4.0]);
        let (idx, centres) = compress_1d(&x, 1000);

        assert_eq!(centres.len(), 5, "expected 5 unique bins");
        assert_eq!(idx.len(), x.len());

        // Each row's x value must equal the centre at its bin.
        for i in 0..x.len() {
            let mu = idx[i] as usize;
            assert_abs_diff_eq!(x[i], centres[mu], epsilon = 0.0);
        }
        // Identical values must share a bin.
        assert_eq!(idx[0], idx[5]); // both 1.0
        assert_eq!(idx[4], idx[6]); // both 5.0
        assert_eq!(idx[2], idx[7]); // both 3.0
    }

    #[test]
    fn compress_1d_quantile_grid() {
        // 2000 unique values in [0,1]; max_bins = 10 forces quantile-grid.
        let n = 2000usize;
        let x: Array1<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let (idx, centres) = compress_1d(&x, 10);

        assert!(
            centres.len() <= 11,
            "quantile grid should give ≤ max_bins+1 (={}) bins, got {}",
            11,
            centres.len()
        );
        // Every index must point at a valid centre.
        for &k in &idx {
            assert!((k as usize) < centres.len());
        }
        // Worst-case error bound: |x_orig - x_binned| ≤ dx/2 = range / (2m) = 0.05.
        for i in 0..n {
            let mu = idx[i] as usize;
            assert!(
                (x[i] - centres[mu]).abs() <= 0.06, // small slop for the 0.05 bound
                "row {} snapped to centre {}, |Δ|={} exceeds 0.06",
                i,
                centres[mu],
                (x[i] - centres[mu]).abs()
            );
        }
    }

    #[test]
    fn compress_1d_constant_column() {
        // All values equal — pure-dedup gives 1 bin.
        let x = Array1::from(vec![3.0; 100]);
        let (idx, centres) = compress_1d(&x, 1000);
        assert_eq!(centres.len(), 1);
        assert_eq!(centres[0], 3.0);
        assert!(idx.iter().all(|&k| k == 0));
    }

    #[test]
    fn compress_1d_empty() {
        let x: Array1<f64> = Array1::from(vec![]);
        let (idx, centres) = compress_1d(&x, 1000);
        assert!(idx.is_empty());
        assert!(centres.is_empty());
    }

    // ------------------ D3: compute_xtwx_discrete (single marginal) ------------------

    #[test]
    fn compute_xtwx_discrete_diagonal_block_pure_dedup() {
        // Single smooth, pure-dedup. Result must match the un-binned
        // X'WX to 1e-12 (no approximation).
        let n = 50usize;
        let nr = 5usize;
        let p = 3usize;
        // Per-bin basis values (small enough to inspect).
        let x_d = Array2::from_shape_vec(
            (nr, p),
            vec![
                1.0, 0.0, 0.0, 1.0, 0.5, 0.25, 1.0, 1.0, 1.0, 1.0, 1.5, 2.25, 1.0, 2.0, 4.0,
            ],
        )
        .unwrap();
        // Indices cycle through bins.
        let indices: Vec<u32> = (0..n).map(|i| (i % nr) as u32).collect();
        let marg = DiscreteMarginal {
            x_d: x_d.clone(),
            indices: indices.clone(),
            nr,
            col_offset: 0,
            num_basis: p,
        };
        let disc = DiscreteDesign {
            marginals: vec![marg],
            total_basis: p,
            n,
        };
        // Reconstruct full X[i, :] = x_d[indices[i], :].
        let full = disc.to_full_matrix();
        // Weight vector with structure.
        let w: Array1<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.07).sin().abs()).collect();
        let xtwx_full = crate::reml::compute_xtwx(&full, &w);
        let xtwx_disc = compute_xtwx_discrete(&disc, &w);
        let err = approx_max_abs(&xtwx_full, &xtwx_disc);
        assert!(
            err < 1e-12,
            "diagonal-block X'WX mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    #[test]
    fn compute_xtwx_discrete_off_diagonal_two_marginals() {
        // Two marginals with independent bin indices. Reconstruct the
        // full design and compare X'WX block-by-block.
        let n = 80usize;
        // Marginal A: nr=4, p=2.
        let nr_a = 4usize;
        let p_a = 2usize;
        let x_d_a = Array2::from_shape_vec(
            (nr_a, p_a),
            vec![1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.5],
        )
        .unwrap();
        let idx_a: Vec<u32> = (0..n).map(|i| (i % nr_a) as u32).collect();
        // Marginal B: nr=3, p=3.
        let nr_b = 3usize;
        let p_b = 3usize;
        let x_d_b = Array2::from_shape_vec(
            (nr_b, p_b),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let idx_b: Vec<u32> = (0..n).map(|i| ((i / nr_a) % nr_b) as u32).collect();

        let disc = DiscreteDesign {
            marginals: vec![
                DiscreteMarginal {
                    x_d: x_d_a.clone(),
                    indices: idx_a.clone(),
                    nr: nr_a,
                    col_offset: 0,
                    num_basis: p_a,
                },
                DiscreteMarginal {
                    x_d: x_d_b.clone(),
                    indices: idx_b.clone(),
                    nr: nr_b,
                    col_offset: p_a,
                    num_basis: p_b,
                },
            ],
            total_basis: p_a + p_b,
            n,
        };

        let full = disc.to_full_matrix();
        let w: Array1<f64> = (0..n).map(|i| 0.5 + 0.5 * ((i as f64) * 0.13).cos().abs()).collect();
        let xtwx_full = crate::reml::compute_xtwx(&full, &w);
        let xtwx_disc = compute_xtwx_discrete(&disc, &w);
        let err = approx_max_abs(&xtwx_full, &xtwx_disc);
        assert!(
            err < 1e-12,
            "off-diagonal X'WX mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    #[test]
    fn compute_xtwx_discrete_intercept_plus_smooth() {
        // Mimic the production layout: an intercept marginal at index 0
        // (m=1, x_d=[[1.0]], indices all zero) plus one "smooth" marginal.
        let n = 60usize;
        let nr_s = 6usize;
        let p_s = 4usize;
        let mut x_d_s = Array2::<f64>::zeros((nr_s, p_s));
        for mu in 0..nr_s {
            for c in 0..p_s {
                x_d_s[[mu, c]] = ((mu as f64 + 1.0) * (c as f64 + 1.0) * 0.1).sin();
            }
        }
        let idx_s: Vec<u32> = (0..n).map(|i| (i % nr_s) as u32).collect();

        let disc = DiscreteDesign {
            marginals: vec![
                DiscreteMarginal {
                    x_d: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                    indices: vec![0u32; n],
                    nr: 1,
                    col_offset: 0,
                    num_basis: 1,
                },
                DiscreteMarginal {
                    x_d: x_d_s.clone(),
                    indices: idx_s,
                    nr: nr_s,
                    col_offset: 1,
                    num_basis: p_s,
                },
            ],
            total_basis: 1 + p_s,
            n,
        };
        let full = disc.to_full_matrix();
        // Intercept column must be ones.
        for i in 0..n {
            assert_abs_diff_eq!(full[[i, 0]], 1.0, epsilon = 0.0);
        }
        let w: Array1<f64> = (0..n)
            .map(|i| 0.7 + 0.3 * ((i as f64) * 0.21).sin().abs())
            .collect();
        let xtwx_full = crate::reml::compute_xtwx(&full, &w);
        let xtwx_disc = compute_xtwx_discrete(&disc, &w);
        let err = approx_max_abs(&xtwx_full, &xtwx_disc);
        assert!(
            err < 1e-12,
            "intercept+smooth X'WX mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    // ------------------ D3: compute_xtwy_discrete ------------------

    #[test]
    fn compute_xtwy_discrete_single_marginal() {
        let n = 100usize;
        let nr = 8usize;
        let p = 3usize;
        let mut x_d = Array2::<f64>::zeros((nr, p));
        for mu in 0..nr {
            for c in 0..p {
                x_d[[mu, c]] = ((mu as f64) + 0.1 * (c as f64 + 1.0)).cos();
            }
        }
        let indices: Vec<u32> = (0..n).map(|i| (i % nr) as u32).collect();
        let disc = DiscreteDesign {
            marginals: vec![DiscreteMarginal {
                x_d,
                indices,
                nr,
                col_offset: 0,
                num_basis: p,
            }],
            total_basis: p,
            n,
        };
        let full = disc.to_full_matrix();
        let w: Array1<f64> = (0..n).map(|i| 1.0 + 0.5 * ((i as f64) * 0.03).sin()).collect();
        let y: Array1<f64> = (0..n).map(|i| (i as f64) * 0.17 - 3.0).collect();

        let xtwy_full = crate::reml::compute_xtwy(&full, &w, &y);
        let xtwy_disc = compute_xtwy_discrete(&disc, &w, &y);
        let err = approx_max_abs_1d(&xtwy_full, &xtwy_disc);
        assert!(
            err < 1e-12,
            "X'Wy mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    #[test]
    fn compute_xtwy_discrete_two_marginals_plus_intercept() {
        // Three marginals (intercept + two "smooths") on n=120 rows.
        let n = 120usize;
        // Smooth A: nr=5, p=2.
        let nr_a = 5usize;
        let p_a = 2usize;
        let x_d_a = Array2::from_shape_vec(
            (nr_a, p_a),
            vec![1.0, 0.0, 1.0, 0.25, 1.0, 0.5, 1.0, 0.75, 1.0, 1.0],
        )
        .unwrap();
        let idx_a: Vec<u32> = (0..n).map(|i| (i % nr_a) as u32).collect();
        // Smooth B: nr=4, p=3.
        let nr_b = 4usize;
        let p_b = 3usize;
        let mut x_d_b = Array2::<f64>::zeros((nr_b, p_b));
        for mu in 0..nr_b {
            for c in 0..p_b {
                x_d_b[[mu, c]] = ((mu as f64 + 1.0) * (c as f64 + 2.0) * 0.07).sin();
            }
        }
        let idx_b: Vec<u32> = (0..n).map(|i| ((i / nr_a) % nr_b) as u32).collect();

        let disc = DiscreteDesign {
            marginals: vec![
                DiscreteMarginal {
                    x_d: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                    indices: vec![0u32; n],
                    nr: 1,
                    col_offset: 0,
                    num_basis: 1,
                },
                DiscreteMarginal {
                    x_d: x_d_a,
                    indices: idx_a,
                    nr: nr_a,
                    col_offset: 1,
                    num_basis: p_a,
                },
                DiscreteMarginal {
                    x_d: x_d_b,
                    indices: idx_b,
                    nr: nr_b,
                    col_offset: 1 + p_a,
                    num_basis: p_b,
                },
            ],
            total_basis: 1 + p_a + p_b,
            n,
        };
        let full = disc.to_full_matrix();
        let w: Array1<f64> = (0..n).map(|i| 0.4 + 0.6 * ((i as f64) * 0.05).cos().abs()).collect();
        let y: Array1<f64> = (0..n).map(|i| ((i as f64) * 0.11).sin()).collect();
        let xtwy_full = crate::reml::compute_xtwy(&full, &w, &y);
        let xtwy_disc = compute_xtwy_discrete(&disc, &w, &y);
        let err = approx_max_abs_1d(&xtwy_full, &xtwy_disc);
        assert!(
            err < 1e-12,
            "X'Wy multi-marginal mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    // ------------------ D3: compute_eta_discrete ------------------

    #[test]
    fn compute_eta_discrete_matches_full_dot() {
        let n = 90usize;
        let nr_a = 6usize;
        let p_a = 3usize;
        let mut x_d_a = Array2::<f64>::zeros((nr_a, p_a));
        for mu in 0..nr_a {
            for c in 0..p_a {
                x_d_a[[mu, c]] = ((mu as f64 - 2.5) * (c as f64 + 1.0) * 0.2).cos();
            }
        }
        let idx_a: Vec<u32> = (0..n).map(|i| (i % nr_a) as u32).collect();
        let disc = DiscreteDesign {
            marginals: vec![
                DiscreteMarginal {
                    x_d: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
                    indices: vec![0u32; n],
                    nr: 1,
                    col_offset: 0,
                    num_basis: 1,
                },
                DiscreteMarginal {
                    x_d: x_d_a,
                    indices: idx_a,
                    nr: nr_a,
                    col_offset: 1,
                    num_basis: p_a,
                },
            ],
            total_basis: 1 + p_a,
            n,
        };
        let beta = array![0.5, -1.0, 2.0, 0.75];
        let full = disc.to_full_matrix();
        let eta_full = full.dot(&beta);
        let eta_disc = compute_eta_discrete(&disc, &beta);
        let err = approx_max_abs_1d(&eta_full, &eta_disc);
        assert!(
            err < 1e-12,
            "η mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }

    // ------------------ D2: DiscreteDesign::new on real basis ------------------

    #[test]
    fn discrete_design_new_roundtrips_via_smooth_term() {
        // Build a tiny SmoothTerm, run DiscreteDesign::new on a covariate
        // with duplicates, and check that the resulting design matrix
        // matches `smooth.evaluate(x_col)` row-for-row.
        use crate::gam::SmoothTerm;
        let x_col = Array1::from(vec![
            0.1, 0.5, 0.9, 0.1, 0.5, 0.5, 0.3, 0.7, 0.9, 0.3,
        ]);
        let n = x_col.len();
        let mut smooth = SmoothTerm::cr_spline_quantile(
            "x".to_string(),
            5, // k=5
            &x_col,
        )
        .unwrap();
        // Apply the sum-to-zero centring so we go through the same
        // constraint path as the real fit cache.
        smooth
            .apply_sum_to_zero_centering(&x_col)
            .expect("centering failed");

        // Full design on all n rows.
        let full_basis = smooth.evaluate(&x_col).unwrap();

        // Discretized version (no intercept here — we test only the smooth slab).
        let x_mat = {
            let mut m = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                m[[i, 0]] = x_col[i];
            }
            m
        };
        let config = DiscreteConfig::default();
        let num_basis_pre = smooth.num_basis();
        let smooth_list = vec![smooth];
        let disc = DiscreteDesign::new(&smooth_list, &x_mat, false, &config);

        // nr should be the number of unique x values (5: {0.1, 0.3, 0.5, 0.7, 0.9}).
        assert_eq!(disc.marginals.len(), 1);
        let m0 = &disc.marginals[0];
        assert_eq!(m0.nr, 5, "expected 5 unique x values, got nr={}", m0.nr);
        assert_eq!(m0.num_basis, num_basis_pre);

        // Reconstructed full design must match the direct evaluation.
        let full_disc = disc.to_full_matrix();
        for i in 0..n {
            for c in 0..num_basis_pre {
                assert_abs_diff_eq!(full_basis[[i, c]], full_disc[[i, c]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn discrete_design_xtwx_via_smooth_term_matches_blas() {
        // End-to-end: build a DiscreteDesign from a real SmoothTerm,
        // compute X'WX via the scatter-gather kernel, and assert 1e-12
        // agreement with the un-binned BLAS X'WX.
        use crate::gam::SmoothTerm;
        let n = 300usize;
        // Repeat a small set of unique x values to force pure-dedup.
        let unique_vals = [0.0_f64, 0.2, 0.4, 0.6, 0.8, 1.0];
        let x_col: Array1<f64> = (0..n)
            .map(|i| unique_vals[i % unique_vals.len()])
            .collect();
        let mut smooth = SmoothTerm::cr_spline_quantile("x".to_string(), 6, &x_col).unwrap();
        smooth.apply_sum_to_zero_centering(&x_col).unwrap();

        let mut x_mat = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_mat[[i, 0]] = x_col[i];
        }
        // Evaluate the full design BEFORE moving the smooth into the
        // design (SmoothTerm isn't Clone because of the Box<dyn>).
        let smooth_full = smooth.evaluate(&x_col).unwrap();
        let p_s = smooth_full.ncols();
        let config = DiscreteConfig::default();
        let smooth_list = vec![smooth];
        let disc = DiscreteDesign::new(&smooth_list, &x_mat, true, &config);
        let mut full = Array2::<f64>::zeros((n, 1 + p_s));
        for i in 0..n {
            full[[i, 0]] = 1.0;
            for c in 0..p_s {
                full[[i, 1 + c]] = smooth_full[[i, c]];
            }
        }
        let w: Array1<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * ((i as f64) * 0.011).sin())
            .collect();

        let xtwx_full = crate::reml::compute_xtwx(&full, &w);
        let xtwx_disc = compute_xtwx_discrete(&disc, &w);
        let err = approx_max_abs(&xtwx_full, &xtwx_disc);
        assert!(
            err < 1e-12,
            "end-to-end X'WX mismatch: max abs err = {} (expected < 1e-12)",
            err
        );
    }
}
