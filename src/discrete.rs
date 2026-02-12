//! Discretized (compressed) basis representation for efficient GAM fitting.
//!
//! This module implements the key optimization from Wood, Li, Shaddick & Augustin (2017):
//! instead of storing the full n x k basis matrix for each smooth term, we bin
//! covariate values into a smaller grid and store only the unique (or binned) rows.
//!
//! For a smooth term with n observations and k basis functions:
//! - Full storage: n x k matrix, O(n*k) memory
//! - Compressed: m x k matrix + n-length index array, O(m*k + n) memory
//!   where m << n (typically m ~ 200-1000)
//!
//! The compressed representation enables:
//! 1. Scatter-gather X'WX computation: O(n*k + m*k^2) instead of O(n*k^2)
//! 2. Efficient eta = X*beta: O(m*k + n) instead of O(n*k)
//! 3. Reduced memory footprint for large n

use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

/// A single smooth term's basis stored in compressed (discretized) form.
///
/// Instead of an n x k matrix, stores:
/// - `values`: m x k matrix of unique/binned basis rows (m << n)
/// - `indices`: n-length array mapping each observation to its compressed row
#[derive(Debug, Clone)]
pub struct CompressedBasis {
    /// Unique (or binned) basis rows: m x k matrix
    pub values: Array2<f64>,
    /// Mapping from observation index to compressed row: indices[i] gives the row
    /// in `values` corresponding to observation i
    pub indices: Vec<u32>,
    /// Column offset in the full p-column design matrix
    pub col_offset: usize,
    /// Number of basis functions (k) for this term
    pub num_basis: usize,
}

/// Configuration for discretization
#[derive(Debug, Clone, Copy)]
pub struct DiscretizeConfig {
    /// Maximum number of unique rows to keep per dimension
    /// For 1D: ~1000, 2D: ~100 per dim, 3D+: ~25 per dim
    pub max_unique_1d: usize,
    /// Minimum number of observations to trigger discretization
    /// Below this, full storage is used (no benefit to compression)
    pub min_n_for_discretize: usize,
}

impl Default for DiscretizeConfig {
    fn default() -> Self {
        DiscretizeConfig {
            max_unique_1d: 1000,
            min_n_for_discretize: 500,
        }
    }
}

impl CompressedBasis {
    /// Create a compressed basis from a full n x k basis matrix.
    ///
    /// Bins observations by rounding covariate values to a grid, then identifies
    /// unique rows in the basis matrix. Observations mapping to the same grid point
    /// share a single compressed row.
    ///
    /// # Arguments
    /// * `full_basis` - The n x k basis matrix evaluated at all observations
    /// * `covariate` - The raw covariate values (used for binning)
    /// * `col_offset` - Column offset in the full design matrix
    /// * `max_bins` - Maximum number of bins (compressed rows)
    pub fn from_basis_1d(
        full_basis: &Array2<f64>,
        covariate: &Array1<f64>,
        col_offset: usize,
        max_bins: usize,
    ) -> Self {
        let n = full_basis.nrows();
        let k = full_basis.ncols();

        // Determine bin edges based on covariate range
        let x_min = covariate.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = covariate.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = x_max - x_min;

        if range < 1e-15 || n <= max_bins {
            // All values are the same or n is small enough - no binning needed
            // But we still deduplicate identical rows
            return Self::from_basis_dedup(full_basis, col_offset);
        }

        // Number of bins = min(max_bins, number of unique values)
        let n_bins = max_bins.min(n);
        let bin_width = range / n_bins as f64;

        // Assign each observation to a bin
        let mut bin_indices: Vec<usize> = Vec::with_capacity(n);
        for i in 0..n {
            let bin = ((covariate[i] - x_min) / bin_width).floor() as usize;
            bin_indices.push(bin.min(n_bins - 1)); // clamp to valid range
        }

        // For each bin, compute the average basis row (or just use the first observation's row)
        // Using first observation per bin is simpler and what bam() does
        let mut bin_to_compressed: HashMap<usize, u32> = HashMap::new();
        let mut compressed_rows: Vec<Vec<f64>> = Vec::new();
        let mut indices: Vec<u32> = Vec::with_capacity(n);

        for i in 0..n {
            let bin = bin_indices[i];
            if let Some(&compressed_idx) = bin_to_compressed.get(&bin) {
                indices.push(compressed_idx);
            } else {
                let compressed_idx = compressed_rows.len() as u32;
                bin_to_compressed.insert(bin, compressed_idx);
                compressed_rows.push(full_basis.row(i).to_vec());
                indices.push(compressed_idx);
            }
        }

        let m = compressed_rows.len();
        let mut values = Array2::zeros((m, k));
        for (row_idx, row) in compressed_rows.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                values[[row_idx, col_idx]] = val;
            }
        }

        CompressedBasis {
            values,
            indices,
            col_offset,
            num_basis: k,
        }
    }

    /// Create a compressed basis by deduplicating identical rows.
    /// Used when n is small or all observations are unique.
    fn from_basis_dedup(full_basis: &Array2<f64>, col_offset: usize) -> Self {
        let n = full_basis.nrows();
        let k = full_basis.ncols();

        // Hash rows by quantizing to find duplicates
        let mut row_map: HashMap<Vec<i64>, u32> = HashMap::new();
        let mut compressed_rows: Vec<Vec<f64>> = Vec::new();
        let mut indices: Vec<u32> = Vec::with_capacity(n);

        for i in 0..n {
            // Quantize row values for hashing (8 decimal digits of precision)
            let key: Vec<i64> = (0..k)
                .map(|j| (full_basis[[i, j]] * 1e8).round() as i64)
                .collect();

            if let Some(&compressed_idx) = row_map.get(&key) {
                indices.push(compressed_idx);
            } else {
                let compressed_idx = compressed_rows.len() as u32;
                row_map.insert(key, compressed_idx);
                compressed_rows.push(full_basis.row(i).to_vec());
                indices.push(compressed_idx);
            }
        }

        let m = compressed_rows.len();
        let mut values = Array2::zeros((m, k));
        for (row_idx, row) in compressed_rows.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                values[[row_idx, col_idx]] = val;
            }
        }

        CompressedBasis {
            values,
            indices,
            col_offset,
            num_basis: k,
        }
    }

    /// Number of compressed (unique) rows
    #[inline]
    pub fn num_compressed(&self) -> usize {
        self.values.nrows()
    }

    /// Number of observations
    #[inline]
    pub fn num_observations(&self) -> usize {
        self.indices.len()
    }

    /// Compression ratio: n / m
    #[inline]
    pub fn compression_ratio(&self) -> f64 {
        self.num_observations() as f64 / self.num_compressed() as f64
    }

    /// Expand compressed row for a single observation (for debugging/verification)
    #[inline]
    pub fn get_row(&self, obs_idx: usize) -> ndarray::ArrayView1<f64> {
        let compressed_idx = self.indices[obs_idx] as usize;
        self.values.row(compressed_idx)
    }
}

/// Collection of compressed basis terms forming the full discretized design matrix.
///
/// This replaces the full n x p design matrix with a set of per-term compressed
/// representations, enabling scatter-gather computation of X'WX.
pub struct DiscretizedDesign {
    /// Compressed basis for each smooth term
    pub terms: Vec<CompressedBasis>,
    /// Total number of basis functions (p = sum of all terms' k)
    pub total_basis: usize,
    /// Number of observations
    pub n: usize,
}

impl DiscretizedDesign {
    /// Create a discretized design from full basis matrices and covariates.
    ///
    /// # Arguments
    /// * `basis_matrices` - Per-term basis matrices (each n x k_i)
    /// * `covariates` - Per-term covariate vectors
    /// * `config` - Discretization configuration
    pub fn new(
        basis_matrices: &[Array2<f64>],
        covariates: &[Array1<f64>],
        config: &DiscretizeConfig,
    ) -> Self {
        let n = basis_matrices[0].nrows();
        let mut terms = Vec::with_capacity(basis_matrices.len());
        let mut col_offset = 0;

        for (basis, covariate) in basis_matrices.iter().zip(covariates.iter()) {
            let k = basis.ncols();
            let max_bins = config.max_unique_1d;

            let compressed = if n >= config.min_n_for_discretize {
                CompressedBasis::from_basis_1d(basis, covariate, col_offset, max_bins)
            } else {
                CompressedBasis::from_basis_dedup(basis, col_offset)
            };

            terms.push(compressed);
            col_offset += k;
        }

        DiscretizedDesign {
            terms,
            total_basis: col_offset,
            n,
        }
    }

    /// Compute X'WX using the scatter-gather algorithm.
    ///
    /// This is the core optimization from bam()'s XWXd() function.
    /// Instead of the naive O(n*p^2) computation, we use:
    ///
    /// For each pair of term blocks (a, b):
    ///   For each column j of block b:
    ///     1. SCATTER: accumulate weighted basis values into compressed buckets
    ///        temp_a[idx_a[i]] += w[i] * X_b[idx_b[i], j]   for i in 0..n  -- O(n)
    ///     2. GATHER: compute block of X'WX via compressed matrix product
    ///        X'WX[a_cols, b_col_j] = X_a_compressed.t() @ temp_a           -- O(m_a * k_a)
    ///
    /// Total: O(n * p + sum_a(m_a * k_a * p))  instead of  O(n * p^2)
    /// For typical cases where m_a << n, this is much faster.
    pub fn compute_xtwx(&self, w: &Array1<f64>) -> Array2<f64> {
        let p = self.total_basis;
        let n = self.n;
        let mut xtwx = Array2::zeros((p, p));

        let num_terms = self.terms.len();

        for a in 0..num_terms {
            let term_a = &self.terms[a];
            let m_a = term_a.num_compressed();
            let k_a = term_a.num_basis;
            let off_a = term_a.col_offset;

            for b in a..num_terms {
                let term_b = &self.terms[b];
                let k_b = term_b.num_basis;
                let off_b = term_b.col_offset;

                // For each column j of block b, scatter weighted values into
                // term_a's compressed buckets, then gather via matrix product
                for j in 0..k_b {
                    // SCATTER: accumulate w[i] * X_b[idx_b[i], j] into buckets for term_a
                    let mut temp = vec![0.0f64; m_a];
                    for i in 0..n {
                        let idx_a = term_a.indices[i] as usize;
                        let idx_b = term_b.indices[i] as usize;
                        temp[idx_a] += w[i] * term_b.values[[idx_b, j]];
                    }

                    // GATHER: X'WX[a_cols, b_col_j] = X_a_compressed.t() @ temp
                    // This is a matrix-vector product: (k_a x m_a) @ (m_a) = (k_a)
                    for ia in 0..k_a {
                        let mut sum = 0.0f64;
                        for im in 0..m_a {
                            sum += term_a.values[[im, ia]] * temp[im];
                        }
                        xtwx[[off_a + ia, off_b + j]] = sum;

                        // Fill symmetric part
                        if a != b {
                            xtwx[[off_b + j, off_a + ia]] = sum;
                        }
                    }
                }

                // For the diagonal block (a == b), fill the lower triangle
                if a == b {
                    for ia in 0..k_a {
                        for ja in 0..ia {
                            xtwx[[off_a + ia, off_a + ja]] = xtwx[[off_a + ja, off_a + ia]];
                        }
                    }
                }
            }
        }

        xtwx
    }

    /// Compute X'Wy using scatter-gather.
    ///
    /// For each term a:
    ///   1. SCATTER: temp_a[idx_a[i]] += w[i] * y[i]   for i in 0..n  -- O(n)
    ///   2. GATHER: X'Wy[a_cols] = X_a_compressed.t() @ temp_a         -- O(m_a * k_a)
    pub fn compute_xtwy(&self, w: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        let p = self.total_basis;
        let n = self.n;
        let mut xtwy = Array1::zeros(p);

        for term in &self.terms {
            let m = term.num_compressed();
            let k = term.num_basis;
            let off = term.col_offset;

            // SCATTER: accumulate w[i] * y[i] into compressed buckets
            let mut temp = vec![0.0f64; m];
            for i in 0..n {
                let idx = term.indices[i] as usize;
                temp[idx] += w[i] * y[i];
            }

            // GATHER: X'Wy[a_cols] = X_compressed.t() @ temp
            for j in 0..k {
                let mut sum = 0.0f64;
                for im in 0..m {
                    sum += term.values[[im, j]] * temp[im];
                }
                xtwy[off + j] = sum;
            }
        }

        xtwy
    }

    /// Compute eta = X * beta efficiently using compressed storage.
    ///
    /// Instead of materializing the full n x p design matrix:
    ///   For each term j:
    ///     eta_j_compressed = X_j_compressed @ beta_j    (m_j x k_j times k_j = m_j vector)
    ///     eta[i] += eta_j_compressed[idx_j[i]]          (gather via index)
    ///
    /// Total: O(sum(m_j * k_j) + n * d) instead of O(n * p)
    pub fn compute_eta(&self, beta: &Array1<f64>) -> Array1<f64> {
        let n = self.n;
        let mut eta = Array1::zeros(n);

        for term in &self.terms {
            let k = term.num_basis;
            let off = term.col_offset;

            // Extract this term's coefficients
            let beta_j = beta.slice(s![off..off + k]);

            // Compute compressed eta: m x k times k = m vector
            let eta_compressed = term.values.dot(&beta_j);

            // GATHER: scatter compressed eta to full observations
            for i in 0..n {
                let idx = term.indices[i] as usize;
                eta[i] += eta_compressed[idx];
            }
        }

        eta
    }

    /// Compute X'Wz for PiRLS where z = w * y (working response already weighted).
    /// Same as compute_xtwy but with pre-weighted values.
    pub fn compute_xtwz(&self, wz: &Array1<f64>) -> Array1<f64> {
        let p = self.total_basis;
        let n = self.n;
        let mut xtwz = Array1::zeros(p);

        for term in &self.terms {
            let m = term.num_compressed();
            let k = term.num_basis;
            let off = term.col_offset;

            // SCATTER: accumulate wz[i] into compressed buckets
            let mut temp = vec![0.0f64; m];
            for i in 0..n {
                let idx = term.indices[i] as usize;
                temp[idx] += wz[i];
            }

            // GATHER
            for j in 0..k {
                let mut sum = 0.0f64;
                for im in 0..m {
                    sum += term.values[[im, j]] * temp[im];
                }
                xtwz[off + j] = sum;
            }
        }

        xtwz
    }

    /// Materialize the full n x p design matrix (for fallback/verification).
    ///
    /// This is O(n*p) and should only be used for debugging or when the
    /// discretized path is not applicable.
    pub fn to_full_matrix(&self) -> Array2<f64> {
        let n = self.n;
        let p = self.total_basis;
        let mut full = Array2::zeros((n, p));

        for term in &self.terms {
            let k = term.num_basis;
            let off = term.col_offset;

            for i in 0..n {
                let idx = term.indices[i] as usize;
                for j in 0..k {
                    full[[i, off + j]] = term.values[[idx, j]];
                }
            }
        }

        full
    }

    /// Compute X'X using scatter-gather (for Gaussian family where W = I).
    /// Specialization of compute_xtwx with w = 1.
    pub fn compute_xtx(&self) -> Array2<f64> {
        let ones = Array1::ones(self.n);
        self.compute_xtwx(&ones)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_compressed_basis_dedup() {
        // Create a basis with duplicate rows
        let full = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, // row 0
                3.0, 4.0, // row 1
                1.0, 2.0, // row 2 = row 0
                3.0, 4.0, // row 3 = row 1
                5.0, 6.0, // row 4
                1.0, 2.0, // row 5 = row 0
            ],
        )
        .unwrap();

        let cb = CompressedBasis::from_basis_dedup(&full, 0);

        assert_eq!(cb.num_compressed(), 3); // 3 unique rows
        assert_eq!(cb.num_observations(), 6);
        assert!(cb.compression_ratio() >= 1.9); // 6/3 = 2.0

        // Verify indices map correctly
        assert_eq!(cb.indices[0], cb.indices[2]); // rows 0,2,5 same
        assert_eq!(cb.indices[0], cb.indices[5]);
        assert_eq!(cb.indices[1], cb.indices[3]); // rows 1,3 same
    }

    #[test]
    fn test_compressed_basis_1d() {
        let n = 1000;
        let k = 10;
        let max_bins = 100;

        // Create synthetic covariate and basis
        let covariate = Array1::linspace(0.0, 1.0, n);
        let mut full_basis = Array2::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                full_basis[[i, j]] = (covariate[i] * (j + 1) as f64 * std::f64::consts::PI).sin();
            }
        }

        let cb = CompressedBasis::from_basis_1d(&full_basis, &covariate, 0, max_bins);

        assert!(cb.num_compressed() <= max_bins);
        assert_eq!(cb.num_observations(), n);
        assert!(cb.compression_ratio() >= 5.0); // should be ~10x
    }

    #[test]
    fn test_xtwx_correctness() {
        // Verify scatter-gather X'WX matches naive computation
        let n = 200;
        let k1 = 5;
        let k2 = 4;
        let p = k1 + k2;

        // Create two basis matrices
        let cov1 = Array1::linspace(0.0, 1.0, n);
        let cov2 = Array1::linspace(-1.0, 1.0, n);

        let mut basis1 = Array2::zeros((n, k1));
        let mut basis2 = Array2::zeros((n, k2));
        for i in 0..n {
            for j in 0..k1 {
                basis1[[i, j]] = (cov1[i] * (j + 1) as f64).powi(2);
            }
            for j in 0..k2 {
                basis2[[i, j]] = (cov2[i] * (j + 1) as f64).cos();
            }
        }

        // Weights
        let w: Array1<f64> = (0..n).map(|i| 1.0 + (i as f64 * 0.01).sin()).collect();

        // Naive X'WX using full design matrix
        let mut full_x = Array2::zeros((n, p));
        full_x.slice_mut(s![.., 0..k1]).assign(&basis1);
        full_x.slice_mut(s![.., k1..k1 + k2]).assign(&basis2);

        let mut naive_xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            for a in 0..p {
                for b in 0..p {
                    naive_xtwx[[a, b]] += full_x[[i, a]] * w[i] * full_x[[i, b]];
                }
            }
        }

        // Discretized X'WX
        let config = DiscretizeConfig {
            max_unique_1d: 50,
            min_n_for_discretize: 100,
        };
        let dd = DiscretizedDesign::new(&[basis1, basis2], &[cov1, cov2], &config);
        let sg_xtwx = dd.compute_xtwx(&w);

        // Check agreement (not exact due to binning, but should be close)
        // For continuous covariates with moderate binning, error should be small
        let max_err: f64 = naive_xtwx
            .iter()
            .zip(sg_xtwx.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let max_val: f64 = naive_xtwx.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        let rel_err = max_err / max_val.max(1e-10);

        assert!(
            rel_err < 0.1,
            "X'WX relative error too large: {} (max_err={}, max_val={})",
            rel_err,
            max_err,
            max_val
        );
    }

    #[test]
    fn test_xtwx_exact_no_binning() {
        // With n small enough that no binning occurs, X'WX should be exact
        let n = 20;
        let k = 3;

        let cov: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let mut basis = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                basis[[i, j]] = cov[i].powi(j as i32);
            }
        }

        let w = Array1::ones(n);

        // Naive
        let naive_xtwx = basis.t().dot(&basis);

        // Discretized (n=20 < min_n_for_discretize=500, so no binning)
        let config = DiscretizeConfig::default();
        let dd = DiscretizedDesign::new(&[basis.clone()], &[cov], &config);
        let sg_xtwx = dd.compute_xtwx(&w);

        for i in 0..k {
            for j in 0..k {
                assert_abs_diff_eq!(naive_xtwx[[i, j]], sg_xtwx[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_eta_computation() {
        let n = 100;
        let k = 4;

        let cov: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let mut basis = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                basis[[i, j]] = cov[i].powi(j as i32);
            }
        }

        let config = DiscretizeConfig::default();
        let dd = DiscretizedDesign::new(&[basis.clone()], &[cov], &config);

        let beta = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        // Naive eta = X * beta
        let naive_eta = basis.dot(&beta);

        // Discretized eta
        let disc_eta = dd.compute_eta(&beta);

        for i in 0..n {
            assert_abs_diff_eq!(naive_eta[i], disc_eta[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_xtwy_computation() {
        let n = 100;
        let k = 3;

        let cov: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let mut basis = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                basis[[i, j]] = cov[i].powi(j as i32);
            }
        }

        let w = Array1::ones(n);
        let y: Array1<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let config = DiscretizeConfig::default();
        let dd = DiscretizedDesign::new(&[basis.clone()], &[cov], &config);

        // Naive X'Wy = X' * (w .* y) -- with w=1, this is X'y
        let naive_xtwy = basis.t().dot(&y);

        // Discretized
        let disc_xtwy = dd.compute_xtwy(&w, &y);

        for j in 0..k {
            assert_abs_diff_eq!(naive_xtwy[j], disc_xtwy[j], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_to_full_matrix_roundtrip() {
        let n = 50;
        let k = 3;

        let cov: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let mut basis = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                basis[[i, j]] = cov[i].powi(j as i32);
            }
        }

        let config = DiscretizeConfig::default();
        let dd = DiscretizedDesign::new(&[basis.clone()], &[cov], &config);

        let full = dd.to_full_matrix();

        for i in 0..n {
            for j in 0..k {
                assert_abs_diff_eq!(basis[[i, j]], full[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_multi_term_xtwx() {
        // Test with multiple terms, verifying block structure
        let n = 100;
        let k1 = 3;
        let k2 = 4;

        let cov1: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let cov2: Array1<f64> = Array1::linspace(-1.0, 1.0, n);

        let mut b1 = Array2::<f64>::zeros((n, k1));
        let mut b2 = Array2::<f64>::zeros((n, k2));
        for i in 0..n {
            for j in 0..k1 {
                b1[[i, j]] = cov1[i].powi(j as i32);
            }
            for j in 0..k2 {
                b2[[i, j]] = cov2[i].powi(j as i32);
            }
        }

        let w = Array1::ones(n);

        // Build full design matrix
        let p = k1 + k2;
        let mut full_x = Array2::zeros((n, p));
        full_x.slice_mut(s![.., 0..k1]).assign(&b1);
        full_x.slice_mut(s![.., k1..p]).assign(&b2);

        let naive = full_x.t().dot(&full_x);

        let config = DiscretizeConfig::default();
        let dd = DiscretizedDesign::new(&[b1, b2], &[cov1, cov2], &config);
        let disc = dd.compute_xtwx(&w);

        for i in 0..p {
            for j in 0..p {
                assert_abs_diff_eq!(naive[[i, j]], disc[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
