# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 8d_neighbourhoods_like_n15000 | ✗ | ✗ | ✓ | 4.91e-04 | 6.53e-03 | 4.47e-04 | 2.98e+00 | 3.29e-05 | 9.69e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 8d_neighbourhoods_like_n15000 | 4.91e-04 | 5.68e+06 | 7.65e+06 | 1.35e+00 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 8d_neighbourhoods_like_n15000 | 7.73e+02 | 7.57e+02 | 6.22e+02 | 1.24e+00 |
| 1d_gaussian_near_linear_n500_k10_cr | 1.96e+00 | 1.96e+00 | 2.55e+01 | 7.69e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 8.39e-01 | 7.39e-01 | 1.90e+01 | 4.41e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 4.32e-01 | 4.30e-01 | 1.56e+01 | 2.76e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 8.74e-01 | 8.36e-01 | 1.98e+01 | 4.41e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 8.33e+00 | 6.19e+00 | 2.49e+01 | 3.35e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.08e-01 | 6.93e-01 | 1.70e+01 | 4.17e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.21e+00 | 2.15e+00 | 2.04e+01 | 1.08e-01 |
| 2d_binomial_logit_n1000_k10_cr | 1.09e+01 | 1.01e+01 | 8.29e+01 | 1.31e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.29e+00 | 6.64e+00 | 1.40e+02 | 5.21e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.72e+00 | 6.92e+00 | 2.96e+01 | 2.61e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.80e+00 | 1.72e+00 | 2.26e+01 | 7.99e-02 |
| 2d_poisson_log_n1000_k10_cr | 6.64e+00 | 6.37e+00 | 1.06e+02 | 6.25e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 6.48e+00 | 6.33e+00 | 3.80e+01 | 1.71e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.40e+01 | 1.34e+01 | 7.20e+01 | 1.95e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.51e+01 | 1.28e+01 | 8.60e+01 | 1.76e-01 |
