# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 1.05e+02 | 1.03e+02 | 3.60e+02 | 2.92e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.84e+00 | 1.76e+00 | 4.27e+01 | 4.32e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 5.50e+00 | 3.70e+00 | 6.34e+01 | 8.67e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 9.87e-01 | 8.56e-01 | 2.51e+01 | 3.93e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 3.68e+01 | 2.76e+01 | 7.28e+01 | 5.06e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 1.10e+00 | 7.80e-01 | 2.41e+01 | 4.57e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.16e+01 | 1.02e+01 | 3.34e+01 | 3.47e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.09e+00 | 9.89e-01 | 2.25e+01 | 4.85e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 3.53e+00 | 3.11e+00 | 3.40e+01 | 1.04e-01 |
| 1d_gaussian_smooth_n50_k10_cr | 5.88e-01 | 5.67e-01 | 2.54e+01 | 2.31e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 1.16e+00 | 1.08e+00 | 2.32e+01 | 5.02e-02 |
| 1d_gaussian_step_n500_k10_cr | 8.62e-01 | 8.27e-01 | 2.13e+01 | 4.04e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.92e+00 | 2.74e+00 | 2.84e+01 | 1.03e-01 |
| 1d_poisson_log_n500_k10_cr | 4.24e+00 | 4.00e+00 | 6.81e+01 | 6.23e-02 |
| 2d_binomial_logit_n1000_k10_cr | 3.00e+01 | 2.80e+01 | 9.90e+01 | 3.04e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.14e+01 | 1.11e+01 | 1.51e+02 | 7.56e-02 |
| 2d_binomial_logit_n5000_k10_cr | 6.80e+02 | 6.60e+02 | 1.73e+02 | 3.94e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.04e+01 | 2.94e+01 | 2.25e+02 | 1.35e-01 |
| 2d_gamma_log_n1000_k10_cr | 3.43e+01 | 3.42e+01 | 1.65e+02 | 2.08e-01 |
| 2d_gamma_log_n200_k10_cr | 7.52e+02 | 7.39e+02 | 1.78e+02 | 4.21e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 7.35e+00 | 6.97e+00 | 3.54e+01 | 2.08e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 4.30e+02 | 3.94e+02 | 3.18e+02 | 1.35e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.87e+00 | 2.05e+00 | 2.85e+01 | 1.01e-01 |
| 2d_gaussian_bs_n1500_k15 | 9.30e+00 | 7.81e+00 | 3.63e+01 | 2.56e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.38e+01 | 3.09e+01 | 1.21e+02 | 2.81e-01 |
| 2d_poisson_log_n200_k10_cr | 9.89e+00 | 9.45e+00 | 1.25e+02 | 7.93e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.69e+02 | 1.62e+02 | 1.70e+02 | 9.94e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 7.10e+00 | 6.48e+00 | 4.21e+01 | 1.69e-01 |
| 3d_poisson_log_n2000_k10_cr | 1.07e+02 | 1.05e+02 | 1.84e+02 | 5.83e-01 |
| 4d_binomial_logit_n2000_k8_cr | 1.08e+02 | 1.04e+02 | 1.83e+02 | 5.88e-01 |
| 4d_gamma_log_n2000_k8_cr | 5.79e+02 | 5.69e+02 | 2.24e+02 | 2.59e+00 |
| 4d_gaussian_bs_n2000_k10 | 2.65e+01 | 2.58e+01 | 7.92e+01 | 3.35e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.75e+01 | 1.55e+01 | 6.43e+01 | 2.73e-01 |
| 4d_small_neighbourhood_n300 | 7.16e+00 | 6.67e+00 | 4.63e+01 | 1.55e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.04e+01 | 9.98e+00 | 9.62e+01 | 1.08e-01 |
| 5d_skewed_features_n5000 | 7.71e+01 | 7.25e+01 | 1.05e+02 | 7.34e-01 |
| 6d_heatmap_pricing_n8000 | 1.67e+02 | 1.59e+02 | 2.06e+02 | 8.11e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.59e+01 | 4.45e+01 | 1.26e+02 | 3.64e-01 |
| 8d_neighbourhoods_like_n15000 | 7.04e+02 | 6.82e+02 | 6.35e+02 | 1.11e+00 |
