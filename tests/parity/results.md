# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 1.14e+02 | 1.06e+02 | 2.93e+02 | 3.88e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.93e+00 | 1.72e+00 | 4.11e+01 | 4.69e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.14e+00 | 2.06e+00 | 3.04e+01 | 7.05e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5.97e-01 | 5.85e-01 | 1.62e+01 | 3.69e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.42e+01 | 2.31e+01 | 6.53e+01 | 3.70e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 5.23e-01 | 5.05e-01 | 1.73e+01 | 3.03e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.10e+01 | 1.03e+01 | 3.74e+01 | 2.94e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.03e+00 | 9.85e-01 | 2.07e+01 | 4.96e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 3.02e+00 | 2.76e+00 | 3.01e+01 | 1.00e-01 |
| 1d_gaussian_smooth_n50_k10_cr | 5.56e-01 | 4.66e-01 | 1.89e+01 | 2.95e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.77e-01 | 7.64e-01 | 2.20e+01 | 3.54e-02 |
| 1d_gaussian_step_n500_k10_cr | 8.13e-01 | 7.71e-01 | 2.15e+01 | 3.79e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.46e+00 | 2.40e+00 | 2.78e+01 | 8.87e-02 |
| 1d_poisson_log_n500_k10_cr | 3.75e+00 | 3.65e+00 | 6.35e+01 | 5.91e-02 |
| 2d_binomial_logit_n1000_k10_cr | 4.88e+01 | 4.84e+01 | 9.93e+01 | 4.91e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.10e+01 | 1.04e+01 | 1.47e+02 | 7.51e-02 |
| 2d_binomial_logit_n5000_k10_cr | 1.14e+03 | 1.11e+03 | 2.09e+02 | 5.44e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.24e+01 | 3.02e+01 | 2.66e+02 | 1.22e-01 |
| 2d_gamma_log_n1000_k10_cr | 3.68e+01 | 3.63e+01 | 1.65e+02 | 2.23e-01 |
| 2d_gamma_log_n200_k10_cr | 7.56e+02 | 7.28e+02 | 1.54e+02 | 4.90e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 7.23e+00 | 6.94e+00 | 3.65e+01 | 1.98e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 4.17e+02 | 4.11e+02 | 2.74e+02 | 1.52e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.10e+00 | 2.04e+00 | 2.67e+01 | 7.84e-02 |
| 2d_gaussian_bs_n1500_k15 | 7.94e+00 | 7.27e+00 | 3.58e+01 | 2.22e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.33e+01 | 3.22e+01 | 1.19e+02 | 2.79e-01 |
| 2d_poisson_log_n200_k10_cr | 9.82e+00 | 9.39e+00 | 1.20e+02 | 8.19e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.64e+02 | 1.62e+02 | 1.70e+02 | 9.66e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.78e+00 | 6.35e+00 | 7.84e+01 | 8.65e-02 |
| 3d_poisson_log_n2000_k10_cr | 1.01e+02 | 9.85e+01 | 1.70e+02 | 5.94e-01 |
| 4d_binomial_logit_n2000_k8_cr | 9.42e+01 | 9.28e+01 | 1.66e+02 | 5.68e-01 |
| 4d_gamma_log_n2000_k8_cr | 5.82e+02 | 5.55e+02 | 2.67e+02 | 2.18e+00 |
| 4d_gaussian_bs_n2000_k10 | 3.01e+01 | 2.72e+01 | 9.47e+01 | 3.18e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.66e+01 | 1.56e+01 | 6.63e+01 | 2.51e-01 |
| 4d_small_neighbourhood_n300 | 6.89e+00 | 6.58e+00 | 5.08e+01 | 1.36e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.17e+01 | 1.02e+01 | 8.64e+01 | 1.35e-01 |
| 5d_skewed_features_n5000 | 1.01e+02 | 7.45e+01 | 1.12e+02 | 9.07e-01 |
| 6d_heatmap_pricing_n8000 | 1.86e+02 | 1.74e+02 | 2.28e+02 | 8.16e-01 |
| 7d_neighbourhoods_compact_n3000 | 5.60e+01 | 5.09e+01 | 1.31e+02 | 4.28e-01 |
| 8d_neighbourhoods_like_n15000 | 7.47e+02 | 6.92e+02 | 6.25e+02 | 1.20e+00 |
