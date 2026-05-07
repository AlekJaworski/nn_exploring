# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | ✓ | ✓ | ✓ | 4.08e-06 | 4.68e-04 | 6.37e-06 | 1.52e-06 | 2.14e-08 | 9.27e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | ✓ | ✓ | ✓ | 7.75e-07 | 9.81e-03 | 1.55e-06 | 8.44e-07 | 7.04e-08 | 3.73e-05 |
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 1.71e-07 | 2.21e-07 | 2.63e-07 | 1.57e-07 | 1.13e-07 | 2.00e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.13e-09 | 2.01e-06 | 1.19e-08 | 5.20e-09 | 1.91e-09 | 8.73e-07 |
| 1d_gaussian_smooth_n1000_k50_cr | ✓ | ✓ | ✓ | 3.47e-07 | 2.47e-05 | 9.96e-07 | 3.39e-07 | 1.48e-07 | 3.42e-05 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 3.49e-10 | 4.11e-09 | 6.17e-10 | 3.97e-10 | 1.30e-10 | 1.64e-08 |
| 1d_gaussian_smooth_n2000_k30_cr | ✓ | ✓ | ✓ | 1.04e-07 | 8.28e-05 | 4.54e-07 | 9.29e-08 | 2.44e-08 | 9.20e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.38e-08 | 2.19e-07 | 1.19e-08 | 2.47e-08 | 1.10e-09 | 5.08e-06 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 2.34e-08 | 6.98e-07 | 6.55e-08 | 4.21e-08 | 3.06e-09 | 1.38e-06 |
| 1d_gaussian_smooth_n50_k10_cr | ✓ | ✓ | ✓ | 5.09e-09 | 8.03e-08 | 1.62e-08 | 5.53e-09 | 2.54e-09 | 1.31e-07 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 5.95e-10 | 1.36e-07 | 1.16e-09 | 3.00e-10 | 1.04e-10 | 4.16e-08 |
| 1d_gaussian_step_n500_k10_cr | ✓ | ✓ | ✓ | 2.57e-08 | 6.23e-07 | 3.92e-08 | 2.72e-08 | 3.92e-09 | 3.65e-06 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 3.85e-10 | 8.18e-08 | 1.68e-09 | 4.09e-10 | 3.40e-11 | 5.52e-08 |
| 1d_poisson_log_n500_k10_cr | ✓ | ✓ | ✓ | 7.23e-06 | 1.79e-06 | 1.68e-05 | 1.85e-06 | 4.14e+00 | 4.77e-05 |
| 1d_tw_log_n400_k20_cr | ✓ | ✓ | ✓ | 1.01e-03 | 7.04e-04 | 2.06e-03 | 7.82e-04 | 9.99e-01 | 1.15e-02 |
| 1d_tweedie_log_n400_k20_cr_p15 | ✓ | ✓ | ✓ | 4.03e-04 | 2.51e-04 | 7.44e-04 | 2.77e-04 | 1.04e+00 | 4.20e-03 |
| 2d_binomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 4.61e-04 | 2.21e-03 | 7.75e-04 | 2.00e-03 | 8.37e-01 | 1.91e-02 |
| 2d_binomial_logit_n200_k10_cr | ✗ | ✗ | ✓ | 5.17e-03 | 2.25e-02 | 2.92e-03 | 3.03e-02 | 8.39e-01 | 9.31e-02 |
| 2d_binomial_logit_n5000_k10_cr | ✓ | ✓ | ✓ | 5.37e-05 | 2.71e-04 | 4.60e-05 | 2.06e-04 | 8.37e-01 | 2.58e-01 |
| 2d_gamma_inverse_n1000_k10_cr | ✗ | ✗ | ✓ | 1.13e-02 | 3.49e-03 | 2.28e-03 | 2.54e-03 | 3.40e+00 | 3.34e+00 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 2.45e-04 | 6.62e-05 | 2.69e-04 | 3.75e-05 | 1.32e+01 | 5.60e-01 |
| 2d_gamma_log_n200_k10_cr | ✓ | ✓ | ✓ | 1.83e-03 | 5.58e-04 | 5.62e-04 | 3.10e-04 | 9.88e+00 | 5.74e-03 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 9.29e-08 | 4.89e-05 | 2.80e-07 | 9.45e-08 | 8.22e-09 | 1.68e-05 |
| 2d_gaussian_additive_n50000_k15_cr | ✓ | ✓ | ✓ | 2.18e-08 | 2.52e-05 | 5.97e-08 | 2.25e-08 | 2.08e-11 | 1.30e-04 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.66e-07 | 1.61e-05 | 3.84e-07 | 1.40e-07 | 1.06e-08 | 1.13e-05 |
| 2d_gaussian_bs_n1500_k15 | ✓ | ✓ | ✓ | 2.26e-07 | 4.20e-05 | 1.69e-06 | 8.22e-07 | 1.49e-08 | 9.24e-06 |
| 2d_invgauss_log_n800_k10_cr | ✗ | ✗ | ✓ | 2.47e-02 | 7.09e-03 | 2.71e-02 | 4.83e-03 | 9.17e+00 | 5.94e-02 |
| 2d_nb_log_n1000_k10_cr_theta2 | ✓ | ✓ | ✓ | 3.68e-03 | 4.43e-04 | 2.79e-03 | 2.82e-04 | 3.11e+01 | 1.44e-02 |
| 2d_nb_profile_log_n1000_k10_cr | ✗ | ✗ | ✓ | 2.76e-02 | 3.23e-03 | 1.91e-02 | 2.14e-03 | 3.04e+01 | 8.95e-02 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 1.63e-04 | 2.41e-05 | 4.43e-04 | 2.29e-05 | 5.45e+00 | 3.29e-01 |
| 2d_poisson_log_n200_k10_cr | ✓ | ✓ | ✓ | 4.96e-04 | 8.62e-05 | 2.64e-03 | 1.00e-04 | 5.15e+00 | 4.06e-01 |
| 2d_poisson_log_n5000_k10_cr | ✓ | ✓ | ✓ | 2.83e-03 | 1.87e-04 | 2.85e-03 | 1.60e-04 | 5.48e+00 | 8.49e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 5.04e-04 | 1.40e-03 | 7.94e-04 | 2.44e-03 | 8.36e-01 | 7.83e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 1.62e-03 | 3.75e-04 | 6.47e-03 | 2.86e-04 | 7.17e+00 | 6.47e-03 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 3.77e-06 | 3.74e-04 | 5.46e-06 | 3.27e-06 | 6.45e-07 | 1.07e-03 |
| 3d_poisson_log_n2000_k10_cr | ✓ | ✓ | ✓ | 2.99e-03 | 3.60e-04 | 3.75e-03 | 1.66e-04 | 3.68e+00 | 6.61e-01 |
| 4d_binomial_logit_n2000_k8_cr | ✓ | ✓ | ✓ | 5.49e-04 | 1.95e-03 | 4.96e-04 | 1.16e-03 | 8.32e-01 | 2.95e-02 |
| 4d_gamma_log_n2000_k8_cr | ✓ | ✓ | ✓ | 2.04e-03 | 3.99e-04 | 2.19e-03 | 2.60e-04 | 7.24e+00 | 3.79e-02 |
| 4d_gaussian_bs_n2000_k10 | ✓ | ✓ | ✓ | 5.48e-07 | 2.51e-04 | 8.23e-07 | 6.47e-07 | 1.06e-07 | 4.08e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.67e-07 | 7.54e-06 | 2.84e-07 | 1.44e-07 | 2.50e-08 | 5.13e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 1.24e-05 | 6.39e-03 | 1.52e-05 | 1.16e-05 | 2.80e-06 | 5.41e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.25e-07 | 1.00e-04 | 1.45e-07 | 1.21e-07 | 2.17e-08 | 1.62e-05 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 3.03e-05 | 3.19e-02 | 8.79e-05 | 2.71e-05 | 3.04e-06 | 1.13e-02 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 1.06e-04 | 7.24e-06 | 1.39e-04 | 1.02e-04 | 3.34e-06 | 5.21e-01 |
| 7d_neighbourhoods_compact_n3000 | ✓ | ✓ | ✓ | 4.14e-05 | 1.24e-03 | 5.79e-05 | 2.78e-05 | 8.14e-06 | 4.91e-01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 4.15e-04 | 7.02e-02 | 6.15e-04 | 2.22e-04 | 2.05e-05 | 9.89e-01 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 8.71e+01 | 8.52e+01 | 2.93e+02 | 2.97e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.87e+00 | 1.79e+00 | 4.56e+01 | 4.11e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.10e+00 | 1.74e+00 | 2.74e+01 | 7.68e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5.93e-01 | 5.69e-01 | 1.87e+01 | 3.17e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.45e+01 | 2.31e+01 | 6.75e+01 | 3.62e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 9.33e-01 | 6.37e-01 | 1.81e+01 | 5.17e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.31e+01 | 1.08e+01 | 3.30e+01 | 3.98e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.01e+00 | 1.01e+00 | 1.98e+01 | 5.10e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.65e+00 | 2.43e+00 | 3.06e+01 | 8.64e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 5.60e-01 | 5.38e-01 | 1.59e+01 | 3.51e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 8.04e-01 | 7.76e-01 | 1.78e+01 | 4.52e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.86e-01 | 7.74e-01 | 2.36e+01 | 3.33e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.90e+00 | 2.76e+00 | 3.03e+01 | 9.58e-02 |
| 1d_poisson_log_n500_k10_cr | 7.34e+00 | 7.18e+00 | 5.79e+01 | 1.27e-01 |
| 1d_tw_log_n400_k20_cr | 1.50e+02 | 1.40e+02 | 2.42e+02 | 6.20e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 9.38e+01 | 9.04e+01 | 8.01e+01 | 1.17e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.92e+01 | 3.59e+01 | 8.47e+01 | 4.63e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.14e+01 | 9.65e+00 | 1.47e+02 | 7.74e-02 |
| 2d_binomial_logit_n5000_k10_cr | 7.69e+02 | 7.53e+02 | 1.48e+02 | 5.21e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.21e+01 | 3.12e+01 | 2.13e+02 | 1.51e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.88e+01 | 2.84e+01 | 1.49e+02 | 1.93e-01 |
| 2d_gamma_log_n200_k10_cr | 5.10e+00 | 4.91e+00 | 1.55e+02 | 3.29e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 8.16e+00 | 7.93e+00 | 3.24e+01 | 2.52e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 3.19e+02 | 3.04e+02 | 3.96e+02 | 8.06e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.39e+00 | 2.25e+00 | 3.83e+01 | 6.24e-02 |
| 2d_gaussian_bs_n1500_k15 | 1.02e+01 | 6.98e+00 | 4.11e+01 | 2.49e-01 |
| 2d_invgauss_log_n800_k10_cr | 2.76e+01 | 2.75e+01 | 1.68e+02 | 1.65e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.47e+01 | 2.43e+01 | 1.29e+02 | 1.91e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 3.12e+01 | 2.98e+01 | 1.64e+02 | 1.91e-01 |
| 2d_poisson_log_n1000_k10_cr | 2.99e+01 | 2.93e+01 | 1.10e+02 | 2.73e-01 |
| 2d_poisson_log_n200_k10_cr | 1.01e+01 | 9.64e+00 | 1.09e+02 | 9.23e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.34e+02 | 1.27e+02 | 1.49e+02 | 8.97e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 4.26e+01 | 4.14e+01 | 1.17e+02 | 3.65e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 2.74e+01 | 2.65e+01 | 1.23e+02 | 2.23e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.01e+00 | 5.75e+00 | 4.15e+01 | 1.45e-01 |
| 3d_poisson_log_n2000_k10_cr | 7.87e+01 | 7.65e+01 | 1.57e+02 | 5.01e-01 |
| 4d_binomial_logit_n2000_k8_cr | 9.56e+01 | 9.48e+01 | 1.63e+02 | 5.88e-01 |
| 4d_gamma_log_n2000_k8_cr | 6.97e+01 | 6.48e+01 | 2.26e+02 | 3.08e-01 |
| 4d_gaussian_bs_n2000_k10 | 2.59e+01 | 2.53e+01 | 8.05e+01 | 3.22e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.45e+01 | 1.42e+01 | 6.17e+01 | 2.35e-01 |
| 4d_small_neighbourhood_n300 | 7.11e+00 | 6.26e+00 | 4.47e+01 | 1.59e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.21e+01 | 1.19e+01 | 8.37e+01 | 1.45e-01 |
| 5d_skewed_features_n5000 | 6.57e+01 | 6.37e+01 | 1.06e+02 | 6.21e-01 |
| 6d_heatmap_pricing_n8000 | 1.38e+02 | 1.26e+02 | 2.07e+02 | 6.66e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.24e+01 | 4.03e+01 | 1.22e+02 | 3.48e-01 |
| 8d_neighbourhoods_like_n15000 | 6.00e+02 | 5.83e+02 | 5.92e+02 | 1.01e+00 |
