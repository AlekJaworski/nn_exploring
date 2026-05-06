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
| 2d_nb_profile_log_n1000_k10_cr | ✗ | ✗ | ✓ | 1.68e-02 | 2.15e-03 | 1.91e-02 | 1.37e-03 | 3.04e+01 | 8.95e-02 |
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
| 10d_gaussian_n3000_k8_cr | 6.83e+01 | 6.76e+01 | 2.54e+02 | 2.69e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.57e+00 | 1.56e+00 | 3.77e+01 | 4.15e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 1.65e+00 | 1.63e+00 | 2.24e+01 | 7.35e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5.42e-01 | 5.25e-01 | 1.60e+01 | 3.40e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 1.81e+01 | 1.74e+01 | 5.88e+01 | 3.08e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 4.57e-01 | 4.39e-01 | 1.65e+01 | 2.77e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.00e+01 | 9.38e+00 | 2.98e+01 | 3.35e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 8.91e-01 | 8.77e-01 | 1.76e+01 | 5.06e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.25e+00 | 2.19e+00 | 2.31e+01 | 9.76e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 8.68e-01 | 8.02e-01 | 1.97e+01 | 4.40e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.51e-01 | 7.34e-01 | 1.65e+01 | 4.54e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.14e-01 | 7.03e-01 | 1.80e+01 | 3.97e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.45e+00 | 2.40e+00 | 2.36e+01 | 1.04e-01 |
| 1d_poisson_log_n500_k10_cr | 3.67e+00 | 3.55e+00 | 5.22e+01 | 7.03e-02 |
| 1d_tw_log_n400_k20_cr | 1.26e+02 | 1.24e+02 | 2.10e+02 | 6.02e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 7.44e+01 | 7.27e+01 | 7.81e+01 | 9.53e-01 |
| 2d_binomial_logit_n1000_k10_cr | 3.23e+01 | 3.02e+01 | 8.62e+01 | 3.75e-01 |
| 2d_binomial_logit_n200_k10_cr | 8.63e+00 | 8.23e+00 | 1.32e+02 | 6.52e-02 |
| 2d_binomial_logit_n5000_k10_cr | 6.19e+02 | 6.15e+02 | 1.39e+02 | 4.47e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 2.62e+01 | 2.60e+01 | 1.92e+02 | 1.36e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.85e+01 | 2.50e+01 | 1.39e+02 | 2.05e-01 |
| 2d_gamma_log_n200_k10_cr | 4.46e+00 | 4.44e+00 | 1.40e+02 | 3.18e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 6.55e+00 | 6.23e+00 | 3.07e+01 | 2.13e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 2.93e+02 | 2.80e+02 | 2.39e+02 | 1.23e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.02e+00 | 1.81e+00 | 2.58e+01 | 7.84e-02 |
| 2d_gaussian_bs_n1500_k15 | 6.55e+00 | 6.23e+00 | 3.12e+01 | 2.10e-01 |
| 2d_invgauss_log_n800_k10_cr | 2.39e+01 | 2.24e+01 | 1.64e+02 | 1.46e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.00e+01 | 1.93e+01 | 1.24e+02 | 1.62e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 2.62e+01 | 2.41e+01 | 1.53e+02 | 1.71e-01 |
| 2d_poisson_log_n1000_k10_cr | 2.47e+01 | 2.33e+01 | 1.01e+02 | 2.45e-01 |
| 2d_poisson_log_n200_k10_cr | 8.19e+00 | 7.94e+00 | 1.04e+02 | 7.90e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.08e+02 | 1.06e+02 | 1.42e+02 | 7.58e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 3.52e+01 | 3.38e+01 | 1.11e+02 | 3.16e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 2.08e+01 | 2.00e+01 | 1.15e+02 | 1.81e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 5.08e+00 | 4.78e+00 | 4.12e+01 | 1.23e-01 |
| 3d_poisson_log_n2000_k10_cr | 6.77e+01 | 6.71e+01 | 1.46e+02 | 4.63e-01 |
| 4d_binomial_logit_n2000_k8_cr | 8.00e+01 | 7.82e+01 | 1.52e+02 | 5.28e-01 |
| 4d_gamma_log_n2000_k8_cr | 5.43e+01 | 5.28e+01 | 2.18e+02 | 2.49e-01 |
| 4d_gaussian_bs_n2000_k10 | 2.22e+01 | 2.05e+01 | 7.32e+01 | 3.03e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.21e+01 | 1.16e+01 | 5.65e+01 | 2.15e-01 |
| 4d_small_neighbourhood_n300 | 5.50e+00 | 4.89e+00 | 4.62e+01 | 1.19e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.05e+01 | 9.46e+00 | 7.24e+01 | 1.46e-01 |
| 5d_skewed_features_n5000 | 5.65e+01 | 5.34e+01 | 1.01e+02 | 5.58e-01 |
| 6d_heatmap_pricing_n8000 | 1.18e+02 | 1.02e+02 | 1.83e+02 | 6.46e-01 |
| 7d_neighbourhoods_compact_n3000 | 3.25e+01 | 2.99e+01 | 1.12e+02 | 2.89e-01 |
| 8d_neighbourhoods_like_n15000 | 5.59e+02 | 5.48e+02 | 5.65e+02 | 9.90e-01 |
