# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 7.36e+01 | 7.17e+01 | 2.72e+02 | 2.70e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.66e+00 | 1.63e+00 | 3.95e+01 | 4.20e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.17e+00 | 1.89e+00 | 2.62e+01 | 8.27e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5.74e-01 | 5.66e-01 | 1.89e+01 | 3.03e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.12e+01 | 2.10e+01 | 6.24e+01 | 3.40e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 4.99e-01 | 4.91e-01 | 1.58e+01 | 3.16e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.10e+01 | 9.78e+00 | 3.45e+01 | 3.18e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.02e+00 | 9.13e-01 | 1.90e+01 | 5.39e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.48e+00 | 2.21e+00 | 2.76e+01 | 8.97e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 4.74e-01 | 4.58e-01 | 1.63e+01 | 2.90e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.64e-01 | 7.50e-01 | 1.82e+01 | 4.20e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.54e-01 | 7.37e-01 | 1.75e+01 | 4.30e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.68e+00 | 2.38e+00 | 2.46e+01 | 1.09e-01 |
| 1d_poisson_log_n500_k10_cr | 3.71e+00 | 3.58e+00 | 5.23e+01 | 7.11e-02 |
| 1d_tw_log_n400_k20_cr | 1.03e+03 | 1.01e+03 | 2.13e+02 | 4.82e+00 |
| 1d_tweedie_log_n400_k20_cr_p15 | 3.81e+02 | 3.69e+02 | 8.08e+01 | 4.71e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.51e+01 | 3.41e+01 | 8.76e+01 | 4.00e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.18e+01 | 1.01e+01 | 1.29e+02 | 9.13e-02 |
| 2d_binomial_logit_n5000_k10_cr | 6.94e+02 | 6.86e+02 | 1.55e+02 | 4.47e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.50e+01 | 3.30e+01 | 2.09e+02 | 1.67e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.92e+01 | 2.87e+01 | 1.49e+02 | 1.96e-01 |
| 2d_gamma_log_n200_k10_cr | 5.68e+00 | 5.01e+00 | 1.50e+02 | 3.78e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.34e+00 | 7.16e+00 | 3.55e+01 | 2.07e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 3.21e+02 | 3.02e+02 | 2.55e+02 | 1.26e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.96e+00 | 1.90e+00 | 2.32e+01 | 8.45e-02 |
| 2d_gaussian_bs_n1500_k15 | 6.75e+00 | 6.58e+00 | 3.15e+01 | 2.14e-01 |
| 2d_invgauss_log_n800_k10_cr | 2.90e+01 | 2.86e+01 | 1.71e+02 | 1.70e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.48e+01 | 2.43e+01 | 1.32e+02 | 1.88e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 3.12e+01 | 3.09e+01 | 1.68e+02 | 1.86e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.06e+01 | 2.90e+01 | 1.06e+02 | 2.89e-01 |
| 2d_poisson_log_n200_k10_cr | 1.03e+01 | 9.86e+00 | 1.18e+02 | 8.70e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.33e+02 | 1.27e+02 | 1.54e+02 | 8.62e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 4.16e+01 | 4.09e+01 | 1.16e+02 | 3.57e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 2.64e+01 | 2.52e+01 | 1.18e+02 | 2.24e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.25e+00 | 6.07e+00 | 4.61e+01 | 1.35e-01 |
| 3d_poisson_log_n2000_k10_cr | 8.18e+01 | 8.02e+01 | 1.59e+02 | 5.14e-01 |
| 4d_binomial_logit_n2000_k8_cr | 9.72e+01 | 9.56e+01 | 1.65e+02 | 5.91e-01 |
| 4d_gamma_log_n2000_k8_cr | 6.96e+01 | 6.37e+01 | 2.30e+02 | 3.02e-01 |
| 4d_gaussian_bs_n2000_k10 | 2.37e+01 | 2.34e+01 | 9.12e+01 | 2.60e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.46e+01 | 1.43e+01 | 6.10e+01 | 2.40e-01 |
| 4d_small_neighbourhood_n300 | 6.24e+00 | 5.96e+00 | 5.22e+01 | 1.19e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.20e+01 | 1.13e+01 | 8.23e+01 | 1.45e-01 |
| 5d_skewed_features_n5000 | 5.89e+01 | 5.74e+01 | 1.04e+02 | 5.66e-01 |
| 6d_heatmap_pricing_n8000 | 1.09e+02 | 1.05e+02 | 2.00e+02 | 5.47e-01 |
| 7d_neighbourhoods_compact_n3000 | 3.55e+01 | 3.41e+01 | 1.23e+02 | 2.88e-01 |
| 8d_neighbourhoods_like_n15000 | 5.53e+02 | 5.45e+02 | 5.90e+02 | 9.36e-01 |
