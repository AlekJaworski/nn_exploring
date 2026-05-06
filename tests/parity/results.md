# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 7.69e+01 | 7.43e+01 | 2.76e+02 | 2.79e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.59e+00 | 1.57e+00 | 4.04e+01 | 3.94e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 1.67e+00 | 1.64e+00 | 2.45e+01 | 6.83e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5.74e-01 | 5.45e-01 | 1.69e+01 | 3.40e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.12e+01 | 1.93e+01 | 6.22e+01 | 3.40e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 4.91e-01 | 4.80e-01 | 1.74e+01 | 2.82e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.13e+01 | 1.03e+01 | 3.18e+01 | 3.55e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.07e+00 | 1.00e+00 | 1.76e+01 | 6.11e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.37e+00 | 2.23e+00 | 2.44e+01 | 9.71e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 4.71e-01 | 4.65e-01 | 1.72e+01 | 2.73e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 9.29e-01 | 8.88e-01 | 2.05e+01 | 4.54e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.93e-01 | 7.13e-01 | 1.85e+01 | 4.27e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.46e+00 | 2.29e+00 | 2.46e+01 | 9.98e-02 |
| 1d_poisson_log_n500_k10_cr | 4.12e+00 | 3.85e+00 | 5.72e+01 | 7.21e-02 |
| 1d_tw_log_n400_k20_cr | 1.36e+02 | 1.31e+02 | 2.06e+02 | 6.61e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 8.16e+01 | 7.82e+01 | 7.71e+01 | 1.06e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.20e+01 | 3.00e+01 | 8.27e+01 | 3.87e-01 |
| 2d_binomial_logit_n200_k10_cr | 9.42e+00 | 8.44e+00 | 1.19e+02 | 7.92e-02 |
| 2d_binomial_logit_n5000_k10_cr | 6.46e+02 | 6.41e+02 | 1.60e+02 | 4.04e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 2.92e+01 | 2.76e+01 | 2.39e+02 | 1.22e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.98e+01 | 2.89e+01 | 1.50e+02 | 1.99e-01 |
| 2d_gamma_log_n200_k10_cr | 5.56e+00 | 5.17e+00 | 1.50e+02 | 3.71e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.23e+00 | 6.84e+00 | 3.51e+01 | 2.06e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 3.43e+02 | 3.21e+02 | 2.67e+02 | 1.28e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.05e+00 | 1.96e+00 | 2.34e+01 | 8.76e-02 |
| 2d_gaussian_bs_n1500_k15 | 7.43e+00 | 6.66e+00 | 3.29e+01 | 2.26e-01 |
| 2d_invgauss_log_n800_k10_cr | 3.12e+01 | 2.85e+01 | 2.02e+02 | 1.55e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.53e+01 | 2.28e+01 | 1.74e+02 | 1.45e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 3.23e+01 | 2.71e+01 | 1.73e+02 | 1.86e-01 |
| 2d_poisson_log_n1000_k10_cr | 2.88e+01 | 2.80e+01 | 1.23e+02 | 2.34e-01 |
| 2d_poisson_log_n200_k10_cr | 9.86e+00 | 9.46e+00 | 1.16e+02 | 8.52e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.48e+02 | 1.40e+02 | 1.67e+02 | 8.87e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 4.13e+01 | 3.83e+01 | 1.24e+02 | 3.35e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 2.41e+01 | 2.31e+01 | 1.53e+02 | 1.57e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.03e+00 | 5.84e+00 | 4.78e+01 | 1.26e-01 |
| 3d_poisson_log_n2000_k10_cr | 9.76e+01 | 9.54e+01 | 1.96e+02 | 4.98e-01 |
| 4d_binomial_logit_n2000_k8_cr | 1.07e+02 | 1.01e+02 | 1.66e+02 | 6.43e-01 |
| 4d_gamma_log_n2000_k8_cr | 7.00e+01 | 6.87e+01 | 2.49e+02 | 2.81e-01 |
| 4d_gaussian_bs_n2000_k10 | 2.55e+01 | 2.48e+01 | 9.93e+01 | 2.56e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.53e+01 | 1.41e+01 | 9.03e+01 | 1.70e-01 |
| 4d_small_neighbourhood_n300 | 9.16e+00 | 7.47e+00 | 5.88e+01 | 1.56e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.10e+01 | 1.06e+01 | 8.29e+01 | 1.32e-01 |
| 5d_skewed_features_n5000 | 5.86e+01 | 5.67e+01 | 1.06e+02 | 5.53e-01 |
| 6d_heatmap_pricing_n8000 | 1.25e+02 | 1.16e+02 | 2.12e+02 | 5.91e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.48e+01 | 3.71e+01 | 1.26e+02 | 3.55e-01 |
| 8d_neighbourhoods_like_n15000 | 6.64e+02 | 6.24e+02 | 6.56e+02 | 1.01e+00 |
