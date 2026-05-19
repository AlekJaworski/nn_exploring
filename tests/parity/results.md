# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | ✓ | ✓ | ✓ | 5.64e-07 | 6.62e-05 | 9.44e-07 | 3.71e-07 | 3.23e-08 | 5.53e-01 |
| 1d_em_nb_seed123_n1000_k4_cr | ✗ | ✗ | ✓ | 1.25e-01 | 1.83e-03 | 1.78e-01 | 1.66e-03 | 1.49e+03 | 1.41e+00 |
| 1d_em_nb_seed124_n1000_k4_cr | ✓ | ✓ | ✓ | 2.49e-03 | 2.91e-05 | 3.70e-03 | 2.70e-05 | 1.54e+03 | 1.84e-01 |
| 1d_em_nb_seed125_n1000_k4_cr | ✓ | ✓ | ✓ | 4.04e-03 | 5.24e-05 | 5.99e-03 | 4.76e-05 | 1.38e+03 | 3.17e-03 |
| 1d_em_nb_seed126_n1000_k4_cr | ✓ | ✓ | ✓ | 1.94e-03 | 2.07e-05 | 3.09e-03 | 1.85e-05 | 1.33e+03 | 7.08e-01 |
| 1d_em_nb_seed127_n1000_k4_cr | ✓ | ✓ | ✓ | 1.15e-02 | 1.24e-04 | 1.73e-02 | 1.16e-04 | 1.35e+03 | 6.87e-01 |
| 1d_em_nb_seed128_n1000_k4_cr | ✓ | ✓ | ✓ | 1.50e-02 | 1.61e-04 | 2.34e-02 | 1.35e-04 | 1.53e+03 | 5.48e-01 |
| 1d_em_nb_seed129_n1000_k4_cr | ✓ | ✓ | ✓ | 2.20e-02 | 2.47e-04 | 3.50e-02 | 2.34e-04 | 1.26e+03 | 1.63e-02 |
| 1d_em_nb_seed130_n1000_k4_cr | ✓ | ✓ | ✓ | 2.24e-04 | 3.92e-06 | 3.42e-04 | 3.48e-06 | 9.79e+02 | 1.68e-04 |
| 1d_em_nb_seed131_n1000_k4_cr | ✓ | ✓ | ✓ | 1.88e-03 | 2.30e-05 | 3.09e-03 | 1.95e-05 | 8.80e+02 | 1.85e-01 |
| 1d_em_nb_seed132_n1000_k4_cr | ✓ | ✓ | ✓ | 4.02e-02 | 5.20e-04 | 5.74e-02 | 4.79e-04 | 1.34e+03 | 6.27e-02 |
| 1d_em_nb_seed133_n1000_k4_cr | ✓ | ✓ | ✓ | 5.79e-03 | 6.67e-05 | 9.22e-03 | 6.18e-05 | 1.11e+03 | 5.74e-01 |
| 1d_em_nb_seed134_n1000_k4_cr | ✓ | ✓ | ✓ | 1.66e-02 | 2.06e-04 | 2.74e-02 | 1.85e-04 | 1.18e+03 | 5.60e-01 |
| 1d_em_nb_seed135_n1000_k4_cr | ✓ | ✓ | ✓ | 5.98e-03 | 8.04e-05 | 9.29e-03 | 7.28e-05 | 1.29e+03 | 4.03e-03 |
| 1d_em_nb_seed136_n1000_k4_cr | ✓ | ✓ | ✓ | 9.62e-04 | 1.40e-05 | 1.34e-03 | 1.23e-05 | 9.91e+02 | 6.10e-04 |
| 1d_em_nb_seed137_n1000_k4_cr | ✓ | ✓ | ✓ | 4.64e-02 | 6.26e-04 | 7.35e-02 | 5.55e-04 | 1.37e+03 | 9.38e-02 |
| 1d_em_nb_seed138_n1000_k4_cr | ✓ | ✓ | ✓ | 2.24e-03 | 2.60e-05 | 3.20e-03 | 2.22e-05 | 1.10e+03 | 1.27e-03 |
| 1d_em_nb_seed139_n1000_k4_cr | ✓ | ✓ | ✓ | 1.14e-03 | 1.84e-05 | 1.60e-03 | 1.70e-05 | 1.98e+03 | 1.01e-03 |
| 1d_em_nb_seed140_n1000_k4_cr | ✓ | ✓ | ✓ | 3.13e-03 | 3.27e-05 | 4.86e-03 | 2.83e-05 | 1.46e+03 | 2.20e-01 |
| 1d_em_nb_seed141_n1000_k4_cr | ✓ | ✓ | ✓ | 3.04e-03 | 3.53e-05 | 4.97e-03 | 2.98e-05 | 1.03e+03 | 2.05e-03 |
| 1d_em_nb_seed142_n1000_k4_cr | ✓ | ✓ | ✓ | 3.35e-04 | 3.85e-06 | 5.37e-04 | 3.50e-06 | 1.22e+03 | 8.63e-02 |
| 1d_gaussian_low_signal_n1000_k10_cr | ✓ | ✓ | ✓ | 7.75e-07 | 9.81e-03 | 1.55e-06 | 8.44e-07 | 7.04e-08 | 3.73e-05 |
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 3.71e-07 | 4.78e-07 | 5.70e-07 | 3.40e-07 | 2.45e-07 | 1.18e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.13e-09 | 2.01e-06 | 1.19e-08 | 5.20e-09 | 1.91e-09 | 8.73e-07 |
| 1d_gaussian_smooth_n1000_k50_cr | ✓ | ✓ | ✓ | 5.76e-09 | 4.11e-07 | 1.66e-08 | 5.63e-09 | 2.46e-09 | 5.69e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 3.49e-10 | 4.11e-09 | 6.17e-10 | 3.97e-10 | 1.30e-10 | 1.64e-08 |
| 1d_gaussian_smooth_n2000_k30_cr | ✓ | ✓ | ✓ | 1.04e-07 | 8.28e-05 | 4.54e-07 | 9.29e-08 | 2.44e-08 | 9.20e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.38e-08 | 2.19e-07 | 1.19e-08 | 2.47e-08 | 1.10e-09 | 5.08e-06 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 2.34e-08 | 6.98e-07 | 6.55e-08 | 4.21e-08 | 3.06e-09 | 1.38e-06 |
| 1d_gaussian_smooth_n50_k10_cr | ✓ | ✓ | ✓ | 5.09e-09 | 8.03e-08 | 1.62e-08 | 5.53e-09 | 2.54e-09 | 1.31e-07 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 5.95e-10 | 1.36e-07 | 1.16e-09 | 3.00e-10 | 1.04e-10 | 4.16e-08 |
| 1d_gaussian_step_n500_k10_cr | ✓ | ✓ | ✓ | 2.57e-08 | 6.23e-07 | 3.92e-08 | 2.72e-08 | 3.92e-09 | 3.65e-06 |
| 1d_gaussian_weighted_n300_k10_cr | ✓ | ✓ | ✓ | 6.99e-10 | 3.70e-07 | 1.81e-09 | 5.41e-10 | 1.99e-01 | 7.52e-08 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 3.85e-10 | 8.18e-08 | 1.68e-09 | 4.09e-10 | 3.40e-11 | 5.52e-08 |
| 1d_poisson_log_n500_k10_cr | ✓ | ✓ | ✓ | 7.23e-06 | 1.79e-06 | 1.68e-05 | 1.85e-06 | 4.14e+00 | 4.77e-05 |
| 1d_poisson_weighted_n500_k10_cr | ✓ | ✓ | ✓ | 1.10e-05 | 1.63e-06 | 3.11e-05 | 1.79e-06 | 3.18e+00 | 6.25e-05 |
| 1d_scat_weighted_n300_k10_cr | ✗ | ✗ | ✓ | 4.42e-03 | 1.42e-01 | 1.40e-02 | 3.97e-03 | 9.63e-01 | 4.68e-03 |
| 1d_tw_log_n400_k20_cr | ✓ | ✓ | ✓ | 3.02e-06 | 1.79e-06 | 5.22e-06 | 1.97e-06 | 9.99e-01 | 2.88e-05 |
| 1d_tweedie_log_n400_k20_cr_p15 | ✓ | ✓ | ✓ | 2.28e-06 | 1.42e-06 | 4.20e-06 | 1.57e-06 | 1.04e+00 | 2.37e-05 |
| 2d_binomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 1.29e-05 | 3.70e-05 | 1.88e-05 | 4.96e-05 | 8.37e-01 | 7.30e-04 |
| 2d_binomial_logit_n200_k10_cr | ✓ | ✓ | ✓ | 4.61e-05 | 1.96e-04 | 2.60e-05 | 2.73e-04 | 8.39e-01 | 3.33e-01 |
| 2d_binomial_logit_n5000_k10_cr | ✓ | ✓ | ✓ | 8.66e-06 | 3.03e-05 | 1.03e-05 | 3.13e-05 | 8.37e-01 | 1.33e-01 |
| 2d_gamma_inverse_n1000_k10_cr | ✓ | ✓ | ✓ | 6.07e-04 | 1.64e-04 | 1.74e-04 | 9.42e-05 | 3.40e+00 | 1.79e-01 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 4.06e-05 | 7.55e-06 | 7.17e-05 | 6.72e-06 | 1.32e+01 | 3.05e-01 |
| 2d_gamma_log_n200_k10_cr | ✓ | ✓ | ✓ | 1.13e-04 | 3.46e-05 | 1.19e-04 | 2.76e-05 | 9.88e+00 | 5.10e-04 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.14e-07 | 5.91e-05 | 3.41e-07 | 1.16e-07 | 9.96e-09 | 2.05e-05 |
| 2d_gaussian_additive_n50000_k15_cr | ✓ | ✓ | ✓ | 2.98e-08 | 2.38e-05 | 7.46e-08 | 2.27e-08 | 1.70e-12 | 1.30e-04 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.42e-08 | 2.60e-06 | 4.17e-08 | 1.25e-08 | 1.30e-09 | 9.73e-07 |
| 2d_gaussian_bs_n1500_k15 | ✓ | ✓ | ✓ | 2.26e-07 | 4.20e-05 | 1.69e-06 | 8.22e-07 | 1.49e-08 | 9.24e-06 |
| 2d_invgauss_log_n800_k10_cr | ✓ | ✓ | ✓ | 2.29e-03 | 8.30e-04 | 3.23e-03 | 6.22e-04 | 9.16e+00 | 6.93e-03 |
| 2d_nb_log_n1000_k10_cr_theta2 | ✓ | ✓ | ✓ | 1.57e-03 | 1.76e-04 | 9.58e-04 | 1.38e-04 | 3.11e+01 | 6.96e-03 |
| 2d_nb_profile_log_n1000_k10_cr | ✓ | ✓ | ✓ | 4.24e-03 | 4.39e-04 | 2.27e-03 | 3.58e-04 | 3.04e+01 | 1.77e-02 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 1.61e-04 | 2.39e-05 | 4.39e-04 | 2.27e-05 | 5.45e+00 | 3.29e-01 |
| 2d_poisson_log_n200_k10_cr | ✓ | ✓ | ✓ | 4.96e-04 | 8.63e-05 | 2.65e-03 | 1.00e-04 | 5.15e+00 | 4.06e-01 |
| 2d_poisson_log_n5000_k10_cr | ✓ | ✓ | ✓ | 7.25e-04 | 4.77e-05 | 7.41e-04 | 4.08e-05 | 5.48e+00 | 5.86e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 1.07e-05 | 3.79e-05 | 1.36e-05 | 3.84e-05 | 8.36e-01 | 5.25e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.12e-03 | 4.30e-04 | 4.73e-03 | 2.83e-04 | 7.17e+00 | 7.45e-03 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 3.76e-06 | 3.73e-04 | 5.45e-06 | 3.26e-06 | 6.43e-07 | 1.07e-03 |
| 3d_poisson_log_n2000_k10_cr | ✓ | ✓ | ✓ | 1.30e-03 | 1.35e-04 | 1.37e-03 | 1.05e-04 | 3.69e+00 | 8.40e-02 |
| 4d_binomial_logit_n2000_k8_cr | ✓ | ✓ | ✓ | 6.24e-05 | 2.03e-04 | 5.78e-05 | 2.01e-04 | 8.32e-01 | 1.20e-02 |
| 4d_gamma_log_n2000_k8_cr | ✓ | ✓ | ✓ | 1.58e-04 | 4.04e-05 | 2.29e-04 | 2.12e-05 | 7.24e+00 | 3.13e-03 |
| 4d_gaussian_bs_n2000_k10 | ✓ | ✓ | ✓ | 3.39e-07 | 1.37e-04 | 5.55e-07 | 6.43e-07 | 6.57e-08 | 4.08e-01 |
| 4d_gaussian_correlated_extrap_slices_n800_k10_cr | ✓ | ✓ | ✓ | 5.25e-06 | 1.02e-04 | 1.31e-05 | 5.61e-06 | 9.78e-07 | 6.66e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 4.11e-08 | 1.11e-06 | 7.71e-08 | 3.04e-08 | 9.34e-10 | 3.25e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 1.21e-05 | 4.99e-03 | 1.74e-05 | 1.19e-05 | 2.28e-06 | 2.49e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.54e-08 | 1.04e-05 | 1.40e-08 | 7.16e-09 | 2.14e-09 | 2.66e-06 |
| 5d_sale_price_correlated_extrap_high_leverage_drops_n1200_k8_cr | ✓ | ✓ | ✓ | 3.57e-06 | 1.62e-03 | 5.50e-06 | 3.36e-06 | 4.72e-07 | 7.62e-01 |
| 5d_sale_price_correlated_extrap_no_drops_n1200_k8_cr | ✓ | ✓ | ✓ | 1.05e-05 | 1.86e-02 | 3.30e-05 | 1.35e-05 | 4.06e-06 | 8.67e-01 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 2.39e-07 | 4.33e-04 | 1.34e-06 | 1.06e-07 | 2.04e-08 | 9.93e-03 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 5.60e-07 | 4.75e-08 | 1.00e-06 | 5.31e-07 | 2.38e-08 | 3.12e-01 |
| 7d_neighbourhoods_compact_n3000 | ✓ | ✓ | ✓ | 6.67e-06 | 8.97e-05 | 9.78e-06 | 7.04e-06 | 1.61e-07 | 3.87e-01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 4.80e-04 | 8.36e-02 | 6.45e-04 | 2.77e-04 | 2.43e-05 | 9.92e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 5.64e-07 | 2.11e+00 | 2.11e+00 | 1.00e+00 |
| 1d_em_nb_seed123_n1000_k4_cr | 1.25e-01 | 5.66e+04 | 2.35e+04 | 4.15e-01 |
| 1d_em_nb_seed124_n1000_k4_cr | 2.49e-03 | 3.17e+05 | 2.67e+05 | 8.44e-01 |
| 1d_em_nb_seed125_n1000_k4_cr | 4.04e-03 | 2.37e+03 | 2.37e+03 | 9.97e-01 |
| 1d_em_nb_seed126_n1000_k4_cr | 1.94e-03 | 7.66e+05 | 4.48e+05 | 5.86e-01 |
| 1d_em_nb_seed127_n1000_k4_cr | 1.15e-02 | 2.84e+05 | 1.68e+05 | 5.93e-01 |
| 1d_em_nb_seed128_n1000_k4_cr | 1.50e-02 | 1.87e+05 | 1.20e+05 | 6.46e-01 |
| 1d_em_nb_seed129_n1000_k4_cr | 2.20e-02 | 2.45e+03 | 2.49e+03 | 1.02e+00 |
| 1d_em_nb_seed130_n1000_k4_cr | 2.24e-04 | 1.09e+03 | 1.09e+03 | 1.00e+00 |
| 1d_em_nb_seed131_n1000_k4_cr | 1.88e-03 | 3.75e+05 | 3.17e+05 | 8.44e-01 |
| 1d_em_nb_seed132_n1000_k4_cr | 4.02e-02 | 7.10e+03 | 7.57e+03 | 1.07e+00 |
| 1d_em_nb_seed133_n1000_k4_cr | 5.79e-03 | 3.92e+05 | 2.49e+05 | 6.35e-01 |
| 1d_em_nb_seed134_n1000_k4_cr | 1.66e-02 | 1.52e+05 | 9.75e+04 | 6.41e-01 |
| 1d_em_nb_seed135_n1000_k4_cr | 5.98e-03 | 1.16e+03 | 1.15e+03 | 9.96e-01 |
| 1d_em_nb_seed136_n1000_k4_cr | 9.62e-04 | 9.67e+02 | 9.66e+02 | 9.99e-01 |
| 1d_em_nb_seed137_n1000_k4_cr | 4.64e-02 | 9.66e+03 | 8.83e+03 | 9.14e-01 |
| 1d_em_nb_seed138_n1000_k4_cr | 2.24e-03 | 1.12e+03 | 1.12e+03 | 9.99e-01 |
| 1d_em_nb_seed139_n1000_k4_cr | 1.14e-03 | 2.36e+03 | 2.36e+03 | 1.00e+00 |
| 1d_em_nb_seed140_n1000_k4_cr | 3.13e-03 | 3.26e+05 | 2.67e+05 | 8.19e-01 |
| 1d_em_nb_seed141_n1000_k4_cr | 3.04e-03 | 1.99e+03 | 1.98e+03 | 9.98e-01 |
| 1d_em_nb_seed142_n1000_k4_cr | 3.35e-04 | 4.92e+05 | 4.53e+05 | 9.21e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 7.75e-07 | 2.37e+03 | 2.37e+03 | 1.00e+00 |
| 1d_gaussian_near_linear_n500_k10_cr | 3.71e-07 | 5.46e+07 | 2.51e+07 | 4.59e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4.13e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n1000_k50_cr | 5.76e-09 | 1.58e+03 | 1.58e+03 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 3.49e-10 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.04e-07 | 3.19e+02 | 3.19e+02 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.38e-08 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k20_bs | 2.34e-08 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n50_k10_cr | 5.09e-09 | 5.08e+00 | 5.08e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5.95e-10 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_step_n500_k10_cr | 2.57e-08 | 6.51e-01 | 6.51e-01 | 1.00e+00 |
| 1d_gaussian_weighted_n300_k10_cr | 6.99e-10 | 6.14e+00 | 6.14e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 3.85e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 1d_poisson_log_n500_k10_cr | 7.23e-06 | 1.60e+02 | 1.60e+02 | 1.00e+00 |
| 1d_poisson_weighted_n500_k10_cr | 1.10e-05 | 9.52e+01 | 9.52e+01 | 1.00e+00 |
| 1d_tw_log_n400_k20_cr | 3.02e-06 | 1.28e+03 | 1.28e+03 | 1.00e+00 |
| 1d_tweedie_log_n400_k20_cr_p15 | 2.28e-06 | 1.21e+03 | 1.21e+03 | 1.00e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.08e-05 | 2.37e+01 | 2.37e+01 | 1.00e+00 |
| 2d_binomial_logit_n200_k10_cr | 4.61e-05 | 7.17e+00 | 7.17e+00 | 9.99e-01 |
| 2d_binomial_logit_n5000_k10_cr | 1.28e-06 | 3.17e+01 | 3.17e+01 | 1.00e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 6.07e-04 | 3.16e+02 | 3.13e+02 | 9.93e-01 |
| 2d_gamma_log_n1000_k10_cr | 4.06e-05 | 7.65e+01 | 7.65e+01 | 1.00e+00 |
| 2d_gamma_log_n200_k10_cr | 1.13e-04 | 7.56e+01 | 7.56e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.14e-07 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n50000_k15_cr | 2.98e-08 | 3.65e+01 | 3.65e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.42e-08 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 2d_gaussian_bs_n1500_k15 | 2.26e-07 | 4.42e+00 | 4.42e+00 | 1.00e+00 |
| 2d_invgauss_log_n800_k10_cr | 2.29e-03 | 1.93e+02 | 1.92e+02 | 9.97e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 1.57e-03 | 8.85e+01 | 8.84e+01 | 1.00e+00 |
| 2d_nb_profile_log_n1000_k10_cr | 4.24e-03 | 8.77e+01 | 8.77e+01 | 1.00e+00 |
| 2d_poisson_log_n1000_k10_cr | 1.61e-04 | 1.63e+02 | 1.63e+02 | 9.99e-01 |
| 2d_poisson_log_n200_k10_cr | 4.96e-04 | 1.40e+02 | 1.40e+02 | 9.98e-01 |
| 2d_poisson_log_n5000_k10_cr | 7.25e-04 | 2.14e+02 | 2.13e+02 | 1.00e+00 |
| 2d_quasibinomial_logit_n1000_k10_cr | 6.71e-06 | 2.19e+01 | 2.19e+01 | 1.00e+00 |
| 2d_quasipoisson_log_n1000_k10_cr | 3.12e-03 | 3.93e+02 | 3.91e+02 | 9.94e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 3.76e-06 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 3d_poisson_log_n2000_k10_cr | 1.30e-03 | 2.52e+02 | 2.52e+02 | 9.99e-01 |
| 4d_binomial_logit_n2000_k8_cr | 6.24e-05 | 1.82e+01 | 1.82e+01 | 1.00e+00 |
| 4d_gamma_log_n2000_k8_cr | 1.58e-04 | 4.15e+01 | 4.15e+01 | 1.00e+00 |
| 4d_gaussian_bs_n2000_k10 | 3.39e-07 | 7.62e-01 | 7.62e-01 | 1.00e+00 |
| 4d_gaussian_correlated_extrap_slices_n800_k10_cr | 5.25e-06 | 3.55e+00 | 3.55e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.11e-08 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 1.21e-05 | 1.83e+08 | 2.16e+08 | 1.18e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.54e-08 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_sale_price_correlated_extrap_high_leverage_drops_n1200_k8_cr | 3.57e-06 | 3.96e+06 | 3.69e+06 | 9.33e-01 |
| 5d_sale_price_correlated_extrap_no_drops_n1200_k8_cr | 1.05e-05 | 4.93e+06 | 3.72e+07 | 7.54e+00 |
| 5d_skewed_features_n5000 | 2.39e-07 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 5.60e-07 | 4.55e+00 | 4.55e+00 | 1.00e+00 |
| 7d_neighbourhoods_compact_n3000 | 6.67e-06 | 9.87e+04 | 9.88e+04 | 1.00e+00 |
| 8d_neighbourhoods_like_n15000 | 4.80e-04 | 1.48e+07 | 1.96e+09 | 1.32e+02 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 14 | 11 | 1 | -4.03e+02 | -4.03e+02 |
| 1d_em_nb_seed123_n1000_k4_cr | 7 | 4 | 1 | 4.80e+03 | 4.80e+03 |
| 1d_em_nb_seed124_n1000_k4_cr | 7 | 5 | 1 | 4.85e+03 | 4.85e+03 |
| 1d_em_nb_seed125_n1000_k4_cr | 4 | 3 | 1 | 4.80e+03 | 4.80e+03 |
| 1d_em_nb_seed126_n1000_k4_cr | 7 | 5 | 1 | 4.80e+03 | 4.80e+03 |
| 1d_em_nb_seed127_n1000_k4_cr | 7 | 5 | 1 | 4.81e+03 | 4.81e+03 |
| 1d_em_nb_seed128_n1000_k4_cr | 7 | 5 | 1 | 4.83e+03 | 4.83e+03 |
| 1d_em_nb_seed129_n1000_k4_cr | 3 | 3 | 1 | 4.80e+03 | 4.80e+03 |
| 1d_em_nb_seed130_n1000_k4_cr | 4 | 2 | 1 | 4.70e+03 | 4.70e+03 |
| 1d_em_nb_seed131_n1000_k4_cr | 7 | 5 | 1 | 4.63e+03 | 4.63e+03 |
| 1d_em_nb_seed132_n1000_k4_cr | 5 | 4 | 1 | 4.80e+03 | 4.80e+03 |
| 1d_em_nb_seed133_n1000_k4_cr | 7 | 5 | 1 | 4.71e+03 | 4.71e+03 |
| 1d_em_nb_seed134_n1000_k4_cr | 7 | 5 | 1 | 4.73e+03 | 4.73e+03 |
| 1d_em_nb_seed135_n1000_k4_cr | 5 | 2 | 1 | 4.76e+03 | 4.76e+03 |
| 1d_em_nb_seed136_n1000_k4_cr | 5 | 2 | 1 | 4.69e+03 | 4.69e+03 |
| 1d_em_nb_seed137_n1000_k4_cr | 6 | 4 | 1 | 4.79e+03 | 4.79e+03 |
| 1d_em_nb_seed138_n1000_k4_cr | 5 | 2 | 1 | 4.75e+03 | 4.76e+03 |
| 1d_em_nb_seed139_n1000_k4_cr | 5 | 3 | 1 | 4.96e+03 | 4.96e+03 |
| 1d_em_nb_seed140_n1000_k4_cr | 7 | 5 | 1 | 4.82e+03 | 4.82e+03 |
| 1d_em_nb_seed141_n1000_k4_cr | 5 | 3 | 1 | 4.72e+03 | 4.72e+03 |
| 1d_em_nb_seed142_n1000_k4_cr | 7 | 5 | 1 | 4.75e+03 | 4.75e+03 |
| 1d_gaussian_low_signal_n1000_k10_cr | 6 | 7 | 1 | 7.55e+02 | 7.55e+02 |
| 1d_gaussian_near_linear_n500_k10_cr | 11 | 8 | 1 | -4.45e+02 | -4.45e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4 | 4 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n1000_k50_cr | 7 | 4 | 1 | -2.13e+02 | -2.13e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 4 | 1 | -7.72e+00 | -7.72e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 6 | 4 | 1 | -3.54e+02 | -3.54e+02 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 5 | 1 | -6.37e+01 | -6.37e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 4 | 4 | 1 | -6.33e+01 | -6.33e+01 |
| 1d_gaussian_smooth_n50_k10_cr | 5 | 3 | 1 | 6.47e+00 | 6.47e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 4 | 1 | -1.69e+02 | -1.69e+02 |
| 1d_gaussian_step_n500_k10_cr | 4 | 5 | — | -2.54e+02 | -2.54e+02 |
| 1d_gaussian_weighted_n300_k10_cr | 5 | 5 | 1 | -4.85e+01 | -4.77e+01 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 6 | — | -1.74e+02 | -1.74e+02 |
| 1d_poisson_log_n500_k10_cr | 4 | 3 | 1 | 1.10e+03 | 1.10e+03 |
| 1d_poisson_weighted_n500_k10_cr | 5 | 4 | 1 | 1.11e+03 | 1.38e+03 |
| 1d_scat_weighted_n300_k10_cr | 7 | 9 | 1 | 1.58e+01 | 1.45e+01 |
| 1d_tw_log_n400_k20_cr | 6 | 7 | 1 | 6.51e+02 | 6.51e+02 |
| 1d_tweedie_log_n400_k20_cr_p15 | 6 | 6 | — | 6.51e+02 | 6.53e+02 |
| 2d_binomial_logit_n1000_k10_cr | 7 | 4 | 1 | 5.23e+02 | 5.23e+02 |
| 2d_binomial_logit_n200_k10_cr | 10 | 9 | 1 | 9.58e+01 | 9.58e+01 |
| 2d_binomial_logit_n5000_k10_cr | 12 | 7 | 1 | 2.53e+03 | 2.53e+03 |
| 2d_gamma_inverse_n1000_k10_cr | 11 | 7 | — | 1.28e+03 | 1.28e+03 |
| 2d_gamma_log_n1000_k10_cr | 10 | 8 | 1 | 1.84e+03 | 1.83e+03 |
| 2d_gamma_log_n200_k10_cr | 7 | 5 | 1 | 3.58e+02 | 3.54e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 5 | 4 | 1 | -3.32e+02 | -3.32e+02 |
| 2d_gaussian_additive_n50000_k15_cr | 4 | 7 | 1 | -9.25e+03 | -9.25e+03 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 5 | 1 | -5.68e+01 | -5.68e+01 |
| 2d_gaussian_bs_n1500_k15 | 4 | 5 | — | -2.47e+02 | -2.47e+02 |
| 2d_invgauss_log_n800_k10_cr | 6 | 6 | 1 | 1.19e+03 | 1.19e+03 |
| 2d_nb_log_n1000_k10_cr_theta2 | 7 | 5 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_nb_profile_log_n1000_k10_cr | 7 | 4 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_poisson_log_n1000_k10_cr | 9 | 7 | 1 | 2.30e+03 | 2.30e+03 |
| 2d_poisson_log_n200_k10_cr | 11 | 9 | 1 | 4.36e+02 | 4.36e+02 |
| 2d_poisson_log_n5000_k10_cr | 9 | 6 | 1 | 1.14e+04 | 1.14e+04 |
| 2d_quasibinomial_logit_n1000_k10_cr | 10 | 10 | 1 | 1.44e+03 | 5.19e+02 |
| 2d_quasipoisson_log_n1000_k10_cr | 6 | 6 | 1 | 2.09e+03 | 1.17e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 8 | 6 | 1 | -1.14e+02 | -1.14e+02 |
| 3d_poisson_log_n2000_k10_cr | 10 | 6 | 1 | 4.31e+03 | 4.31e+03 |
| 4d_binomial_logit_n2000_k8_cr | 9 | 6 | 1 | 1.15e+03 | 1.15e+03 |
| 4d_gamma_log_n2000_k8_cr | 9 | 7 | 1 | 3.34e+03 | 3.33e+03 |
| 4d_gaussian_bs_n2000_k10 | 11 | 9 | 1 | -3.56e+02 | -3.56e+02 |
| 4d_gaussian_correlated_extrap_slices_n800_k10_cr | 11 | 10 | 1 | -4.76e+02 | -4.76e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 13 | 10 | 1 | -1.46e+02 | -1.46e+02 |
| 4d_small_neighbourhood_n300 | 13 | 10 | 1 | -1.30e+02 | -1.30e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.01e+02 |
| 5d_sale_price_correlated_extrap_high_leverage_drops_n1200_k8_cr | 10 | 8 | 1 | -1.45e+03 | -1.45e+03 |
| 5d_sale_price_correlated_extrap_no_drops_n1200_k8_cr | 10 | 8 | 1 | -1.42e+03 | -1.42e+03 |
| 5d_skewed_features_n5000 | 11 | 7 | 1 | -4.35e+03 | -4.35e+03 |
| 6d_heatmap_pricing_n8000 | 10 | 6 | 1 | -6.79e+03 | -6.79e+03 |
| 7d_neighbourhoods_compact_n3000 | 10 | 7 | 1 | -2.55e+03 | -2.55e+03 |
| 8d_neighbourhoods_like_n15000 | 10 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 5.65e+01 | 5.46e+01 | 3.98e+02 | 1.42e-01 |
| 1d_em_nb_seed123_n1000_k4_cr | 2.18e+01 | 2.16e+01 | 2.29e+02 | 9.52e-02 |
| 1d_em_nb_seed124_n1000_k4_cr | 2.57e+01 | 2.38e+01 | 1.78e+02 | 1.44e-01 |
| 1d_em_nb_seed125_n1000_k4_cr | 1.11e+01 | 1.01e+01 | 1.87e+02 | 5.91e-02 |
| 1d_em_nb_seed126_n1000_k4_cr | 2.51e+01 | 2.34e+01 | 1.79e+02 | 1.41e-01 |
| 1d_em_nb_seed127_n1000_k4_cr | 2.52e+01 | 2.37e+01 | 2.15e+02 | 1.17e-01 |
| 1d_em_nb_seed128_n1000_k4_cr | 2.01e+01 | 1.87e+01 | 1.62e+02 | 1.24e-01 |
| 1d_em_nb_seed129_n1000_k4_cr | 1.12e+01 | 1.11e+01 | 1.65e+02 | 6.80e-02 |
| 1d_em_nb_seed130_n1000_k4_cr | 1.46e+01 | 1.26e+01 | 1.61e+02 | 9.04e-02 |
| 1d_em_nb_seed131_n1000_k4_cr | 2.56e+01 | 2.15e+01 | 1.31e+02 | 1.96e-01 |
| 1d_em_nb_seed132_n1000_k4_cr | 1.37e+01 | 1.20e+01 | 1.30e+02 | 1.05e-01 |
| 1d_em_nb_seed133_n1000_k4_cr | 2.54e+01 | 2.52e+01 | 1.77e+02 | 1.44e-01 |
| 1d_em_nb_seed134_n1000_k4_cr | 2.55e+01 | 1.54e+01 | 1.41e+02 | 1.81e-01 |
| 1d_em_nb_seed135_n1000_k4_cr | 1.94e+01 | 1.72e+01 | 1.25e+02 | 1.55e-01 |
| 1d_em_nb_seed136_n1000_k4_cr | 1.64e+01 | 1.36e+01 | 1.55e+02 | 1.05e-01 |
| 1d_em_nb_seed137_n1000_k4_cr | 1.78e+01 | 1.68e+01 | 1.20e+02 | 1.48e-01 |
| 1d_em_nb_seed138_n1000_k4_cr | 1.15e+01 | 1.08e+01 | 1.27e+02 | 9.05e-02 |
| 1d_em_nb_seed139_n1000_k4_cr | 1.16e+01 | 1.08e+01 | 1.37e+02 | 8.49e-02 |
| 1d_em_nb_seed140_n1000_k4_cr | 1.69e+01 | 1.51e+01 | 1.37e+02 | 1.23e-01 |
| 1d_em_nb_seed141_n1000_k4_cr | 1.47e+01 | 1.24e+01 | 1.38e+02 | 1.06e-01 |
| 1d_em_nb_seed142_n1000_k4_cr | 1.75e+01 | 1.58e+01 | 1.27e+02 | 1.39e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.49e+00 | 1.24e+00 | 7.38e+01 | 2.02e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.98e+00 | 2.38e+00 | 4.60e+01 | 6.48e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 1.11e+00 | 9.69e-01 | 2.87e+01 | 3.87e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 3.84e+01 | 3.57e+01 | 1.09e+02 | 3.51e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 7.11e-01 | 6.94e-01 | 2.80e+01 | 2.54e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.40e+01 | 1.07e+01 | 5.56e+01 | 2.51e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 9.55e-01 | 9.42e-01 | 3.20e+01 | 2.98e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 4.97e+00 | 4.81e+00 | 3.99e+01 | 1.25e-01 |
| 1d_gaussian_smooth_n50_k10_cr | 6.83e-01 | 6.67e-01 | 2.70e+01 | 2.52e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 9.93e-01 | 8.27e-01 | 3.39e+01 | 2.93e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.96e-01 | 7.66e-01 | 2.72e+01 | 2.93e-02 |
| 1d_gaussian_weighted_n300_k10_cr | 1.42e+00 | 1.35e+00 | 3.79e+01 | 3.74e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 3.19e+00 | 2.93e+00 | 4.11e+01 | 7.77e-02 |
| 1d_poisson_log_n500_k10_cr | 6.35e+00 | 5.89e+00 | 9.41e+01 | 6.75e-02 |
| 1d_poisson_weighted_n500_k10_cr | 8.33e+00 | 6.98e+00 | 9.91e+01 | 8.40e-02 |
| 1d_scat_weighted_n300_k10_cr | 1.65e+01 | 1.49e+01 | 3.33e+02 | 4.96e-02 |
| 1d_tw_log_n400_k20_cr | 2.34e+02 | 2.26e+02 | 3.19e+02 | 7.32e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 1.19e+02 | 1.18e+02 | 1.08e+02 | 1.10e+00 |
| 2d_binomial_logit_n1000_k10_cr | 5.00e+01 | 4.80e+01 | 1.18e+02 | 4.23e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.27e+01 | 1.16e+01 | 1.82e+02 | 6.96e-02 |
| 2d_binomial_logit_n5000_k10_cr | 5.08e+02 | 4.64e+02 | 1.78e+02 | 2.85e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.38e+01 | 3.17e+01 | 2.51e+02 | 1.35e-01 |
| 2d_gamma_log_n1000_k10_cr | 4.04e+01 | 3.68e+01 | 1.82e+02 | 2.23e-01 |
| 2d_gamma_log_n200_k10_cr | 7.38e+00 | 7.05e+00 | 1.87e+02 | 3.95e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 5.08e+00 | 4.55e+00 | 3.98e+01 | 1.28e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 1.06e+02 | 1.02e+02 | 2.77e+02 | 3.84e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.58e+00 | 1.48e+00 | 2.70e+01 | 5.84e-02 |
| 2d_gaussian_bs_n1500_k15 | 5.27e+00 | 5.16e+00 | 4.04e+01 | 1.30e-01 |
| 2d_invgauss_log_n800_k10_cr | 6.82e+01 | 6.41e+01 | 2.07e+02 | 3.29e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 4.21e+01 | 3.51e+01 | 1.64e+02 | 2.57e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 3.71e+01 | 3.59e+01 | 2.00e+02 | 1.85e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.32e+01 | 3.17e+01 | 1.24e+02 | 2.68e-01 |
| 2d_poisson_log_n200_k10_cr | 1.12e+01 | 1.00e+01 | 1.40e+02 | 8.01e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.70e+02 | 1.54e+02 | 1.87e+02 | 9.11e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 7.14e+01 | 6.63e+01 | 1.65e+02 | 4.34e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 3.76e+01 | 3.51e+01 | 2.28e+02 | 1.65e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.95e+00 | 6.16e+00 | 9.64e+01 | 7.21e-02 |
| 3d_poisson_log_n2000_k10_cr | 1.44e+02 | 1.32e+02 | 2.59e+02 | 5.58e-01 |
| 4d_binomial_logit_n2000_k8_cr | 1.63e+02 | 1.54e+02 | 2.01e+02 | 8.13e-01 |
| 4d_gamma_log_n2000_k8_cr | 9.57e+01 | 7.97e+01 | 2.87e+02 | 3.33e-01 |
| 4d_gaussian_bs_n2000_k10 | 1.61e+01 | 1.39e+01 | 8.87e+01 | 1.81e-01 |
| 4d_gaussian_correlated_extrap_slices_n800_k10_cr | 8.42e+00 | 8.10e+00 | 6.80e+01 | 1.24e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.11e+01 | 1.02e+01 | 6.96e+01 | 1.60e-01 |
| 4d_small_neighbourhood_n300 | 5.40e+00 | 5.15e+00 | 5.20e+01 | 1.04e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 7.31e+00 | 7.00e+00 | 1.05e+02 | 6.96e-02 |
| 5d_sale_price_correlated_extrap_high_leverage_drops_n1200_k8_cr | 9.21e+00 | 8.60e+00 | 7.49e+01 | 1.23e-01 |
| 5d_sale_price_correlated_extrap_no_drops_n1200_k8_cr | 7.78e+00 | 7.68e+00 | 7.43e+01 | 1.05e-01 |
| 5d_skewed_features_n5000 | 2.37e+01 | 2.30e+01 | 1.19e+02 | 1.99e-01 |
| 6d_heatmap_pricing_n8000 | 4.11e+01 | 4.09e+01 | 2.21e+02 | 1.86e-01 |
| 7d_neighbourhoods_compact_n3000 | 1.93e+01 | 1.84e+01 | 1.32e+02 | 1.46e-01 |
| 8d_neighbourhoods_like_n15000 | 9.23e+01 | 8.92e+01 | 6.31e+02 | 1.46e-01 |
