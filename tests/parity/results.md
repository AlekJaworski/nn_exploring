# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | ✓ | ✓ | ✓ | 4.06e-06 | 4.53e-04 | 6.18e-06 | 1.52e-06 | 2.40e-08 | 9.27e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | ✓ | ✓ | ✓ | 7.75e-07 | 9.81e-03 | 1.55e-06 | 8.44e-07 | 7.04e-08 | 3.73e-05 |
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 5.70e-07 | 7.35e-07 | 8.75e-07 | 5.23e-07 | 3.76e-07 | 4.92e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.13e-09 | 2.01e-06 | 1.19e-08 | 5.20e-09 | 1.91e-09 | 8.73e-07 |
| 1d_gaussian_smooth_n1000_k50_cr | ✓ | ✓ | ✓ | 5.76e-09 | 4.11e-07 | 1.66e-08 | 5.63e-09 | 2.46e-09 | 5.69e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 3.49e-10 | 4.11e-09 | 6.17e-10 | 3.97e-10 | 1.30e-10 | 1.64e-08 |
| 1d_gaussian_smooth_n2000_k30_cr | ✓ | ✓ | ✓ | 1.04e-07 | 8.28e-05 | 4.54e-07 | 9.29e-08 | 2.44e-08 | 9.20e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.38e-08 | 2.19e-07 | 1.19e-08 | 2.47e-08 | 1.10e-09 | 5.08e-06 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 1.02e-10 | 3.05e-09 | 3.93e-10 | 3.32e-10 | 7.65e-12 | 2.78e-09 |
| 1d_gaussian_smooth_n50_k10_cr | ✓ | ✓ | ✓ | 5.09e-09 | 8.03e-08 | 1.62e-08 | 5.53e-09 | 2.54e-09 | 1.31e-07 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 5.95e-10 | 1.36e-07 | 1.16e-09 | 3.00e-10 | 1.04e-10 | 4.16e-08 |
| 1d_gaussian_step_n500_k10_cr | ✓ | ✓ | ✓ | 2.57e-08 | 6.23e-07 | 3.92e-08 | 2.72e-08 | 3.92e-09 | 3.65e-06 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 3.85e-10 | 8.18e-08 | 1.68e-09 | 4.09e-10 | 3.40e-11 | 5.52e-08 |
| 1d_poisson_log_n500_k10_cr | ✓ | ✓ | ✓ | 7.23e-06 | 1.79e-06 | 1.68e-05 | 1.85e-06 | 4.14e+00 | 4.77e-05 |
| 1d_tw_log_n400_k20_cr | ✓ | ✓ | ✓ | 2.26e-04 | 1.86e-04 | 5.51e-04 | 2.11e-04 | 9.99e-01 | 3.13e-03 |
| 1d_tweedie_log_n400_k20_cr_p15 | ✓ | ✓ | ✓ | 4.03e-04 | 2.51e-04 | 7.44e-04 | 2.77e-04 | 1.04e+00 | 4.20e-03 |
| 2d_binomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 4.49e-05 | 1.31e-04 | 6.75e-05 | 1.74e-04 | 8.37e-01 | 2.56e-03 |
| 2d_binomial_logit_n200_k10_cr | ✓ | ✓ | ✓ | 4.75e-05 | 1.99e-04 | 2.72e-05 | 2.81e-04 | 8.39e-01 | 8.84e+00 |
| 2d_binomial_logit_n5000_k10_cr | ✓ | ✓ | ✓ | 5.98e-05 | 2.12e-04 | 7.14e-05 | 2.16e-04 | 8.37e-01 | 4.24e+00 |
| 2d_gamma_inverse_n1000_k10_cr | ✓ | ✓ | ✓ | 6.03e-04 | 1.62e-04 | 1.74e-04 | 9.24e-05 | 3.40e+00 | 1.79e-01 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 2.27e-04 | 4.39e-05 | 4.55e-04 | 2.47e-05 | 1.32e+01 | 7.86e+00 |
| 2d_gamma_log_n200_k10_cr | ✓ | ✓ | ✓ | 1.83e-03 | 5.58e-04 | 5.62e-04 | 3.10e-04 | 9.88e+00 | 5.74e-03 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 9.29e-08 | 4.89e-05 | 2.80e-07 | 9.45e-08 | 8.22e-09 | 1.68e-05 |
| 2d_gaussian_additive_n50000_k15_cr | ✓ | ✓ | ✓ | 2.27e-08 | 2.57e-05 | 6.63e-08 | 2.34e-08 | 1.53e-11 | 1.34e-04 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.66e-07 | 1.61e-05 | 3.84e-07 | 1.40e-07 | 1.06e-08 | 1.13e-05 |
| 2d_gaussian_bs_n1500_k15 | ✓ | ✓ | ✓ | 2.20e-07 | 7.05e-05 | 1.64e-06 | 8.99e-07 | 9.09e-09 | 1.01e-05 |
| 2d_invgauss_log_n800_k10_cr | ✓ | ✓ | ✓ | 1.72e-03 | 4.92e-04 | 1.68e-03 | 2.86e-04 | 9.16e+00 | 3.67e-03 |
| 2d_nb_log_n1000_k10_cr_theta2 | ✓ | ✓ | ✓ | 3.57e-03 | 4.26e-04 | 2.29e-03 | 2.82e-04 | 3.11e+01 | 1.44e-02 |
| 2d_nb_profile_log_n1000_k10_cr | ✓ | ✓ | ✓ | 4.76e-03 | 5.43e-04 | 1.05e-02 | 4.37e-04 | 3.04e+01 | 3.00e-03 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 1.65e-04 | 2.49e-05 | 3.04e-04 | 2.29e-05 | 5.45e+00 | 3.97e+00 |
| 2d_poisson_log_n200_k10_cr | ✓ | ✓ | ✓ | 6.30e-04 | 8.19e-05 | 3.11e-03 | 9.90e-05 | 5.15e+00 | 3.39e+00 |
| 2d_poisson_log_n5000_k10_cr | ✓ | ✓ | ✓ | 5.97e-05 | 3.90e-06 | 7.38e-05 | 3.30e-06 | 5.48e+00 | 1.29e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 3.75e-06 | 1.32e-05 | 4.60e-06 | 1.34e-05 | 8.36e-01 | 6.30e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.12e-03 | 4.30e-04 | 4.73e-03 | 2.83e-04 | 7.17e+00 | 7.46e-03 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 9.83e-07 | 1.45e-04 | 1.39e-06 | 8.31e-07 | 1.40e-07 | 2.72e-04 |
| 3d_poisson_log_n2000_k10_cr | ✓ | ✓ | ✓ | 1.32e-03 | 1.64e-04 | 8.30e-04 | 1.06e-04 | 3.69e+00 | 1.48e+00 |
| 4d_binomial_logit_n2000_k8_cr | ✓ | ✓ | ✓ | 4.67e-05 | 1.38e-04 | 2.89e-05 | 1.25e-04 | 8.32e-01 | 7.48e-03 |
| 4d_gamma_log_n2000_k8_cr | ✓ | ✓ | ✓ | 2.03e-03 | 3.97e-04 | 2.30e-03 | 2.60e-04 | 7.24e+00 | 3.79e-02 |
| 4d_gaussian_bs_n2000_k10 | ✓ | ✓ | ✓ | 6.59e-07 | 8.61e-05 | 1.30e-06 | 9.90e-07 | 3.28e-08 | 3.38e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.25e-07 | 7.34e-06 | 2.35e-07 | 1.41e-07 | 3.19e-08 | 3.25e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 1.18e-05 | 4.93e-03 | 1.73e-05 | 1.16e-05 | 2.14e-06 | 5.41e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.25e-07 | 1.00e-04 | 1.45e-07 | 1.21e-07 | 2.17e-08 | 1.62e-05 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 2.67e-05 | 3.83e-02 | 3.44e-05 | 2.71e-05 | 2.44e-06 | 1.13e-02 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 1.06e-04 | 7.18e-06 | 1.39e-04 | 1.03e-04 | 3.22e-06 | 3.12e-01 |
| 7d_neighbourhoods_compact_n3000 | ✓ | ✓ | ✓ | 4.14e-05 | 1.24e-03 | 5.79e-05 | 2.78e-05 | 8.14e-06 | 4.91e-01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 4.88e-04 | 8.55e-02 | 6.98e-04 | 2.77e-04 | 2.48e-05 | 9.92e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 4.06e-06 | 2.11e+00 | 2.11e+00 | 1.00e+00 |
| 1d_gaussian_low_signal_n1000_k10_cr | 7.75e-07 | 2.37e+03 | 2.37e+03 | 1.00e+00 |
| 1d_gaussian_near_linear_n500_k10_cr | 5.70e-07 | 1.49e+08 | 2.51e+07 | 1.69e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4.13e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n1000_k50_cr | 5.76e-09 | 1.58e+03 | 1.58e+03 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 3.49e-10 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.04e-07 | 3.19e+02 | 3.19e+02 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.38e-08 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k20_bs | 1.02e-10 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n50_k10_cr | 5.09e-09 | 5.08e+00 | 5.08e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5.95e-10 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_step_n500_k10_cr | 2.57e-08 | 6.51e-01 | 6.51e-01 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 3.85e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 1d_poisson_log_n500_k10_cr | 7.23e-06 | 1.60e+02 | 1.60e+02 | 1.00e+00 |
| 1d_tw_log_n400_k20_cr | 2.26e-04 | 1.29e+03 | 1.28e+03 | 9.97e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 4.03e-04 | 1.22e+03 | 1.21e+03 | 9.96e-01 |
| 2d_binomial_logit_n1000_k10_cr | 4.49e-05 | 2.37e+01 | 2.37e+01 | 1.00e+00 |
| 2d_binomial_logit_n200_k10_cr | 4.75e-05 | 7.17e+00 | 7.17e+00 | 9.99e-01 |
| 2d_binomial_logit_n5000_k10_cr | 5.98e-05 | 3.17e+01 | 3.17e+01 | 1.00e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 6.03e-04 | 3.15e+02 | 3.13e+02 | 9.94e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.27e-04 | 7.66e+01 | 7.65e+01 | 9.99e-01 |
| 2d_gamma_log_n200_k10_cr | 1.83e-03 | 7.52e+01 | 7.56e+01 | 1.01e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 9.29e-08 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n50000_k15_cr | 2.27e-08 | 3.65e+01 | 3.65e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.66e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 2d_gaussian_bs_n1500_k15 | 2.20e-07 | 4.42e+00 | 4.42e+00 | 1.00e+00 |
| 2d_invgauss_log_n800_k10_cr | 1.72e-03 | 1.91e+02 | 1.92e+02 | 1.00e+00 |
| 2d_nb_log_n1000_k10_cr_theta2 | 3.57e-03 | 8.87e+01 | 8.84e+01 | 9.97e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 4.76e-03 | 8.76e+01 | 8.77e+01 | 1.00e+00 |
| 2d_poisson_log_n1000_k10_cr | 1.65e-04 | 1.63e+02 | 1.63e+02 | 9.99e-01 |
| 2d_poisson_log_n200_k10_cr | 6.30e-04 | 1.40e+02 | 1.40e+02 | 9.98e-01 |
| 2d_poisson_log_n5000_k10_cr | 5.97e-05 | 2.14e+02 | 2.13e+02 | 1.00e+00 |
| 2d_quasibinomial_logit_n1000_k10_cr | 3.75e-06 | 2.19e+01 | 2.19e+01 | 1.00e+00 |
| 2d_quasipoisson_log_n1000_k10_cr | 3.12e-03 | 3.93e+02 | 3.91e+02 | 9.94e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 9.83e-07 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 3d_poisson_log_n2000_k10_cr | 1.32e-03 | 2.52e+02 | 2.52e+02 | 9.99e-01 |
| 4d_binomial_logit_n2000_k8_cr | 4.67e-05 | 1.82e+01 | 1.82e+01 | 1.00e+00 |
| 4d_gamma_log_n2000_k8_cr | 2.03e-03 | 4.14e+01 | 4.15e+01 | 1.00e+00 |
| 4d_gaussian_bs_n2000_k10 | 6.59e-07 | 7.62e-01 | 7.62e-01 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.25e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 1.18e-05 | 1.83e+08 | 2.16e+08 | 1.18e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.25e-07 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 2.67e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 1.06e-04 | 4.55e+00 | 4.55e+00 | 1.00e+00 |
| 7d_neighbourhoods_compact_n3000 | 4.14e-05 | 9.57e+04 | 9.88e+04 | 1.03e+00 |
| 8d_neighbourhoods_like_n15000 | 4.88e-04 | 1.48e+07 | 1.96e+09 | 1.32e+02 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 14 | 11 | 1 | -4.03e+02 | -4.03e+02 |
| 1d_gaussian_low_signal_n1000_k10_cr | 6 | 7 | 1 | 7.55e+02 | 7.55e+02 |
| 1d_gaussian_near_linear_n500_k10_cr | 12 | 8 | 1 | -4.45e+02 | -4.45e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4 | 4 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n1000_k50_cr | 7 | 4 | 1 | -2.13e+02 | -2.13e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 4 | 1 | -7.72e+00 | -7.72e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 6 | 4 | 1 | -3.54e+02 | -3.54e+02 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 5 | 1 | -6.37e+01 | -6.37e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 5 | 4 | 1 | -6.33e+01 | -6.33e+01 |
| 1d_gaussian_smooth_n50_k10_cr | 5 | 3 | 1 | 6.47e+00 | 6.47e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 4 | 1 | -1.69e+02 | -1.69e+02 |
| 1d_gaussian_step_n500_k10_cr | 4 | 5 | — | -2.54e+02 | -2.54e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 6 | — | -1.74e+02 | -1.74e+02 |
| 1d_poisson_log_n500_k10_cr | 4 | 3 | 1 | 1.10e+03 | 1.10e+03 |
| 1d_tw_log_n400_k20_cr | 6 | 7 | 1 | 6.51e+02 | 6.51e+02 |
| 1d_tweedie_log_n400_k20_cr_p15 | 6 | 6 | — | 6.51e+02 | 6.53e+02 |
| 2d_binomial_logit_n1000_k10_cr | 8 | 4 | 1 | 5.23e+02 | 5.23e+02 |
| 2d_binomial_logit_n200_k10_cr | 12 | 9 | 1 | 9.58e+01 | 9.58e+01 |
| 2d_binomial_logit_n5000_k10_cr | 13 | 7 | 1 | 2.53e+03 | 2.53e+03 |
| 2d_gamma_inverse_n1000_k10_cr | 11 | 7 | — | 1.28e+03 | 1.28e+03 |
| 2d_gamma_log_n1000_k10_cr | 12 | 8 | 1 | 1.84e+03 | 1.83e+03 |
| 2d_gamma_log_n200_k10_cr | 7 | 5 | 1 | 3.58e+02 | 3.54e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 5 | 4 | 1 | -3.32e+02 | -3.32e+02 |
| 2d_gaussian_additive_n50000_k15_cr | 5 | 7 | 1 | -9.25e+03 | -9.25e+03 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 5 | 1 | -5.68e+01 | -5.68e+01 |
| 2d_gaussian_bs_n1500_k15 | 5 | 5 | — | -2.47e+02 | -2.47e+02 |
| 2d_invgauss_log_n800_k10_cr | 4 | 6 | 1 | 1.19e+03 | 1.19e+03 |
| 2d_nb_log_n1000_k10_cr_theta2 | 8 | 5 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_nb_profile_log_n1000_k10_cr | 8 | 4 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_poisson_log_n1000_k10_cr | 11 | 7 | 1 | 2.30e+03 | 2.30e+03 |
| 2d_poisson_log_n200_k10_cr | 13 | 9 | 1 | 4.36e+02 | 4.36e+02 |
| 2d_poisson_log_n5000_k10_cr | 11 | 6 | 1 | 1.14e+04 | 1.14e+04 |
| 2d_quasibinomial_logit_n1000_k10_cr | 12 | 10 | 1 | 1.44e+03 | 5.19e+02 |
| 2d_quasipoisson_log_n1000_k10_cr | 6 | 6 | 1 | 2.09e+03 | 1.17e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 9 | 6 | 1 | -1.14e+02 | -1.14e+02 |
| 3d_poisson_log_n2000_k10_cr | 12 | 6 | 1 | 4.31e+03 | 4.31e+03 |
| 4d_binomial_logit_n2000_k8_cr | 9 | 6 | 1 | 1.15e+03 | 1.15e+03 |
| 4d_gamma_log_n2000_k8_cr | 9 | 7 | 1 | 3.34e+03 | 3.33e+03 |
| 4d_gaussian_bs_n2000_k10 | 13 | 9 | 1 | -3.56e+02 | -3.56e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 13 | 10 | 1 | -1.46e+02 | -1.46e+02 |
| 4d_small_neighbourhood_n300 | 14 | 10 | 1 | -1.30e+02 | -1.30e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.01e+02 |
| 5d_skewed_features_n5000 | 12 | 7 | 1 | -4.35e+03 | -4.35e+03 |
| 6d_heatmap_pricing_n8000 | 11 | 6 | 1 | -6.79e+03 | -6.79e+03 |
| 7d_neighbourhoods_compact_n3000 | 10 | 7 | 1 | -2.55e+03 | -2.55e+03 |
| 8d_neighbourhoods_like_n15000 | 10 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 9.45e+01 | 8.79e+01 | 2.92e+02 | 3.23e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.86e+00 | 1.78e+00 | 4.75e+01 | 3.91e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.09e+00 | 2.00e+00 | 2.63e+01 | 7.95e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.47e-01 | 6.31e-01 | 1.59e+01 | 4.07e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.86e+01 | 2.73e+01 | 6.25e+01 | 4.58e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 5.58e-01 | 5.48e-01 | 1.58e+01 | 3.54e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.28e+01 | 1.21e+01 | 3.25e+01 | 3.93e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.09e+00 | 1.02e+00 | 1.95e+01 | 5.58e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.97e+00 | 2.92e+00 | 2.65e+01 | 1.12e-01 |
| 1d_gaussian_smooth_n50_k10_cr | 5.15e-01 | 4.98e-01 | 1.76e+01 | 2.92e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 8.43e-01 | 8.18e-01 | 1.75e+01 | 4.83e-02 |
| 1d_gaussian_step_n500_k10_cr | 9.28e-01 | 8.01e-01 | 1.71e+01 | 5.42e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 3.98e+00 | 2.44e+00 | 2.32e+01 | 1.72e-01 |
| 1d_poisson_log_n500_k10_cr | 3.92e+00 | 3.63e+00 | 5.97e+01 | 6.56e-02 |
| 1d_tw_log_n400_k20_cr | 1.51e+02 | 1.50e+02 | 2.09e+02 | 7.22e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 9.23e+01 | 8.85e+01 | 7.80e+01 | 1.18e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.91e+01 | 3.84e+01 | 9.18e+01 | 4.26e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.22e+01 | 1.19e+01 | 1.37e+02 | 8.93e-02 |
| 2d_binomial_logit_n5000_k10_cr | 4.09e+02 | 3.96e+02 | 1.65e+02 | 2.48e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 4.63e+01 | 4.34e+01 | 2.54e+02 | 1.82e-01 |
| 2d_gamma_log_n1000_k10_cr | 4.20e+01 | 3.94e+01 | 1.69e+02 | 2.49e-01 |
| 2d_gamma_log_n200_k10_cr | 5.81e+00 | 5.35e+00 | 1.64e+02 | 3.55e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 8.68e+00 | 8.24e+00 | 3.75e+01 | 2.31e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 4.19e+02 | 4.03e+02 | 2.82e+02 | 1.49e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.02e+00 | 1.94e+00 | 2.45e+01 | 8.23e-02 |
| 2d_gaussian_bs_n1500_k15 | 8.01e+00 | 7.14e+00 | 3.54e+01 | 2.26e-01 |
| 2d_invgauss_log_n800_k10_cr | 3.95e+01 | 3.76e+01 | 1.90e+02 | 2.08e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.94e+01 | 2.59e+01 | 1.42e+02 | 2.07e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 3.69e+01 | 3.48e+01 | 1.91e+02 | 1.93e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.98e+01 | 3.90e+01 | 1.25e+02 | 3.20e-01 |
| 2d_poisson_log_n200_k10_cr | 1.27e+01 | 1.19e+01 | 1.38e+02 | 9.17e-02 |
| 2d_poisson_log_n5000_k10_cr | 2.34e+02 | 2.26e+02 | 2.07e+02 | 1.13e+00 |
| 2d_quasibinomial_logit_n1000_k10_cr | 7.03e+01 | 6.92e+01 | 1.67e+02 | 4.22e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 3.14e+01 | 3.00e+01 | 1.62e+02 | 1.94e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 9.20e+00 | 8.54e+00 | 5.91e+01 | 1.56e-01 |
| 3d_poisson_log_n2000_k10_cr | 1.39e+02 | 1.34e+02 | 2.20e+02 | 6.33e-01 |
| 4d_binomial_logit_n2000_k8_cr | 1.41e+02 | 1.36e+02 | 1.92e+02 | 7.34e-01 |
| 4d_gamma_log_n2000_k8_cr | 8.10e+01 | 7.58e+01 | 2.52e+02 | 3.22e-01 |
| 4d_gaussian_bs_n2000_k10 | 3.17e+01 | 2.94e+01 | 8.79e+01 | 3.60e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.89e+01 | 1.75e+01 | 7.44e+01 | 2.54e-01 |
| 4d_small_neighbourhood_n300 | 7.48e+00 | 6.81e+00 | 5.74e+01 | 1.30e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.35e+01 | 1.30e+01 | 9.95e+01 | 1.35e-01 |
| 5d_skewed_features_n5000 | 8.29e+01 | 8.19e+01 | 1.30e+02 | 6.39e-01 |
| 6d_heatmap_pricing_n8000 | 1.64e+02 | 1.59e+02 | 2.19e+02 | 7.49e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.67e+01 | 4.30e+01 | 1.42e+02 | 3.29e-01 |
| 8d_neighbourhoods_like_n15000 | 4.41e+02 | 4.32e+02 | 6.30e+02 | 7.00e-01 |
