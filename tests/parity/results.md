# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | ✓ | ✓ | ✓ | 4.21e-06 | 4.74e-04 | 6.99e-06 | 1.61e+00 | 1.39e-08 | 8.63e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | ✓ | ✓ | ✓ | 8.10e-07 | 1.03e-02 | 1.62e-06 | 1.32e-01 | 7.36e-08 | 3.90e-05 |
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 5.06e-07 | 6.52e-07 | 7.77e-07 | 1.05e+00 | 3.34e-07 | 5.90e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.16e-09 | 2.03e-06 | 1.20e-08 | 4.11e-01 | 1.92e-09 | 8.80e-07 |
| 1d_gaussian_smooth_n1000_k50_cr | ✓ | ✓ | ✓ | 3.52e-07 | 2.50e-05 | 1.01e-06 | 2.52e-02 | 1.50e-07 | 3.47e-05 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 8.38e-09 | 9.87e-08 | 1.47e-08 | 9.29e-02 | 3.07e-09 | 3.93e-07 |
| 1d_gaussian_smooth_n2000_k30_cr | ✓ | ✓ | ✓ | 9.45e-10 | 7.51e-07 | 4.14e-09 | 2.63e-01 | 2.22e-10 | 8.35e-08 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.09e-09 | 1.93e-08 | 1.04e-09 | 7.89e-01 | 9.64e-11 | 4.48e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 2.34e-08 | 6.97e-07 | 6.54e-08 | 7.40e-01 | 3.05e-09 | 1.38e-06 |
| 1d_gaussian_smooth_n50_k10_cr | ✓ | ✓ | ✓ | 5.30e-09 | 8.36e-08 | 1.68e-08 | 9.67e-02 | 2.65e-09 | 1.36e-07 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 6.65e-10 | 1.53e-07 | 1.30e-09 | 3.30e-02 | 1.16e-10 | 4.66e-08 |
| 1d_gaussian_step_n500_k10_cr | ✓ | ✓ | ✓ | 2.50e-08 | 6.07e-07 | 3.82e-08 | 1.67e+00 | 3.81e-09 | 3.55e-06 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.28e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 1d_poisson_log_n500_k10_cr | ✓ | ✓ | ✓ | 1.19e-04 | 2.94e-05 | 2.76e-04 | 4.82e-02 | 4.14e+00 | 7.85e-04 |
| 2d_binomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 3.61e-04 | 1.03e-03 | 5.25e-04 | 5.24e-01 | 8.37e-01 | 2.03e-02 |
| 2d_binomial_logit_n200_k10_cr | ✓ | ✓ | ✓ | 1.21e-07 | 3.75e-07 | 5.88e-08 | 2.47e+00 | 8.39e-01 | 7.08e-02 |
| 2d_binomial_logit_n5000_k10_cr | ✓ | ✓ | ✓ | 9.73e-05 | 3.39e-04 | 1.16e-04 | 1.78e+00 | 8.37e-01 | 9.92e-01 |
| 2d_gamma_inverse_n1000_k10_cr | ✗ | ✗ | ✓ | 1.49e-02 | 4.15e-03 | 2.93e-03 | 7.63e-02 | 3.40e+00 | 2.01e+00 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.70e-03 | 1.08e-03 | 6.49e-03 | 1.38e-01 | 1.32e+01 | 8.74e-01 |
| 2d_gamma_log_n200_k10_cr | ✗ | ✗ | ✓ | 6.95e-02 | 2.30e-02 | 8.26e-02 | 2.58e-01 | 9.91e+00 | 3.34e-01 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.99e-08 | 9.76e-06 | 5.78e-08 | 9.47e-01 | 1.67e-09 | 3.59e-06 |
| 2d_gaussian_additive_n50000_k15_cr | ✓ | ✓ | ✓ | 2.18e-08 | 2.52e-05 | 5.97e-08 | 2.25e-08 | 2.08e-11 | 1.30e-04 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.61e-07 | 1.34e-05 | 3.70e-07 | 1.02e+00 | 1.19e-08 | 1.13e-05 |
| 2d_gaussian_bs_n1500_k15 | ✓ | ✓ | ✓ | 4.58e-07 | 1.01e-04 | 3.43e-06 | 1.51e+00 | 2.74e-08 | 1.93e-05 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 4.13e-05 | 3.35e-06 | 5.91e-05 | 2.02e-01 | 5.45e+00 | 3.42e-01 |
| 2d_poisson_log_n200_k10_cr | ✓ | ✓ | ✓ | 1.20e-04 | 1.05e-05 | 2.10e-04 | 4.38e-01 | 5.15e+00 | 3.88e-01 |
| 2d_poisson_log_n5000_k10_cr | ✓ | ✓ | ✓ | 2.83e-03 | 1.86e-04 | 2.85e-03 | 2.06e-01 | 5.48e+00 | 8.47e-01 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 1.40e-05 | 1.39e-03 | 2.01e-05 | 1.54e-01 | 2.30e-06 | 3.95e-03 |
| 3d_poisson_log_n2000_k10_cr | ✓ | ✓ | ✓ | 1.98e-03 | 2.50e-04 | 2.44e-03 | 4.63e-01 | 3.68e+00 | 6.50e-01 |
| 4d_binomial_logit_n2000_k8_cr | ✓ | ✓ | ✓ | 3.76e-04 | 1.24e-03 | 3.33e-04 | 9.77e-01 | 8.32e-01 | 6.99e-02 |
| 4d_gamma_log_n2000_k8_cr | ✗ | ✗ | ✓ | 1.26e-02 | 2.67e-03 | 2.62e-02 | 4.85e-01 | 7.24e+00 | 2.05e-01 |
| 4d_gaussian_bs_n2000_k10 | ✓ | ✓ | ✓ | 5.41e-07 | 2.48e-04 | 8.13e-07 | 1.97e+00 | 1.04e-07 | 3.97e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.23e-07 | 7.29e-06 | 2.46e-07 | 5.04e-01 | 2.87e-08 | 1.78e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 1.24e-05 | 6.11e-03 | 1.55e-05 | 2.09e+00 | 2.77e-06 | 6.28e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 4.70e-07 | 2.12e-04 | 5.54e-07 | 8.09e-01 | 2.36e-08 | 1.02e-04 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 3.03e-05 | 3.15e-02 | 8.69e-05 | 8.23e-01 | 2.97e-06 | 5.26e-02 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 1.06e-04 | 7.22e-06 | 1.38e-04 | 1.54e+00 | 3.30e-06 | 3.30e-01 |
| 7d_neighbourhoods_compact_n3000 | ✓ | ✓ | ✓ | 4.11e-05 | 1.23e-03 | 5.89e-05 | 9.30e-01 | 7.99e-06 | 4.26e-01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 4.15e-04 | 7.02e-02 | 6.15e-04 | 2.22e-04 | 2.05e-05 | 9.89e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 4.21e-06 | 2.11e+00 | 2.11e+00 | 1.00e+00 |
| 1d_gaussian_low_signal_n1000_k10_cr | 8.10e-07 | 2.37e+03 | 2.37e+03 | 1.00e+00 |
| 1d_gaussian_near_linear_n500_k10_cr | 5.06e-07 | 2.01e+07 | 4.90e+07 | 2.44e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4.16e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n1000_k50_cr | 3.52e-07 | 1.58e+03 | 1.58e+03 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 8.38e-09 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 9.45e-10 | 3.19e+02 | 3.19e+02 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.09e-09 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n50_k10_cr | 5.30e-09 | 5.08e+00 | 5.08e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 6.65e-10 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_step_n500_k10_cr | 2.50e-08 | 6.51e-01 | 6.51e-01 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.02e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.99e-08 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n50000_k15_cr | 2.18e-08 | 3.65e+01 | 3.65e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.61e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 2d_gaussian_bs_n1500_k15 | 4.58e-07 | 4.42e+00 | 4.42e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.40e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_bs_n2000_k10 | 5.41e-07 | 7.62e-01 | 7.62e-01 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.23e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 1.24e-05 | 1.83e+08 | 4.29e+08 | 2.34e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 4.70e-07 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 3.03e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 1.06e-04 | 4.56e+00 | 4.55e+00 | 9.99e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.11e-05 | 9.57e+04 | 9.88e+04 | 1.03e+00 |
| 8d_neighbourhoods_like_n15000 | 4.15e-04 | 2.09e+07 | 1.96e+09 | 9.39e+01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 13 | 10 | 2 | -4.03e+02 | -4.09e+02 |
| 1d_gaussian_low_signal_n1000_k10_cr | 6 | 7 | 1 | 7.55e+02 | 7.54e+02 |
| 1d_gaussian_near_linear_n500_k10_cr | 10 | 10 | — | -4.45e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4 | 5 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n1000_k50_cr | 6 | 6 | — | -2.13e+02 | -2.14e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 3 | 1 | -7.72e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 6 | 5 | 1 | -3.54e+02 | -3.55e+02 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 6 | 1 | -6.37e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 4 | 5 | 1 | -6.33e+01 | -6.40e+01 |
| 1d_gaussian_smooth_n50_k10_cr | 5 | 4 | 1 | 6.47e+00 | 5.80e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 6 | 1 | -1.69e+02 | -1.70e+02 |
| 1d_gaussian_step_n500_k10_cr | 4 | 6 | 1 | -2.54e+02 | -2.54e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 5 | — | -1.74e+02 | -1.75e+02 |
| 1d_poisson_log_n500_k10_cr | 4 | 3 | 1 | 1.10e+03 | 7.38e+02 |
| 2d_binomial_logit_n1000_k10_cr | 7 | 4 | 1 | 5.23e+02 | 1.43e+03 |
| 2d_binomial_logit_n200_k10_cr | 10 | 9 | 1 | 9.58e+01 | 2.78e+02 |
| 2d_binomial_logit_n5000_k10_cr | 14 | 6 | 1 | 2.53e+03 | 7.09e+03 |
| 2d_gamma_inverse_n1000_k10_cr | 10 | 6 | 1 | 1.28e+03 | 7.73e+02 |
| 2d_gamma_log_n1000_k10_cr | 11 | 10 | 1 | 1.84e+03 | 7.39e+02 |
| 2d_gamma_log_n200_k10_cr | 129 | 5 | 1 | 3.58e+02 | 1.43e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 5 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n50000_k15_cr | 4 | 7 | 1 | -9.25e+03 | -9.25e+03 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 6 | 1 | -5.68e+01 | -5.80e+01 |
| 2d_gaussian_bs_n1500_k15 | 4 | 5 | 1 | -2.47e+02 | -2.49e+02 |
| 2d_poisson_log_n1000_k10_cr | 9 | 7 | 1 | 2.30e+03 | 1.45e+03 |
| 2d_poisson_log_n200_k10_cr | 11 | 9 | 1 | 4.36e+02 | 2.67e+02 |
| 2d_poisson_log_n5000_k10_cr | 8 | 6 | 1 | 1.14e+04 | 7.11e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 8 | 8 | 1 | -1.14e+02 | -1.16e+02 |
| 3d_poisson_log_n2000_k10_cr | 9 | 6 | 1 | 4.31e+03 | 2.87e+03 |
| 4d_binomial_logit_n2000_k8_cr | 8 | 5 | 1 | 1.15e+03 | 2.85e+03 |
| 4d_gamma_log_n2000_k8_cr | 22 | 6 | 1 | 3.34e+03 | 1.47e+03 |
| 4d_gaussian_bs_n2000_k10 | 11 | 10 | 1 | -3.56e+02 | -3.59e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 12 | 10 | 1 | -1.46e+02 | -1.48e+02 |
| 4d_small_neighbourhood_n300 | 13 | 11 | 1 | -1.30e+02 | -1.32e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.04e+02 |
| 5d_skewed_features_n5000 | 11 | 8 | 1 | -4.35e+03 | -4.36e+03 |
| 6d_heatmap_pricing_n8000 | 9 | 8 | 1 | -6.79e+03 | -6.80e+03 |
| 7d_neighbourhoods_compact_n3000 | 10 | 8 | — | -2.55e+03 | -2.55e+03 |
| 8d_neighbourhoods_like_n15000 | 13 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 8.51e+01 | 8.35e+01 | 2.90e+02 | 2.93e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.63e+00 | 1.59e+00 | 4.71e+01 | 3.47e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 1.97e+00 | 1.86e+00 | 2.61e+01 | 7.55e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.30e-01 | 6.10e-01 | 1.89e+01 | 3.34e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.36e+01 | 2.35e+01 | 6.60e+01 | 3.58e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 5.52e-01 | 5.21e-01 | 1.47e+01 | 3.75e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.04e+01 | 1.02e+01 | 3.21e+01 | 3.25e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 1.01e+00 | 1.01e+00 | 1.78e+01 | 5.71e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.61e+00 | 2.39e+00 | 2.67e+01 | 9.77e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 5.68e-01 | 4.49e-01 | 1.60e+01 | 3.54e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 8.99e-01 | 7.62e-01 | 1.70e+01 | 5.29e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.56e-01 | 7.11e-01 | 1.63e+01 | 4.63e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.30e+00 | 2.22e+00 | 2.15e+01 | 1.07e-01 |
| 1d_poisson_log_n500_k10_cr | 3.90e+00 | 3.85e+00 | 5.31e+01 | 7.34e-02 |
| 2d_binomial_logit_n1000_k10_cr | 3.47e+01 | 3.31e+01 | 8.27e+01 | 4.20e-01 |
| 2d_binomial_logit_n200_k10_cr | 1.10e+01 | 1.00e+01 | 1.34e+02 | 8.18e-02 |
| 2d_binomial_logit_n5000_k10_cr | 7.02e+02 | 6.94e+02 | 1.48e+02 | 4.74e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.28e+01 | 3.16e+01 | 2.18e+02 | 1.50e-01 |
| 2d_gamma_log_n1000_k10_cr | 3.43e+01 | 3.29e+01 | 1.52e+02 | 2.25e-01 |
| 2d_gamma_log_n200_k10_cr | 7.01e+02 | 6.81e+02 | 1.55e+02 | 4.51e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.34e+01 | 9.89e+00 | 3.49e+01 | 3.83e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 3.21e+02 | 3.09e+02 | 2.85e+02 | 1.12e+00 |
| 2d_gaussian_additive_n500_k10_cr | 2.14e+00 | 2.13e+00 | 3.64e+01 | 5.88e-02 |
| 2d_gaussian_bs_n1500_k15 | 7.26e+00 | 6.39e+00 | 3.26e+01 | 2.23e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.04e+01 | 2.95e+01 | 1.22e+02 | 2.50e-01 |
| 2d_poisson_log_n200_k10_cr | 9.13e+00 | 8.38e+00 | 1.21e+02 | 7.56e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.44e+02 | 1.39e+02 | 1.91e+02 | 7.54e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 8.21e+00 | 6.83e+00 | 5.45e+01 | 1.51e-01 |
| 3d_poisson_log_n2000_k10_cr | 8.37e+01 | 7.83e+01 | 1.73e+02 | 4.82e-01 |
| 4d_binomial_logit_n2000_k8_cr | 9.81e+01 | 9.51e+01 | 1.71e+02 | 5.75e-01 |
| 4d_gamma_log_n2000_k8_cr | 1.14e+03 | 1.14e+03 | 2.30e+02 | 4.98e+00 |
| 4d_gaussian_bs_n2000_k10 | 2.84e+01 | 2.70e+01 | 8.86e+01 | 3.20e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.57e+01 | 1.42e+01 | 6.82e+01 | 2.30e-01 |
| 4d_small_neighbourhood_n300 | 6.73e+00 | 6.37e+00 | 4.72e+01 | 1.43e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.12e+01 | 1.04e+01 | 7.89e+01 | 1.42e-01 |
| 5d_skewed_features_n5000 | 6.32e+01 | 6.10e+01 | 1.10e+02 | 5.76e-01 |
| 6d_heatmap_pricing_n8000 | 1.25e+02 | 1.24e+02 | 2.26e+02 | 5.56e-01 |
| 7d_neighbourhoods_compact_n3000 | 3.99e+01 | 3.77e+01 | 1.32e+02 | 3.03e-01 |
| 8d_neighbourhoods_like_n15000 | 6.66e+02 | 6.40e+02 | 5.94e+02 | 1.12e+00 |
