# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 1.88e-07 | 2.43e-07 | 2.89e-07 | 1.05e+00 | 1.24e-07 | 1.16e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 3.72e-09 | 4.23e-06 | 1.07e-08 | 4.11e-01 | 1.71e-09 | 7.86e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 1.11e-07 | 1.31e-06 | 1.94e-07 | 9.29e-02 | 4.07e-08 | 5.21e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.09e-09 | 1.94e-08 | 1.04e-09 | 7.89e-01 | 9.64e-11 | 4.48e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.92e-01 | 9.48e-02 | 1.00e+00 | 2.76e-03 | 9.74e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 4.63e-09 | 1.25e-06 | 9.13e-09 | 3.30e-02 | 8.20e-10 | 3.30e-07 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.48e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✓ | 4.02e-03 | 1.28e-02 | 3.88e-03 | 5.39e-01 | 8.37e-01 | 2.23e-01 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 3.55e-01 | 9.98e-02 | 2.88e+00 | 8.54e-01 | 1.32e+01 | 1.30e+01 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.32e-09 | 1.45e-06 | 4.45e-09 | 9.47e-01 | 1.18e-10 | 1.61e-07 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.34e-07 | 3.13e-05 | 4.20e-07 | 1.02e+00 | 1.35e-08 | 9.45e-06 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.47e-03 | 5.30e-04 | 8.54e-03 | 2.02e-01 | 5.45e+00 | 1.18e+01 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 1.79e-05 | 2.20e-03 | 2.57e-05 | 1.54e-01 | 2.96e-06 | 5.05e-03 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 4.93e-08 | 1.41e-06 | 7.64e-08 | 5.04e-01 | 2.88e-09 | 4.88e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 7.10e-06 | 3.62e-03 | 1.11e-05 | 2.09e+00 | 1.26e-06 | 7.49e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.35e-06 | 1.77e-03 | 1.80e-06 | 8.09e-01 | 1.62e-07 | 3.22e-04 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 1.59e-05 | 4.67e-02 | 4.07e-05 | 8.23e-01 | 1.62e-06 | 1.60e+01 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 3.01e-06 | 2.52e-07 | 4.88e-06 | 1.54e+00 | 1.21e-07 | 1.14e+01 |
| 8d_neighbourhoods_like_n15000 | ✗ | ✗ | ✓ | 4.63e-04 | 7.30e-02 | 6.45e-04 | 2.99e+00 | 2.91e-05 | 9.74e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.88e-07 | 1.06e+08 | 4.90e+07 | 4.64e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 3.72e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 1.11e-07 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.09e-09 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 4.63e-09 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.02e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.32e-09 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.34e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.79e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.93e-08 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 7.10e-06 | 1.08e+08 | 4.29e+08 | 3.99e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.35e-06 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 1.59e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 3.01e-06 | 4.55e+00 | 4.55e+00 | 1.00e+00 |
| 8d_neighbourhoods_like_n15000 | 4.63e-04 | 2.01e+07 | 7.76e+08 | 3.87e+01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 12 | 10 | — | -4.45e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5 | 5 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 4 | 3 | 1 | -7.72e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 6 | 1 | -6.37e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 5 | 5 | — | -6.11e+01 | -6.40e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 6 | 1 | -1.69e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 5 | — | -1.74e+02 | -1.75e+02 |
| 2d_binomial_logit_n1000_k10_cr | 7 | 4 | 1 | 1.44e+03 | 1.43e+03 |
| 2d_gamma_log_n1000_k10_cr | 13 | 10 | 1 | 7.45e+02 | 7.39e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 6 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 6 | 1 | -5.68e+01 | -5.80e+01 |
| 2d_poisson_log_n1000_k10_cr | 13 | 7 | 1 | 1.45e+03 | 1.45e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 9 | 8 | 1 | -1.14e+02 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 13 | 10 | 1 | -1.46e+02 | -1.48e+02 |
| 4d_small_neighbourhood_n300 | 13 | 11 | 1 | -1.30e+02 | -1.32e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.04e+02 |
| 5d_skewed_features_n5000 | 14 | 8 | 1 | -4.35e+03 | -4.36e+03 |
| 6d_heatmap_pricing_n8000 | 13 | 8 | 1 | -6.79e+03 | -6.80e+03 |
| 8d_neighbourhoods_like_n15000 | 15 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.40e+00 | 2.12e+00 | 3.11e+01 | 7.72e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 8.69e-01 | 7.95e-01 | 1.99e+01 | 4.37e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 4.65e-01 | 4.37e-01 | 1.69e+01 | 2.76e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 8.86e-01 | 8.81e-01 | 2.07e+01 | 4.29e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 6.45e+00 | 6.18e+00 | 2.97e+01 | 2.17e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.94e-01 | 7.73e-01 | 2.11e+01 | 3.76e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.40e+00 | 2.28e+00 | 2.31e+01 | 1.04e-01 |
| 2d_binomial_logit_n1000_k10_cr | 5.61e+00 | 5.41e+00 | 9.39e+01 | 5.98e-02 |
| 2d_gamma_log_n1000_k10_cr | 7.90e+00 | 7.46e+00 | 1.64e+02 | 4.81e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 9.47e+00 | 8.47e+00 | 3.73e+01 | 2.54e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.01e+00 | 1.89e+00 | 2.65e+01 | 7.59e-02 |
| 2d_poisson_log_n1000_k10_cr | 7.19e+00 | 6.56e+00 | 1.24e+02 | 5.78e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 6.75e+00 | 6.32e+00 | 5.92e+01 | 1.14e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.45e+01 | 1.35e+01 | 6.68e+01 | 2.17e-01 |
| 4d_small_neighbourhood_n300 | 5.82e+00 | 5.59e+00 | 5.14e+01 | 1.13e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.06e+01 | 1.02e+01 | 9.27e+01 | 1.14e-01 |
| 5d_skewed_features_n5000 | 8.73e+01 | 8.39e+01 | 1.15e+02 | 7.62e-01 |
| 6d_heatmap_pricing_n8000 | 2.15e+02 | 2.00e+02 | 2.14e+02 | 1.00e+00 |
| 8d_neighbourhoods_like_n15000 | 8.56e+02 | 8.49e+02 | 6.35e+02 | 1.35e+00 |
