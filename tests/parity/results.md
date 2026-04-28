# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 1.88e-07 | 2.43e-07 | 2.89e-07 | 1.05e+00 | 1.24e-07 | 1.16e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 3.72e-09 | 6.27e-06 | 1.07e-08 | 4.11e-01 | 1.71e-09 | 7.86e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 1.11e-07 | 1.31e-06 | 1.94e-07 | 9.29e-02 | 4.07e-08 | 5.21e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.09e-09 | 1.94e-08 | 1.04e-09 | 7.89e-01 | 9.64e-11 | 4.48e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.94e-01 | 9.48e-02 | 1.00e+00 | 2.76e-03 | 9.74e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 4.63e-09 | 1.30e-06 | 9.13e-09 | 3.30e-02 | 8.20e-10 | 3.30e-07 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.54e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.74e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 9.99e-01 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.91e+00 | 3.73e-01 | 5.50e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.32e-09 | 1.85e-06 | 4.45e-09 | 9.47e-01 | 1.18e-10 | 1.61e-07 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.34e-07 | 3.37e-05 | 4.20e-07 | 1.02e+00 | 1.35e-08 | 9.45e-06 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.10e+00 | 9.22e-02 | 1.47e+00 | 3.01e-01 | 5.40e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | ✗ | ✓ | ✓ | 1.79e-05 | 2.43e-03 | 2.57e-05 | 1.54e-01 | 2.96e-06 | 5.05e-03 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 4.93e-08 | 1.42e-06 | 7.64e-08 | 5.04e-01 | 2.88e-09 | 4.88e-01 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 7.10e-06 | 4.19e-03 | 1.11e-05 | 2.09e+00 | 1.26e-06 | 7.49e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.35e-06 | 2.18e-03 | 1.80e-06 | 8.09e-01 | 1.62e-07 | 3.22e-04 |
| 5d_skewed_features_n5000 | ✗ | ✓ | ✓ | 1.59e-05 | 5.11e-02 | 4.07e-05 | 8.23e-01 | 1.62e-06 | 1.60e+01 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 3.01e-06 | 2.52e-07 | 4.88e-06 | 1.54e+00 | 1.21e-07 | 1.14e+01 |
| 8d_neighbourhoods_like_n15000 | ✗ | ✗ | ✓ | 4.63e-04 | 7.52e-02 | 6.45e-04 | 2.99e+00 | 2.91e-05 | 9.74e-01 |

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
| 2d_gaussian_additive_n2000_k15_cr | 6 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 6 | 1 | -5.68e+01 | -5.80e+01 |
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
| 1d_gaussian_near_linear_n500_k10_cr | 1.82e+00 | 1.67e+00 | 2.67e+01 | 6.83e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.63e-01 | 6.59e-01 | 1.81e+01 | 3.66e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 4.03e-01 | 3.99e-01 | 1.43e+01 | 2.81e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 8.88e-01 | 8.55e-01 | 1.84e+01 | 4.83e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 6.28e+00 | 5.58e+00 | 2.70e+01 | 2.33e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 1.01e+00 | 6.83e-01 | 1.73e+01 | 5.84e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.33e+00 | 2.19e+00 | 2.39e+01 | 9.72e-02 |
| 2d_binomial_logit_n1000_k10_cr | 1.08e+01 | 1.03e+01 | 8.36e+01 | 1.30e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.35e+00 | 6.92e+00 | 1.44e+02 | 5.12e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.62e+00 | 7.15e+00 | 3.08e+01 | 2.47e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.81e+00 | 1.72e+00 | 2.31e+01 | 7.87e-02 |
| 2d_poisson_log_n1000_k10_cr | 7.41e+00 | 6.64e+00 | 1.07e+02 | 6.91e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 5.65e+00 | 5.18e+00 | 4.12e+01 | 1.37e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.29e+01 | 1.17e+01 | 5.68e+01 | 2.27e-01 |
| 4d_small_neighbourhood_n300 | 5.30e+00 | 4.98e+00 | 4.22e+01 | 1.25e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.08e+01 | 9.36e+00 | 7.94e+01 | 1.36e-01 |
| 5d_skewed_features_n5000 | 7.02e+01 | 6.67e+01 | 1.02e+02 | 6.90e-01 |
| 6d_heatmap_pricing_n8000 | 1.73e+02 | 1.68e+02 | 1.88e+02 | 9.18e-01 |
| 8d_neighbourhoods_like_n15000 | 7.98e+02 | 7.55e+02 | 7.55e+02 | 1.06e+00 |
