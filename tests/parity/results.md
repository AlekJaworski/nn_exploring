# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 2.92e-07 | 3.75e-07 | 4.48e-07 | 1.05e+00 | 1.92e-07 | 4.83e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 3.72e-09 | 4.23e-06 | 1.07e-08 | 4.11e-01 | 1.71e-09 | 7.86e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 8.38e-09 | 9.90e-08 | 1.47e-08 | 9.29e-02 | 3.07e-09 | 3.93e-07 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.09e-09 | 1.94e-08 | 1.04e-09 | 7.89e-01 | 9.64e-11 | 4.48e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.92e-01 | 9.48e-02 | 1.00e+00 | 2.76e-03 | 9.74e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 4.63e-09 | 1.25e-06 | 9.13e-09 | 3.30e-02 | 8.20e-10 | 3.30e-07 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.48e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✓ | 4.02e-03 | 1.28e-02 | 3.88e-03 | 5.39e-01 | 8.37e-01 | 2.23e-01 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 3.55e-01 | 9.98e-02 | 2.88e+00 | 8.54e-01 | 1.32e+01 | 3.69e+01 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.32e-09 | 1.45e-06 | 4.45e-09 | 9.47e-01 | 1.18e-10 | 1.61e-07 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 4.18e-11 | 5.31e-09 | 7.81e-11 | 1.02e+00 | 3.78e-12 | 1.79e-09 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.47e-03 | 5.30e-04 | 8.54e-03 | 2.02e-01 | 5.45e+00 | 1.18e+01 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 1.79e-05 | 2.20e-03 | 2.57e-05 | 1.54e-01 | 2.96e-06 | 5.05e-03 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.20e-07 | 6.03e-06 | 4.17e-07 | 5.04e-01 | 5.22e-09 | 9.98e+00 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 9.64e-05 | 6.37e-02 | 1.38e-04 | 2.09e+00 | 1.82e-05 | 4.18e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 4.26e-07 | 5.43e-04 | 5.27e-07 | 8.09e-01 | 4.47e-08 | 1.03e-04 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 1.59e-05 | 4.67e-02 | 4.07e-05 | 8.23e-01 | 1.62e-06 | 1.60e+01 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 3.01e-06 | 2.52e-07 | 4.88e-06 | 1.54e+00 | 1.21e-07 | 1.14e+01 |
| 8d_neighbourhoods_like_n15000 | ✗ | ✓ | ✓ | 3.85e-04 | 7.63e-02 | 5.36e-04 | 2.28e-04 | 2.04e-05 | 9.90e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.92e-07 | 2.86e+08 | 4.90e+07 | 1.71e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 3.72e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 8.38e-09 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.09e-09 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 4.63e-09 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.02e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.32e-09 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 4.18e-11 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.79e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.20e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 9.64e-05 | 1.31e+09 | 4.29e+08 | 3.27e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 4.26e-07 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 1.59e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 3.01e-06 | 4.55e+00 | 4.55e+00 | 1.00e+00 |
| 8d_neighbourhoods_like_n15000 | 3.85e-04 | 2.01e+07 | 1.96e+09 | 9.78e+01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 13 | 10 | — | -4.45e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5 | 5 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 3 | 1 | -7.72e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 6 | 1 | -6.37e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 5 | 5 | — | -6.11e+01 | -6.40e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 6 | 1 | -1.69e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 5 | — | -1.74e+02 | -1.75e+02 |
| 2d_binomial_logit_n1000_k10_cr | 7 | 4 | 1 | 1.44e+03 | 1.43e+03 |
| 2d_gamma_log_n1000_k10_cr | 14 | 10 | 1 | 7.45e+02 | 7.39e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 6 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 6 | 6 | 1 | -5.68e+01 | -5.80e+01 |
| 2d_poisson_log_n1000_k10_cr | 13 | 7 | 1 | 1.45e+03 | 1.45e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 9 | 8 | 1 | -1.14e+02 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 15 | 10 | 1 | -1.46e+02 | -1.48e+02 |
| 4d_small_neighbourhood_n300 | 16 | 11 | 1 | -1.30e+02 | -1.32e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 9 | 7 | 1 | -2.01e+02 | -2.04e+02 |
| 5d_skewed_features_n5000 | 14 | 8 | 1 | -4.35e+03 | -4.36e+03 |
| 6d_heatmap_pricing_n8000 | 13 | 8 | 1 | -6.79e+03 | -6.80e+03 |
| 8d_neighbourhoods_like_n15000 | 15 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.52e+00 | 2.29e+00 | 3.24e+01 | 7.77e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 8.29e-01 | 7.90e-01 | 1.86e+01 | 4.46e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 6.50e-01 | 5.66e-01 | 2.21e+01 | 2.95e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 9.82e-01 | 9.74e-01 | 1.64e+01 | 6.00e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 6.28e+00 | 5.97e+00 | 2.97e+01 | 2.12e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.72e-01 | 7.51e-01 | 1.88e+01 | 4.12e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.58e+00 | 2.39e+00 | 2.20e+01 | 1.17e-01 |
| 2d_binomial_logit_n1000_k10_cr | 5.74e+00 | 5.32e+00 | 9.48e+01 | 6.06e-02 |
| 2d_gamma_log_n1000_k10_cr | 7.21e+00 | 7.04e+00 | 1.57e+02 | 4.58e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.69e+00 | 7.18e+00 | 3.49e+01 | 2.20e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.24e+00 | 2.05e+00 | 2.40e+01 | 9.32e-02 |
| 2d_poisson_log_n1000_k10_cr | 6.75e+00 | 6.53e+00 | 1.23e+02 | 5.51e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 6.17e+00 | 5.71e+00 | 4.57e+01 | 1.35e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.68e+01 | 1.54e+01 | 6.50e+01 | 2.59e-01 |
| 4d_small_neighbourhood_n300 | 7.01e+00 | 6.55e+00 | 6.57e+01 | 1.07e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.19e+01 | 1.06e+01 | 7.92e+01 | 1.50e-01 |
| 5d_skewed_features_n5000 | 7.79e+01 | 7.18e+01 | 1.27e+02 | 6.13e-01 |
| 6d_heatmap_pricing_n8000 | 1.84e+02 | 1.83e+02 | 2.12e+02 | 8.65e-01 |
| 8d_neighbourhoods_like_n15000 | 7.81e+02 | 7.38e+02 | 7.22e+02 | 1.08e+00 |
