# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 3.33e-07 | 4.34e-07 | 5.13e-07 | 1.05e+00 | 2.21e-07 | 1.99e+01 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.16e-09 | 2.03e-06 | 1.20e-08 | 4.11e-01 | 1.92e-09 | 8.80e-07 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 8.38e-09 | 9.87e-08 | 1.47e-08 | 9.29e-02 | 3.07e-09 | 3.93e-07 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 1.29e-11 | 7.92e-11 | 1.45e-11 | 7.89e-01 | 4.00e-13 | 4.50e-10 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 1.45e-10 | 4.31e-09 | 5.12e-10 | 7.40e-01 | 1.32e-11 | 5.28e-09 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 6.65e-10 | 1.53e-07 | 1.30e-09 | 3.30e-02 | 1.16e-10 | 4.66e-08 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.28e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✓ | ✓ | 6.11e-04 | 1.96e-03 | 5.69e-04 | 5.27e-01 | 8.37e-01 | 2.99e-02 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 4.98e-05 | 9.08e-06 | 8.78e-05 | 1.38e-01 | 1.32e+01 | 4.97e+01 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 8.47e-10 | 4.78e-07 | 2.61e-09 | 9.47e-01 | 7.68e-11 | 1.42e-07 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.39e-10 | 1.77e-08 | 4.37e-10 | 1.02e+00 | 1.81e-11 | 8.38e-09 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.62e-03 | 5.53e-04 | 8.91e-03 | 2.02e-01 | 5.45e+00 | 8.94e+01 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 1.87e-05 | 1.86e-03 | 2.70e-05 | 1.54e-01 | 3.11e-06 | 5.30e-03 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.23e-07 | 6.04e-06 | 4.24e-07 | 5.04e-01 | 5.07e-09 | 8.98e+00 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 8.86e-05 | 3.75e-02 | 1.27e-04 | 2.09e+00 | 1.67e-05 | 6.46e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 4.21e-07 | 3.02e-04 | 5.13e-07 | 8.09e-01 | 4.32e-08 | 1.01e-04 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 1.56e-05 | 3.36e-02 | 3.91e-05 | 8.23e-01 | 1.59e-06 | 1.27e+01 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 3.17e-06 | 2.58e-07 | 4.95e-06 | 1.54e+00 | 1.19e-07 | 3.60e+01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 3.85e-04 | 6.84e-02 | 5.23e-04 | 2.28e-04 | 1.95e-05 | 9.90e-01 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 3.33e-07 | 1.02e+09 | 4.90e+07 | 4.80e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4.16e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 8.38e-09 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 1.29e-11 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 6.65e-10 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.02e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 8.47e-10 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.39e-10 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.87e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.23e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 8.86e-05 | 1.36e+09 | 4.29e+08 | 3.16e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 4.21e-07 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 1.56e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 3.17e-06 | 4.55e+00 | 4.55e+00 | 1.00e+00 |
| 8d_neighbourhoods_like_n15000 | 3.85e-04 | 2.00e+07 | 1.96e+09 | 9.80e+01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 14 | 10 | — | -4.45e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4 | 5 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 3 | 1 | -7.72e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 6 | 6 | 1 | -6.37e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 5 | 5 | 1 | -6.33e+01 | -6.40e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 6 | 1 | -1.69e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 5 | — | -1.74e+02 | -1.75e+02 |
| 2d_binomial_logit_n1000_k10_cr | 35 | 4 | 1 | 1.43e+03 | 1.43e+03 |
| 2d_gamma_log_n1000_k10_cr | 150 | 10 | 1 | 7.47e+02 | 7.39e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 5 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 6 | 1 | -5.68e+01 | -5.80e+01 |
| 2d_poisson_log_n1000_k10_cr | 140 | 7 | 1 | 1.45e+03 | 1.45e+03 |
| 3d_gaussian_mixed_n800_k10_cr | 9 | 8 | 1 | -1.14e+02 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 15 | 10 | 1 | -1.46e+02 | -1.48e+02 |
| 4d_small_neighbourhood_n300 | 16 | 11 | 1 | -1.30e+02 | -1.32e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.04e+02 |
| 5d_skewed_features_n5000 | 15 | 8 | 1 | -4.35e+03 | -4.36e+03 |
| 6d_heatmap_pricing_n8000 | 13 | 8 | 1 | -6.79e+03 | -6.80e+03 |
| 8d_neighbourhoods_like_n15000 | 15 | 9 | 1 | -1.32e+04 | -1.32e+04 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.60e+00 | 2.35e+00 | 3.30e+01 | 7.87e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.94e-01 | 6.48e-01 | 1.86e+01 | 3.73e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 6.12e-01 | 5.46e-01 | 1.43e+01 | 4.28e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.01e+00 | 9.52e-01 | 2.13e+01 | 4.76e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.39e+00 | 2.31e+00 | 2.83e+01 | 8.45e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.49e-01 | 7.16e-01 | 1.87e+01 | 4.01e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.24e+00 | 2.14e+00 | 2.44e+01 | 9.19e-02 |
| 2d_binomial_logit_n1000_k10_cr | 2.08e+01 | 2.08e+01 | 9.32e+01 | 2.24e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.49e+01 | 6.75e+01 | 1.63e+02 | 4.59e-01 |
| 2d_gaussian_additive_n2000_k15_cr | 8.33e+00 | 7.83e+00 | 3.39e+01 | 2.45e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.92e+00 | 1.82e+00 | 2.39e+01 | 8.01e-02 |
| 2d_poisson_log_n1000_k10_cr | 6.96e+01 | 6.76e+01 | 1.14e+02 | 6.11e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.84e+00 | 5.72e+00 | 4.45e+01 | 1.54e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.60e+01 | 1.51e+01 | 6.50e+01 | 2.46e-01 |
| 4d_small_neighbourhood_n300 | 7.62e+00 | 7.15e+00 | 4.95e+01 | 1.54e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.06e+01 | 9.85e+00 | 8.19e+01 | 1.29e-01 |
| 5d_skewed_features_n5000 | 9.17e+01 | 8.96e+01 | 1.14e+02 | 8.07e-01 |
| 6d_heatmap_pricing_n8000 | 2.04e+02 | 1.62e+02 | 2.04e+02 | 9.99e-01 |
| 8d_neighbourhoods_like_n15000 | 9.76e+02 | 8.88e+02 | 7.22e+02 | 1.35e+00 |
