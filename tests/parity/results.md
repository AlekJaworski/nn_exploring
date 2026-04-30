# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | ✓ | ✓ | ✓ | 8.88e-07 | 1.04e-04 | 4.02e-07 | 1.61e+00 | 8.66e-08 | 3.56e+01 |
| 1d_gaussian_low_signal_n1000_k10_cr | ✓ | ✓ | ✓ | 8.10e-07 | 1.03e-02 | 1.62e-06 | 1.32e-01 | 7.36e-08 | 3.90e-05 |
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 3.07e-07 | 3.97e-07 | 4.72e-07 | 1.05e+00 | 2.03e-07 | 6.99e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 4.16e-09 | 2.03e-06 | 1.20e-08 | 4.11e-01 | 1.92e-09 | 8.80e-07 |
| 1d_gaussian_smooth_n1000_k50_cr | ✓ | ✓ | ✓ | 7.99e-10 | 5.53e-08 | 2.25e-09 | 2.52e-02 | 3.29e-10 | 7.63e-08 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 8.38e-09 | 9.87e-08 | 1.47e-08 | 9.29e-02 | 3.07e-09 | 3.93e-07 |
| 1d_gaussian_smooth_n2000_k30_cr | ✓ | ✓ | ✓ | 9.45e-10 | 7.51e-07 | 4.14e-09 | 2.63e-01 | 2.22e-10 | 8.35e-08 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 2.09e-09 | 1.93e-08 | 1.04e-09 | 7.89e-01 | 9.64e-11 | 4.48e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✓ | ✓ | ✓ | 1.45e-10 | 4.31e-09 | 5.12e-10 | 7.40e-01 | 1.32e-11 | 5.28e-09 |
| 1d_gaussian_smooth_n50_k10_cr | ✓ | ✓ | ✓ | 5.30e-09 | 8.36e-08 | 1.68e-08 | 9.67e-02 | 2.65e-09 | 1.36e-07 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 6.65e-10 | 1.53e-07 | 1.30e-09 | 3.30e-02 | 1.16e-10 | 4.66e-08 |
| 1d_gaussian_step_n500_k10_cr | ✓ | ✓ | ✓ | 2.50e-08 | 6.07e-07 | 3.82e-08 | 1.67e+00 | 3.81e-09 | 3.55e-06 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 6.02e-10 | 1.28e-07 | 2.76e-09 | 1.51e+00 | 5.73e-11 | 8.74e-08 |
| 1d_poisson_log_n500_k10_cr | ✓ | ✓ | ✓ | 1.19e-04 | 2.94e-05 | 2.76e-04 | 4.82e-02 | 4.14e+00 | 7.85e-04 |
| 2d_binomial_logit_n1000_k10_cr | ✓ | ✓ | ✓ | 3.61e-04 | 1.03e-03 | 5.25e-04 | 5.24e-01 | 8.37e-01 | 2.03e-02 |
| 2d_binomial_logit_n200_k10_cr | ✓ | ✓ | ✓ | 1.74e-06 | 4.30e-06 | 1.06e-06 | 2.47e+00 | 8.39e-01 | 2.05e+01 |
| 2d_binomial_logit_n5000_k10_cr | ✓ | ✓ | ✓ | 1.83e-04 | 6.41e-04 | 2.18e-04 | 1.78e+00 | 8.37e-01 | 1.35e+01 |
| 2d_gamma_inverse_n1000_k10_cr | ✗ | ✗ | ✓ | 1.49e-02 | 4.14e-03 | 2.93e-03 | 7.63e-02 | 3.40e+00 | 2.01e+00 |
| 2d_gamma_log_n1000_k10_cr | ✓ | ✓ | ✓ | 3.61e-03 | 1.09e-03 | 6.95e-03 | 1.38e-01 | 1.32e+01 | 1.53e+00 |
| 2d_gamma_log_n200_k10_cr | ✗ | ✗ | ✓ | 6.95e-02 | 2.30e-02 | 8.26e-02 | 2.58e-01 | 9.91e+00 | 3.34e-01 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 8.47e-10 | 4.78e-07 | 2.61e-09 | 9.47e-01 | 7.68e-11 | 1.42e-07 |
| 2d_gaussian_additive_n50000_k15_cr | ✓ | ✓ | ✓ | 3.00e-08 | 2.72e-05 | 7.97e-08 | 2.33e-08 | 3.43e-11 | 1.34e-04 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.39e-10 | 1.77e-08 | 4.37e-10 | 1.02e+00 | 1.81e-11 | 8.38e-09 |
| 2d_gaussian_bs_n1500_k15 | ✓ | ✓ | ✓ | 7.99e-10 | 1.90e-07 | 5.73e-09 | 1.51e+00 | 1.18e-11 | 3.56e-08 |
| 2d_poisson_log_n1000_k10_cr | ✓ | ✓ | ✓ | 6.31e-05 | 5.07e-06 | 8.38e-05 | 2.02e-01 | 5.45e+00 | 3.87e+00 |
| 2d_poisson_log_n200_k10_cr | ✓ | ✓ | ✓ | 1.74e-04 | 1.52e-05 | 3.02e-04 | 4.38e-01 | 5.15e+00 | 1.13e+01 |
| 2d_poisson_log_n5000_k10_cr | ✓ | ✓ | ✓ | 3.57e-04 | 2.35e-05 | 3.70e-04 | 2.06e-01 | 5.48e+00 | 2.12e+00 |
| 3d_gaussian_mixed_n800_k10_cr | ✓ | ✓ | ✓ | 1.87e-05 | 1.86e-03 | 2.70e-05 | 1.54e-01 | 3.11e-06 | 5.30e-03 |
| 3d_poisson_log_n2000_k10_cr | ✓ | ✓ | ✓ | 6.89e-04 | 9.15e-05 | 8.18e-04 | 4.63e-01 | 3.69e+00 | 5.97e+00 |
| 4d_binomial_logit_n2000_k8_cr | ✓ | ✓ | ✓ | 5.35e-04 | 1.74e-03 | 4.95e-04 | 9.77e-01 | 8.32e-01 | 1.04e-01 |
| 4d_gamma_log_n2000_k8_cr | ✗ | ✗ | ✓ | 1.25e-02 | 2.67e-03 | 2.61e-02 | 4.85e-01 | 7.24e+00 | 2.02e-01 |
| 4d_gaussian_bs_n2000_k10 | ✓ | ✓ | ✓ | 4.61e-07 | 1.47e-04 | 5.31e-07 | 1.97e+00 | 7.01e-08 | 3.46e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 2.23e-07 | 6.04e-06 | 4.24e-07 | 5.04e-01 | 5.07e-09 | 8.98e+00 |
| 4d_small_neighbourhood_n300 | ✓ | ✓ | ✓ | 8.84e-05 | 3.74e-02 | 1.26e-04 | 2.09e+00 | 1.67e-05 | 6.45e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 4.21e-07 | 3.02e-04 | 5.13e-07 | 8.09e-01 | 4.32e-08 | 1.01e-04 |
| 5d_skewed_features_n5000 | ✓ | ✓ | ✓ | 1.54e-05 | 3.33e-02 | 3.76e-05 | 8.23e-01 | 1.58e-06 | 1.19e+01 |
| 6d_heatmap_pricing_n8000 | ✓ | ✓ | ✓ | 3.03e-06 | 2.53e-07 | 4.91e-06 | 1.54e+00 | 1.21e-07 | 1.26e+01 |
| 7d_neighbourhoods_compact_n3000 | ✓ | ✓ | ✓ | 2.60e-05 | 2.99e-04 | 3.34e-05 | 9.30e-01 | 1.26e-06 | 1.06e+01 |
| 8d_neighbourhoods_like_n15000 | ✓ | ✓ | ✓ | 3.85e-04 | 6.84e-02 | 5.23e-04 | 2.28e-04 | 1.95e-05 | 9.90e-01 |
