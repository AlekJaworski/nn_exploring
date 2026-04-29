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
