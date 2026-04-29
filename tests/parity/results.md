# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.24e+00 | 2.22e+00 | 3.05e+01 | 7.35e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.41e-01 | 6.15e-01 | 1.87e+01 | 3.43e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 5.83e-01 | 5.64e-01 | 1.63e+01 | 3.59e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.04e+00 | 1.01e+00 | 2.04e+01 | 5.11e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 3.03e+00 | 2.75e+00 | 2.81e+01 | 1.08e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 8.93e-01 | 8.15e-01 | 1.98e+01 | 4.50e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.34e+00 | 2.34e+00 | 2.18e+01 | 1.08e-01 |
| 2d_binomial_logit_n1000_k10_cr | 2.43e+01 | 2.14e+01 | 8.89e+01 | 2.74e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.91e+01 | 7.60e+01 | 1.67e+02 | 4.74e-01 |
| 2d_gaussian_additive_n2000_k15_cr | 8.76e+00 | 7.71e+00 | 4.32e+01 | 2.03e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.14e+00 | 1.88e+00 | 3.48e+01 | 6.15e-02 |
| 2d_poisson_log_n1000_k10_cr | 9.20e+01 | 8.19e+01 | 1.37e+02 | 6.72e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 6.77e+00 | 6.26e+00 | 4.91e+01 | 1.38e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.64e+01 | 2.11e+01 | 8.28e+01 | 3.19e-01 |
| 4d_small_neighbourhood_n300 | 1.11e+01 | 8.41e+00 | 6.95e+01 | 1.59e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.80e+01 | 1.34e+01 | 1.01e+02 | 1.79e-01 |
| 5d_skewed_features_n5000 | 1.35e+02 | 1.29e+02 | 1.54e+02 | 8.76e-01 |
| 6d_heatmap_pricing_n8000 | 2.50e+02 | 2.26e+02 | 2.24e+02 | 1.12e+00 |
| 8d_neighbourhoods_like_n15000 | 1.02e+03 | 9.68e+02 | 6.76e+02 | 1.51e+00 |
