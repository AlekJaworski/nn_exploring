# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.31e+00 | 2.01e+00 | 2.75e+01 | 8.42e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.09e-01 | 5.85e-01 | 1.96e+01 | 3.11e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 6.61e-01 | 6.49e-01 | 2.08e+01 | 3.17e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 9.20e-01 | 9.04e-01 | 2.10e+01 | 4.38e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.72e+00 | 2.63e+00 | 2.98e+01 | 9.13e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.60e-01 | 7.47e-01 | 1.92e+01 | 3.96e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.61e+00 | 2.30e+00 | 2.29e+01 | 1.14e-01 |
| 2d_binomial_logit_n1000_k10_cr | 2.36e+01 | 2.18e+01 | 9.27e+01 | 2.54e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.61e+01 | 2.57e+01 | 1.53e+02 | 1.71e-01 |
| 2d_gaussian_additive_n2000_k15_cr | 8.04e+00 | 7.71e+00 | 3.95e+01 | 2.03e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.21e+00 | 2.17e+00 | 3.07e+01 | 7.22e-02 |
| 2d_poisson_log_n1000_k10_cr | 6.42e+01 | 6.27e+01 | 1.41e+02 | 4.56e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 7.86e+00 | 7.47e+00 | 5.75e+01 | 1.37e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.25e+01 | 2.04e+01 | 7.68e+01 | 2.93e-01 |
| 4d_small_neighbourhood_n300 | 8.52e+00 | 7.97e+00 | 6.22e+01 | 1.37e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.48e+01 | 1.44e+01 | 1.07e+02 | 1.38e-01 |
| 5d_skewed_features_n5000 | 1.13e+02 | 1.06e+02 | 1.40e+02 | 8.11e-01 |
| 6d_heatmap_pricing_n8000 | 1.98e+02 | 1.93e+02 | 2.56e+02 | 7.73e-01 |
| 8d_neighbourhoods_like_n15000 | 6.89e+02 | 6.67e+02 | 7.32e+02 | 9.41e-01 |
