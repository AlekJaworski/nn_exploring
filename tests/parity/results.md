# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.79e+00 | 1.77e+00 | 3.06e+01 | 5.85e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.60e-01 | 6.34e-01 | 1.69e+01 | 3.89e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 6.30e-01 | 6.12e-01 | 1.58e+01 | 3.99e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 9.01e-01 | 8.58e-01 | 2.13e+01 | 4.23e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.68e+00 | 2.53e+00 | 2.89e+01 | 9.29e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.45e-01 | 7.15e-01 | 1.72e+01 | 4.33e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.36e+00 | 2.30e+00 | 2.09e+01 | 1.13e-01 |
| 2d_binomial_logit_n1000_k10_cr | 4.61e+01 | 4.21e+01 | 9.17e+01 | 5.03e-01 |
| 2d_gamma_log_n1000_k10_cr | 3.36e+01 | 3.32e+01 | 1.53e+02 | 2.19e-01 |
| 2d_gaussian_additive_n2000_k15_cr | 1.05e+01 | 7.13e+00 | 3.25e+01 | 3.22e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.93e+00 | 1.76e+00 | 2.30e+01 | 8.37e-02 |
| 2d_poisson_log_n1000_k10_cr | 3.31e+01 | 3.13e+01 | 1.17e+02 | 2.83e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 5.89e+00 | 5.49e+00 | 4.73e+01 | 1.24e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.57e+01 | 1.48e+01 | 6.06e+01 | 2.60e-01 |
| 4d_small_neighbourhood_n300 | 5.97e+00 | 5.76e+00 | 4.83e+01 | 1.24e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 9.99e+00 | 9.22e+00 | 7.50e+01 | 1.33e-01 |
| 5d_skewed_features_n5000 | 7.14e+01 | 6.29e+01 | 1.05e+02 | 6.78e-01 |
| 6d_heatmap_pricing_n8000 | 1.41e+02 | 1.37e+02 | 2.01e+02 | 7.01e-01 |
| 8d_neighbourhoods_like_n15000 | 6.63e+02 | 6.50e+02 | 5.89e+02 | 1.13e+00 |
