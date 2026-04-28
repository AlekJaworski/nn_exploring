# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.55e+00 | 1.52e+00 | 3.25e+01 | 4.77e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 1.53e+00 | 1.49e+00 | 2.01e+01 | 7.58e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 8.64e-01 | 5.66e-01 | 1.88e+01 | 4.59e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.19e+00 | 1.13e+00 | 2.21e+01 | 5.38e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 1.16e+01 | 1.04e+01 | 3.04e+01 | 3.81e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.09e-01 | 6.95e-01 | 2.00e+01 | 3.55e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 4.67e+00 | 4.07e+00 | 2.22e+01 | 2.10e-01 |
| 2d_binomial_logit_n1000_k10_cr | 4.47e+01 | 4.11e+01 | 8.74e+01 | 5.11e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.64e+00 | 7.49e+00 | 1.60e+02 | 4.76e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 1.76e+01 | 1.57e+01 | 3.46e+01 | 5.07e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.05e+00 | 1.88e+00 | 2.54e+01 | 8.09e-02 |
| 2d_poisson_log_n1000_k10_cr | 7.58e+00 | 7.32e+00 | 1.18e+02 | 6.41e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 2.50e+01 | 2.27e+01 | 4.34e+01 | 5.76e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.27e+01 | 3.92e+01 | 6.89e+01 | 6.19e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 5.58e+01 | 5.40e+01 | 8.14e+01 | 6.85e-01 |
