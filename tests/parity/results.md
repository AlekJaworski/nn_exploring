# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✗ | ✗ | ✗ | 1.10e-02 | 2.15e-02 | 2.42e-02 | 1.06e+00 | 1.27e-02 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | ✗ | ✗ | ✗ | 1.66e-03 | 1.96e-02 | 3.04e-03 | 9.18e-02 | 6.39e-04 | 4.57e-01 |
| 1d_gaussian_smooth_n500_k10_cr | ✗ | ✗ | ✓ | 2.11e-04 | 1.13e-03 | 2.67e-04 | 7.89e-01 | 6.14e-06 | 6.50e-01 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.94e-01 | 9.09e-02 | 1.01e+00 | 2.38e-03 | 9.92e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | ✗ | ✗ | ✓ | 6.83e-04 | 1.77e-02 | 1.04e-03 | 1.51e+00 | 2.90e-05 | 7.75e-01 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.75e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 1.00e+00 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.62e+00 | 3.15e-01 | 4.18e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 2.14e-03 | 7.21e-01 | 3.56e-03 | 9.47e-01 | 2.10e-04 | 7.94e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✓ | 1.04e-03 | 3.29e-02 | 1.79e-03 | 1.02e+00 | 4.03e-06 | 6.95e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e+00 | 1.13e-01 | 1.38e+00 | 3.12e-01 | 5.39e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✓ | 2.61e-03 | 1.41e-01 | 1.67e-03 | 5.04e-01 | 1.15e-04 | 1.00e+00 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 5.55e+00 | 5.16e+00 | 4.25e+01 | 1.31e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 1.55e+00 | 1.53e+00 | 2.04e+01 | 7.57e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 3.99e+00 | 3.77e+00 | 1.81e+01 | 2.20e-01 |
| 1d_gaussian_smooth_n500_k20_bs | 2.32e+01 | 2.10e+01 | 2.67e+01 | 8.70e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | 1.14e+01 | 1.05e+01 | 2.14e+01 | 5.34e-01 |
| 2d_binomial_logit_n1000_k10_cr | 4.17e+01 | 3.91e+01 | 9.25e+01 | 4.51e-01 |
| 2d_gamma_log_n1000_k10_cr | 8.58e+00 | 7.95e+00 | 1.61e+02 | 5.34e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 3.63e+01 | 3.47e+01 | 3.90e+01 | 9.30e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.08e+01 | 9.58e+00 | 3.05e+01 | 3.54e-01 |
| 2d_poisson_log_n1000_k10_cr | 1.09e+01 | 9.20e+00 | 1.34e+02 | 8.15e-02 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.53e+02 | 2.45e+02 | 5.94e+01 | 4.25e+00 |
