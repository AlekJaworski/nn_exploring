# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✗ | ✗ | ✗ | 2.12e-02 | 4.10e-02 | 4.52e-02 | 1.08e+00 | 2.04e-02 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | ✗ | ✗ | ✓ | 1.10e-03 | 1.31e-02 | 1.82e-03 | 9.37e-02 | 3.75e-04 | 5.22e-01 |
| 1d_gaussian_smooth_n500_k10_cr | ✗ | ✗ | ✓ | 1.48e-03 | 1.23e-02 | 1.13e-03 | 7.89e-01 | 6.13e-05 | 7.69e-01 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.90e-02 | 1.09e+00 | 1.78e-01 | 9.77e-01 | 7.51e-03 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | ✗ | ✗ | ✓ | 5.48e-04 | 1.14e-01 | 1.04e-03 | 1.51e+00 | 6.31e-06 | 7.87e-01 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.75e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 1.00e+00 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.62e+00 | 3.15e-01 | 4.18e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 2.14e-03 | 7.21e-01 | 3.56e-03 | 9.47e-01 | 2.10e-04 | 7.94e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✓ | 7.39e-03 | 9.19e-01 | 2.52e-02 | 1.01e+00 | 7.74e-04 | 7.93e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e+00 | 1.13e-01 | 1.38e+00 | 3.12e-01 | 5.39e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✗ | 5.63e-02 | 3.67e+00 | 9.45e-02 | 5.07e-01 | 7.05e-03 | 1.00e+00 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 4.13e-01 | 3.96e-01 | 2.78e+01 | 1.48e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 2.36e-01 | 2.18e-01 | 2.13e+01 | 1.11e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 7.12e-01 | 5.02e-01 | 1.99e+01 | 3.57e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 5.16e+00 | 4.53e+00 | 2.99e+01 | 1.73e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | 8.39e-01 | 7.60e-01 | 2.51e+01 | 3.34e-02 |
| 2d_binomial_logit_n1000_k10_cr | 4.25e+01 | 3.99e+01 | 9.86e+01 | 4.31e-01 |
| 2d_gamma_log_n1000_k10_cr | 8.21e+00 | 7.66e+00 | 1.72e+02 | 4.78e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 3.72e+01 | 3.69e+01 | 4.05e+01 | 9.17e-01 |
| 2d_gaussian_additive_n500_k10_cr | 8.33e-01 | 8.08e-01 | 2.71e+01 | 3.07e-02 |
| 2d_poisson_log_n1000_k10_cr | 9.54e+00 | 9.04e+00 | 1.33e+02 | 7.15e-02 |
| 4d_gaussian_mixed_n1000_k10_cr | 3.11e+00 | 2.66e+00 | 7.06e+01 | 4.40e-02 |
