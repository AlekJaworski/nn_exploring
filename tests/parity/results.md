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
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 1.54e-03 | 7.87e-01 | 3.91e-03 | 9.46e-01 | 3.64e-05 | 7.94e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✓ | 7.39e-03 | 9.19e-01 | 2.52e-02 | 1.01e+00 | 7.74e-04 | 7.93e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e+00 | 1.13e-01 | 1.38e+00 | 3.12e-01 | 5.39e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✗ | 5.63e-02 | 3.67e+00 | 9.45e-02 | 5.07e-01 | 7.05e-03 | 1.00e+00 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 3.74e-01 | 3.59e-01 | 2.68e+01 | 1.40e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 2.20e-01 | 2.16e-01 | 1.69e+01 | 1.30e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 7.65e-01 | 7.42e-01 | 1.78e+01 | 4.29e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 5.43e+00 | 4.93e+00 | 2.76e+01 | 1.97e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | 8.38e-01 | 7.86e-01 | 2.24e+01 | 3.74e-02 |
| 2d_binomial_logit_n1000_k10_cr | 4.13e+01 | 3.78e+01 | 9.06e+01 | 4.56e-01 |
| 2d_gamma_log_n1000_k10_cr | 7.17e+00 | 6.83e+00 | 1.45e+02 | 4.96e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 3.21e+01 | 3.02e+01 | 3.19e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 8.07e-01 | 7.98e-01 | 2.46e+01 | 3.28e-02 |
| 2d_poisson_log_n1000_k10_cr | 7.71e+00 | 6.78e+00 | 1.07e+02 | 7.23e-02 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.45e+00 | 2.44e+00 | 5.88e+01 | 4.16e-02 |
