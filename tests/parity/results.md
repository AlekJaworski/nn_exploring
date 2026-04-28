# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✗ | ✗ | ✗ | 1.11e-02 | 2.17e-02 | 2.44e-02 | 1.07e+00 | 1.28e-02 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | ✗ | ✗ | ✗ | 1.63e-03 | 1.92e-02 | 2.98e-03 | 9.18e-02 | 6.26e-04 | 4.58e-01 |
| 1d_gaussian_smooth_n500_k10_cr | ✗ | ✗ | ✓ | 2.11e-04 | 1.13e-03 | 2.67e-04 | 7.89e-01 | 6.14e-06 | 6.50e-01 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.94e-01 | 9.09e-02 | 1.01e+00 | 2.39e-03 | 9.92e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | ✗ | ✗ | ✓ | 6.84e-04 | 1.68e-02 | 1.05e-03 | 1.51e+00 | 2.93e-05 | 7.74e-01 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.75e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 1.00e+00 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.62e+00 | 3.15e-01 | 4.18e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 2.14e-03 | 9.68e-01 | 3.26e-03 | 9.46e-01 | 2.29e-04 | 7.94e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✓ | 1.04e-03 | 3.29e-02 | 1.79e-03 | 1.02e+00 | 4.03e-06 | 6.95e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e+00 | 1.13e-01 | 1.38e+00 | 3.12e-01 | 5.39e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✓ | 1.90e-03 | 8.15e-02 | 1.61e-03 | 5.04e-01 | 2.88e-04 | 1.00e+00 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.49e+00 | 1.48e+00 | 2.76e+01 | 5.39e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 6.58e-01 | 4.70e-01 | 1.95e+01 | 3.37e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.35e+00 | 1.29e+00 | 1.84e+01 | 7.32e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 1.11e+01 | 9.80e+00 | 3.08e+01 | 3.61e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | 4.07e+00 | 4.00e+00 | 2.47e+01 | 1.65e-01 |
| 2d_binomial_logit_n1000_k10_cr | 5.55e+01 | 5.11e+01 | 1.26e+02 | 4.41e-01 |
| 2d_gamma_log_n1000_k10_cr | 1.18e+01 | 1.13e+01 | 2.14e+02 | 5.49e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 2.41e+01 | 2.28e+01 | 5.05e+01 | 4.78e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.89e+00 | 2.70e+00 | 3.52e+01 | 8.21e-02 |
| 2d_poisson_log_n1000_k10_cr | 1.30e+01 | 1.04e+01 | 1.53e+02 | 8.50e-02 |
| 4d_gaussian_mixed_n1000_k10_cr | 5.82e+01 | 5.74e+01 | 8.65e+01 | 6.73e-01 |
