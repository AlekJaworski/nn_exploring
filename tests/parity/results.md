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

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 6 | 10 | 1 | -4.37e+02 | -4.46e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 4 | 3 | 1 | -2.99e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 4 | 6 | — | -5.86e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 6 | 5 | 1 | -5.55e+01 | -6.40e+01 |
| 1d_gaussian_wiggly_n500_k20_cr | 6 | 5 | 1 | -1.58e+02 | -1.75e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 3 | 5 | 1 | -3.15e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 3 | 6 | 1 | -4.62e+01 | -5.80e+01 |
| 4d_gaussian_mixed_n1000_k10_cr | 14 | 10 | 1 | -1.24e+02 | -1.48e+02 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.88e+00 | 1.59e+00 | 3.62e+01 | 5.19e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 5.47e-01 | 5.25e-01 | 2.37e+01 | 2.31e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.67e+00 | 1.46e+00 | 2.34e+01 | 7.15e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 1.18e+01 | 1.12e+01 | 3.57e+01 | 3.31e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | 4.63e+00 | 4.34e+00 | 2.80e+01 | 1.65e-01 |
| 2d_binomial_logit_n1000_k10_cr | 4.48e+01 | 4.25e+01 | 1.16e+02 | 3.85e-01 |
| 2d_gamma_log_n1000_k10_cr | 1.02e+01 | 9.85e+00 | 2.11e+02 | 4.83e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 1.90e+01 | 1.80e+01 | 4.72e+01 | 4.02e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.51e+00 | 2.22e+00 | 2.78e+01 | 9.00e-02 |
| 2d_poisson_log_n1000_k10_cr | 8.33e+00 | 7.92e+00 | 1.65e+02 | 5.06e-02 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.39e+01 | 4.36e+01 | 7.20e+01 | 6.10e-01 |
