# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✓ | ✓ | ✓ | 6.04e-08 | 7.81e-08 | 9.28e-08 | 1.05e+00 | 3.99e-08 | 1.47e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✓ | ✓ | ✓ | 8.98e-09 | 1.49e-05 | 2.59e-08 | 4.11e-01 | 4.16e-09 | 1.91e-06 |
| 1d_gaussian_smooth_n100_k10_cr | ✓ | ✓ | ✓ | 3.25e-08 | 3.85e-07 | 5.70e-08 | 9.29e-02 | 1.19e-08 | 1.53e-06 |
| 1d_gaussian_smooth_n500_k10_cr | ✓ | ✓ | ✓ | 4.31e-09 | 3.99e-08 | 2.15e-09 | 7.89e-01 | 1.98e-10 | 9.16e-07 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.94e-01 | 9.48e-02 | 1.00e+00 | 2.76e-03 | 9.74e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✓ | ✓ | ✓ | 1.10e-09 | 3.06e-07 | 2.15e-09 | 3.30e-02 | 1.93e-10 | 7.75e-08 |
| 1d_gaussian_wiggly_n500_k20_cr | ✓ | ✓ | ✓ | 8.08e-10 | 2.07e-07 | 3.67e-09 | 1.51e+00 | 7.63e-11 | 1.17e-07 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.74e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 9.99e-01 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.91e+00 | 3.73e-01 | 5.50e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✓ | ✓ | ✓ | 1.19e-09 | 1.67e-06 | 3.96e-09 | 9.47e-01 | 1.07e-10 | 1.53e-07 |
| 2d_gaussian_additive_n500_k10_cr | ✓ | ✓ | ✓ | 1.26e-07 | 3.20e-05 | 3.75e-07 | 1.02e+00 | 1.18e-08 | 8.68e-06 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.10e+00 | 9.22e-02 | 1.47e+00 | 3.01e-01 | 5.40e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | ✗ | ✓ | ✓ | 1.79e-05 | 2.41e-03 | 2.58e-05 | 1.54e-01 | 2.98e-06 | 5.08e-03 |
| 4d_gaussian_mixed_n1000_k10_cr | ✓ | ✓ | ✓ | 3.76e-07 | 1.50e-05 | 2.05e-07 | 5.04e-01 | 5.28e-08 | 3.78e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | ✓ | ✓ | ✓ | 1.18e-06 | 1.92e-03 | 1.57e-06 | 8.09e-01 | 1.39e-07 | 2.80e-04 |

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 6.04e-08 | 4.18e+07 | 4.90e+07 | 1.17e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | 8.98e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 3.25e-08 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 4.31e-09 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 1.10e-09 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 8.08e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.19e-09 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.26e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.79e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 3.76e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.18e-06 | 2.42e+00 | 2.42e+00 | 1.00e+00 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 9 | 10 | — | -4.45e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 5 | 5 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 4 | 3 | 1 | -7.72e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 6 | 1 | -6.37e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 5 | 5 | — | -6.11e+01 | -6.40e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 6 | 1 | -1.69e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 5 | — | -1.74e+02 | -1.75e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 6 | 5 | 1 | -3.32e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 6 | 1 | -5.68e+01 | -5.80e+01 |
| 3d_gaussian_mixed_n800_k10_cr | 9 | 8 | 1 | -1.14e+02 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 14 | 10 | 1 | -1.46e+02 | -1.48e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.04e+02 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 3.24e+00 | 3.12e+00 | 2.52e+01 | 1.28e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 1.39e+00 | 1.32e+00 | 3.30e+01 | 4.21e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 1.78e+00 | 1.76e+00 | 2.79e+01 | 6.37e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 2.71e+00 | 2.68e+00 | 2.31e+01 | 1.18e-01 |
| 1d_gaussian_smooth_n500_k20_bs | 1.08e+01 | 9.87e+00 | 2.97e+01 | 3.63e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 1.38e+00 | 1.29e+00 | 1.52e+01 | 9.04e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 4.97e+00 | 4.32e+00 | 2.18e+01 | 2.28e-01 |
| 2d_binomial_logit_n1000_k10_cr | 1.37e+01 | 1.24e+01 | 8.91e+01 | 1.53e-01 |
| 2d_gamma_log_n1000_k10_cr | 8.65e+00 | 7.40e+00 | 1.55e+02 | 5.58e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 2.94e+01 | 2.79e+01 | 3.30e+01 | 8.93e-01 |
| 2d_gaussian_additive_n500_k10_cr | 7.97e+00 | 6.38e+00 | 2.57e+01 | 3.10e-01 |
| 2d_poisson_log_n1000_k10_cr | 9.43e+00 | 7.92e+00 | 1.11e+02 | 8.48e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 4.03e+01 | 3.87e+01 | 4.41e+01 | 9.15e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.68e+02 | 1.62e+02 | 6.15e+01 | 2.73e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.44e+02 | 1.39e+02 | 8.09e+01 | 1.79e+00 |
