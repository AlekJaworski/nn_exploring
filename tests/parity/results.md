# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✗ | ✗ | ✗ | 1.11e-02 | 2.17e-02 | 2.44e-02 | 1.07e+00 | 1.28e-02 | 1.00e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | ✗ | ✗ | ✗ | 1.26e-03 | 6.64e-01 | 3.00e-03 | 4.11e-01 | 6.43e-04 | 6.13e-01 |
| 1d_gaussian_smooth_n100_k10_cr | ✗ | ✗ | ✗ | 1.63e-03 | 1.92e-02 | 2.98e-03 | 9.18e-02 | 6.26e-04 | 4.58e-01 |
| 1d_gaussian_smooth_n500_k10_cr | ✗ | ✗ | ✓ | 2.11e-04 | 1.13e-03 | 2.67e-04 | 7.89e-01 | 6.14e-06 | 6.50e-01 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.30e-02 | 9.94e-01 | 9.09e-02 | 1.01e+00 | 2.39e-03 | 9.92e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | ✗ | ✗ | ✓ | 2.26e-04 | 1.42e-02 | 2.46e-04 | 3.30e-02 | 1.01e-05 | 4.55e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | ✗ | ✗ | ✓ | 6.84e-04 | 1.68e-02 | 1.05e-03 | 1.51e+00 | 2.93e-05 | 7.74e-01 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e-01 | 6.75e-01 | 3.31e-01 | 9.31e-01 | 8.38e-01 | 1.00e+00 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.62e+00 | 3.15e-01 | 4.18e+01 | 8.53e-01 | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 2.14e-03 | 9.68e-01 | 3.26e-03 | 9.46e-01 | 2.29e-04 | 7.94e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✓ | 1.04e-03 | 3.29e-02 | 1.79e-03 | 1.02e+00 | 4.03e-06 | 6.95e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.25e+00 | 1.13e-01 | 1.38e+00 | 3.12e-01 | 5.39e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | ✗ | ✗ | ✗ | 1.31e-02 | 1.14e+00 | 2.02e-02 | 1.53e-01 | 1.77e-03 | 9.70e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✓ | 1.90e-03 | 8.15e-02 | 1.61e-03 | 5.04e-01 | 2.88e-04 | 1.00e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | ✗ | ✗ | ✗ | 1.12e-02 | 2.01e+01 | 2.00e-02 | 8.08e-01 | 1.69e-03 | 8.78e-01 |

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
| 1d_gaussian_near_linear_n500_k10_cr | 6 | 10 | 1 | -4.37e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 3 | 5 | 1 | -3.07e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 4 | 3 | 1 | -2.99e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 4 | 6 | — | -5.86e+01 | -6.43e+01 |
| 1d_gaussian_smooth_n500_k20_bs | 6 | 5 | 1 | -5.55e+01 | -6.40e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 3 | 6 | 2 | -1.62e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 6 | 5 | 1 | -1.58e+02 | -1.75e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 3 | 5 | 1 | -3.15e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 3 | 6 | 1 | -4.62e+01 | -5.80e+01 |
| 3d_gaussian_mixed_n800_k10_cr | 16 | 8 | 1 | -9.87e+01 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 14 | 10 | 1 | -1.24e+02 | -1.48e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 14 | 7 | 1 | -1.79e+02 | -2.04e+02 |

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 2.57e+00 | 1.83e+00 | 2.74e+01 | 9.37e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 1.50e+00 | 1.41e+00 | 1.75e+01 | 8.57e-02 |
| 1d_gaussian_smooth_n100_k10_cr | 4.96e-01 | 4.80e-01 | 1.94e+01 | 2.55e-02 |
| 1d_gaussian_smooth_n500_k10_cr | 1.59e+00 | 1.24e+00 | 2.21e+01 | 7.16e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 1.32e+01 | 1.14e+01 | 3.18e+01 | 4.15e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 9.30e-01 | 8.17e-01 | 1.93e+01 | 4.81e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5.69e+00 | 5.04e+00 | 2.20e+01 | 2.59e-01 |
| 2d_binomial_logit_n1000_k10_cr | 5.13e+01 | 4.62e+01 | 9.22e+01 | 5.57e-01 |
| 2d_gamma_log_n1000_k10_cr | 9.62e+00 | 8.85e+00 | 1.63e+02 | 5.92e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 2.01e+01 | 1.75e+01 | 3.93e+01 | 5.11e-01 |
| 2d_gaussian_additive_n500_k10_cr | 2.51e+00 | 2.37e+00 | 2.65e+01 | 9.49e-02 |
| 2d_poisson_log_n1000_k10_cr | 9.31e+00 | 8.54e+00 | 1.18e+02 | 7.92e-02 |
| 3d_gaussian_mixed_n800_k10_cr | 2.62e+01 | 2.55e+01 | 4.53e+01 | 5.77e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.39e+01 | 4.33e+01 | 6.82e+01 | 6.43e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 6.40e+01 | 6.14e+01 | 8.66e+01 | 7.39e-01 |
