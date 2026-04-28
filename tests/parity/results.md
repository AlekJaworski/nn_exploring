# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.09e-02 | 3.31e+02 | 4.90e+07 | 1.48e+05 |
| 1d_gaussian_sigmoid_n300_k10_cr | 2.48e-04 | 1.14e+01 | 1.13e+01 | 9.95e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 2.30e-04 | 5.57e+00 | 5.55e+00 | 9.98e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 2.24e-04 | 5.10e+00 | 5.10e+00 | 9.99e-01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 2.19e-04 | 9.92e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.87e-04 | 4.14e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 2.20e-03 | 2.79e+01 | 2.63e+01 | 9.41e-01 |
| 2d_gaussian_additive_n500_k10_cr | 1.02e-03 | 5.25e+00 | 5.21e+00 | 9.92e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 1.31e-02 | 5.28e+00 | 5.22e+00 | 9.88e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.87e-03 | 5.21e+00 | 5.16e+00 | 9.90e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.11e-02 | 2.43e+00 | 2.42e+00 | 9.97e-01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 6 | 10 | 1 | -4.37e+02 | -4.46e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 3 | 5 | 1 | -3.07e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 4 | 3 | 1 | -2.99e+00 | -8.36e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 4 | 6 | — | -5.86e+01 | -6.43e+01 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 3 | 6 | 2 | -1.62e+02 | -1.70e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 6 | 5 | 1 | -1.58e+02 | -1.75e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 3 | 5 | 1 | -3.15e+02 | -3.33e+02 |
| 2d_gaussian_additive_n500_k10_cr | 3 | 6 | 1 | -4.62e+01 | -5.80e+01 |
| 3d_gaussian_mixed_n800_k10_cr | 16 | 8 | 1 | -9.87e+01 | -1.16e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 14 | 10 | 1 | -1.24e+02 | -1.48e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 14 | 7 | 1 | -1.79e+02 | -2.04e+02 |
| 1d_gaussian_smooth_n500_k20_bs | 6 | 5 | 1 | -5.55e+01 | -6.40e+01 |
