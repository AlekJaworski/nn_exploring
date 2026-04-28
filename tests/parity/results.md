# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | 1.88e-07 | 1.06e+08 | 4.90e+07 | 4.64e-01 |
| 1d_gaussian_sigmoid_n300_k10_cr | 3.72e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 1.11e-07 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.09e-09 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 4.63e-09 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 6.02e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 1.32e-09 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.34e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 1.79e-05 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 4.93e-08 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.35e-06 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
