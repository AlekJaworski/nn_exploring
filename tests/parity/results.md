# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## mgcv_exact mode (Stage 4)

Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. Bar at 1e-3 absolute. λ values shown for the first smooth only.

| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 4.08e-06 | 2.11e+00 | 2.11e+00 | 1.00e+00 |
| 1d_gaussian_low_signal_n1000_k10_cr | 7.75e-07 | 2.37e+03 | 2.37e+03 | 1.00e+00 |
| 1d_gaussian_near_linear_n500_k10_cr | 1.71e-07 | 2.01e+07 | 2.51e+07 | 1.25e+00 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4.13e-09 | 1.13e+01 | 1.13e+01 | 1.00e+00 |
| 1d_gaussian_smooth_n1000_k50_cr | 3.47e-07 | 1.58e+03 | 1.58e+03 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | 3.49e-10 | 5.55e+00 | 5.55e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.04e-07 | 3.19e+02 | 3.19e+02 | 1.00e+00 |
| 1d_gaussian_smooth_n500_k10_cr | 2.38e-08 | 5.10e+00 | 5.10e+00 | 1.00e+00 |
| 1d_gaussian_smooth_n50_k10_cr | 5.09e-09 | 5.08e+00 | 5.08e+00 | 1.00e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5.95e-10 | 9.95e+00 | 9.95e+00 | 1.00e+00 |
| 1d_gaussian_step_n500_k10_cr | 2.57e-08 | 6.51e-01 | 6.51e-01 | 1.00e+00 |
| 1d_gaussian_wiggly_n500_k20_cr | 3.85e-10 | 4.15e-01 | 4.15e-01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | 9.29e-08 | 2.63e+01 | 2.63e+01 | 1.00e+00 |
| 2d_gaussian_additive_n50000_k15_cr | 2.18e-08 | 3.65e+01 | 3.65e+01 | 1.00e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.66e-07 | 5.21e+00 | 5.21e+00 | 1.00e+00 |
| 2d_gaussian_bs_n1500_k15 | 2.26e-07 | 4.42e+00 | 4.42e+00 | 1.00e+00 |
| 3d_gaussian_mixed_n800_k10_cr | 3.77e-06 | 5.22e+00 | 5.22e+00 | 1.00e+00 |
| 4d_gaussian_bs_n2000_k10 | 5.48e-07 | 7.62e-01 | 7.62e-01 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | 2.67e-07 | 5.16e+00 | 5.16e+00 | 1.00e+00 |
| 4d_small_neighbourhood_n300 | 1.24e-05 | 1.83e+08 | 2.16e+08 | 1.18e+00 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.25e-07 | 2.42e+00 | 2.42e+00 | 1.00e+00 |
| 5d_skewed_features_n5000 | 3.03e-05 | 1.36e+01 | 1.36e+01 | 1.00e+00 |
| 6d_heatmap_pricing_n8000 | 1.06e-04 | 4.56e+00 | 4.55e+00 | 9.99e-01 |
| 7d_neighbourhoods_compact_n3000 | 4.14e-05 | 9.57e+04 | 9.88e+04 | 1.03e+00 |
| 8d_neighbourhoods_like_n15000 | 4.15e-04 | 2.09e+07 | 1.96e+09 | 9.39e+01 |

## Newton trajectory vs mgcv (Stage 3)

First Newton iter where rust's REML score diverges from mgcv's by >5% of mgcv's score range. `—` means mgcv_rust stayed within 5% throughout the run.

| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |
|---|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 13 | 11 | 1 | -4.03e+02 | -4.03e+02 |
| 1d_gaussian_low_signal_n1000_k10_cr | 6 | 7 | 1 | 7.55e+02 | 7.55e+02 |
| 1d_gaussian_near_linear_n500_k10_cr | 10 | 8 | 1 | -4.45e+02 | -4.45e+02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 4 | 4 | 1 | -3.18e+02 | -3.18e+02 |
| 1d_gaussian_smooth_n1000_k50_cr | 6 | 4 | 1 | -2.13e+02 | -2.13e+02 |
| 1d_gaussian_smooth_n100_k10_cr | 5 | 4 | 1 | -7.72e+00 | -7.72e+00 |
| 1d_gaussian_smooth_n2000_k30_cr | 6 | 4 | 1 | -3.54e+02 | -3.54e+02 |
| 1d_gaussian_smooth_n500_k10_cr | 5 | 5 | 1 | -6.37e+01 | -6.37e+01 |
| 1d_gaussian_smooth_n50_k10_cr | 5 | 3 | 1 | 6.47e+00 | 6.47e+00 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 5 | 4 | 1 | -1.69e+02 | -1.69e+02 |
| 1d_gaussian_step_n500_k10_cr | 4 | 5 | — | -2.54e+02 | -2.54e+02 |
| 1d_gaussian_wiggly_n500_k20_cr | 5 | 6 | — | -1.74e+02 | -1.74e+02 |
| 2d_gaussian_additive_n2000_k15_cr | 5 | 4 | 1 | -3.32e+02 | -3.32e+02 |
| 2d_gaussian_additive_n50000_k15_cr | 4 | 7 | 1 | -9.25e+03 | -9.25e+03 |
| 2d_gaussian_additive_n500_k10_cr | 5 | 5 | 1 | -5.68e+01 | -5.68e+01 |
| 2d_gaussian_bs_n1500_k15 | 4 | 5 | — | -2.47e+02 | -2.47e+02 |
| 3d_gaussian_mixed_n800_k10_cr | 8 | 6 | 1 | -1.14e+02 | -1.14e+02 |
| 4d_gaussian_bs_n2000_k10 | 11 | 9 | 1 | -3.56e+02 | -3.56e+02 |
| 4d_gaussian_mixed_n1000_k10_cr | 12 | 10 | 1 | -1.46e+02 | -1.46e+02 |
| 4d_small_neighbourhood_n300 | 13 | 10 | 1 | -1.30e+02 | -1.30e+02 |
| 5d_gaussian_mixed_n1500_k8_cr | 8 | 7 | 1 | -2.01e+02 | -2.01e+02 |
| 5d_skewed_features_n5000 | 11 | 7 | 1 | -4.35e+03 | -4.35e+03 |
| 6d_heatmap_pricing_n8000 | 9 | 6 | 1 | -6.79e+03 | -6.79e+03 |
| 7d_neighbourhoods_compact_n3000 | 10 | 7 | 1 | -2.55e+03 | -2.55e+03 |
| 8d_neighbourhoods_like_n15000 | 13 | 9 | 1 | -1.32e+04 | -1.32e+04 |
| 1d_gaussian_smooth_n500_k20_bs | 4 | 4 | 1 | -6.33e+01 | -6.33e+01 |
| 1d_poisson_log_n500_k10_cr | 4 | 3 | 1 | 1.10e+03 | 1.10e+03 |
| 1d_tweedie_log_n400_k20_cr_p15 | 6 | 6 | — | 6.51e+02 | 6.53e+02 |
| 2d_binomial_logit_n1000_k10_cr | 7 | 4 | 1 | 5.23e+02 | 5.23e+02 |
| 2d_binomial_logit_n200_k10_cr | 10 | 9 | 1 | 9.58e+01 | 9.58e+01 |
| 2d_binomial_logit_n5000_k10_cr | 14 | 7 | 1 | 2.53e+03 | 2.53e+03 |
| 2d_gamma_inverse_n1000_k10_cr | 10 | 7 | — | 1.28e+03 | 1.28e+03 |
| 2d_gamma_log_n1000_k10_cr | 11 | 8 | 1 | 1.84e+03 | 1.83e+03 |
| 2d_gamma_log_n200_k10_cr | 106 | 5 | 1 | 3.58e+02 | 3.54e+02 |
| 2d_invgauss_log_n800_k10_cr | 5 | 6 | 1 | 1.19e+03 | 1.19e+03 |
| 2d_nb_log_n1000_k10_cr_theta2 | 7 | 5 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_nb_profile_log_n1000_k10_cr | 7 | 4 | 1 | 2.76e+03 | 2.76e+03 |
| 2d_poisson_log_n1000_k10_cr | 9 | 7 | 1 | 2.30e+03 | 2.30e+03 |
| 2d_poisson_log_n200_k10_cr | 11 | 9 | 1 | 4.36e+02 | 4.36e+02 |
| 2d_poisson_log_n5000_k10_cr | 8 | 6 | 1 | 1.14e+04 | 1.14e+04 |
| 2d_quasibinomial_logit_n1000_k10_cr | 9 | 10 | 1 | 1.44e+03 | 5.19e+02 |
| 2d_quasipoisson_log_n1000_k10_cr | 7 | 6 | 1 | 2.09e+03 | 1.17e+03 |
| 3d_poisson_log_n2000_k10_cr | 9 | 6 | 1 | 4.31e+03 | 4.31e+03 |
| 4d_binomial_logit_n2000_k8_cr | 8 | 6 | 1 | 1.15e+03 | 1.15e+03 |
| 4d_gamma_log_n2000_k8_cr | 8 | 7 | 1 | 3.34e+03 | 3.33e+03 |
