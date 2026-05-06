# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

## Bar A / B / C

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|

## Perf (median over N=5 fits, lower is better)

| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |
|---|---|---|---|---|
| 10d_gaussian_n3000_k8_cr | 8.30e+01 | 7.58e+01 | 2.93e+02 | 2.84e-01 |
| 1d_gaussian_low_signal_n1000_k10_cr | 1.66e+00 | 1.58e+00 | 5.02e+01 | 3.30e-02 |
| 1d_gaussian_near_linear_n500_k10_cr | 2.02e+00 | 1.75e+00 | 3.04e+01 | 6.66e-02 |
| 1d_gaussian_sigmoid_n300_k10_cr | 6.06e-01 | 5.78e-01 | 1.93e+01 | 3.15e-02 |
| 1d_gaussian_smooth_n1000_k50_cr | 2.02e+01 | 1.82e+01 | 6.68e+01 | 3.03e-01 |
| 1d_gaussian_smooth_n100_k10_cr | 5.09e-01 | 4.93e-01 | 1.80e+01 | 2.83e-02 |
| 1d_gaussian_smooth_n2000_k30_cr | 1.06e+01 | 9.88e+00 | 3.37e+01 | 3.15e-01 |
| 1d_gaussian_smooth_n500_k10_cr | 9.50e-01 | 8.92e-01 | 1.97e+01 | 4.81e-02 |
| 1d_gaussian_smooth_n500_k20_bs | 2.50e+00 | 2.32e+00 | 2.57e+01 | 9.71e-02 |
| 1d_gaussian_smooth_n50_k10_cr | 4.66e-01 | 4.52e-01 | 1.92e+01 | 2.43e-02 |
| 1d_gaussian_sparse_edges_n400_k10_cr | 7.22e-01 | 7.13e-01 | 2.10e+01 | 3.44e-02 |
| 1d_gaussian_step_n500_k10_cr | 7.85e-01 | 7.52e-01 | 1.96e+01 | 4.00e-02 |
| 1d_gaussian_wiggly_n500_k20_cr | 2.37e+00 | 2.27e+00 | 2.45e+01 | 9.67e-02 |
| 1d_poisson_log_n500_k10_cr | 3.81e+00 | 3.62e+00 | 6.47e+01 | 5.89e-02 |
| 1d_tw_log_n400_k20_cr | 2.06e+02 | 2.04e+02 | 2.17e+02 | 9.47e-01 |
| 1d_tweedie_log_n400_k20_cr_p15 | 7.68e+01 | 7.19e+01 | 7.37e+01 | 1.04e+00 |
| 2d_binomial_logit_n1000_k10_cr | 3.38e+01 | 2.81e+01 | 1.01e+02 | 3.35e-01 |
| 2d_binomial_logit_n200_k10_cr | 9.49e+00 | 8.58e+00 | 1.38e+02 | 6.88e-02 |
| 2d_binomial_logit_n5000_k10_cr | 6.67e+02 | 6.56e+02 | 1.71e+02 | 3.91e+00 |
| 2d_gamma_inverse_n1000_k10_cr | 3.00e+01 | 2.89e+01 | 1.94e+02 | 1.55e-01 |
| 2d_gamma_log_n1000_k10_cr | 2.64e+01 | 2.61e+01 | 1.42e+02 | 1.87e-01 |
| 2d_gamma_log_n200_k10_cr | 5.07e+00 | 4.88e+00 | 1.48e+02 | 3.44e-02 |
| 2d_gaussian_additive_n2000_k15_cr | 7.31e+00 | 7.17e+00 | 3.23e+01 | 2.26e-01 |
| 2d_gaussian_additive_n50000_k15_cr | 3.47e+02 | 3.01e+02 | 2.97e+02 | 1.17e+00 |
| 2d_gaussian_additive_n500_k10_cr | 1.98e+00 | 1.78e+00 | 2.63e+01 | 7.53e-02 |
| 2d_gaussian_bs_n1500_k15 | 5.68e+00 | 5.23e+00 | 3.35e+01 | 1.70e-01 |
| 2d_invgauss_log_n800_k10_cr | 2.69e+01 | 2.20e+01 | 2.10e+02 | 1.28e-01 |
| 2d_nb_log_n1000_k10_cr_theta2 | 2.37e+01 | 2.22e+01 | 1.56e+02 | 1.52e-01 |
| 2d_nb_profile_log_n1000_k10_cr | 2.87e+01 | 2.69e+01 | 1.97e+02 | 1.45e-01 |
| 2d_poisson_log_n1000_k10_cr | 3.22e+01 | 3.14e+01 | 1.53e+02 | 2.11e-01 |
| 2d_poisson_log_n200_k10_cr | 8.85e+00 | 7.90e+00 | 1.15e+02 | 7.68e-02 |
| 2d_poisson_log_n5000_k10_cr | 1.25e+02 | 1.19e+02 | 1.64e+02 | 7.60e-01 |
| 2d_quasibinomial_logit_n1000_k10_cr | 4.24e+01 | 3.93e+01 | 1.22e+02 | 3.47e-01 |
| 2d_quasipoisson_log_n1000_k10_cr | 2.45e+01 | 2.38e+01 | 1.28e+02 | 1.91e-01 |
| 3d_gaussian_mixed_n800_k10_cr | 5.64e+00 | 4.83e+00 | 4.66e+01 | 1.21e-01 |
| 3d_poisson_log_n2000_k10_cr | 8.00e+01 | 7.80e+01 | 1.64e+02 | 4.89e-01 |
| 4d_binomial_logit_n2000_k8_cr | 9.46e+01 | 9.36e+01 | 1.81e+02 | 5.24e-01 |
| 4d_gamma_log_n2000_k8_cr | 6.35e+01 | 5.70e+01 | 2.19e+02 | 2.90e-01 |
| 4d_gaussian_bs_n2000_k10 | 2.29e+01 | 2.23e+01 | 8.03e+01 | 2.85e-01 |
| 4d_gaussian_mixed_n1000_k10_cr | 1.34e+01 | 1.30e+01 | 6.36e+01 | 2.11e-01 |
| 4d_small_neighbourhood_n300 | 6.32e+00 | 5.98e+00 | 4.23e+01 | 1.49e-01 |
| 5d_gaussian_mixed_n1500_k8_cr | 1.05e+01 | 1.03e+01 | 8.27e+01 | 1.27e-01 |
| 5d_skewed_features_n5000 | 5.33e+01 | 5.19e+01 | 9.82e+01 | 5.43e-01 |
| 6d_heatmap_pricing_n8000 | 1.08e+02 | 1.02e+02 | 1.93e+02 | 5.56e-01 |
| 7d_neighbourhoods_compact_n3000 | 3.52e+01 | 3.43e+01 | 1.15e+02 | 3.06e-01 |
| 8d_neighbourhoods_like_n15000 | 5.94e+02 | 5.71e+02 | 5.68e+02 | 1.05e+00 |
