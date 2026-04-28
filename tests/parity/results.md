# Parity results

Bar A is `ok` flag from `predict()` agreement at the given tolerance (rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` columns give the actual numbers so ratchets are easy to see. Bar B and C are tracked, not blocking.

| case | A train | A test | A extrap | train abs | train rel | extrap abs | β maxabsdiff | dev rel | λ relerr |
|---|---|---|---|---|---|---|---|---|---|
| 1d_gaussian_near_linear_n500_k10_cr | ✗ | ✗ | ✗ | 2.39e-02 | 4.46e-02 | 6.64e-02 | 1.99e+00 | 2.10e-02 | 1.00e+00 |
| 1d_gaussian_smooth_n100_k10_cr | ✗ | ✗ | ✗ | 1.39e-02 | 5.26e-02 | 1.59e-02 | 2.23e-01 | 5.63e-03 | 3.67e-01 |
| 1d_gaussian_smooth_n500_k10_cr | ✗ | ✗ | ✗ | 7.08e-03 | 1.22e-01 | 2.49e-02 | 7.39e-01 | 7.32e-04 | 2.58e-01 |
| 1d_gaussian_smooth_n500_k20_bs | ✗ | ✗ | ✗ | 3.28e-02 | 9.89e-01 | 1.10e-01 | 9.35e-01 | 3.60e-03 | 9.81e-01 |
| 1d_gaussian_wiggly_n500_k20_cr | ✗ | ✗ | ✗ | 3.23e-02 | 2.60e+00 | 1.17e-02 | 1.68e+00 | 6.22e-04 | 3.47e-02 |
| 2d_binomial_logit_n1000_k10_cr | ✗ | ✗ | ✗ | 1.23e-01 | 6.77e-01 | 3.29e-01 | — | 8.38e-01 | 9.99e-01 |
| 2d_gamma_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.88e+00 | 3.67e-01 | 5.22e+01 | — | 1.31e+01 | 1.00e+00 |
| 2d_gaussian_additive_n2000_k15_cr | ✗ | ✗ | ✓ | 5.62e-03 | 1.57e+01 | 1.55e-02 | — | 2.78e-04 | 1.87e-01 |
| 2d_gaussian_additive_n500_k10_cr | ✗ | ✗ | ✗ | 1.73e-02 | 4.18e+00 | 3.46e-02 | — | 1.81e-03 | 3.55e-01 |
| 2d_poisson_log_n1000_k10_cr | ✗ | ✗ | ✗ | 1.12e+00 | 9.32e-02 | 1.50e+00 | — | 5.40e+00 | 1.00e+00 |
| 4d_gaussian_mixed_n1000_k10_cr | ✗ | ✗ | ✗ | 6.50e-02 | 4.52e+00 | 1.14e-01 | — | 6.51e-03 | 1.00e+00 |
