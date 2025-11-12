#!/usr/bin/env python3
"""
TDD Test: Compare exact REML criterion computation

This tests the REML criterion at the SAME lambda value to see if
the formula itself differs.
"""

import numpy as np
import pytest

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    importr('mgcv')
    HAS_RPY2 = True
except:
    HAS_RPY2 = False
    pytest.skip("rpy2 not available", allow_module_level=True)

# Test data
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.2 * np.random.randn(n)

def compute_reml_mgcv(k_val, lambda_val):
    """Compute REML using mgcv's data at a specific lambda"""
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['x_r'] = x
        ro.globalenv['y_r'] = y
        ro.globalenv['k_val'] = k_val
        ro.globalenv['lambda_test'] = lambda_val

        # Fit GAM to get S and X, then recompute at test lambda
        ro.r('''
            gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="bs"), method="REML")
            smooth_obj <- gam_fit$smooth[[1]]
            S <- smooth_obj$S[[1]]
            X_full <- predict(gam_fit, type="lpmatrix")
            smooth_first <- smooth_obj$first.para
            smooth_last <- smooth_obj$last.para
            X <- X_full[, smooth_first:smooth_last, drop=FALSE]
            w <- gam_fit$prior.weights
            if (is.null(w)) w <- rep(1, length(y_r))

            # Compute at test lambda
            W <- diag(w)
            XtWX <- t(X) %*% W %*% X
            A <- XtWX + lambda_test * S

            # Solve for beta
            y_weighted <- w * y_r
            b_rhs <- t(X) %*% y_weighted
            beta <- solve(A, b_rhs)

            # Compute RSS
            fitted <- as.vector(X %*% beta)
            residuals <- y_r - fitted
            RSS <- sum(w * residuals^2)

            # Penalty
            penalty <- as.numeric(t(beta) %*% S %*% beta)

            # RSS + lambda * penalty
            RSS_penalized <- RSS + lambda_test * penalty

            # Rank and phi
            rank_S <- qr(S)$rank
            phi <- RSS / (length(y_r) - rank_S)

            # Log determinants
            log_det_A <- determinant(A, logarithm=TRUE)$modulus[1]
            log_lambda_term <- rank_S * log(lambda_test)

            # REML
            REML <- (RSS_penalized/phi +
                     (length(y_r) - rank_S) * log(2*pi*phi) +
                     log_det_A -
                     log_lambda_term) / 2
        ''')

        result = {
            'lambda': lambda_val,
            'k': k_val,
            'RSS': float(ro.r('RSS')[0]),
            'penalty': float(ro.r('penalty')[0]),
            'RSS_penalized': float(ro.r('RSS_penalized')[0]),
            'rank_S': int(ro.r('rank_S')[0]),
            'phi': float(ro.r('phi')[0]),
            'log_det_A': float(ro.r('log_det_A')[0]),
            'log_lambda_term': float(ro.r('log_lambda_term')[0]),
            'REML': float(ro.r('REML')[0]),
            'S_norm': np.linalg.norm(np.array(ro.r('S')), 'fro'),
        }

        return result

@pytest.mark.parametrize("k_val", [5, 10, 20])
def test_reml_at_different_lambdas(k_val):
    """Test REML criterion at different lambda values"""
    # Test at several lambda values
    if k_val == 5:
        lambda_vals = [0.001, 0.003, 0.01, 0.1]
    elif k_val == 10:
        lambda_vals = [0.001, 0.1, 0.5, 1.0]
    else:  # k=20
        lambda_vals = [0.001, 1.0, 5.0, 10.0]

    print(f"\n{'='*80}")
    print(f"k = {k_val}: REML vs Lambda")
    print(f"{'='*80}")
    print(f"{'lambda':<12} {'REML':<15} {'RSS':<12} {'penalty':<12} {'log|A|':<12}")
    print(f"{'-'*80}")

    reml_values = []
    for lam in lambda_vals:
        result = compute_reml_mgcv(k_val, lam)
        reml_values.append(result['REML'])

        print(f"{lam:<12.6f} {result['REML']:<15.6f} {result['RSS']:<12.6f} "
              f"{result['penalty']:<12.6f} {result['log_det_A']:<12.6f}")

    # Find minimum
    min_idx = np.argmin(reml_values)
    opt_lambda = lambda_vals[min_idx]
    print(f"\nApproximate optimal lambda (from tested): {opt_lambda:.6f}")

@pytest.mark.parametrize("k_val", [10, 20])
def test_reml_gradient_direction(k_val):
    """Test REML gradient to see where it's decreasing"""
    # Start from a small lambda and see how REML changes

    if k_val == 10:
        lambdas = np.logspace(-3, 0, 20)  # 0.001 to 1
    else:  # k=20
        lambdas = np.logspace(-3, 1.5, 20)  # 0.001 to ~30

    print(f"\n{'='*80}")
    print(f"k = {k_val}: REML Gradient Test")
    print(f"{'='*80}")

    results = [compute_reml_mgcv(k_val, lam) for lam in lambdas]
    reml_vals = [r['REML'] for r in results]

    # Find minimum
    min_idx = np.argmin(reml_vals)
    opt_lambda = lambdas[min_idx]

    print(f"Optimal lambda found: {opt_lambda:.6f}")
    print(f"Minimum REML: {reml_vals[min_idx]:.6f}")

    # Show values around optimum
    print(f"\nValues around optimum:")
    start = max(0, min_idx - 2)
    end = min(len(lambdas), min_idx + 3)
    for i in range(start, end):
        marker = " <-- MIN" if i == min_idx else ""
        print(f"  lambda={lambdas[i]:.6f}: REML={reml_vals[i]:.6f}{marker}")

    # THIS IS THE KEY TEST:
    # For k=10, mgcv finds lambda ~ 0.54
    # For k=20, mgcv finds lambda ~ 9.25
    # But our Rust code finds lambda ~ 0.0005 for both!

    if k_val == 10:
        # mgcv's actual optimum
        mgcv_lambda = 0.542645
        assert abs(opt_lambda - mgcv_lambda) / mgcv_lambda < 0.5, \
            f"Found lambda {opt_lambda:.6f}, expected around {mgcv_lambda:.6f}"

    elif k_val == 20:
        # mgcv's actual optimum
        mgcv_lambda = 9.246667
        assert abs(opt_lambda - mgcv_lambda) / mgcv_lambda < 0.5, \
            f"Found lambda {opt_lambda:.6f}, expected around {mgcv_lambda:.6f}"

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
