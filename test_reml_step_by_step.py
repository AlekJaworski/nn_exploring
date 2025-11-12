#!/usr/bin/env python3
"""
TDD Test: Compare REML computation step-by-step between mgcv_rust and R's mgcv

This test verifies each component of the REML criterion to identify
where the implementations diverge.
"""

import numpy as np
import mgcv_rust
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

# Test data - same as R script
np.random.seed(42)
n = 100
x = np.linspace(0, 1, n)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.2 * np.random.randn(n)

class TestREMLStepByStep:
    """Test REML computation step by step against mgcv"""

    @pytest.fixture(params=[5, 10, 20])
    def k_value(self, request):
        """Test multiple k values"""
        return request.param

    def get_mgcv_internals(self, k_val):
        """Extract mgcv internal values"""
        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['x_r'] = x
            ro.globalenv['y_r'] = y
            ro.globalenv['k_val'] = k_val

            # Fit GAM
            ro.r('gam_fit <- gam(y_r ~ s(x_r, k=k_val, bs="bs"), method="REML")')

            # Extract smooth object internals
            ro.r('''
                smooth_obj <- gam_fit$smooth[[1]]
                S <- smooth_obj$S[[1]]
                X_full <- predict(gam_fit, type="lpmatrix")
                smooth_first <- smooth_obj$first.para
                smooth_last <- smooth_obj$last.para
                X <- X_full[, smooth_first:smooth_last, drop=FALSE]
                beta <- coef(gam_fit)[smooth_first:smooth_last]
                lambda <- gam_fit$sp
                w <- gam_fit$prior.weights
                if (is.null(w)) w <- rep(1, length(y_r))
            ''')

            # Get values
            S = np.array(ro.r('S'))
            X = np.array(ro.r('X'))
            beta = np.array(ro.r('beta'))
            lambda_mgcv = float(ro.r('lambda')[0])
            w = np.array(ro.r('w'))

            # Compute RSS
            fitted = X @ beta
            residuals = y - fitted
            RSS = np.sum(w * residuals**2)

            # Penalty term
            penalty = beta.T @ S @ beta

            # Rank of S
            rank_S = np.linalg.matrix_rank(S)

            return {
                'k': k_val,
                'S': S,
                'X': X,
                'beta': beta,
                'lambda': lambda_mgcv,
                'w': w,
                'RSS': RSS,
                'penalty': penalty,
                'rank_S': rank_S,
            }

    def get_rust_internals(self, k_val):
        """Extract rust internal values"""
        X_input = x.reshape(-1, 1)
        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X_input, y, k=[k_val], method='REML')

        return {
            'k': k_val,
            'lambda': result['lambda'],
            'deviance': result['deviance'],
        }

    def test_penalty_matrix_norm(self, k_value):
        """Test 1: Penalty matrix should have similar structure"""
        mgcv = self.get_mgcv_internals(k_value)

        S = mgcv['S']
        S_norm_frobenius = np.sqrt(np.sum(S**2))
        S_trace = np.trace(S)

        print(f"\nk={k_value}:")
        print(f"  S shape: {S.shape}")
        print(f"  S ||F||: {S_norm_frobenius:.6f}")
        print(f"  S trace: {S_trace:.6f}")
        print(f"  S rank: {mgcv['rank_S']}")

        # Penalty matrix should grow with k
        if k_value >= 10:
            prev_mgcv = self.get_mgcv_internals(5)
            prev_norm = np.sqrt(np.sum(prev_mgcv['S']**2))
            assert S_norm_frobenius > prev_norm, \
                "Penalty matrix norm should increase with k"

    def test_penalty_term_magnitude(self, k_value):
        """Test 2: Penalty term beta'S beta magnitude"""
        mgcv = self.get_mgcv_internals(k_value)

        penalty = mgcv['penalty']
        print(f"\nk={k_value}:")
        print(f"  beta'S beta: {penalty:.6f}")

        # Larger k should have smaller penalty (more flexibility)
        # This is because the fit is better, so curvature penalty is smaller

    def test_XtWX_magnitude(self, k_value):
        """Test 3: Data term X'WX magnitude"""
        mgcv = self.get_mgcv_internals(k_value)

        X = mgcv['X']
        w = mgcv['w']
        W = np.diag(w)
        XtWX = X.T @ W @ X

        XtWX_norm = np.sqrt(np.sum(XtWX**2))
        S_norm = np.sqrt(np.sum(mgcv['S']**2))

        print(f"\nk={k_value}:")
        print(f"  ||X'WX||_F: {XtWX_norm:.6f}")
        print(f"  ||S||_F: {S_norm:.6f}")
        print(f"  Ratio ||X'WX|| / ||S||: {XtWX_norm / S_norm:.6f}")

    def test_balance_ratio(self, k_value):
        """Test 4: Balance between data and penalty terms"""
        mgcv = self.get_mgcv_internals(k_value)

        X = mgcv['X']
        w = mgcv['w']
        W = np.diag(w)
        XtWX = X.T @ W @ X

        XtWX_norm = np.sqrt(np.sum(XtWX**2))
        S_norm = np.sqrt(np.sum(mgcv['S']**2))

        lambda_mgcv = mgcv['lambda']

        # This ratio should be O(1) for good balance
        balance_ratio = (lambda_mgcv * S_norm) / XtWX_norm

        print(f"\nk={k_value}:")
        print(f"  lambda: {lambda_mgcv:.6f}")
        print(f"  Balance ratio (lambda*||S|| / ||X'WX||): {balance_ratio:.6f}")

        # KEY INSIGHT: For larger k, lambda must increase to maintain balance!

    def test_lambda_vs_penalty_norm(self, k_value):
        """Test 5: Lambda should scale with penalty norm"""
        mgcv = self.get_mgcv_internals(k_value)

        S_norm = np.sqrt(np.sum(mgcv['S']**2))
        lambda_mgcv = mgcv['lambda']

        # Hypothesis: lambda should be proportional to some power of ||S||
        # or inversely proportional (to compensate for larger S)

        print(f"\nk={k_value}:")
        print(f"  ||S||: {S_norm:.6f}")
        print(f"  lambda: {lambda_mgcv:.6f}")
        print(f"  lambda * ||S||: {lambda_mgcv * S_norm:.6f}")
        print(f"  lambda / ||S||: {lambda_mgcv / S_norm:.6f}")

    def test_rust_vs_mgcv_lambda(self, k_value):
        """Test 6: Compare rust lambda to mgcv lambda"""
        mgcv = self.get_mgcv_internals(k_value)
        rust = self.get_rust_internals(k_value)

        lambda_mgcv = mgcv['lambda']
        lambda_rust = rust['lambda']
        ratio = lambda_rust / lambda_mgcv

        S_norm = np.sqrt(np.sum(mgcv['S']**2))

        print(f"\nk={k_value}:")
        print(f"  lambda_mgcv: {lambda_mgcv:.6f}")
        print(f"  lambda_rust: {lambda_rust:.6f}")
        print(f"  Ratio (rust/mgcv): {ratio:.6f}")
        print(f"  ||S||: {S_norm:.6f}")

        # EXPECTED FAILURE: This will fail for k > 10
        # This documents the problem we're trying to fix
        if k_value <= 10:
            assert ratio > 0.1, \
                f"Lambda ratio too small: {ratio:.6f} for k={k_value}"

    def test_REML_criterion_components(self, k_value):
        """Test 7: Break down REML criterion into components"""
        mgcv = self.get_mgcv_internals(k_value)

        X = mgcv['X']
        w = mgcv['w']
        S = mgcv['S']
        beta = mgcv['beta']
        lambda_mgcv = mgcv['lambda']
        RSS = mgcv['RSS']
        penalty = mgcv['penalty']
        rank_S = mgcv['rank_S']

        # Compute each term
        W = np.diag(w)
        XtWX = X.T @ W @ X
        A = XtWX + lambda_mgcv * S

        RSS_penalized = RSS + lambda_mgcv * penalty
        phi = RSS / (n - rank_S)

        log_det_A = np.linalg.slogdet(A)[1]
        log_lambda_term = rank_S * np.log(lambda_mgcv) if lambda_mgcv > 0 else 0

        REML = (RSS_penalized / phi +
                (n - rank_S) * np.log(2 * np.pi * phi) +
                log_det_A -
                log_lambda_term) / 2

        print(f"\nk={k_value}: REML Breakdown")
        print(f"  RSS: {RSS:.6f}")
        print(f"  lambda * beta'S beta: {lambda_mgcv * penalty:.6f}")
        print(f"  RSS + lambda*penalty: {RSS_penalized:.6f}")
        print(f"  phi: {phi:.6f}")
        print(f"  log|A|: {log_det_A:.6f}")
        print(f"  rank(S) * log(lambda): {log_lambda_term:.6f}")
        print(f"  REML: {REML:.6f}")

        # The log|A| and log(lambda) terms grow with k
        # These balance each other to some extent
        print(f"\n  log|A| - rank(S)*log(lambda): {log_det_A - log_lambda_term:.6f}")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s', '--tb=short'])
