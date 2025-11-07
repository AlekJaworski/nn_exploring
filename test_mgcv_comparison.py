#!/usr/bin/env python3
"""
Unit tests comparing mgcv_rust to R's mgcv package using rpy2
"""

import unittest
import numpy as np
import mgcv_rust

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    numpy2ri.activate()
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("Warning: rpy2 not available. Install with: pip install rpy2")


class TestMgcvComparison(unittest.TestCase):
    """Compare mgcv_rust results to R's mgcv package"""

    @classmethod
    def setUpClass(cls):
        if not HAS_RPY2:
            raise unittest.SkipTest("rpy2 not available")

        # Import R packages
        try:
            cls.mgcv = importr('mgcv')
            cls.stats = importr('stats')
        except Exception as e:
            raise unittest.SkipTest(f"R mgcv package not available: {e}")

    def setUp(self):
        """Set up test data"""
        np.random.seed(42)

        # Simple sine wave
        self.n = 100
        self.x = np.linspace(0, 1, self.n)
        self.y = np.sin(2 * np.pi * self.x) + 0.2 * np.random.randn(self.n)

    def test_1var_predictions_match(self):
        """Test that predictions match between implementations"""

        # Fit with mgcv_rust
        X = self.x.reshape(-1, 1)
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X, self.y, k=[10], method='REML')
        pred_rust = gam_rust.predict(X)

        # Fit with R mgcv
        ro.globalenv['x'] = self.x
        ro.globalenv['y'] = self.y
        ro.r('gam_fit <- gam(y ~ s(x, k=10, bs="bs"), method="REML")')
        pred_r = np.array(ro.r('predict(gam_fit)'))

        # Compare predictions
        # Allow for scale differences (mgcv centers differently)
        corr = np.corrcoef(pred_rust, pred_r)[0, 1]
        rmse_diff = np.sqrt(np.mean((pred_rust - pred_r)**2))

        print(f"\n1-var predictions:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE difference: {rmse_diff:.4f}")

        # High correlation expected (>0.99)
        self.assertGreater(corr, 0.95,
                          f"Predictions should be highly correlated (got {corr:.4f})")

    def test_lambda_similar(self):
        """Test that smoothing parameters are in similar range"""

        # Fit with mgcv_rust
        X = self.x.reshape(-1, 1)
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X, self.y, k=[10], method='REML')
        lambda_rust = result_rust['lambda']

        # Fit with R mgcv
        ro.globalenv['x'] = self.x
        ro.globalenv['y'] = self.y
        ro.r('gam_fit <- gam(y ~ s(x, k=10, bs="bs"), method="REML")')
        lambda_r = np.array(ro.r('gam_fit$sp'))[0]

        print(f"\nLambda comparison:")
        print(f"  Rust:  {lambda_rust:.6f}")
        print(f"  R mgcv: {lambda_r:.6f}")
        print(f"  Ratio: {lambda_rust/lambda_r:.4f}")

        # Lambdas should be within same order of magnitude
        ratio = lambda_rust / lambda_r
        self.assertGreater(ratio, 0.1, "Lambda should be in similar range")
        self.assertLess(ratio, 10.0, "Lambda should be in similar range")

    def test_linear_function_exact(self):
        """Test that both implementations handle linear function well"""

        # Perfect linear data (should get high lambda)
        x_lin = np.linspace(0, 1, 50)
        y_lin = 2 * x_lin + 1 + 0.01 * np.random.randn(50)

        # Fit with mgcv_rust
        X_lin = x_lin.reshape(-1, 1)
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X_lin, y_lin, k=[10], method='REML')
        pred_rust = gam_rust.predict(X_lin)

        # Fit with R mgcv
        ro.globalenv['x_lin'] = x_lin
        ro.globalenv['y_lin'] = y_lin
        ro.r('gam_fit <- gam(y_lin ~ s(x_lin, k=10, bs="bs"), method="REML")')
        pred_r = np.array(ro.r('predict(gam_fit)'))

        # Both should fit the line well
        rmse_rust = np.sqrt(np.mean((pred_rust - y_lin)**2))
        rmse_r = np.sqrt(np.mean((pred_r - y_lin)**2))

        print(f"\nLinear function fit:")
        print(f"  Rust RMSE:  {rmse_rust:.6f}")
        print(f"  R RMSE:     {rmse_r:.6f}")

        # Both should have low RMSE
        self.assertLess(rmse_rust, 0.1, "Should fit linear function well")
        self.assertLess(rmse_r, 0.1, "R should fit linear function well")

    def test_multi_variable_predictions(self):
        """Test multi-variable GAM matches R"""

        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 1, n)
        x2 = np.linspace(-1, 1, n)
        y_multi = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(n)

        # Fit with mgcv_rust
        X_multi = np.column_stack([x1, x2])
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X_multi, y_multi, k=[10, 10], method='REML')
        pred_rust = gam_rust.predict(X_multi)

        # Fit with R mgcv
        ro.globalenv['x1'] = x1
        ro.globalenv['x2'] = x2
        ro.globalenv['y_multi'] = y_multi
        ro.r('gam_fit <- gam(y_multi ~ s(x1, k=10, bs="bs") + s(x2, k=10, bs="bs"), method="REML")')
        pred_r = np.array(ro.r('predict(gam_fit)'))

        # Compare predictions
        corr = np.corrcoef(pred_rust, pred_r)[0, 1]
        rmse_diff = np.sqrt(np.mean((pred_rust - pred_r)**2))

        print(f"\nMulti-variable predictions:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  RMSE difference: {rmse_diff:.4f}")

        self.assertGreater(corr, 0.95,
                          f"Multi-var predictions should be highly correlated (got {corr:.4f})")

    def test_extrapolation_behavior(self):
        """Test that extrapolation produces reasonable values (not zero)"""

        # Train on subset
        x_train = np.linspace(0.3, 0.7, 50)
        y_train = np.sin(2*np.pi*x_train) + 0.1*np.random.randn(50)

        # Fit with mgcv_rust
        X_train = x_train.reshape(-1, 1)
        gam_rust = mgcv_rust.GAM()
        result_rust = gam_rust.fit_auto(X_train, y_train, k=[10], method='REML')

        # Predict outside range
        x_test = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        X_test = x_test.reshape(-1, 1)
        pred_rust = gam_rust.predict(X_test)

        # Fit with R mgcv
        ro.globalenv['x_train'] = x_train
        ro.globalenv['y_train'] = y_train
        ro.globalenv['x_test'] = x_test
        ro.r('gam_fit <- gam(y_train ~ s(x_train, k=10, bs="bs"), method="REML")')
        ro.r('pred_r <- predict(gam_fit, newdata=data.frame(x_train=x_test))')
        pred_r = np.array(ro.r('pred_r'))

        print(f"\nExtrapolation comparison:")
        print(f"  {'x':>6s} {'Rust':>10s} {'R mgcv':>10s} {'Diff':>10s}")
        for i, x in enumerate(x_test):
            diff = abs(pred_rust[i] - pred_r[i])
            region = "extrap" if (x < 0.3 or x > 0.7) else "in"
            print(f"  {x:6.2f} {pred_rust[i]:10.4f} {pred_r[i]:10.4f} {diff:10.4f} ({region})")

        # Check no zeros in extrapolation
        has_zeros_rust = np.any(np.abs(pred_rust) < 1e-6)
        has_zeros_r = np.any(np.abs(pred_r) < 1e-6)

        self.assertFalse(has_zeros_rust, "Rust should not produce zeros in extrapolation")
        self.assertFalse(has_zeros_r, "R should not produce zeros in extrapolation")


class TestMgcvConsistency(unittest.TestCase):
    """Test internal consistency of mgcv_rust (without R comparison)"""

    def test_reproducibility(self):
        """Test that same data produces same results"""

        np.random.seed(42)
        x = np.linspace(0, 1, 100)
        y = np.sin(2*np.pi*x) + 0.2*np.random.randn(100)
        X = x.reshape(-1, 1)

        # Fit twice
        gam1 = mgcv_rust.GAM()
        result1 = gam1.fit_auto(X, y, k=[10], method='REML')
        pred1 = gam1.predict(X)

        gam2 = mgcv_rust.GAM()
        result2 = gam2.fit_auto(X, y, k=[10], method='REML')
        pred2 = gam2.predict(X)

        # Should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10,
                                            err_msg="Results should be reproducible")
        self.assertAlmostEqual(result1['lambda'], result2['lambda'], places=10)

    def test_fit_auto_vs_fit_formula(self):
        """Test that fit_auto and fit_formula produce same results"""

        np.random.seed(42)
        x1 = np.linspace(0, 1, 50)
        x2 = np.linspace(-1, 1, 50)
        y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.1*np.random.randn(50)
        X = np.column_stack([x1, x2])

        # fit_auto
        gam_auto = mgcv_rust.GAM()
        result_auto = gam_auto.fit_auto(X, y, k=[10, 10], method='REML')
        pred_auto = gam_auto.predict(X)

        # fit_formula
        gam_formula = mgcv_rust.GAM()
        result_formula = gam_formula.fit_formula(X, y, formula="s(0, k=10) + s(1, k=10)", method='REML')
        pred_formula = gam_formula.predict(X)

        # Should be very similar
        np.testing.assert_array_almost_equal(pred_auto, pred_formula, decimal=8,
                                            err_msg="fit_auto and fit_formula should match")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
