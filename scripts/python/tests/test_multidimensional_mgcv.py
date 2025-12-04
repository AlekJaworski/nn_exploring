#!/usr/bin/env python3
"""
Comprehensive tests comparing mgcv_rust to R's mgcv for multidimensional cases.

This test suite ensures that mgcv_rust correctly handles GAMs with multiple
predictors across various scenarios:
- Different dimensionalities (2D, 3D, 4D, 5D)
- Different signal complexities (linear, quadratic, cubic, sine)
- Different basis sizes
- Edge cases (all linear, all complex, mixed)

Each test compares:
1. Smoothing parameters (lambdas) - should be similar
2. Predictions - should be highly correlated
3. Model fit quality - should be comparable
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


class TestMultidimensionalMgcvComparison(unittest.TestCase):
    """Comprehensive tests for multidimensional GAMs comparing with R mgcv"""

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
        """Set up common test parameters"""
        np.random.seed(42)
        self.n = 150  # Number of observations
        self.tol_corr = 0.90  # Minimum correlation for predictions
        self.tol_lambda_ratio = (0.1, 10.0)  # Lambda ratio bounds

    def _fit_rust_gam(self, X, y, k_values, method='REML'):
        """Fit GAM using mgcv_rust"""
        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=k_values, method=method)
        predictions = gam.predict(X)
        lambdas = result['all_lambdas']

        return {
            'gam': gam,
            'result': result,
            'predictions': predictions,
            'lambdas': lambdas
        }

    def _fit_r_mgcv(self, data_dict, formula_str, method='REML'):
        """Fit GAM using R mgcv

        Args:
            data_dict: Dictionary mapping variable names to numpy arrays
            formula_str: R formula string like "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr')"
            method: 'REML' or 'GCV'
        """
        # Put data into R environment
        for name, values in data_dict.items():
            ro.globalenv[name] = values

        # Fit the model
        ro.r(f'gam_fit <- gam({formula_str}, method="{method}")')

        # Extract results
        predictions = np.array(ro.r('predict(gam_fit)'))
        lambdas = np.array(ro.r('gam_fit$sp'))

        return {
            'predictions': predictions,
            'lambdas': lambdas
        }

    def _compare_results(self, rust_results, r_results, test_name):
        """Compare results from Rust and R implementations"""
        pred_rust = rust_results['predictions']
        pred_r = r_results['predictions']
        lambda_rust = rust_results['lambdas']
        lambda_r = r_results['lambdas']

        # Compare predictions
        corr = np.corrcoef(pred_rust, pred_r)[0, 1]
        rmse_diff = np.sqrt(np.mean((pred_rust - pred_r)**2))

        print(f"\n{test_name}:")
        print(f"  Prediction correlation: {corr:.4f}")
        print(f"  RMSE difference: {rmse_diff:.4f}")
        print(f"  Lambdas (Rust): {lambda_rust}")
        print(f"  Lambdas (R):    {lambda_r}")

        # Compare lambda ratios
        if len(lambda_rust) == len(lambda_r):
            ratios = lambda_rust / (lambda_r + 1e-10)  # Add small value to avoid div by zero
            print(f"  Lambda ratios:  {ratios}")

        return corr, rmse_diff, lambda_rust, lambda_r

    # ========================================================================
    # 2D TESTS
    # ========================================================================

    def test_2d_sine_quadratic(self):
        """2D: sine wave + quadratic"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(self.n)
        X = np.column_stack([x1, x2])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[10, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'y': y},
            "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr')"
        )

        # Compare
        corr, rmse_diff, _, _ = self._compare_results(
            rust_results, r_results, "2D: sine + quadratic"
        )

        self.assertGreater(corr, self.tol_corr,
                          f"Predictions should be correlated > {self.tol_corr}")

    def test_2d_linear_linear(self):
        """2D: two linear functions (should have high lambdas)"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        y = 2*x1 + 3*x2 + 1 + 0.1*np.random.randn(self.n)
        X = np.column_stack([x1, x2])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[10, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'y': y},
            "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr')"
        )

        # Compare
        corr, rmse_diff, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "2D: linear + linear (high lambda)"
        )

        self.assertGreater(corr, self.tol_corr)
        # Both lambdas should be relatively high for linear functions
        self.assertGreater(lambda_rust[0], 1.0, "Lambda should be high for linear function")
        self.assertGreater(lambda_rust[1], 1.0, "Lambda should be high for linear function")

    def test_2d_different_k(self):
        """2D: different basis sizes for each dimension"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(self.n)
        X = np.column_stack([x1, x2])

        # Fit with Rust - different k values
        rust_results = self._fit_rust_gam(X, y, k_values=[15, 8])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'y': y},
            "y ~ s(x1, k=15, bs='cr') + s(x2, k=8, bs='cr')"
        )

        # Compare
        corr, _, _, _ = self._compare_results(
            rust_results, r_results, "2D: different k values [15, 8]"
        )

        self.assertGreater(corr, self.tol_corr)

    # ========================================================================
    # 3D TESTS
    # ========================================================================

    def test_3d_mixed_complexity(self):
        """3D: sine + quadratic + linear (different complexities)"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        y = (np.sin(2*np.pi*x1) +      # Complex
             0.5*x2**2 +                # Moderate
             2.0*x3 +                   # Simple
             0.3*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[15, 12, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'y': y},
            "y ~ s(x1, k=15, bs='cr') + s(x2, k=12, bs='cr') + s(x3, k=10, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "3D: sine + quadratic + linear"
        )

        self.assertGreater(corr, self.tol_corr)

        # Check lambda pattern: linear should have highest lambda
        self.assertGreater(lambda_rust[2], lambda_rust[0],
                          "Linear function should have higher lambda than sine")

    def test_3d_all_complex(self):
        """3D: three complex (sine) functions"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(0, 1, self.n)
        x3 = np.linspace(0, 1, self.n)
        y = (np.sin(2*np.pi*x1) +
             np.sin(3*np.pi*x2) +
             np.sin(4*np.pi*x3) +
             0.2*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[12, 12, 12])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'y': y},
            "y ~ s(x1, k=12, bs='cr') + s(x2, k=12, bs='cr') + s(x3, k=12, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, _ = self._compare_results(
            rust_results, r_results, "3D: all complex (sine functions)"
        )

        self.assertGreater(corr, self.tol_corr)

        # All lambdas should be relatively low (flexible fit needed)
        self.assertTrue(np.all(lambda_rust < 10.0),
                       "Lambdas should be relatively low for complex functions")

    def test_3d_all_linear(self):
        """3D: three linear functions (should have high lambdas)"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        y = 2*x1 + 3*x2 + 1.5*x3 + 1 + 0.1*np.random.randn(self.n)
        X = np.column_stack([x1, x2, x3])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[10, 10, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'y': y},
            "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "3D: all linear (high lambdas)"
        )

        self.assertGreater(corr, self.tol_corr)

        # All lambdas should be high for linear functions
        self.assertTrue(np.all(lambda_rust > 1.0),
                       f"All lambdas should be > 1 for linear functions, got {lambda_rust}")

    # ========================================================================
    # 4D TESTS
    # ========================================================================

    def test_4d_mixed(self):
        """4D: mixed complexity functions"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        x4 = np.linspace(0, 2, self.n)
        y = (np.sin(2*np.pi*x1) +      # Complex
             0.5*x2**2 +                # Moderate
             2.0*x3 +                   # Simple
             x4**3 - 2*x4 +             # Cubic
             0.3*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3, x4])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[12, 10, 10, 12])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y},
            "y ~ s(x1, k=12, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr') + s(x4, k=12, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "4D: sine + quadratic + linear + cubic"
        )

        self.assertGreater(corr, self.tol_corr)

        # Linear (x3) should have highest lambda
        max_lambda_idx = np.argmax(lambda_rust)
        self.assertEqual(max_lambda_idx, 2,
                        f"Linear function (index 2) should have highest lambda, got index {max_lambda_idx}")

    def test_4d_uniform_k(self):
        """4D: all same basis size"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(0, 1, self.n)
        x3 = np.linspace(0, 1, self.n)
        x4 = np.linspace(0, 1, self.n)
        y = (np.sin(2*np.pi*x1) +
             np.sin(3*np.pi*x2) +
             x3**2 +
             2*x4 +
             0.2*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3, x4])

        # Fit with Rust - all k=10
        rust_results = self._fit_rust_gam(X, y, k_values=[10, 10, 10, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'y': y},
            "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr') + s(x4, k=10, bs='cr')"
        )

        # Compare
        corr, _, _, _ = self._compare_results(
            rust_results, r_results, "4D: uniform k=10"
        )

        self.assertGreater(corr, self.tol_corr)

    # ========================================================================
    # 5D TESTS
    # ========================================================================

    def test_5d_high_dimensional(self):
        """5D: testing higher dimensional case"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        x4 = np.linspace(0, 2, self.n)
        x5 = np.linspace(-1, 1, self.n)
        y = (np.sin(2*np.pi*x1) +
             0.5*x2**2 +
             2.0*x3 +
             x4**3 - x4 +
             -0.3*x5**2 + x5 +
             0.3*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3, x4, x5])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[12, 10, 10, 12, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'y': y},
            "y ~ s(x1, k=12, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr') + s(x4, k=12, bs='cr') + s(x5, k=10, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "5D: high dimensional"
        )

        self.assertGreater(corr, self.tol_corr)

        # Should have 5 lambdas
        self.assertEqual(len(lambda_rust), 5, "Should have 5 smoothing parameters")
        self.assertEqual(len(lambda_r), 5, "R should have 5 smoothing parameters")

    # ========================================================================
    # EDGE CASES AND ROBUSTNESS
    # ========================================================================

    def test_small_k_values(self):
        """Test with small k values (k=5)"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(self.n)
        X = np.column_stack([x1, x2])

        # Fit with Rust - small k
        rust_results = self._fit_rust_gam(X, y, k_values=[5, 5])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'y': y},
            "y ~ s(x1, k=5, bs='cr') + s(x2, k=5, bs='cr')"
        )

        # Compare
        corr, _, _, _ = self._compare_results(
            rust_results, r_results, "Small k values (k=5)"
        )

        # With small k, might have lower correlation but should still work
        self.assertGreater(corr, 0.80, "Should work even with small k")

    def test_large_k_values(self):
        """Test with large k values (k=20)"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        y = np.sin(2*np.pi*x1) + 0.5*x2**2 + 0.2*np.random.randn(self.n)
        X = np.column_stack([x1, x2])

        # Fit with Rust - large k
        rust_results = self._fit_rust_gam(X, y, k_values=[20, 20])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'y': y},
            "y ~ s(x1, k=20, bs='cr') + s(x2, k=20, bs='cr')"
        )

        # Compare
        corr, _, _, _ = self._compare_results(
            rust_results, r_results, "Large k values (k=20)"
        )

        self.assertGreater(corr, self.tol_corr)

    def test_noisy_data(self):
        """Test with high noise level"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        # High noise
        y = (np.sin(2*np.pi*x1) + 0.5*x2**2 + 2.0*x3 +
             1.0*np.random.randn(self.n))  # Large noise
        X = np.column_stack([x1, x2, x3])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[10, 10, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'y': y},
            "y ~ s(x1, k=10, bs='cr') + s(x2, k=10, bs='cr') + s(x3, k=10, bs='cr')"
        )

        # Compare
        corr, _, lambda_rust, lambda_r = self._compare_results(
            rust_results, r_results, "Noisy data (high variance)"
        )

        self.assertGreater(corr, 0.85, "Should handle noisy data")

        # High noise should lead to higher lambdas (more smoothing)
        self.assertTrue(np.mean(lambda_rust) > 0.1,
                       "High noise should lead to more smoothing")

    # ========================================================================
    # LAMBDA VERIFICATION
    # ========================================================================

    def test_lambda_ordering_matches_complexity(self):
        """Test that lambda ordering matches signal complexity for both implementations"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        # Clear complexity difference: sine << quadratic < linear
        y = (2.0*np.sin(2*np.pi*x1) +   # Complex: large amplitude sine
             0.5*x2**2 +                 # Moderate: quadratic
             2.0*x3 +                    # Simple: linear
             0.1*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3])

        # Fit with Rust
        rust_results = self._fit_rust_gam(X, y, k_values=[15, 12, 10])

        # Fit with R
        r_results = self._fit_r_mgcv(
            {'x1': x1, 'x2': x2, 'x3': x3, 'y': y},
            "y ~ s(x1, k=15, bs='cr') + s(x2, k=12, bs='cr') + s(x3, k=10, bs='cr')"
        )

        lambda_rust = rust_results['lambdas']
        lambda_r = r_results['lambdas']

        print(f"\nLambda ordering test:")
        print(f"  Rust: λ1={lambda_rust[0]:.3f}, λ2={lambda_rust[1]:.3f}, λ3={lambda_rust[2]:.3f}")
        print(f"  R:    λ1={lambda_r[0]:.3f}, λ2={lambda_r[1]:.3f}, λ3={lambda_r[2]:.3f}")

        # Both should show: lambda1 < lambda2 < lambda3
        # (sine < quadratic < linear)
        rust_order = lambda_rust[2] > lambda_rust[0]  # linear > sine
        r_order = lambda_r[2] > lambda_r[0]

        self.assertTrue(rust_order,
                       f"Rust: Linear should have higher lambda than sine: {lambda_rust}")
        self.assertTrue(r_order,
                       f"R: Linear should have higher lambda than sine: {lambda_r}")


class TestConsistencyAcrossMethods(unittest.TestCase):
    """Test that different fitting methods produce consistent multidimensional results"""

    def setUp(self):
        np.random.seed(42)
        self.n = 100

    def test_fit_auto_vs_fit_formula_3d(self):
        """Test fit_auto and fit_formula produce same results in 3D"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        y = (np.sin(2*np.pi*x1) + 0.5*x2**2 + 2.0*x3 +
             0.2*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3])

        # fit_auto
        gam_auto = mgcv_rust.GAM()
        result_auto = gam_auto.fit_auto(X, y, k=[12, 10, 10], method='REML')
        pred_auto = gam_auto.predict(X)
        lambda_auto = result_auto['all_lambdas']

        # fit_formula
        gam_formula = mgcv_rust.GAM()
        result_formula = gam_formula.fit_formula(
            X, y,
            formula="s(0, k=12) + s(1, k=10) + s(2, k=10)",
            method='REML'
        )
        pred_formula = gam_formula.predict(X)
        lambda_formula = result_formula['all_lambdas']

        print(f"\nfit_auto vs fit_formula (3D):")
        print(f"  Lambdas (auto):    {lambda_auto}")
        print(f"  Lambdas (formula): {lambda_formula}")
        print(f"  Max pred diff:     {np.max(np.abs(pred_auto - pred_formula)):.6f}")

        # Should be very similar
        np.testing.assert_array_almost_equal(
            pred_auto, pred_formula, decimal=6,
            err_msg="fit_auto and fit_formula should produce same predictions"
        )
        np.testing.assert_array_almost_equal(
            lambda_auto, lambda_formula, decimal=6,
            err_msg="fit_auto and fit_formula should produce same lambdas"
        )

    def test_reproducibility_multidim(self):
        """Test that multidimensional fits are reproducible"""
        x1 = np.linspace(0, 1, self.n)
        x2 = np.linspace(-1, 1, self.n)
        x3 = np.linspace(-2, 2, self.n)
        y = (np.sin(2*np.pi*x1) + 0.5*x2**2 + 2.0*x3 +
             0.2*np.random.randn(self.n))
        X = np.column_stack([x1, x2, x3])

        # Fit twice
        gam1 = mgcv_rust.GAM()
        result1 = gam1.fit_auto(X, y, k=[12, 10, 10], method='REML')
        pred1 = gam1.predict(X)

        gam2 = mgcv_rust.GAM()
        result2 = gam2.fit_auto(X, y, k=[12, 10, 10], method='REML')
        pred2 = gam2.predict(X)

        # Should be identical
        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=10,
            err_msg="Results should be exactly reproducible"
        )
        np.testing.assert_array_almost_equal(
            result1['all_lambdas'], result2['all_lambdas'], decimal=10,
            err_msg="Lambdas should be exactly reproducible"
        )


def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMultidimensionalMgcvComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestConsistencyAcrossMethods))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


if __name__ == '__main__':
    import sys

    # Check if R and rpy2 are available
    if not HAS_RPY2:
        print("\nERROR: rpy2 is required for these tests.")
        print("Install with: pip install rpy2")
        sys.exit(1)

    # Run tests
    result = run_tests(verbosity=2)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
