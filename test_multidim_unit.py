#!/usr/bin/env python3
"""
Unit tests for multidimensional GAMs.

These tests ensure multidimensional functionality works correctly across
different dimensions and configurations.
"""
import unittest
import numpy as np
import mgcv_rust


class TestMultidimensionalGAM(unittest.TestCase):
    """Unit tests for multidimensional GAM functionality"""

    def setUp(self):
        """Set up common test parameters"""
        np.random.seed(42)

    def test_2d_basic(self):
        """Test basic 2D GAM fitting"""
        n = 100
        X = np.column_stack([
            np.linspace(0, 1, n),
            np.linspace(-1, 1, n)
        ])
        y = np.sin(2*np.pi*X[:,0]) + 0.5*X[:,1]**2 + 0.1*np.random.randn(n)

        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=[10, 10], method='REML')
        preds = gam.predict(X)

        # Check result structure
        self.assertIn('lambda', result)
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(np.isfinite(preds)))

        # Check fit quality
        r2 = 1 - np.var(y - preds) / np.var(y)
        self.assertGreater(r2, 0.85, "R² should be > 0.85 for 2D GAM")

    def test_2d_different_k(self):
        """Test 2D GAM with different k values"""
        n = 100
        X = np.column_stack([
            np.linspace(0, 1, n),
            np.linspace(-1, 1, n)
        ])
        y = np.sin(2*np.pi*X[:,0]) + 0.3*X[:,1] + 0.1*np.random.randn(n)

        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=[12, 8], method='REML')
        preds = gam.predict(X)

        # Check basics
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(np.isfinite(preds)))

        # Check fit quality
        r2 = 1 - np.var(y - preds) / np.var(y)
        self.assertGreater(r2, 0.80)

    def test_3d_basic(self):
        """Test basic 3D GAM fitting"""
        n = 150
        X = np.random.randn(n, 3)
        y = np.sin(X[:,0]) + 0.5*X[:,1]**2 + np.cos(X[:,2]) + 0.1*np.random.randn(n)

        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=[10, 10, 10], method='REML')
        preds = gam.predict(X)

        # Check result structure
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(np.isfinite(preds)))

        # Check fit quality
        r2 = 1 - np.var(y - preds) / np.var(y)
        self.assertGreater(r2, 0.90, "R² should be > 0.90 for 3D GAM")

    def test_4d_basic(self):
        """Test basic 4D GAM fitting"""
        n = 200
        X = np.random.randn(n, 4)
        y = (np.sin(X[:,0]) + 0.5*X[:,1]**2 +
             np.cos(X[:,2]) + 0.3*X[:,3] + 0.1*np.random.randn(n))

        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=[12, 12, 12, 12], method='REML')
        preds = gam.predict(X)

        # Check result structure
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(np.isfinite(preds)))

        # Check fit quality
        r2 = 1 - np.var(y - preds) / np.var(y)
        self.assertGreater(r2, 0.92, "R² should be > 0.92 for 4D GAM")

    def test_5d_basic(self):
        """Test basic 5D GAM fitting"""
        n = 200
        X = np.random.randn(n, 5)
        y = (np.sin(X[:,0]) + 0.5*X[:,1]**2 + np.cos(X[:,2]) +
             0.3*X[:,3] + 0.2*X[:,4]**3 + 0.1*np.random.randn(n))

        gam = mgcv_rust.GAM()
        result = gam.fit_auto(X, y, k=[10, 10, 10, 10, 10], method='REML')
        preds = gam.predict(X)

        # Check result structure
        self.assertEqual(len(preds), n)
        self.assertTrue(np.all(np.isfinite(preds)))

        # Check fit quality
        r2 = 1 - np.var(y - preds) / np.var(y)
        self.assertGreater(r2, 0.90, "R² should be > 0.90 for 5D GAM")

    def test_prediction_shape(self):
        """Test that prediction shape matches input"""
        n = 50
        for d in [2, 3, 4]:
            with self.subTest(dimensions=d):
                X = np.random.randn(n, d)
                y = np.random.randn(n)

                gam = mgcv_rust.GAM()
                gam.fit_auto(X, y, k=[8]*d, method='REML')

                # Test prediction on same data
                preds_train = gam.predict(X)
                self.assertEqual(len(preds_train), n)

                # Test prediction on new data
                X_new = np.random.randn(25, d)
                preds_new = gam.predict(X_new)
                self.assertEqual(len(preds_new), 25)

    def test_lambda_dimensions(self):
        """Test that lambda has correct dimensions"""
        n = 100
        for d in [2, 3, 4]:
            with self.subTest(dimensions=d):
                X = np.random.randn(n, d)
                y = np.random.randn(n)

                gam = mgcv_rust.GAM()
                result = gam.fit_auto(X, y, k=[8]*d, method='REML')

                lam = result.get('lambda', result.get('all_lambdas'))
                if isinstance(lam, (list, np.ndarray)):
                    self.assertEqual(len(lam), d,
                                   f"Lambda should have {d} values for {d}D GAM")

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        n = 100
        d = 3
        X = np.random.randn(n, d)
        y = np.random.randn(n)

        # Fit twice with same seed
        np.random.seed(42)
        gam1 = mgcv_rust.GAM()
        result1 = gam1.fit_auto(X, y, k=[8]*d, method='REML')
        preds1 = gam1.predict(X)

        np.random.seed(42)
        gam2 = mgcv_rust.GAM()
        result2 = gam2.fit_auto(X, y, k=[8]*d, method='REML')
        preds2 = gam2.predict(X)

        # Results should be identical
        np.testing.assert_array_almost_equal(preds1, preds2,
                                            decimal=10,
                                            err_msg="Results should be reproducible")

    def test_performance_scaling(self):
        """Test that performance scales reasonably with dimensions"""
        import time

        n = 150
        k = 10
        timings = {}

        for d in [2, 3, 4]:
            X = np.random.randn(n, d)
            y = np.random.randn(n)

            start = time.time()
            gam = mgcv_rust.GAM()
            gam.fit_auto(X, y, k=[k]*d, method='REML')
            elapsed = time.time() - start

            timings[d] = elapsed

            # Should complete in reasonable time
            self.assertLess(elapsed, 1.0,
                          f"{d}D GAM should complete in < 1s")

        # Print timings for reference
        print(f"\nPerformance scaling:")
        for d, t in timings.items():
            print(f"  {d}D: {t:.3f}s")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
