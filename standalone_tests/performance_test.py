#!/usr/bin/env python3
"""
Comprehensive performance testing framework for GAM implementations.
Tests various scenarios: low/high n, low/high dimensions, low/high k values.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple
import sys

try:
    import mgcv_rust
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: mgcv_rust not available. Build first with 'maturin develop'")
    RUST_AVAILABLE = False


class PerformanceTester:
    """Framework for testing GAM performance across multiple scenarios."""

    def __init__(self):
        self.results = []

    def generate_test_data(self, n: int, d: int, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic test data for GAM fitting.

        Args:
            n: Number of data points
            d: Number of dimensions
            noise_level: Standard deviation of Gaussian noise

        Returns:
            X: n x d array of predictors
            y: n array of responses
        """
        np.random.seed(42)  # For reproducibility

        # Generate X uniformly in [0, 1]^d
        X = np.random.uniform(0, 1, (n, d))

        # Generate y as a sum of smooth functions + noise
        y = np.zeros(n)
        for i in range(d):
            # Different smooth function for each dimension
            if i % 3 == 0:
                y += np.sin(2 * np.pi * X[:, i])
            elif i % 3 == 1:
                y += (X[:, i] - 0.5) ** 2
            else:
                y += np.exp(-5 * (X[:, i] - 0.5) ** 2)

        # Add noise
        y += np.random.normal(0, noise_level, n)

        return X, y

    def run_single_test(self, n: int, d: int, k_values: List[int], method: str = "REML",
                       bs: str = "cr", max_iter: int = 10) -> Dict:
        """
        Run a single performance test.

        Args:
            n: Number of data points
            d: Number of dimensions
            k_values: List of k values for each dimension
            method: Optimization method ("REML" or "GCV")
            bs: Basis type ("cr" or "bs")
            max_iter: Maximum iterations

        Returns:
            Dictionary with timing and convergence info
        """
        if not RUST_AVAILABLE:
            return {"error": "Rust library not available"}

        if len(k_values) != d:
            raise ValueError(f"k_values length ({len(k_values)}) must match d ({d})")

        # Generate data
        X, y = self.generate_test_data(n, d)

        # Fit GAM and measure time
        gam = mgcv_rust.GAM()

        start_time = time.perf_counter()
        try:
            result = gam.fit_auto(X, y, k=k_values, method=method, bs=bs, max_iter=max_iter)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time

            # Extract results
            fitted_values = result.get('fitted_values', None)
            lambdas = result.get('all_lambdas', None)
            deviance = result.get('deviance', None)

            # Compute R-squared
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = deviance if deviance is not None else np.sum((y - fitted_values) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "n": n,
                "d": d,
                "k_values": k_values,
                "method": method,
                "bs": bs,
                "time_seconds": elapsed_time,
                "deviance": float(deviance) if deviance is not None else None,
                "r_squared": float(r_squared),
                "lambdas": lambdas.tolist() if lambdas is not None else None,
                "status": "success"
            }
        except Exception as e:
            end_time = time.perf_counter()
            return {
                "n": n,
                "d": d,
                "k_values": k_values,
                "method": method,
                "bs": bs,
                "time_seconds": end_time - start_time,
                "status": "failed",
                "error": str(e)
            }

    def run_benchmark_suite(self, save_file: str = "performance_results.json"):
        """
        Run comprehensive benchmark suite covering various scenarios.
        """
        test_configs = []

        # Small n, small d, small k
        test_configs.append({"n": 50, "d": 1, "k": [5]})
        test_configs.append({"n": 50, "d": 1, "k": [10]})
        test_configs.append({"n": 50, "d": 2, "k": [5, 5]})

        # Small n, small d, larger k
        test_configs.append({"n": 50, "d": 1, "k": [15]})
        test_configs.append({"n": 50, "d": 1, "k": [20]})
        test_configs.append({"n": 50, "d": 1, "k": [30]})
        test_configs.append({"n": 50, "d": 2, "k": [10, 10]})

        # Medium n, small d, various k
        test_configs.append({"n": 200, "d": 1, "k": [10]})
        test_configs.append({"n": 200, "d": 1, "k": [20]})
        test_configs.append({"n": 200, "d": 1, "k": [30]})
        test_configs.append({"n": 200, "d": 2, "k": [10, 10]})
        test_configs.append({"n": 200, "d": 3, "k": [10, 10, 10]})

        # Large n, small d, various k
        test_configs.append({"n": 1000, "d": 1, "k": [10]})
        test_configs.append({"n": 1000, "d": 1, "k": [20]})
        test_configs.append({"n": 1000, "d": 1, "k": [30]})
        test_configs.append({"n": 1000, "d": 2, "k": [15, 15]})
        test_configs.append({"n": 1000, "d": 3, "k": [10, 10, 10]})

        # Very large n, small d
        test_configs.append({"n": 5000, "d": 1, "k": [10]})
        test_configs.append({"n": 5000, "d": 1, "k": [20]})
        test_configs.append({"n": 5000, "d": 2, "k": [10, 10]})

        # Medium n, larger d (up to 10)
        test_configs.append({"n": 500, "d": 5, "k": [8] * 5})
        test_configs.append({"n": 500, "d": 10, "k": [6] * 10})

        # Large n, larger d
        test_configs.append({"n": 2000, "d": 5, "k": [10] * 5})

        print("="*70)
        print("Running comprehensive GAM performance benchmark suite")
        print("="*70)
        print(f"Total test configurations: {len(test_configs)}")
        print()

        results = []
        for i, config in enumerate(test_configs, 1):
            n, d, k = config["n"], config["d"], config["k"]
            print(f"Test {i}/{len(test_configs)}: n={n:5d}, d={d:2d}, k={k}", end=" ... ")
            sys.stdout.flush()

            result = self.run_single_test(n, d, k)
            results.append(result)

            if result["status"] == "success":
                print(f"✓ {result['time_seconds']:.3f}s (R²={result['r_squared']:.3f})")
            else:
                print(f"✗ Failed: {result.get('error', 'unknown')}")

        # Save results
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)

        print()
        print("="*70)
        print("Benchmark Summary")
        print("="*70)

        successful_tests = [r for r in results if r["status"] == "success"]
        failed_tests = [r for r in results if r["status"] == "failed"]

        print(f"Successful: {len(successful_tests)}/{len(results)}")
        print(f"Failed: {len(failed_tests)}/{len(results)}")

        if successful_tests:
            times = [r["time_seconds"] for r in successful_tests]
            print(f"\nTiming statistics:")
            print(f"  Min:    {min(times):.3f}s")
            print(f"  Max:    {max(times):.3f}s")
            print(f"  Mean:   {np.mean(times):.3f}s")
            print(f"  Median: {np.median(times):.3f}s")

        print(f"\nResults saved to: {save_file}")

        return results


def main():
    """Main entry point for performance testing."""
    if not RUST_AVAILABLE:
        print("Error: mgcv_rust module not available.")
        print("Please build it first with: maturin develop")
        return 1

    tester = PerformanceTester()
    tester.run_benchmark_suite("baseline_performance.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
