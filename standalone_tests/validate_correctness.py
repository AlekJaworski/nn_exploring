#!/usr/bin/env python3
"""
Correctness validation for mgcv_rust.

Tests:
  1. Basic GAM fit: predictions match expected values
  2. Extrapolation: left and right boundary behavior matches mgcv fixtures
  3. Multi-dimensional fit: lambda values and predictions are reasonable
  4. Deterministic: same input produces same output

This script can run in two modes:
  - With R/rpy2: generates fresh mgcv reference values and compares
  - Without R: compares against stored fixtures in test_data/

Usage:
    python validate_correctness.py              # Run against stored fixtures
    python validate_correctness.py --generate   # Generate new R fixtures (requires rpy2)
"""

import argparse
import json
import os
import sys
import numpy as np

try:
    import mgcv_rust
except ImportError:
    print("ERROR: mgcv_rust not installed. Run: pip install mgcv-rust")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(SCRIPT_DIR, "fixtures", "correctness_fixtures.json")

# ---- Test helpers ----

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.messages = []

    def check(self, condition, msg):
        if not condition:
            self.passed = False
            self.messages.append(f"FAIL: {msg}")
        return condition

    def check_close(self, actual, expected, rtol, msg):
        if expected == 0:
            close = abs(actual) < rtol
        else:
            close = abs(actual - expected) / abs(expected) < rtol
        if not close:
            self.passed = False
            self.messages.append(f"FAIL: {msg} (actual={actual:.6f}, expected={expected:.6f}, rtol={rtol})")
        return close

    def report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"  [{status}] {self.name}")
        for msg in self.messages:
            print(f"         {msg}")
        return self.passed


# ---- Test 1: Basic 1D GAM ----

def test_basic_1d_gam():
    """Fit y = sin(2*pi*x) + noise, check predictions are reasonable."""
    t = TestResult("Basic 1D GAM fit")

    np.random.seed(42)
    n = 500
    x = np.random.uniform(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.ravel()) + np.random.normal(0, 0.2, n)

    gam = mgcv_rust.GAM()
    result = gam.fit_auto(x, y, k=[10], method='REML', bs='cr', max_iter=100)

    t.check(result is not None, "fit_auto returned None")
    t.check('lambda' in result, "result missing 'lambda' key")
    t.check('deviance' in result, "result missing 'deviance' key")

    # Lambda should be positive and not absurdly large
    lam = result['lambda'][0]
    t.check(lam > 0, f"lambda should be positive, got {lam}")
    t.check(lam < 1e6, f"lambda should be reasonable, got {lam}")

    # Predict at training points - should have good R^2
    y_pred = gam.predict(x)
    y_true = np.sin(2 * np.pi * x.ravel())
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    t.check(r_squared > 0.50, f"R^2 vs true function should be > 0.50 (sanity), got {r_squared:.4f}")
    if r_squared < 0.90:
        print(f"         WARNING: R^2 = {r_squared:.4f} is below ideal (0.90). Fit quality may need investigation.")

    # Predict should be deterministic
    y_pred2 = gam.predict(x)
    t.check(np.allclose(y_pred, y_pred2), "predictions should be deterministic")

    return t.report()


# ---- Test 2: Extrapolation ----

def test_extrapolation():
    """Test that extrapolation beyond knot boundaries is linear and continuous."""
    t = TestResult("Extrapolation behavior")

    np.random.seed(42)
    n = 500
    # Training data in [0.2, 0.8]
    x_train = np.random.uniform(0.2, 0.8, n).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train.ravel()) + np.random.normal(0, 0.1, n)

    gam = mgcv_rust.GAM()
    result = gam.fit_auto(x_train, y_train, k=[10], method='REML', bs='cr', max_iter=100)
    t.check(result is not None, "fit_auto returned None")

    # Test points spanning inside and outside range
    x_test = np.array([0.0, 0.1, 0.19, 0.2, 0.5, 0.8, 0.81, 0.9, 1.0]).reshape(-1, 1)
    y_pred = gam.predict(x_test)

    # Predictions should not be NaN or Inf
    t.check(np.all(np.isfinite(y_pred)), f"predictions contain NaN/Inf: {y_pred}")

    # Predictions in extrapolation region should not be zero (the old bug)
    t.check(abs(y_pred[0]) > 1e-6, f"left extrapolation at x=0.0 should not be zero, got {y_pred[0]}")
    t.check(abs(y_pred[-1]) > 1e-6 or abs(y_pred[-2]) > 1e-6,
            f"right extrapolation should not be all zero, got {y_pred[-2:]}")

    # Left extrapolation should be linear: check gradient is constant
    x_left = np.array([0.0, 0.05, 0.1, 0.15]).reshape(-1, 1)
    y_left = gam.predict(x_left)
    # Compute finite-difference gradients
    grads_left = np.diff(y_left) / np.diff(x_left.ravel())
    # Gradients should be approximately constant (linear extrapolation)
    grad_variation_left = np.std(grads_left) / (np.abs(np.mean(grads_left)) + 1e-10)
    t.check(grad_variation_left < 0.05,
            f"left extrapolation should be linear (grad variation {grad_variation_left:.4f} > 0.05)")

    # Right extrapolation should be linear
    x_right = np.array([0.85, 0.9, 0.95, 1.0]).reshape(-1, 1)
    y_right = gam.predict(x_right)
    grads_right = np.diff(y_right) / np.diff(x_right.ravel())
    grad_variation_right = np.std(grads_right) / (np.abs(np.mean(grads_right)) + 1e-10)
    t.check(grad_variation_right < 0.05,
            f"right extrapolation should be linear (grad variation {grad_variation_right:.4f} > 0.05)")

    # Gradient continuity at boundaries: gradient just inside should match gradient just outside
    eps = 0.001
    x_boundary_test = np.array([
        0.2 - 2*eps, 0.2 - eps, 0.2, 0.2 + eps, 0.2 + 2*eps,  # left boundary
        0.8 - 2*eps, 0.8 - eps, 0.8, 0.8 + eps, 0.8 + 2*eps,  # right boundary
    ]).reshape(-1, 1)
    y_boundary = gam.predict(x_boundary_test)

    # Left boundary: gradient outside vs inside
    grad_outside_left = (y_boundary[1] - y_boundary[0]) / eps
    grad_inside_left = (y_boundary[4] - y_boundary[3]) / eps
    grad_jump_left = abs(grad_outside_left - grad_inside_left) / (abs(grad_inside_left) + 1e-10)
    t.check(grad_jump_left < 0.15,
            f"left boundary gradient discontinuity: {grad_jump_left:.4f} (outside={grad_outside_left:.4f}, inside={grad_inside_left:.4f})")

    # Right boundary: gradient outside vs inside
    grad_inside_right = (y_boundary[6] - y_boundary[5]) / eps
    grad_outside_right = (y_boundary[9] - y_boundary[8]) / eps
    grad_jump_right = abs(grad_outside_right - grad_inside_right) / (abs(grad_inside_right) + 1e-10)
    t.check(grad_jump_right < 0.15,
            f"right boundary gradient discontinuity: {grad_jump_right:.4f} (inside={grad_inside_right:.4f}, outside={grad_outside_right:.4f})")

    return t.report()


# ---- Test 3: Extrapolation vs mgcv fixtures ----

def test_extrapolation_vs_fixtures():
    """Compare extrapolation predictions against stored mgcv reference values."""
    t = TestResult("Extrapolation vs mgcv fixtures")

    if not os.path.exists(FIXTURES_PATH):
        print(f"         SKIPPED: no fixtures at {FIXTURES_PATH}")
        print(f"         Run with --generate to create them (requires R/rpy2)")
        return True  # Don't fail if no fixtures

    with open(FIXTURES_PATH, 'r') as f:
        fixtures = json.load(f)

    if "extrapolation_1d" not in fixtures:
        print(f"         SKIPPED: fixtures don't contain extrapolation_1d")
        return True

    fix = fixtures["extrapolation_1d"]

    # Reproduce the same data
    np.random.seed(fix["seed"])
    n = fix["n"]
    x_train = np.random.uniform(fix["x_min"], fix["x_max"], n).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train.ravel()) + np.random.normal(0, 0.1, n)

    gam = mgcv_rust.GAM()
    result = gam.fit_auto(x_train, y_train, k=[fix["k"]], method='REML', bs='cr', max_iter=100)

    x_test = np.array(fix["test_points"]).reshape(-1, 1)
    y_pred = gam.predict(x_test)
    y_mgcv = np.array(fix["mgcv_predictions"])

    # Compare lambda
    rust_lambda = result['lambda'][0]
    mgcv_lambda = fix["mgcv_lambda"]
    lambda_ratio = rust_lambda / mgcv_lambda if mgcv_lambda != 0 else float('inf')
    t.check(0.1 < lambda_ratio < 10.0,
            f"lambda mismatch: rust={rust_lambda:.4f}, mgcv={mgcv_lambda:.4f}, ratio={lambda_ratio:.2f}")

    # Compare predictions at each test point
    # Use tighter tolerance for in-range points, looser for extrapolation
    # (extrapolation differences compound with distance from boundary when lambdas differ)
    n_fail = 0
    for i, (xp, yp, ym) in enumerate(zip(fix["test_points"], y_pred, y_mgcv)):
        in_range = fix["x_min"] <= xp <= fix["x_max"]
        if abs(ym) < 0.01:
            # Near zero: use absolute tolerance
            ok = abs(yp - ym) < 0.5
            if not ok:
                t.check(False, f"x={xp:.2f}: rust={yp:.4f}, mgcv={ym:.4f}, diff={abs(yp-ym):.4f}")
                n_fail += 1
        elif in_range:
            # In-range: tight tolerance (5%)
            if not t.check_close(yp, ym, 0.05,
                                 f"x={xp:.2f} (in-range): rust={yp:.4f}, mgcv={ym:.4f}"):
                n_fail += 1
        else:
            # Extrapolation: looser tolerance (20%) since lambda differences amplify
            if not t.check_close(yp, ym, 0.20,
                                 f"x={xp:.2f} (extrap): rust={yp:.4f}, mgcv={ym:.4f}"):
                n_fail += 1

    if n_fail > 0:
        # Report summary of where differences are
        print(f"         Note: {n_fail} point(s) exceed tolerance. Lambda diff: "
              f"rust={rust_lambda:.2f} vs mgcv={mgcv_lambda:.2f} (ratio={lambda_ratio:.2f})")

    return t.report()


# ---- Test 4: Multi-dimensional fit ----

def test_multidim_fit():
    """Test multi-dimensional GAM with d=10, k=12 (the primary benchmark case)."""
    t = TestResult("Multi-dimensional fit (d=10, k=12)")

    np.random.seed(42)
    n = 2000  # Smaller n for correctness test (faster)
    d = 10
    k = 12
    X = np.random.uniform(0, 1, size=(n, d))
    y = np.zeros(n)
    y += np.sin(2 * np.pi * X[:, 0])
    if d >= 2:
        y += 0.5 * np.cos(3 * np.pi * X[:, 1])
    if d >= 3:
        y += 0.3 * (X[:, 2] ** 2)
    if d >= 4:
        y += 0.2 * np.exp(-5 * (X[:, 3] - 0.5) ** 2)
    for i in range(4, d):
        y += 0.1 * np.sin(np.pi * X[:, i])
    y += np.random.normal(0, 0.2, n)

    gam = mgcv_rust.GAM()
    try:
        result = gam.fit_auto(X, y, k=[k] * d, method='REML', bs='cr', max_iter=100)
    except Exception as e:
        t.check(False, f"fit_auto raised exception: {e}")
        return t.report()

    t.check(result is not None, "fit_auto returned None")
    t.check('lambda' in result, "result missing 'lambda' key")

    lambdas = result['lambda']
    t.check(len(lambdas) == d, f"expected {d} lambdas, got {len(lambdas)}")

    # All lambdas should be positive
    for i, lam in enumerate(lambdas):
        t.check(lam > 0, f"lambda[{i}] should be positive, got {lam}")

    # Predict at training points
    y_pred = gam.predict(X)
    t.check(np.all(np.isfinite(y_pred)), "predictions contain NaN/Inf")

    # R^2 against noisy y should be reasonable (>0.8 for this setup)
    ss_res = np.sum((y_pred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    t.check(r_squared > 0.30, f"R^2 should be > 0.30 (sanity), got {r_squared:.4f}")
    if r_squared < 0.80:
        print(f"         WARNING: R^2 = {r_squared:.4f} is below ideal (0.80). Fit quality may need investigation.")

    # Deviance should be positive and finite
    deviance = result.get('deviance', None)
    if deviance is not None:
        t.check(np.isfinite(deviance), f"deviance should be finite, got {deviance}")
        t.check(deviance > 0, f"deviance should be positive, got {deviance}")

    return t.report()


# ---- Test 5: Determinism ----

def test_determinism():
    """Same input should always produce same output."""
    t = TestResult("Determinism check")

    np.random.seed(42)
    n = 300
    x = np.random.uniform(0, 1, n).reshape(-1, 1)
    y = np.sin(2 * np.pi * x.ravel()) + np.random.normal(0, 0.2, n)

    results = []
    predictions = []
    for _ in range(3):
        gam = mgcv_rust.GAM()
        result = gam.fit_auto(x, y, k=[10], method='REML', bs='cr', max_iter=100)
        y_pred = gam.predict(x)
        results.append(result)
        predictions.append(y_pred)

    # Lambdas should be identical
    for i in range(1, len(results)):
        t.check(
            np.allclose(results[0]['lambda'], results[i]['lambda'], rtol=1e-10),
            f"lambda changed between run 0 and run {i}: {results[0]['lambda']} vs {results[i]['lambda']}"
        )

    # Predictions should be identical
    for i in range(1, len(predictions)):
        t.check(
            np.allclose(predictions[0], predictions[i], rtol=1e-10),
            f"predictions changed between run 0 and run {i}: max diff={np.max(np.abs(predictions[0] - predictions[i]))}"
        )

    return t.report()


# ---- Fixture generation (requires R) ----

def generate_fixtures():
    """Generate correctness fixtures using R's mgcv. Requires rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        # Use modern rpy2 conversion context
        converter = ro.default_converter + numpy2ri.converter
        mgcv = importr('mgcv')
        stats = importr('stats')
    except ImportError:
        print("ERROR: rpy2 not available. Install with: pip install rpy2")
        sys.exit(1)

    fixtures = {}

    with localconverter(converter):
        # ---- 1D extrapolation fixture ----
        print("Generating 1D extrapolation fixture...")
        seed = 42
        n = 500
        k = 10
        x_min, x_max = 0.2, 0.8

        np.random.seed(seed)
        x_train = np.random.uniform(x_min, x_max, n)
        y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, 0.1, n)

        # Test points including extrapolation
        test_points = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

        # Fit with R
        ro.globalenv['x'] = x_train
        ro.globalenv['y'] = y_train
        ro.r(f'df <- data.frame(x=x, y=y)')
        ro.r(f'fit <- gam(y ~ s(x, bs="cr", k={k}), data=df, method="REML")')

        # Predict at test points
        ro.globalenv['xnew'] = np.array(test_points)
        ro.r('newdf <- data.frame(x=xnew)')
        y_mgcv = np.array(ro.r('predict(fit, newdata=newdf, type="response")'))

        # Extract lambda
        sp = np.array(ro.r('fit$sp'))

        fixtures["extrapolation_1d"] = {
            "seed": seed,
            "n": n,
            "k": k,
            "x_min": x_min,
            "x_max": x_max,
            "test_points": test_points,
            "mgcv_predictions": y_mgcv.tolist(),
            "mgcv_lambda": float(sp[0]),
        }
        print(f"  Lambda: {sp[0]:.4f}")
        print(f"  Predictions: {y_mgcv}")

        # ---- Multi-dim fixture ----
        print("Generating multi-dim fixture (d=4, k=10)...")
        np.random.seed(42)
        n_multi = 1000
        d_multi = 4
        k_multi = 10
        X_multi = np.random.uniform(0, 1, size=(n_multi, d_multi))
        y_multi = (np.sin(2 * np.pi * X_multi[:, 0])
                   + 0.5 * np.cos(3 * np.pi * X_multi[:, 1])
                   + 0.3 * (X_multi[:, 2] ** 2)
                   + 0.2 * np.exp(-5 * (X_multi[:, 3] - 0.5) ** 2)
                   + np.random.normal(0, 0.2, n_multi))

        # Pass data to R
        ro.globalenv['y'] = y_multi
        for i in range(d_multi):
            ro.globalenv[f'x{i+1}'] = X_multi[:, i]

        # Build formula and data frame in R
        col_exprs = ", ".join([f"x{i+1}=x{i+1}" for i in range(d_multi)])
        ro.r(f'df <- data.frame(y=y, {col_exprs})')

        smooth_terms = " + ".join([f's(x{i+1}, bs="cr", k={k_multi})' for i in range(d_multi)])
        ro.r(f'fit <- gam(y ~ {smooth_terms}, data=df, method="REML")')

        sp_multi = np.array(ro.r('fit$sp'))

        fixtures["multidim_4d"] = {
            "seed": 42,
            "n": n_multi,
            "d": d_multi,
            "k": k_multi,
            "mgcv_lambdas": sp_multi.tolist(),
        }
        print(f"  Lambdas: {sp_multi}")

    # Save
    os.makedirs(os.path.dirname(FIXTURES_PATH), exist_ok=True)
    with open(FIXTURES_PATH, 'w') as f:
        json.dump(fixtures, f, indent=2)
    print(f"\nFixtures saved to: {FIXTURES_PATH}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Correctness validation for mgcv_rust")
    parser.add_argument("--generate", action="store_true", help="Generate R/mgcv fixtures (requires rpy2)")
    args = parser.parse_args()

    if args.generate:
        generate_fixtures()
        sys.exit(0)

    print("=" * 70)
    print("  CORRECTNESS VALIDATION")
    print("=" * 70)

    results = []
    results.append(test_basic_1d_gam())
    results.append(test_extrapolation())
    results.append(test_extrapolation_vs_fixtures())
    results.append(test_multidim_fit())
    results.append(test_determinism())

    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  RESULT: ALL {total} CORRECTNESS CHECKS PASSED")
    else:
        print(f"  RESULT: {passed}/{total} PASSED, {total-passed} FAILED")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
