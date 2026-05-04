"""Tests for Ergo-7: bs="re" random effects basis.

Verifies that `predictor_basis_map={"cluster": "re"}` correctly:
1. Recovers per-cluster random effects (β̂ correlates with true β).
2. Shows EDF < #levels (shrinkage happened — λ > 0).
3. Handles unseen test-level gracefully (zero smooth contribution).
4. Reports k = #unique training levels (no sum-to-zero reduction).
5. (Optional) Compares to mgcv via Rscript when R is on PATH.
"""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np
import pytest

from mgcv_rust import GAMFitter


# ------------------------------------------------------------------ #
# Shared fixture                                                      #
# ------------------------------------------------------------------ #


def _make_cluster_data(
    n: int = 1000,
    n_clusters: int = 50,
    sigma_b: float = 1.5,
    sigma_e: float = 0.5,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with cluster random effects.

    Returns (X, y, true_betas) where:
        X[:, 0] = cluster IDs (1.0 .. n_clusters, as floats)
        y = beta_cluster[cluster] + noise
        true_betas = beta array of length n_clusters
    """
    rng = np.random.default_rng(seed)
    true_betas = rng.normal(0.0, sigma_b, n_clusters)
    cluster_ids = rng.integers(1, n_clusters + 1, size=n).astype(float)
    y = true_betas[cluster_ids.astype(int) - 1] + rng.normal(0.0, sigma_e, n)
    X = cluster_ids.reshape(-1, 1)
    return X, y, true_betas


# ------------------------------------------------------------------ #
# Test 1: Random-effect recovery (r > 0.7)                           #
# ------------------------------------------------------------------ #


def test_re_recovery():
    """Fitted cluster random effects should correlate strongly with truth.

    We fit intercept + bs="re" on cluster. With σ_b >> σ_e the shrinkage
    estimators should track the true β at r ≥ 0.7 (generous; typical is 0.9+).
    """
    n_clusters = 50
    X, y, true_betas = _make_cluster_data(n=1000, n_clusters=n_clusters)

    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        k_default=n_clusters,  # k is ignored for re, but avoids auto-k path
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)

    # Extract fitted cluster effects.  The design matrix has columns
    # [intercept | level_1 | level_2 | ... | level_50].
    # get_term_indices() returns (name, first, last) for the "cluster" smooth.
    ti = dict(
        (name, (first, last))
        for name, first, last in gam._native.get_term_indices()
    )
    first, last = ti["x0"]  # native names are x0, x1, ...
    coef = gam.get_coefficients()
    fitted_betas = coef[first : last + 1]  # length n_clusters

    assert len(fitted_betas) == n_clusters, (
        f"Expected {n_clusters} level coefficients, got {len(fitted_betas)}"
    )

    # Sort by cluster ID to align with true_betas (sorted 1..50 ↔ index 0..49).
    # Level IDs are 1.0..50.0 in sorted order — matching true_betas[0..49].
    r = np.corrcoef(true_betas, fitted_betas)[0, 1]
    print(f"\n  recovery test: r={r:.4f}", end="")
    assert r > 0.7, f"Correlation too low: r={r:.4f} (expected > 0.7)"
    print(f" PASS (r={r:.4f})")


# ------------------------------------------------------------------ #
# Test 2: EDF < #levels (shrinkage present)                          #
# ------------------------------------------------------------------ #


def test_re_shrinkage_edf():
    """Effective degrees of freedom should be strictly less than #levels.

    If λ=0 (unpenalized) EDF equals #levels. Any positive λ from REML
    shrinks it below that. We verify EDF < n_clusters.
    """
    n_clusters = 30
    X, y, _ = _make_cluster_data(n=600, n_clusters=n_clusters, seed=13)

    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)

    edf_map = dict(gam._native.get_edf_per_smooth())
    edf = edf_map.get("x0", None)
    assert edf is not None, "EDF not returned for x0"
    print(f"\n  shrinkage EDF: {edf:.2f} < {n_clusters} ", end="")
    assert edf < n_clusters, (
        f"EDF={edf:.2f} should be strictly less than n_clusters={n_clusters}"
    )
    print("PASS")


# ------------------------------------------------------------------ #
# Test 3: Unseen test-level → smooth contribution = 0               #
# ------------------------------------------------------------------ #


def test_re_unseen_level():
    """Prediction for an unseen cluster ID must have zero smooth contribution.

    We fit on clusters {1..50}, then predict on cluster=999. The random-effect
    smooth's columns should all be 0 for that row, so the smooth contributes 0
    and only the intercept remains.
    """
    n_clusters = 50
    X, y, _ = _make_cluster_data(n=1000, n_clusters=n_clusters, seed=99)

    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)

    # Build X for prediction with unseen cluster 999.
    X_unseen = np.array([[999.0]])

    # Evaluate lpmatrix: shape (1, 1 + n_clusters).
    lp = gam.evaluate_lpmatrix(X_unseen)  # (1, 1 + n_clusters)
    coef = gam.get_coefficients()

    # Smooth contribution = lp[0, 1:] @ coef[1:]
    smooth_contrib = float(lp[0, 1:] @ coef[1:])
    print(f"\n  unseen level smooth contribution: {smooth_contrib:.6f} ", end="")
    assert abs(smooth_contrib) < 1e-10, (
        f"Smooth contribution for unseen level should be ~0, got {smooth_contrib}"
    )
    print("PASS")


# ------------------------------------------------------------------ #
# Test 4: k = #unique training levels, no sum-to-zero reduction      #
# ------------------------------------------------------------------ #


def test_re_k_equals_n_levels():
    """The design matrix width for the RE smooth must equal #unique levels.

    Random effects skip sum-to-zero centering (which would reduce k to k-1),
    so get_term_indices() should report a span of exactly n_clusters columns.
    """
    n_clusters = 40
    X, y, _ = _make_cluster_data(n=800, n_clusters=n_clusters, seed=55)

    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)

    ti = gam._native.get_term_indices()  # [(name, first, last), ...]
    assert len(ti) == 1, f"Expected 1 smooth term, got {len(ti)}"
    _name, first, last = ti[0]
    n_cols = last - first + 1
    print(f"\n  RE k cols: {n_cols}, expected {n_clusters} ", end="")
    assert n_cols == n_clusters, (
        f"Expected {n_clusters} columns (one per level, no centering), got {n_cols}"
    )
    print("PASS")


# ------------------------------------------------------------------ #
# Test 5: GAMFitter API works with predictor_basis_map               #
# ------------------------------------------------------------------ #


def test_re_fitter_api():
    """Sanity check: GAMFitter with predictor_basis_map='re' fits and predicts."""
    n_clusters = 20
    X, y, _ = _make_cluster_data(n=400, n_clusters=n_clusters, seed=2)

    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)

    # Predict on training data — should return array of length n.
    preds = gam.predict(X)
    assert preds.shape == (400,), f"Unexpected prediction shape: {preds.shape}"

    # Predictions should be finite.
    assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf"
    print("\n  fitter API PASS")


# ------------------------------------------------------------------ #
# Test 6: Mixed model — continuous + random effect                   #
# ------------------------------------------------------------------ #


def test_re_mixed_continuous_and_re():
    """Fit a model with one continuous smooth and one random effect.

    Verifies that the bs_list dispatch routes correctly for a 2-column X.
    """
    rng = np.random.default_rng(42)
    n = 500
    n_clusters = 25

    x_cont = rng.uniform(0.0, 1.0, n)
    cluster = rng.integers(1, n_clusters + 1, size=n).astype(float)
    true_betas = rng.normal(0.0, 1.0, n_clusters)
    y = np.sin(2 * np.pi * x_cont) + true_betas[cluster.astype(int) - 1] + rng.normal(0.0, 0.3, n)

    X = np.column_stack([x_cont, cluster])

    gam = GAMFitter(
        predictors=["x_cont", "cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"x_cont": 10, "cluster": n_clusters},
    )
    gam.fit(X, y)

    preds = gam.predict(X)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))

    # Check term indices: x_cont has k-1 cols (centred), cluster has n_clusters cols.
    ti = gam._native.get_term_indices()
    assert len(ti) == 2, f"Expected 2 terms, got {len(ti)}"
    _n1, f1, l1 = ti[0]
    _n2, f2, l2 = ti[1]
    re_k = l2 - f2 + 1
    assert re_k == n_clusters, f"RE term has {re_k} cols, expected {n_clusters}"
    print(f"\n  mixed model: x_cont cols={(l1-f1+1)}, cluster cols={re_k} PASS")


# ------------------------------------------------------------------ #
# Test 7: MGCV parity (skip if R not available)                      #
# ------------------------------------------------------------------ #


def test_re_mgcv_parity():
    """Compare predictions against mgcv's bs='re' when Rscript is available."""
    # Check if Rscript is available.
    try:
        result = subprocess.run(
            ["Rscript", "--version"], capture_output=True, timeout=10
        )
        if result.returncode != 0:
            pytest.skip("Rscript not available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Rscript not available")

    n_clusters = 10
    n = 200
    rng = np.random.default_rng(77)
    true_betas = rng.normal(0.0, 1.5, n_clusters)
    cluster = rng.integers(1, n_clusters + 1, size=n).astype(float)
    y = true_betas[cluster.astype(int) - 1] + rng.normal(0.0, 0.3, n)

    X = cluster.reshape(-1, 1)

    # Fit our model.
    gam = GAMFitter(
        predictors=["cluster"],
        predictor_basis_map={"cluster": "re"},
        term_k_mapping={"cluster": n_clusters},
    )
    gam.fit(X, y)
    our_preds = gam.predict(X)

    # Write data and R script to temp files.
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("cluster,y\n")
        for c, yi in zip(cluster, y):
            f.write(f"{int(c)},{yi:.10f}\n")
        data_path = f.name

    r_script = f"""
library(mgcv)
dat <- read.csv('{data_path}')
dat$cluster <- as.factor(dat$cluster)
fit <- gam(y ~ s(cluster, bs='re'), data=dat, method='REML')
preds <- predict(fit, type='response')
cat(paste(preds, collapse=','))
"""
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(r_script)
        r_path = f.name

    try:
        result = subprocess.run(
            ["Rscript", r_path], capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            pytest.skip(f"R script failed: {result.stderr[:200]}")
        mgcv_preds = np.array([float(x) for x in result.stdout.strip().split(",")])
    except subprocess.TimeoutExpired:
        pytest.skip("R script timed out")
    finally:
        import os
        os.unlink(data_path)
        os.unlink(r_path)

    corr = np.corrcoef(our_preds, mgcv_preds)[0, 1]
    print(f"\n  mgcv parity: r={corr:.4f}", end="")
    assert corr > 0.95, f"Poor correlation with mgcv: r={corr:.4f}"
    print(" PASS")
