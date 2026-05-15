# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mgcv_rust` is a Rust port of R's `mgcv` package (Generalized Additive Models) with Python bindings via PyO3. The goal is byte-for-byte parity with R's mgcv on common families while being ~4× faster. Python API is sklearn-like (`fit`, `predict`, `score`).

## Build Commands

```bash
# Development build (Linux)
maturin develop --features python,blas,blas-system

# Faster iteration (release-quality, faster compile)
maturin develop --profile release-fast --features python,blas,blas-system

# Release builds (platform-specific)
maturin build --release --features python,blas,blas-static       # Linux/macOS
maturin build --release --features python,blas,blas-mkl-static   # Windows
```

## Test Commands

```bash
# Python test suites
pytest tests/usability -q       # smoke tests
pytest tests/parity/ -q         # 554 parity fixtures vs R/mgcv
pytest tests/ergonomics/ -q     # API surface (predict modes, CI, subsetting)
pytest tests/real_data/ -q      # real-world dataset stress tests

# Rust tests
cargo test --release --features blas,blas-system

# Validation scripts
python scripts/python/validate_correctness.py
python scripts/python/validate_performance.py --ci   # non-blocking CI mode
```

## Lint / Check

```bash
cargo check --features blas,blas-system
cargo clippy --features blas,blas-system
```

## Architecture

### Two-Layer API

- **`mgcv_rust.Gam`** (`python/mgcv_rust/_fitter.py`): ergonomic sklearn-like wrapper. Takes DataFrames, column names, supports subset views (`gam[["x0"]]`), `predict_ci`, `predict_diff`.
- **`mgcv_rust.GAM`** (`python/mgcv_rust/_low_level.py`): coercion layer that wraps the Rust core directly with raw ndarray inputs. Used when you need full control or byte-for-byte mgcv parity.
- **`GamPredictor`** (`python/mgcv_rust/_predictor.py`): deployment wrapper with strict schema validation for serialized models.

### Rust Core (PIRLS-centric)

The numerical heart is `src/pirls.rs` — the penalized IRLS inner loop. It integrates with the REML outer loop (`src/reml/`) for automatic smoothing-parameter selection.

Key modules:
- `src/gam.rs` — GAM struct, `fit`/`predict` entry points
- `src/pirls.rs` — penalized IRLS inner loop (~191 KB, most of the numerical work)
- `src/basis.rs` — cubic-regression splines, B-splines (de Boor), random effects (`bs="re"`), tensor products
- `src/penalty.rs`, `src/block_penalty.rs` — second-derivative penalty matrices
- `src/reml/` — outer REML/LAML optimization for smoothing parameters
- `src/family_theta.rs` — family-specific theta profiling (negative binomial, Tweedie, t)
- `src/reparam.rs` — reparameterization for constraints
- `src/linalg.rs` — linear algebra utilities

### PyO3 Bindings

`src/lib.rs` defines `PyGAM` and wires it into the `python` feature. `python/mgcv_rust/mgcv_rust.*.so` is the compiled extension; `python/mgcv_rust/__init__.py` re-exports everything.

### BLAS Feature Flag

- Default (no BLAS): pure ndarray, no external deps, slower.
- `--features blas,blas-system`: links ndarray-linalg to system OpenBLAS + LAPACK. Required for benchmarks and performance comparisons. Use `blas-static` for portable builds, `blas-mkl-static` on Windows.

### Testing Strategy

- **Parity tests** (`tests/parity/`): fixtures generated from R/mgcv; numerical diffs must be within tolerance.
- **Ergonomics tests** (`tests/ergonomics/`): API shape, predict scales, marginal views, confidence intervals.
- **Usability tests** (`tests/usability/`): multi-family smoke tests.
- **Real-data tests** (`tests/real_data/`): stress tests on actual datasets.
- **Rust unit tests** (`tests/*.rs`): component-level (gradient correctness, Cholesky stability, batch solves).

### Cargo Profiles

- `dev`: `opt-level=1` (fast compile, some optimization)
- `release-fast`: custom profile for faster iteration at release-quality performance
- `release`: aggressive LTO (`lto="fat"`, `codegen-units=1`, `opt-level=3`)
