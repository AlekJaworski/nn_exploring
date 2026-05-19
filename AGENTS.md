# AGENTS.md

Project context and working instructions for any agent (Claude Code, Codex, etc.) operating in this repo. **Read this on session start, before answering any question that depends on project context.**

## Obsidian is the source of truth for project context

This codebase has a deep, evolving narrative around its numerical algorithms (mgcv parity, REML/fREML, scat/quantile, etc.) that does **not** live in code comments or in this file. It lives in `~/ObsidianVault/Projects/mgcv_rust/`. Roughly 80+ notes, organised into release reports, checkpoints, investigations, plans, and customer-feedback writeups.

**On session start, when the user's request involves any of these topics, read Obsidian first:**

| Topic | Start here |
|---|---|
| What's the current state / known issues / open xfails | `mgcv_rust - Known Issues and Status.md`, `mgcv_rust - Backlog - Next Numerical Steps.md` |
| Why we did X — release rationale | `mgcv_rust - 0.<N>.<M> release report ...md` (latest is 0.16.0 path-B fREML; commits beyond that are not yet written up) |
| Specific algorithm history | search `investigations/` for the keyword (`reparam`, `saturation`, `Tk`, `LAML`, `qgam`, `Sigma chain`, etc.) |
| How does mgcv handle X | `mgcv_analysis/mgcv/` (cloned mgcv source, checked into this repo at `/home/alex/vibe_coding/nn_exploring/mgcv_analysis/mgcv/`). Authoritative. |
| What's planned next | `mgcv_rust - Backlog - Next Numerical Steps.md` + the linked plan notes |
| Specific subsystem deep-dive | `mgcv_rust - Architecture.md`, `mgcv_rust - REML Optimization.md`, `mgcv_rust - Math Foundations.md`, `mgcv_rust - B5 Tk Reference.md`, `mgcv_rust - C Code Verification.md` |
| Production-integration concerns | `customer feedback 0.16.md` (latest), `mgcv_rust - parity handoff 2026-05-14.md` |

Open notes with `Read` (they're plain markdown). Don't guess at history from `git log` — the *why* lives in Obsidian, not the commit messages.

## Active epic — read this before any numerical-algorithm work

**[[mgcv_rust - Epic Numerical Step Control 2026-05-19]]** sequences five mgcv numerical-step-control algorithms (β-step blending, inner PIRLS step-halving, joint ρ–log φ Newton, joint ρ–log θ NB, β̇ IFT line search) into a shared 3-5 week deliverable with shared Phase-0 refactors. Status: Phase 1 (β-step blending) landed `1d3ba8f`; Phases 2-5 pending. **If the user asks for any of items 2-5, open the epic first** — it has the dependency order and the shared-infrastructure list that determines what to build.

## Project basics (the rest of this file is CLAUDE.md content)

`mgcv_rust` is a Rust port of R's `mgcv` package (Generalized Additive Models) with Python bindings via PyO3. The goal is byte-for-byte parity with R's mgcv on common families while being ~4× faster. Python API is sklearn-like (`fit`, `predict`, `score`).

### Build commands

```bash
# Development build (Linux)
maturin develop --features python,blas,blas-system

# Faster iteration (release-quality, faster compile)
maturin develop --profile release-fast --features python,blas,blas-system

# Release builds (platform-specific)
maturin build --release --features python,blas,blas-static       # Linux/macOS
maturin build --release --features python,blas,blas-mkl-static   # Windows
```

### Test commands

```bash
# Python test suites
pytest tests/usability -q       # smoke tests
pytest tests/parity/ -q         # parity fixtures vs R/mgcv
pytest tests/ergonomics/ -q     # API surface
pytest tests/real_data/ -q      # real-world dataset stress tests

# Rust tests (use --features python,blas,blas-system — without `python`, lib tests
# fail to compile because pyo3 attrs are feature-gated)
cargo test --release --features python,blas,blas-system --lib

# Validation scripts
python scripts/python/validate_correctness.py
python scripts/python/validate_performance.py --ci   # non-blocking CI mode
```

### Lint / check

```bash
cargo check --features blas,blas-system
cargo clippy --features blas,blas-system
```

## Architecture

### Two-layer Python API

- **`mgcv_rust.Gam`** (`python/mgcv_rust/_fitter.py`): ergonomic sklearn-like wrapper. Takes DataFrames, column names, supports subset views (`gam[["x0"]]`), `predict_ci`, `predict_diff`.
- **`mgcv_rust.GAM`** (`python/mgcv_rust/_low_level.py`): coercion layer wrapping the Rust core directly with raw ndarray inputs. Used when you need full control or byte-for-byte mgcv parity.
- **`GamPredictor`** (`python/mgcv_rust/_predictor.py`): deployment wrapper with strict schema validation for serialized models.

### Rust core (PIRLS-centric)

The numerical heart is `src/pirls.rs` — the penalized IRLS inner loop. It integrates with the REML outer loop (`src/reml/`) for automatic smoothing-parameter selection.

Key modules:
- `src/gam.rs` — GAM struct, `fit`/`predict` entry points
- `src/pirls.rs` — penalized IRLS inner loop (~190 KB, most of the numerical work; multiple `fit_pirls*` variants for different families)
- `src/link.rs` — `Link` enum with the conjugation invariant (any safety clamp in `inverse_link` is mirrored in every derivative — read the module-level docs)
- `src/basis.rs` — cubic-regression splines, B-splines (de Boor), random effects (`bs="re"`), tensor products
- `src/penalty.rs`, `src/block_penalty.rs` — second-derivative penalty matrices
- `src/reml/` — outer REML/LAML optimization for smoothing parameters; `fastreml.rs` is the bam-port (`fit_pirls_fastreml`)
- `src/family_theta.rs` — family-specific theta profiling (negative binomial, Tweedie, t)
- `src/reparam.rs` — reparameterization for constraints
- `src/linalg.rs` — linear algebra utilities

### PyO3 bindings

`src/lib.rs` defines `PyGAM` and wires it into the `python` feature. `python/mgcv_rust/mgcv_rust.*.so` is the compiled extension; `python/mgcv_rust/__init__.py` re-exports everything.

### BLAS feature flag

- Default (no BLAS): pure ndarray, no external deps, slower.
- `--features blas,blas-system`: links ndarray-linalg to system OpenBLAS + LAPACK. Required for benchmarks and performance comparisons. Use `blas-static` for portable builds, `blas-mkl-static` on Windows.

### Testing strategy

- **Parity tests** (`tests/parity/`): fixtures generated from R/mgcv; numerical diffs must be within tolerance. `_KNOWN_FEATURE_GAPS` in `conftest.py` and `_KNOWN_FREML_GAPS` in `test_parity_freml.py` track xfailed cases — each entry has a one-paragraph reason linked to an Obsidian investigation note.
- **Ergonomics tests** (`tests/ergonomics/`): API shape, predict scales, marginal views, confidence intervals.
- **Usability tests** (`tests/usability/`): multi-family smoke + targeted regressions (e.g. `test_em_nb_freml_smoke.py` pins the captured fREML non-finite-gradient regression).
- **Real-data tests** (`tests/real_data/`): stress tests on real customer data (gitignored — `~/vibe_coding/mgcv_rust_parity_handoff/`; skip gracefully when absent).
- **Rust unit tests**: `cargo test ... --lib` for in-module tests; `cargo test ... --tests` for `tests/*.rs` integration tests (note: `test_sigma_chain_reference` is currently broken on master from a signature change, unrelated).

### qgam OOS presets

- qgam parity contracts remain diagnostic; product decisions should use OOS pinball, coverage, and wall time.
- Real qgam OOS contracts live in `test_data/qgam_oos_real_contracts.json`; regenerate/expand them with `scripts/r/tests/extract_qgam_oos_real_contracts.R`.
- Benchmark Rust variants with `python scripts/python/bench_quantile_oos.py --no-synthetic --variants heuristic,heuristic_covcal,pin_cv_covcal`.
- Current 26-case real OOS finding: `heuristic_covcal` is the ultra-fast approximate preset (~72× qgam speedup, mean pinball ratio ~1.033); `pin_cv_covcal` is the quality preset (~1.4× speedup, mean ratio ~1.010). Uncalibrated q95 variants overcover and are not recommended for production q95.

### Cargo profiles

- `dev`: `opt-level=1` (fast compile, some optimization)
- `release-fast`: custom profile for faster iteration at release-quality performance
- `release`: aggressive LTO (`lto="fat"`, `codegen-units=1`, `opt-level=3`)

## Working norms

- **No new features** during the active epic unless the user explicitly asks. Items 2-5 of the epic close numerical gaps; don't add tensor product smooths / select=TRUE / ocat etc. in the same patch.
- **Don't paper over symptoms with conditionals.** If a fix needs an `if family == Foo`, justify it from mgcv source — e.g. `additive=TRUE` in `bam.r:567` is a real mgcv branch, not a hack. Document the mgcv reference in the comment.
- **When in doubt, read `mgcv_analysis/mgcv/` first.** The R source is the authoritative reference for what mgcv actually does. Obsidian notes annotate it; they don't replace it.
- **Commit message style**: see recent commits for the convention (`<subsystem>: <verb> <object>`, multi-paragraph body explaining mgcv source references and what changed). Examples: `reml: retry scat fREML with Fisher weights`, `fastreml: mgcv-style β init + penalised-deviance β-step blending`.
