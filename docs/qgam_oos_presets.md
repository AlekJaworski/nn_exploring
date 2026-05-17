# qgam OOS Presets

`fit_quantile` exposes named presets for the two recommended out-of-sample qgam-style paths. They are convenience aliases only; the explicit arguments still work and keep the same behavior.

```python
from mgcv_rust import fit_quantile

# Fast OOS path: Rust heuristic sigma plus empirical coverage calibration.
gam_fast, sigma_fast, info_fast = fit_quantile(
    X_train,
    y_train,
    tau=0.95,
    k=[10, 10, 10],
    preset="fast_oos",
)

# Quality OOS path: pinball-CV sigma plus empirical coverage calibration.
gam_quality, sigma_quality, info_quality = fit_quantile(
    X_train,
    y_train,
    tau=0.95,
    k=[10, 10, 10],
    preset="quality_oos",
)
```

Preset mappings:

| Preset | Equivalent explicit options | Intended use |
|---|---|---|
| `fast_oos` | `coverage_calibrate=True` | Ultra-fast approximate production path. |
| `quality_oos` | `calibrate=True, loss="pin", coverage_calibrate=True` | Higher-quality path when extra fit time is acceptable. |

Current real-data benchmark summary on 26 qgam-referenced OOS cases:

| Path | Mean pinball ratio vs qgam | Within 5% | Mean speedup |
|---|---:|---:|---:|
| `fast_oos` | ~1.033 | 23/26 | ~72x |
| `quality_oos` | ~1.010 | 25/26 | ~1.4x |

`coverage_calibrate=True` shifts the fitted quantile intercept so empirical training coverage matches `tau`. This is an OOS-product correction, not a qgam parity mode. For parity diagnostics against R/qgam internals, keep using the explicit lower-level arguments and contracts.

`final_algorithm=` remains available for diagnostics. Passing it to `fit_quantile` emits a `RuntimeWarning` because the quantile-specific ELF Newton/LAML gradient is not complete yet; calibration folds still use the default fast path, and the final fit should not be treated as a production preset.

Holdout and K-fold coverage-calibration variants are still benchmark-only diagnostics in `scripts/python/bench_quantile_oos.py`. If they become public API, add new named presets rather than changing these mappings.

Benchmark the public preset names directly with:

```bash
python scripts/python/bench_quantile_oos.py --no-synthetic --variants fast_oos,quality_oos
```

The runner still supports the older explicit-option variant names (`heuristic_covcal`, `pin_cv_covcal`) for continuity with historical result files.
