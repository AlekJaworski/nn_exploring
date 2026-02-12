# mgcv-rust Standalone Tests

Tests for verifying a clean `pip install mgcv-rust` installation.

## Setup

```bash
python -m venv test_env
source test_env/bin/activate
pip install mgcv-rust numpy
# Only needed for comprehensive_benchmark.py:
pip install matplotlib
```

## Scripts

### Correctness

| Script | Description | Deps |
|--------|-------------|------|
| `validate_correctness.py` | 5 correctness checks: basic 1D GAM, extrapolation, extrapolation vs R fixtures, multi-dim (d=10), determinism | numpy |

```bash
python validate_correctness.py
```

### Performance

| Script | Description | Deps |
|--------|-------------|------|
| `validate_performance.py` | Regression test against saved baseline (4 configs). Use `--establish` to create a new baseline on your machine | numpy |
| `bench_vs_r.py` | 14-config comparison against hardcoded historical R bam()/gam() times | numpy |
| `quick_benchmark.py` | Fast smoke test: 4 problem sizes (1D) | numpy |
| `benchmark_rust_only.py` | 1D benchmark across 5 sizes with hardcoded R reference times | numpy |
| `performance_test.py` | Comprehensive 23-config benchmark suite (various n/d/k), saves JSON results | numpy |
| `comprehensive_benchmark.py` | Scaling analysis (n, k, d) with visualization plots | numpy, matplotlib |
| `profile_large_n.py` | Single large-n profiling (n=5000, d=1, k=20) | numpy |

### Recommended order

```bash
# 1. Quick smoke test (~5 seconds)
python quick_benchmark.py

# 2. Correctness validation (~2 seconds)
python validate_correctness.py

# 3. Performance vs R comparison (~30 seconds)
python bench_vs_r.py

# 4. Establish a performance baseline for your machine (~60 seconds)
python validate_performance.py --establish

# 5. Full benchmark suite (~2 minutes)
python performance_test.py

# 6. Scaling analysis with plots (requires matplotlib, ~3 minutes)
python comprehensive_benchmark.py
```

## Notes

- The `fixtures/` directory contains R mgcv reference values for correctness checks
  and a performance baseline from the development machine.
- `validate_performance.py --establish` creates a new baseline for YOUR machine.
  Subsequent runs compare against it to detect regressions.
- `bench_vs_r.py --run-r` requires R + the source tree (not available standalone).
  Without the flag, it compares against hardcoded historical R times.
