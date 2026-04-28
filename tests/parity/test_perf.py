"""
Performance regression tests for the parity battery.

For every fixture, this:

1. Times N=5 mgcv_rust fits (median + min stored).
2. Asserts the median is within the per-case budget in
   `perf_budgets.json` (cases not listed are recorded but not
   asserted).
3. If rpy2 + mgcv are available, also times mgcv::bam over N=5 fits
   and records the rust/mgcv ratio for visibility — *not* asserted,
   since mgcv timing is much noisier under rpy2 round-trips and we
   don't want CI to thrash on it.

Output is appended to tests/parity/results.{json,md} alongside the
parity numbers, in a `perf` block per case. Updating budgets is a
single-file edit; see `perf_budgets.json` for the protocol.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

mgcv_rust = pytest.importorskip("mgcv_rust")
from schema import Fixture  # noqa: E402

_BUDGETS_PATH = HERE / "perf_budgets.json"
_BUDGET_SLACK = float(os.environ.get("MGCV_PERF_SLACK", "1.5"))
_N_RUNS = int(os.environ.get("MGCV_PERF_RUNS", "5"))


def _load_budgets() -> dict[str, float]:
    if not _BUDGETS_PATH.exists():
        return {}
    raw = json.loads(_BUDGETS_PATH.read_text())
    return {k: float(v) for k, v in raw.items() if not k.startswith("_")}


def _make_gam(family: str):
    fam = family.lower()
    if fam == "gaussian":
        return mgcv_rust.GAM()
    return mgcv_rust.GAM(fam if fam != "gamma" else "gamma")


def _time_rust_fit(fix: Fixture) -> dict[str, Any]:
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    k = list(inp.k)
    bs0 = inp.bs[0]

    # One warmup run to amortize first-call overhead (BLAS init, etc).
    g = _make_gam(inp.family)
    g.fit(x, y, k=k, method=inp.method, bs=bs0)

    times_ms: list[float] = []
    for _ in range(_N_RUNS):
        g = _make_gam(inp.family)
        t0 = time.perf_counter()
        g.fit(x, y, k=k, method=inp.method, bs=bs0)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_runs": _N_RUNS,
    }


# ---- mgcv timing via rpy2 (optional) ------------------------------------

_RPY2_AVAILABLE = False
_mgcv = None
_stats = None
_pandas2ri = None
_default_converter = None
_localconverter = None
_ro = None
try:
    import pandas as pd  # noqa: F401
    import rpy2.robjects as _ro  # noqa: F811
    from rpy2.rinterface_lib.callbacks import logger as _rpy2_logger
    from rpy2.robjects import default_converter as _default_converter  # noqa: F811
    from rpy2.robjects import pandas2ri as _pandas2ri  # noqa: F811
    from rpy2.robjects.conversion import localconverter as _localconverter  # noqa: F811
    from rpy2.robjects.packages import importr

    import logging as _logging

    _rpy2_logger.setLevel(_logging.ERROR)
    _mgcv = importr("mgcv")
    _stats = importr("stats")
    _ro.r("options(warn = -1)")
    _RPY2_AVAILABLE = True
except Exception:
    _RPY2_AVAILABLE = False


_FAMILY_R = {
    "gaussian": 'gaussian(link="{}")',
    "binomial": 'binomial(link="{}")',
    "poisson":  'poisson(link="{}")',
    "Gamma":    'Gamma(link="{}")',
    "gamma":    'Gamma(link="{}")',
}


def _time_mgcv_fit(fix: Fixture) -> dict[str, Any] | None:
    if not _RPY2_AVAILABLE:
        return None
    import pandas as pd

    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    df = pd.DataFrame({f"x{i}": x[:, i] for i in range(inp.d)})
    df["y"] = y
    rhs = " + ".join(f's(x{i}, k={inp.k[i]}, bs="{inp.bs[i]}")' for i in range(inp.d))
    family_call = _ro.r(_FAMILY_R[inp.family].format(inp.link))
    formula = _ro.Formula(f"y ~ {rhs}")

    with _localconverter(_pandas2ri.converter + _default_converter):
        rdf = _pandas2ri.py2rpy(df)

    # Warmup
    _mgcv.bam(formula, data=rdf, method=inp.method, family=family_call)

    times_ms: list[float] = []
    for _ in range(_N_RUNS):
        t0 = time.perf_counter()
        _mgcv.bam(formula, data=rdf, method=inp.method, family=family_call)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "n_runs": _N_RUNS,
    }


# ---- the test ----------------------------------------------------------


def test_perf(fixture_path, fixture: Fixture, parity_results: list[dict[str, Any]]) -> None:
    fix = fixture
    rust = _time_rust_fit(fix)
    mgcv = _time_mgcv_fit(fix)

    rec: dict[str, Any] = {
        "rust": rust,
        "mgcv": mgcv,
    }
    if mgcv is not None and mgcv["median_ms"] > 0:
        rec["rust_over_mgcv"] = rust["median_ms"] / mgcv["median_ms"]

    # Stitch into the existing parity record for this fixture, if any —
    # otherwise create a perf-only one. This keeps results.{json,md}
    # one row per case.
    matched = next((r for r in parity_results if r.get("name") == fix.name), None)
    if matched is None:
        parity_results.append({"name": fix.name, "perf": rec})
    else:
        matched["perf"] = rec

    budgets = _load_budgets()
    budget = budgets.get(fix.name)
    if budget is None:
        pytest.skip(
            f"no perf budget for {fix.name} in perf_budgets.json — "
            f"observed median={rust['median_ms']:.2f}ms"
        )
    limit = budget * _BUDGET_SLACK
    assert rust["median_ms"] <= limit, (
        f"perf regression on {fix.name}: "
        f"median={rust['median_ms']:.2f}ms exceeds budget*{_BUDGET_SLACK} = {limit:.2f}ms "
        f"(budget={budget}ms, slack from MGCV_PERF_SLACK env var)"
    )
