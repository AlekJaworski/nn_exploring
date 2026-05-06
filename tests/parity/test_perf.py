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
from typing import Any, Optional

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


# Map fixture-side family names to mgcv_rust.GAM constructor names.
# mgcv_rust accepts these family strings: gaussian, binomial, poisson, gamma,
# quasipoisson, quasibinomial, t-dist, tweedie, inverse.gaussian, negbin, nb.
_RUST_FAM = {
    "gaussian":          "gaussian",
    "binomial":          "binomial",
    "poisson":           "poisson",
    "Gamma":             "gamma",
    "gamma":             "gamma",
    "tw":                "tweedie",          # profile-p (p kwarg omitted)
    "tweedie":           "tweedie",          # fixed-p (p kwarg from fixture)
    "Tweedie":           "tweedie",
    "inverse.gaussian":  "inverse.gaussian",
    "negative.binomial": "negbin",           # fixed-θ (theta kwarg from fixture)
    "nb":                "nb",               # profile-θ
    "quasipoisson":      "quasipoisson",
    "quasibinomial":     "quasibinomial",
    "t-dist":            "t-dist",
    "scat":              "t-dist",
}


import re as _re


def _parse_family_param(name: str, kind: str) -> Optional[float]:
    """Parse a family parameter encoded in the fixture name suffix.

    Conventions used by the fixture generator:
    - `_p15` → tweedie p=1.5 (last two digits are the decimal part)
    - `_theta2` / `_theta20` → negbin theta (whole number)
    - `_df5` → t-dist df

    Returns None if no match — the fixture may simply not encode the parameter
    (e.g. profile-p tw() and profile-θ nb() don't need one).
    """
    if kind == "p":
        m = _re.search(r"_p(\d+)(?=$|[_.])", name)
        if m:
            digits = m.group(1)
            return float(digits) / (10 ** (len(digits) - 1))
    elif kind == "theta":
        m = _re.search(r"_theta(\d+(?:\.\d+)?)(?=$|[_.])", name)
        if m:
            return float(m.group(1))
    elif kind == "df":
        m = _re.search(r"_df(\d+(?:\.\d+)?)(?=$|[_.])", name)
        if m:
            return float(m.group(1))
    return None


def _make_gam(name: str, inp: Any):
    """Construct a fresh mgcv_rust.GAM matching the fixture family/link/kwargs."""
    fam = _RUST_FAM.get(inp.family, inp.family)
    kwargs: dict[str, Any] = {"link": inp.link}
    # Family-specific parameters; omit if not encoded so rust falls back to
    # the right default (profile mode for tw()/nb(), p=1.5 for Tweedie etc.).
    if inp.family in ("tweedie", "Tweedie"):
        p = _parse_family_param(name, "p")
        if p is not None:
            kwargs["p"] = p
    elif inp.family == "negative.binomial":
        theta = _parse_family_param(name, "theta")
        if theta is not None:
            kwargs["theta"] = theta
    elif inp.family in ("t-dist", "scat"):
        df = _parse_family_param(name, "df")
        if df is not None:
            kwargs["df"] = df
    return mgcv_rust.GAM(fam, **kwargs)


def _time_rust_fit(fix: Fixture) -> dict[str, Any]:
    inp = fix.inputs
    x = np.asarray(inp.x_train, dtype=float)
    y = np.asarray(inp.y_train, dtype=float)
    k = list(inp.k)
    bs0 = inp.bs[0]

    # One warmup run to amortize first-call overhead (BLAS init, etc).
    g = _make_gam(fix.name, inp)
    g.fit(x, y, k=k, method=inp.method, bs=bs0)

    times_ms: list[float] = []
    for _ in range(_N_RUNS):
        g = _make_gam(fix.name, inp)
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


# R-side mgcv family() constructors, parameterized over (link, p, theta, df).
# Single-arg families ignore unused kwargs via .format_map.
_FAMILY_R = {
    "gaussian":          'gaussian(link="{link}")',
    "binomial":          'binomial(link="{link}")',
    "poisson":           'poisson(link="{link}")',
    "Gamma":             'Gamma(link="{link}")',
    "gamma":             'Gamma(link="{link}")',
    "tw":                'tw(link="{link}")',                  # profile-p
    "tweedie":           'Tweedie(p={p}, link="{link}")',      # fixed-p
    "Tweedie":           'Tweedie(p={p}, link="{link}")',
    "inverse.gaussian":  'inverse.gaussian(link="{link}")',
    "negative.binomial": 'negbin(theta={theta}, link="{link}")',  # mgcv's fixed-θ
    "nb":                'nb(link="{link}")',                  # profile-θ
    "quasipoisson":      'quasipoisson(link="{link}")',
    "quasibinomial":     'quasibinomial(link="{link}")',
    "t-dist":            'scat(link="{link}")',
    "scat":              'scat(link="{link}")',
}


class _Dflt(dict):
    """Format-map helper: missing keys render as the empty string,
    which is fine for single-arg family() templates that ignore p/theta/df."""
    def __missing__(self, key: str) -> str:
        return ""


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
    template = _FAMILY_R.get(inp.family)
    if template is None:
        return None  # silently skip cases we don't have an R-side mapping for
    fam_args = _Dflt(
        link=inp.link,
        p=_parse_family_param(fix.name, "p") or 1.5,
        theta=_parse_family_param(fix.name, "theta") or 1.0,
        df=_parse_family_param(fix.name, "df") or 5.0,
    )
    family_call = _ro.r(template.format_map(fam_args))
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
