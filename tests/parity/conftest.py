"""
Pytest plumbing for the parity battery.

Discovers every fixture under tests/parity/fixtures/ (skipping ones whose
name starts with `EXAMPLE`), parameterizes test_parity over them, and
collects a per-case results record that gets dumped to results.json and
results.md at session end. The dump is the source of truth for ratcheting
the Bar B / Bar C tolerances over time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from schema import DEFAULT_TOLERANCES, Fixture, Tolerances  # noqa: E402

FIXTURES_DIR = HERE / "fixtures"
RESULTS_JSON = HERE / "results.json"
RESULTS_MD = HERE / "results.md"


def _discover_fixture_paths() -> list[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(
        p for p in FIXTURES_DIR.glob("*.json")
        if not p.name.startswith("EXAMPLE")
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize any test that takes a `fixture_path` arg over discovered fixtures."""
    if "fixture_path" not in metafunc.fixturenames:
        return
    paths = _discover_fixture_paths()
    if not paths:
        metafunc.parametrize(
            "fixture_path",
            [pytest.param(None, marks=pytest.mark.skip(
                reason="No fixtures generated yet — run scripts/r/generate_parity_fixtures.R"
            ))],
            ids=["no-fixtures"],
        )
        return
    metafunc.parametrize(
        "fixture_path",
        paths,
        ids=[p.stem for p in paths],
    )


@pytest.fixture(scope="session")
def tolerances() -> Tolerances:
    return DEFAULT_TOLERANCES


@pytest.fixture
def fixture(fixture_path: Path) -> Fixture:
    return Fixture.load(fixture_path)


# ---- session-level results accumulator ----------------------------------

_PARITY_RECORDS: list[dict[str, Any]] = []


@pytest.fixture(scope="session")
def parity_results() -> list[dict[str, Any]]:
    """Mutable list test_parity appends per-case records to."""
    return _PARITY_RECORDS


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _PARITY_RECORDS:
        return
    _write_results(_PARITY_RECORDS)


def _write_results(records: list[dict[str, Any]]) -> None:
    RESULTS_JSON.write_text(json.dumps(records, indent=2, sort_keys=True))
    RESULTS_MD.write_text(_render_markdown(records))


def _render_markdown(records: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# Parity results",
        "",
        "Bar A is `ok` flag from `predict()` agreement at the given tolerance "
        "(rtol/atol from `schema.Tolerances`). The `max_absdiff` / `max_relerr` "
        "columns give the actual numbers so ratchets are easy to see. Bar B "
        "and C are tracked, not blocking.",
        "",
    ]
    lines.append(
        "| case | A train | A test | A extrap | train abs | train rel | "
        "extrap abs | β maxabsdiff | dev rel | λ relerr |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        a = r["bar_a"]
        b = r["bar_b"]
        c = r["bar_c"]
        lines.append(
            "| {name} | {atr} | {ate} | {aex} | {tabs} | {trel} | {eabs} "
            "| {bma} | {dr} | {lr} |".format(
                name=r["name"],
                atr=_fmt_ok(a["train"].get("ok")),
                ate=_fmt_ok(a["test"].get("ok")),
                aex=_fmt_ok(a["extrap"].get("ok")),
                tabs=_fmt_num(a["train"].get("max_absdiff")),
                trel=_fmt_num(a["train"].get("max_relerr")),
                eabs=_fmt_num(a["extrap"].get("max_absdiff")),
                bma=_fmt_num(b.get("beta_maxabsdiff")),
                dr=_fmt_num(b.get("deviance_relerr")),
                lr=_fmt_num(c.get("lambda_max_relerr")),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt_ok(v: Any) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2e}"
    return str(v)
