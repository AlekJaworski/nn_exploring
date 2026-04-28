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
        "## Bar A / B / C",
        "",
    ]
    lines.append(
        "| case | A train | A test | A extrap | train abs | train rel | "
        "extrap abs | β maxabsdiff | dev rel | λ relerr |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        a = r.get("bar_a")
        b = r.get("bar_b") or {}
        c = r.get("bar_c") or {}
        if a is None:
            continue  # perf-only record
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

    # ---- trajectory section -----------------------------------------------
    traj_records = [r for r in records if "trajectory" in r]
    if traj_records:
        lines.append("")
        lines.append("## Newton trajectory vs mgcv (Stage 3)")
        lines.append("")
        lines.append(
            "First Newton iter where rust's REML score diverges from mgcv's by "
            ">5% of mgcv's score range. `—` means mgcv_rust stayed within 5% "
            "throughout the run."
        )
        lines.append("")
        lines.append(
            "| case | rust iters | mgcv iters | first diverged | rust final REML | mgcv final REML |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in traj_records:
            t = r["trajectory"]
            lines.append(
                "| {name} | {nri} | {nmi} | {fdi} | {rrf} | {mrf} |".format(
                    name=r["name"],
                    nri=t.get("n_rust_iters", "—"),
                    nmi=t.get("n_mgcv_iters", "—"),
                    fdi=t.get("first_diverged_iter") if t.get("first_diverged_iter") is not None else "—",
                    rrf=_fmt_num(t.get("rust_final_reml")),
                    mrf=_fmt_num(t.get("mgcv_final_reml")),
                )
            )

    # ---- perf section -----------------------------------------------------
    perf_records = [r for r in records if "perf" in r]
    if perf_records:
        lines.append("")
        lines.append("## Perf (median over N=5 fits, lower is better)")
        lines.append("")
        lines.append(
            "| case | rust median ms | rust min ms | mgcv median ms | rust/mgcv |"
        )
        lines.append("|---|---|---|---|---|")
        for r in perf_records:
            perf = r["perf"]
            rust = perf.get("rust") or {}
            mgcv = perf.get("mgcv")
            ratio = perf.get("rust_over_mgcv")
            lines.append(
                "| {name} | {rmed} | {rmin} | {mmed} | {ratio} |".format(
                    name=r["name"],
                    rmed=_fmt_num(rust.get("median_ms")),
                    rmin=_fmt_num(rust.get("min_ms")),
                    mmed=_fmt_num((mgcv or {}).get("median_ms")) if mgcv else "—",
                    ratio=_fmt_num(ratio),
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
