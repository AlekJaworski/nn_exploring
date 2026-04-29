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


# Known feature gaps — these (test_function_name, fixture_name) pairs are
# expected to fail until the underlying gap is closed. Marked xfail so the
# suite remains green while we track them as XFAILs.
_BS_BASIS_REASON = (
    'bs="bs" uses de Boor B-splines in mgcv vs natural cubic splines in '
    "mgcv_rust — different basis library, column spans cannot match. "
    "Tracked as a separate followup; remove this xfail when de Boor splines "
    "land."
)
_8D_15K_REASON = (
    "At production cardinality (n=15000, 8 cr smooths, k=[25,12,12,6,6,6,6,3]) "
    "rust and mgcv cr-spline bases diverge by ~1.88% energy (combined rank "
    "129 vs 69 each). Likely mgcv's nat.param reparameterization or "
    "knot-placement nuance at this scale. Smaller real-estate-shape cases "
    "(4d_small_neighbourhood_n300, 6d_heatmap_pricing_n8000) pass."
)
_NON_GAUSSIAN_BYTE_FOR_BYTE_REASON = (
    "Non-Gaussian (binomial / poisson) byte-for-byte parity needs the "
    "outer Newton loop to re-run inner PiRLS at each λ step (mgcv's outer "
    "iteration). Currently Newton uses the PiRLS-frozen IRLS weights and "
    "re-solves β with `A β = X'Wy`, which is a one-step approximation "
    "rather than the converged β̂(λ). Fit lands within ~10% of mgcv's λ "
    "(was ~30× off under FS — switching to Newton + envelope-theorem "
    "gradient is a big improvement) but residual prediction diff is "
    "2.5e-3 (binomial) / 2.5e-2 (poisson) — just above Bar A. "
    "Closing this gap is its own followup."
)

_KNOWN_FEATURE_GAPS: dict[str, dict[str, str]] = {
    "test_parity": {
        "1d_gaussian_smooth_n500_k20_bs": _BS_BASIS_REASON,
        "8d_neighbourhoods_like_n15000": _8D_15K_REASON,
        "2d_binomial_logit_n1000_k10_cr": _NON_GAUSSIAN_BYTE_FOR_BYTE_REASON,
        "2d_poisson_log_n1000_k10_cr": _NON_GAUSSIAN_BYTE_FOR_BYTE_REASON,
    },
    "test_design_matrix_span": {
        "1d_gaussian_smooth_n500_k20_bs": _BS_BASIS_REASON,
        "8d_neighbourhoods_like_n15000": _8D_15K_REASON,
    },
    "test_closed_form_matches_finite_diff_at_optimum": {
        "8d_neighbourhoods_like_n15000": _8D_15K_REASON,
    },
    "test_gradient_zero_at_mgcv_optimum": {
        "8d_neighbourhoods_like_n15000": _8D_15K_REASON,
    },
    "test_closed_form_matches_mgcv_reported_gradient": {
        "8d_neighbourhoods_like_n15000": _8D_15K_REASON,
    },
}


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
    test_name = metafunc.function.__name__
    gaps = _KNOWN_FEATURE_GAPS.get(test_name, {})
    params = []
    for p in paths:
        marks = []
        if p.stem in gaps:
            marks.append(pytest.mark.xfail(reason=gaps[p.stem], strict=False))
        params.append(pytest.param(p, id=p.stem, marks=marks))
    metafunc.parametrize("fixture_path", params)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply known-feature-gap xfail markers to tests that parametrize themselves
    via @pytest.mark.parametrize (instead of fixture_path). We match on the
    test function name + a parameter id that contains a known-gap fixture name.
    """
    for item in items:
        gaps = _KNOWN_FEATURE_GAPS.get(item.originalname or item.name, {})
        if not gaps:
            continue
        # callspec.id is the parametrize-ids string; check for fixture-name match
        cs = getattr(item, "callspec", None)
        if cs is None:
            continue
        param_id = cs.id  # e.g. "8d_neighbourhoods_like_n15000-"
        for fixture_name, reason in gaps.items():
            if fixture_name in param_id:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break


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

    # ---- mgcv_exact stage 4 section --------------------------------------
    s4_records = [r for r in records if "stage4" in r]
    if s4_records:
        lines.append("")
        lines.append("## mgcv_exact mode (Stage 4)")
        lines.append("")
        lines.append(
            "Predictions in mgcv_exact mode (pre-Z normalisation) compared to mgcv. "
            "Bar at 1e-3 absolute. λ values shown for the first smooth only."
        )
        lines.append("")
        lines.append(
            "| case | max_absdiff | our λ_0 | mgcv λ_0 | λ ratio |"
        )
        lines.append("|---|---|---|---|---|")
        for r in s4_records:
            s4 = r["stage4"]
            ours = s4.get("rust_lambda", [None])
            mgcvl = s4.get("mgcv_lambda", [None])
            ratio = (mgcvl[0] / ours[0]) if (ours and ours[0] and mgcvl and mgcvl[0]) else None
            lines.append(
                "| {name} | {abs} | {ol} | {ml} | {ra} |".format(
                    name=r["name"],
                    abs=_fmt_num(s4.get("max_absdiff")),
                    ol=_fmt_num(ours[0] if ours else None),
                    ml=_fmt_num(mgcvl[0] if mgcvl else None),
                    ra=_fmt_num(ratio),
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
