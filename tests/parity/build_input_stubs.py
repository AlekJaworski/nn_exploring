"""
Materialize each parity case's inputs to JSON in tests/parity/_stubs/.

This is the bridge between Python (where data-generating functions live)
and R (which reads the stubs, runs gam(), and dumps the full fixture).

Usage:
    python -m tests.parity.build_input_stubs
    # or, from this directory:
    python build_input_stubs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ is None:
    sys.path.insert(0, str(HERE))
    from cases import all_cases  # type: ignore
else:
    from .cases import all_cases

STUBS_DIR = HERE / "_stubs"


def main() -> None:
    STUBS_DIR.mkdir(exist_ok=True)
    for case in all_cases():
        inputs = case.realize()
        stub = {
            "schema_version": 1,
            "name": case.name,
            "description": case.description,
            "inputs": inputs,
        }
        out = STUBS_DIR / f"{case.name}.json"
        with open(out, "w") as f:
            json.dump(stub, f, indent=2)
        print(f"  stub -> {out.relative_to(HERE.parent.parent)}")
    print(f"Wrote {len(all_cases())} stubs to {STUBS_DIR.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
