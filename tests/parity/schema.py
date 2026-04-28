"""
Parity fixture schema (v1).

Mirrors the JSON layout described in tests/parity/README.md. The R
fixture generator and the pytest harness both round-trip through this
module, so any schema drift breaks loudly at load time instead of inside
an assertion.

This module is intentionally dependency-free (stdlib only) — pytest
machinery in conftest.py is allowed to add numpy etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


@dataclass
class Metadata:
    mgcv_version: str
    r_version: str
    generated_at: str


@dataclass
class Inputs:
    seed: int
    n: int
    d: int
    k: list[int]
    bs: list[str]
    family: str
    link: str
    method: str
    weights: Optional[list[float]]
    x_train: list[list[float]]
    y_train: list[float]
    x_test: list[list[float]]
    x_extrap: list[list[float]]


@dataclass
class MgcvOutput:
    beta: list[float]
    vcov: list[list[float]]
    lambda_: list[float]            # `lambda` is reserved
    edf_per_smooth: dict[str, float]
    edf_total: float
    deviance: float
    scale: float
    n_iter: int
    predictions_train: list[float]
    predictions_test: list[float]
    predictions_extrap: list[float]
    predictions_train_se: list[float]
    predictions_test_se: list[float]


@dataclass
class Fixture:
    schema_version: int
    name: str
    description: str
    metadata: Metadata
    inputs: Inputs
    mgcv_output: MgcvOutput

    @classmethod
    def load(cls, path: str | Path) -> "Fixture":
        with open(path) as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Fixture":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Fixture schema version mismatch: got {raw.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}"
            )
        out = raw["mgcv_output"]
        return cls(
            schema_version=raw["schema_version"],
            name=raw["name"],
            description=raw.get("description", ""),
            metadata=Metadata(**raw["metadata"]),
            inputs=Inputs(**raw["inputs"]),
            mgcv_output=MgcvOutput(
                beta=out["beta"],
                vcov=out["vcov"],
                lambda_=out["lambda"],
                edf_per_smooth=out["edf_per_smooth"],
                edf_total=out["edf_total"],
                deviance=out["deviance"],
                scale=out["scale"],
                n_iter=out["n_iter"],
                predictions_train=out["predictions_train"],
                predictions_test=out["predictions_test"],
                predictions_extrap=out["predictions_extrap"],
                predictions_train_se=out["predictions_train_se"],
                predictions_test_se=out["predictions_test_se"],
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Remap `lambda_` -> `lambda` on the way out
        d["mgcv_output"]["lambda"] = d["mgcv_output"].pop("lambda_")
        return d

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---- tolerance config (tunable; not part of the on-disk fixture) ----


@dataclass
class Tolerances:
    """Per-bar tolerance profile. All comparisons use np.allclose semantics."""

    # Bar A: predictions
    pred_rtol: float = 1e-3
    pred_atol: float = 1e-6
    # Extrapolation tolerances are looser (compounded by lambda diff)
    pred_extrap_rtol: float = 5e-2
    pred_extrap_atol: float = 1e-3

    # Bar B: fitted model
    beta_rtol: float = 1e-3
    beta_atol: float = 1e-6
    vcov_diag_rtol: float = 5e-3
    vcov_diag_atol: float = 1e-8
    edf_per_smooth_atol: float = 0.05
    edf_total_atol: float = 0.1
    deviance_rtol: float = 1e-3
    scale_rtol: float = 1e-3

    # Bar C: smoothing parameters
    lambda_rtol: float = 5e-2


DEFAULT_TOLERANCES = Tolerances()
