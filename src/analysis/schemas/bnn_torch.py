"""Bnn torch report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from typing_extensions import Required

from analysis.schemas.primitives import (
    ReportSchemaError,
    SchemaDefinition,
    _check,
    _raise_missing,
    _raise_type_error,
)


class BnnTorchOkReport(TypedDict):
    """Executed PyTorch complement payload written to ``bnn_torch.json``."""

    accuracy_by_config: dict[str, object]
    beta: float
    consensus_max_simplex_deviation: float
    contamination_levels: list[float]
    deterministic: bool
    hidden_dim: int
    n_clients: int
    n_steps: int
    reported_contamination: float
    robust_accuracy: float
    robustness: float
    seed: int
    standard_accuracy: float
    status: Literal["ok"]
    torch_version: str


class BnnTorchSkippedReport(TypedDict, total=False):
    """Degradation payload when the PyTorch optional extra is unavailable."""

    status: Required[str]


def _check_bnn_torch(payload: Mapping[str, object]) -> None:
    if "status" not in payload:
        _raise_missing("bnn_torch", "status", "str")
    status = payload["status"]
    if not isinstance(status, str):
        _raise_type_error("bnn_torch", "status", "str", status)
    if status == "ok":
        _check(
            "bnn_torch",
            payload,
            SchemaDefinition(
                required={
                    "accuracy_by_config": "dict",
                    "beta": "number",
                    "consensus_max_simplex_deviation": "number",
                    "contamination_levels": "list",
                    "deterministic": "bool",
                    "hidden_dim": "int",
                    "n_clients": "int",
                    "n_steps": "int",
                    "reported_contamination": "number",
                    "robust_accuracy": "number",
                    "robustness": "number",
                    "seed": "int",
                    "standard_accuracy": "number",
                    "status": "str",
                    "torch_version": "str",
                }
            ),
        )
        return
    if status.startswith("skipped"):
        return
    raise ReportSchemaError("bnn_torch payload field 'status' must be 'ok' or start with 'skipped'")
