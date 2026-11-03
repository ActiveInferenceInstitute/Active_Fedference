"""Belief sharing report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _number, _number_list


class BeliefSharingReport(TypedDict):
    """Top-level payload written to ``belief_sharing.json``."""

    schema_version: str
    analysis_unit: str
    communicating_free_energy: list[float]
    communicating_mean: float
    communication_helps: bool
    difference_definition: str
    free_energy_gap: float
    incommunicado_free_energy: list[float]
    incommunicado_mean: float
    interval_method: str
    interval_percent: int
    n_agents: int
    n_seeds: int
    paired_difference_ci: list[float]
    paired_difference_mean: float
    paired_free_energy_difference: list[float]
    replication_unit: str


def _check_belief_sharing(payload: Mapping[str, object]) -> None:
    """Bind the paired free-energy estimand to its source arrays."""
    communicating = _number_list(
        payload["communicating_free_energy"], schema="belief_sharing", field="communicating_free_energy"
    )
    incommunicado = _number_list(
        payload["incommunicado_free_energy"], schema="belief_sharing", field="incommunicado_free_energy"
    )
    paired = _number_list(
        payload["paired_free_energy_difference"],
        schema="belief_sharing",
        field="paired_free_energy_difference",
    )
    n_seeds = payload["n_seeds"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    if n_seeds < 2 or len(communicating) != n_seeds or len(incommunicado) != n_seeds:
        raise ReportSchemaError("belief_sharing paired arrays must each contain n_seeds >= 2 values")
    expected = [inco - comm for inco, comm in zip(incommunicado, communicating)]
    if len(paired) != n_seeds or any(
        not math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12)
        for observed, target in zip(paired, expected)
    ):
        raise ReportSchemaError(
            "belief_sharing paired_free_energy_difference must equal "
            "incommunicado_free_energy - communicating_free_energy elementwise"
        )
    comm_mean = _number(payload["communicating_mean"], schema="belief_sharing", field="communicating_mean")
    incom_mean = _number(payload["incommunicado_mean"], schema="belief_sharing", field="incommunicado_mean")
    paired_mean = _number(
        payload["paired_difference_mean"],
        schema="belief_sharing",
        field="paired_difference_mean",
    )
    if not math.isclose(comm_mean, sum(communicating) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing communicating_mean disagrees with source values")
    if not math.isclose(incom_mean, sum(incommunicado) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing incommunicado_mean disagrees with source values")
    if not math.isclose(paired_mean, sum(paired) / n_seeds, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing paired_difference_mean disagrees with paired values")
    free_energy_gap = _number(payload["free_energy_gap"], schema="belief_sharing", field="free_energy_gap")
    if not math.isclose(free_energy_gap, paired_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ReportSchemaError("belief_sharing free_energy_gap must equal paired_difference_mean")
    if payload["communication_helps"] is not (comm_mean < incom_mean):
        raise ReportSchemaError("belief_sharing communication_helps disagrees with paired means")
    interval = _number_list(
        payload["paired_difference_ci"], schema="belief_sharing", field="paired_difference_ci"
    )
    if len(interval) != 2 or interval[0] > paired_mean or paired_mean > interval[1]:
        raise ReportSchemaError("belief_sharing paired_difference_ci must be ordered around the mean")
    if payload["difference_definition"] != "incommunicado_minus_communicating":
        raise ReportSchemaError(
            "belief_sharing difference_definition must be 'incommunicado_minus_communicating'"
        )
    if payload["replication_unit"] != "configured seed":
        raise ReportSchemaError("belief_sharing replication_unit must be 'configured seed'")
    analysis_unit = payload["analysis_unit"]
    if analysis_unit != "configured seed with within-seed paired conditions":
        raise ReportSchemaError("belief_sharing analysis_unit must identify within-seed pairing")
    if payload["interval_percent"] != 95:
        raise ReportSchemaError("belief_sharing interval_percent must be 95")


_BELIEF_SHARING_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "analysis_unit": "str",
        "communicating_free_energy": "list",
        "communicating_mean": "number",
        "communication_helps": "bool",
        "difference_definition": "str",
        "free_energy_gap": "number",
        "incommunicado_free_energy": "list",
        "incommunicado_mean": "number",
        "interval_method": "str",
        "interval_percent": "int",
        "n_agents": "int",
        "n_seeds": "int",
        "paired_difference_ci": "list",
        "paired_difference_mean": "number",
        "paired_free_energy_difference": "list",
        "replication_unit": "str",
    }
)
