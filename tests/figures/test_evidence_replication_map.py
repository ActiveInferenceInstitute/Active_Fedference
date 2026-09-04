"""Tests for the evidence-class and replication-unit map."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import pytest

from analysis.visual_contracts import evidence_replication_map_contract
from figures.evidence_replication_map import (
    _INDEPENDENT_UNIT_HEADER,
    generate_evidence_replication_map,
)


def test_evidence_map_names_the_report_as_independent_unit_owner() -> None:
    assert _INDEPENDENT_UNIT_HEADER == "Declared\nindependent unit"


def test_evidence_replication_map_writes_large_type_landscape_pair(tmp_path: Path) -> None:
    output = generate_evidence_replication_map(
        evidence_replication_map_contract(),
        project_root=tmp_path,
    )

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    pixels = mpimg.imread(output)
    assert pixels.shape[1] > pixels.shape[0]


def test_evidence_replication_map_is_deterministic(tmp_path: Path) -> None:
    report = evidence_replication_map_contract()
    first = generate_evidence_replication_map(report, project_root=tmp_path)
    first_png = first.read_bytes()
    first_pdf = first.with_suffix(".pdf").read_bytes()

    second = generate_evidence_replication_map(report, project_root=tmp_path)

    assert second.read_bytes() == first_png
    assert second.with_suffix(".pdf").read_bytes() == first_pdf


def test_evidence_replication_map_rejects_unknown_class(tmp_path: Path) -> None:
    report = evidence_replication_map_contract()
    report["lanes"][0]["evidence_class"] = "not_registered"

    with pytest.raises(ValueError, match="unknown evidence class"):
        generate_evidence_replication_map(report, project_root=tmp_path)


def test_evidence_replication_map_rejects_broken_nesting_chain(tmp_path: Path) -> None:
    report = evidence_replication_map_contract()
    report["nesting_edges"][0]["target"] = "trial"

    with pytest.raises(ValueError, match="nesting edges must connect"):
        generate_evidence_replication_map(report, project_root=tmp_path)


def test_evidence_replication_map_rejects_an_overdense_matrix_cell(tmp_path: Path) -> None:
    report = evidence_replication_map_contract()
    report["lanes"][2]["display"]["prohibited"] = " ".join(["overdense"] * 80)

    with pytest.raises(ValueError, match="exceeds three lines"):
        generate_evidence_replication_map(report, project_root=tmp_path)
