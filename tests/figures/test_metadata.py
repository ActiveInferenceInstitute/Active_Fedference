"""Figure provenance and publication-style contracts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figures import FIGURE_METADATA, apply_style
from figures._common import (
    FIGURE_EXPORT_DPI,
    MIN_QUANTITATIVE_FONT_SIZE,
    MIN_SCHEMATIC_FONT_SIZE,
)


def test_every_figure_generator_has_complete_metadata() -> None:
    figure_dir = Path(__file__).resolve().parents[2] / "src" / "figures"
    generators = {
        path.stem
        for path in figure_dir.glob("*.py")
        if path.name not in {"__init__.py", "_common.py", "_metadata.py"}
    }
    assert generators == set(FIGURE_METADATA)
    required = {
        "status",
        "source_relation",
        "source_figure",
        "source_equation",
        "source_citation",
        "estimand",
        "unit",
        "uncertainty",
        "replication_unit",
        "alt_text",
    }
    for generator, metadata in FIGURE_METADATA.items():
        assert required <= set(metadata), generator
        assert all(isinstance(metadata[key], str) for key in required)
        assert metadata["status"]
        assert metadata["source_relation"]
        assert metadata["estimand"]
        assert metadata["unit"]
        assert metadata["uncertainty"]
        assert metadata["replication_unit"]
        assert metadata["alt_text"]
        assert len(metadata["alt_text"]) <= 500
        assert metadata["estimand"] != "project-specific diagnostic quantity"
        assert metadata["unit"] != "declared in the embedded caption"
        assert metadata["uncertainty"] != "caption declares the interval or deterministic status"
        assert metadata["replication_unit"] != "caption declares the replication unit"


def test_publication_style_contract_is_readable() -> None:
    apply_style()
    assert plt.rcParams["savefig.dpi"] == FIGURE_EXPORT_DPI
    assert plt.rcParams["axes.titlesize"] >= 15
    assert plt.rcParams["axes.labelsize"] >= 13
    assert plt.rcParams["xtick.labelsize"] >= 11
    assert plt.rcParams["legend.fontsize"] >= MIN_QUANTITATIVE_FONT_SIZE
    assert MIN_SCHEMATIC_FONT_SIZE >= 8.5


def test_source_analogue_uncertainty_metadata_matches_captions() -> None:
    assert (
        FIGURE_METADATA["free_energy_comparison"]["uncertainty"]
        == "across-seed standard-deviation spread; not a confidence interval"
    )
    assert FIGURE_METADATA["emergence_bmr"]["uncertainty"] == (
        "none; deterministic closed-form comparison on a single posterior"
    )
    assert FIGURE_METADATA["emergence_bmr"]["replication_unit"] == "not applicable"
