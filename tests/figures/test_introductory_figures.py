from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from figures import graphical_abstract, system_overview


def test_graphical_abstract_generator_writes_png_and_pdf(tmp_path: Path) -> None:
    graphical_abstract.generate_graphical_abstract(project_root=tmp_path)

    figures_root = tmp_path / "output" / "figures"
    png = figures_root / "graphical_abstract.png"
    pdf = figures_root / "graphical_abstract.pdf"
    cover = tmp_path / "manuscript" / "cover_image.png"
    assert png.exists()
    assert pdf.exists()
    assert cover.exists()
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 10_000
    assert cover.stat().st_size > 10_000
    assert all("axis" not in label.lower() for label in graphical_abstract.METHOD_LABELS)
    assert graphical_abstract.COVER_NETWORK_N_AGENTS == system_overview.SYSTEM_OVERVIEW_METADATA["n_agents"]
    assert (
        graphical_abstract.COVER_NETWORK_N_ADVERSARIAL
        == system_overview.SYSTEM_OVERVIEW_METADATA["n_adversarial"]
    )


def test_system_overview_generator_writes_png_and_pdf(tmp_path: Path) -> None:
    system_overview.generate_system_overview(project_root=tmp_path)

    figures_root = tmp_path / "output" / "figures"
    png = figures_root / "system_overview.png"
    pdf = figures_root / "system_overview.pdf"
    assert png.exists()
    assert pdf.exists()
    assert png.stat().st_size > 10_000
    assert pdf.stat().st_size > 10_000


def test_system_overview_metadata_is_derived_from_drawn_data() -> None:
    """Displayed percentages and cover metadata share the plotted computation."""
    data = system_overview.build_data()
    metadata = system_overview.SYSTEM_OVERVIEW_METADATA

    assert metadata["naive_acc_pct"] == round(
        100 * float(data["naive"][system_overview.TRUE_STATE])
    )
    assert metadata["robust_acc_pct"] == round(
        100 * float(data["robust"][system_overview.TRUE_STATE])
    )
    assert metadata["n_agents"] == len(data["beliefs"])
    assert data["weights"].shape == (metadata["n_agents"],)
    assert np.isclose(data["weights"].sum(), 1.0)


def test_introductory_pdf_generation_is_byte_reproducible(tmp_path: Path) -> None:
    """PDF metadata must not inject wall-clock timestamps into release artifacts."""
    figures_root = tmp_path / "output" / "figures"
    system_overview.generate_system_overview(project_root=tmp_path)
    graphical_abstract.generate_graphical_abstract(project_root=tmp_path)
    first = {
        name: hashlib.sha256((figures_root / name).read_bytes()).hexdigest()
        for name in ("system_overview.pdf", "graphical_abstract.pdf")
    }

    system_overview.generate_system_overview(project_root=tmp_path)
    graphical_abstract.generate_graphical_abstract(project_root=tmp_path)
    second = {
        name: hashlib.sha256((figures_root / name).read_bytes()).hexdigest()
        for name in first
    }
    assert second == first
