"""Real Matplotlib, raster and typed-manifest presentation-boundary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analysis.report_schemas import ReportSchemaError, validate_report
from figures._common import FIGURE_EXPORT_DPI, apply_style, plt
from figures._presentation import (
    PresentationPanel,
    save_presentation_panel,
    save_presentation_panels,
    validate_presentation_files,
)


def _panel(identifier: str = "first", *, size: float = 24.0) -> PresentationPanel:
    apply_style()
    figure = plt.figure(figsize=(5.0, 1.4))
    figure.set_layout_engine("none")
    figure.text(0.5, 0.5, "Source-owned evidence", ha="center", va="center", fontsize=size)
    return PresentationPanel(identifier, figure, "The source-owned evidence panel, without a broader claim.")


def test_presentation_pair_and_manifest_are_measured_and_deterministic(tmp_path: Path) -> None:
    canonical = tmp_path / "evidence.png"
    manifest = save_presentation_panels([_panel()], canonical_path=canonical, expected_identifiers=("first",))
    payload = json.loads(manifest.read_text())
    validate_report("presentation_manifest", payload)
    raster = tmp_path / "evidence.slide-first.png"
    vector = raster.with_suffix(".pdf")
    original = (raster.read_bytes(), vector.read_bytes(), manifest.read_bytes())
    row = payload["panels"][0]
    assert row["minimum_label_px"] == pytest.approx(24 * FIGURE_EXPORT_DPI / 72)
    assert row["sha256"] == hashlib.sha256(raster.read_bytes()).hexdigest()
    assert row["src"] == "../figures/evidence.slide-first.png"
    assert validate_presentation_files(manifest) == [raster, vector]
    save_presentation_panels([_panel()], canonical_path=canonical, expected_identifiers=("first",))
    assert (raster.read_bytes(), vector.read_bytes(), manifest.read_bytes()) == original


def test_presentation_file_verification_rejects_tampering_and_missing_vectors(tmp_path: Path) -> None:
    canonical = tmp_path / "evidence.png"
    manifest = save_presentation_panels([_panel()], canonical_path=canonical, expected_identifiers=("first",))
    raster, vector = validate_presentation_files(manifest)
    original = raster.read_bytes()
    raster.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="source digest"):
        validate_presentation_files(manifest)
    raster.write_bytes(original)
    vector.unlink()
    with pytest.raises(ValueError, match="pair is missing"):
        validate_presentation_files(manifest)


def test_presentation_rejects_clipped_native_label(tmp_path: Path) -> None:
    panel = _panel()
    axis = panel.figure.add_axes((0.2, 0.2, 0.2, 0.2))
    axis.axis("off")
    label = axis.text(1, 0.5, "Clipped label", fontsize=24, clip_on=True)
    label.set_clip_box(axis.bbox)
    try:
        with pytest.raises(ValueError, match="label is clipped"):
            save_presentation_panel(panel, tmp_path / "clipped.png")
    finally:
        plt.close(panel.figure)


def test_presentation_inventory_does_not_accept_an_omitted_or_reordered_panel(tmp_path: Path) -> None:
    panel = _panel()
    try:
        with pytest.raises(ValueError, match="complete ordered source inventory"):
            save_presentation_panels(
                [panel], canonical_path=tmp_path / "evidence.png", expected_identifiers=("first", "second")
            )
        assert not list(tmp_path.iterdir())
    finally:
        plt.close(panel.figure)


def test_presentation_design_rejects_labels_shrunk_by_a_large_export(tmp_path: Path) -> None:
    panel = _panel(size=16.0)
    panel.figure.text(0.5, 8.0, "Tall source extent", fontsize=16.0)
    path = tmp_path / "panel.png"
    with pytest.raises(ValueError, match="design envelope"):
        save_presentation_panel(panel, path)
    assert not path.exists() and not path.with_suffix(".pdf").exists()


def test_presentation_failure_invalidates_prior_manifest(tmp_path: Path) -> None:
    canonical = tmp_path / "evidence.png"
    manifest = canonical.with_suffix(".slides.json")
    manifest.write_text("prior successful manifest")
    with pytest.raises(ValueError, match="below 16"):
        save_presentation_panels([_panel(size=12)], canonical_path=canonical, expected_identifiers=("first",))
    assert not manifest.exists()


@pytest.mark.parametrize(
    "field,value",
    (
        ("src", "../figures/../escape.png"),
        ("alt", ""),
        ("sha256", "not-a-digest"),
        ("minimum_label_px", True),
        ("minimum_label_px", 0),
        ("minimum_label_px", float("nan")),
    ),
)
def test_presentation_manifest_rejects_corrupt_records(field: str, value: object) -> None:
    panel = {"src": "../figures/panel.png", "alt": "Evidence", "sha256": "a" * 64, "minimum_label_px": 50.0}
    panel[field] = value
    with pytest.raises(ReportSchemaError):
        validate_report("presentation_manifest", {"schema_version": "1.0", "panels": [panel]})


def test_invalid_inventory_invalidates_prior_success_and_closes_figures(tmp_path: Path) -> None:
    canonical = tmp_path / "evidence.png"
    manifest = save_presentation_panels(
        [_panel()], canonical_path=canonical, expected_identifiers=("first",)
    )
    panel = _panel()
    number = panel.figure.number
    with pytest.raises(ValueError, match="complete ordered source inventory"):
        save_presentation_panels(
            [panel], canonical_path=canonical, expected_identifiers=("first", "missing")
        )
    assert not manifest.exists()
    assert not plt.fignum_exists(number)
