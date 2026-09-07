"""Measured, source-owned figure panels for projection-size publication.

The canonical portrait figure remains owned by its existing generator. This
module supplies the same explicit save boundary for a generator's complete,
ordered presentation panels; it never crops an existing image or infers which
scientific evidence may be omitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.text import Text
from PIL import Image

from ._common import FIGURE_EXPORT_DPI, plt, validate_figure_text

MIN_EFFECTIVE_SLIDE_FONT_SIZE = 16.0
# Conservative design envelope; actual placement is independently measured by
# Template after PDF compilation. Two-line headings can reduce the safe height.
PRESENTATION_DESIGN_WIDTH_PT = 380.0
PRESENTATION_DESIGN_HEIGHT_PT = 125.0


@dataclass(frozen=True)
class PresentationPanel:
    """One explicitly selected semantic panel in source-defined reading order."""

    identifier: str
    figure: Figure
    alt_text: str


class PresentationPanelRecord(TypedDict):
    """Exact generic Template manifest row; labels are measured from artists."""

    src: str
    alt: str
    sha256: str
    minimum_label_px: float


def _visible_labels(fig: Figure) -> list[Text]:
    """Traverse painted parents, excluding hidden artists and hidden axes."""
    result: list[Text] = []

    def visit(artist: Artist) -> None:
        if not artist.get_visible():
            return
        if isinstance(artist, Text) and artist.get_text().strip():
            result.append(artist)
        for child in artist.get_children():
            visit(child)

    visit(fig)
    return result


def save_presentation_panel(panel: PresentationPanel, path: Path) -> PresentationPanelRecord:
    """Save one deterministic pair only after measuring labels and clipping.

    The design check uses the exported raster dimensions, including its tight
    bounding box, rather than the nominal canvas. The final embedded PDF gate
    remains mandatory and may reject a panel which passed this design envelope.
    """
    fig = panel.figure
    if not panel.identifier or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in panel.identifier
    ):
        raise ValueError("presentation identifier requires lowercase ASCII words, digits, or hyphens")
    if not panel.alt_text.strip() or len(panel.alt_text) > 500:
        raise ValueError("presentation alternative must contain 1 to 500 characters")
    if path.suffix != ".png":
        raise ValueError("presentation panels require a PNG destination")
    validate_figure_text(fig, minimum_font_size=16.0)
    labels = _visible_labels(fig)
    if not labels:
        raise ValueError("presentation panel requires measured visible labels")
    for label in labels:
        clip = label.get_clip_box() if label.get_clip_on() else None
        extent = label.get_window_extent()
        if clip is not None and (
            extent.x0 < clip.x0 or extent.x1 > clip.x1 or extent.y0 < clip.y0 or extent.y1 > clip.y1
        ):
            raise ValueError(f"presentation label is clipped: {label.get_text()!r}")
    minimum = min(float(label.get_fontsize()) for label in labels)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("presentation label measurements must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=FIGURE_EXPORT_DPI)
        with Image.open(path) as raster:
            width, height = raster.size
        minimum_px = minimum * FIGURE_EXPORT_DPI / 72.0
        effective = minimum_px * min(
            PRESENTATION_DESIGN_WIDTH_PT / width,
            PRESENTATION_DESIGN_HEIGHT_PT / height,
        )
        if effective + 1e-9 < MIN_EFFECTIVE_SLIDE_FONT_SIZE:
            raise ValueError(
                f"presentation panel {panel.identifier!r} has {effective:.2f} pt labels "
                f"in the design envelope; requires {MIN_EFFECTIVE_SLIDE_FONT_SIZE:g} pt"
            )
        fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        return {
            "src": f"../figures/{path.name}",
            "alt": panel.alt_text,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "minimum_label_px": minimum_px,
        }
    except BaseException:
        path.unlink(missing_ok=True)
        path.with_suffix(".pdf").unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def save_presentation_panels(
    panels: Sequence[PresentationPanel],
    *,
    canonical_path: Path,
    expected_identifiers: Sequence[str],
) -> Path:
    """Bind a complete source-declared panel inventory to a typed manifest."""
    from analysis.workflow import _write_json

    manifest = canonical_path.with_suffix(".slides.json")
    try:
        manifest.unlink(missing_ok=True)
        identifiers = [panel.identifier for panel in panels]
        if (
            not 1 <= len(panels) <= 64
            or len(set(identifiers)) != len(identifiers)
            or identifiers != list(expected_identifiers)
        ):
            raise ValueError("presentation panels must exactly match the complete ordered source inventory")
        records = [
            save_presentation_panel(
                panel,
                canonical_path.with_name(f"{canonical_path.stem}.slide-{panel.identifier}.png"),
            )
            for panel in panels
        ]
        return _write_json(
            {"schema_version": "1.0", "panels": records},
            manifest,
            schema="presentation_manifest",
        )
    finally:
        for panel in panels:
            plt.close(panel.figure)


def validate_presentation_files(manifest: Path) -> list[Path]:
    """Verify every declared pair and source digest without directory discovery."""
    from analysis.report_schemas import validate_report

    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1024 * 1024:
        raise ValueError("presentation manifest must be a bounded local regular file")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("presentation manifest contains duplicate JSON keys")
            result[key] = value
        return result

    payload = json.loads(manifest.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    validate_report("presentation_manifest", payload)
    paths: list[Path] = []
    prefix = manifest.name.removesuffix(".slides.json") + ".slide-"
    for panel in payload["panels"]:
        raster = manifest.parent / Path(panel["src"]).name
        if not raster.name.startswith(prefix):
            raise ValueError("presentation panel does not belong to its canonical figure")
        for path in (raster, raster.with_suffix(".pdf")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("presentation panel pair is missing or symlinked")
            paths.append(path)
        if hashlib.sha256(raster.read_bytes()).hexdigest() != panel["sha256"]:
            raise ValueError("presentation raster does not match its source digest")
    return paths
