"""Tests for the metadata-backed layered graphical abstract."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from figures import generate_graphical_abstract
from figures.graphical_abstract import _build_graphical_abstract_figure

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"


def test_graphical_abstract_writes_figure_and_cover(tmp_path: Path) -> None:
    path = generate_graphical_abstract(project_root=tmp_path)
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.with_suffix(".pdf").read_bytes()[:4] == _PDF_MAGIC
    cover = tmp_path / "manuscript" / "cover_image.png"
    assert cover.read_bytes()[:8] == _PNG_MAGIC
    assert Image.open(path).mode == "RGB"
    assert Image.open(cover).mode == "RGB"


def test_graphical_abstract_is_deterministic(tmp_path: Path) -> None:
    first = generate_graphical_abstract(project_root=tmp_path)
    first_bytes = first.read_bytes()
    second = generate_graphical_abstract(project_root=tmp_path)
    assert second.read_bytes() == first_bytes


def _artist_bbox(fig, gid: str):
    matches = [artist for artist in fig.findobj() if artist.get_gid() == gid]
    assert len(matches) == 1, f"expected exactly one artist tagged {gid!r}"
    artist = matches[0]
    bbox_patch = getattr(artist, "get_bbox_patch", lambda: None)()
    if bbox_patch is not None:
        return bbox_patch.get_window_extent(fig.canvas.get_renderer())
    return artist.get_window_extent(fig.canvas.get_renderer())


def test_graphical_abstract_title_and_performance_geometry_do_not_overlap() -> None:
    fig = _build_graphical_abstract_figure()
    fig.canvas.draw()

    title_stack = [
        _artist_bbox(fig, gid)
        for gid in (
            "ga-main-title",
            "ga-main-subtitle",
            "ga-identity",
            "ga-eq7-qualification",
        )
    ]
    for upper, lower in zip(title_stack, title_stack[1:], strict=False):
        assert upper.y0 > lower.y1

    performance_title = _artist_bbox(fig, "ga-performance-title")
    performance_subtitle = _artist_bbox(fig, "ga-performance-subtitle")
    naive_card = _artist_bbox(fig, "ga-card-naive")
    robust_card = _artist_bbox(fig, "ga-card-robust")
    gain_callout = _artist_bbox(fig, "ga-gain-callout")
    assert not performance_title.overlaps(performance_subtitle)
    assert not performance_subtitle.overlaps(naive_card)
    assert not naive_card.overlaps(robust_card)
    assert not gain_callout.overlaps(naive_card)
    assert not gain_callout.overlaps(robust_card)
    for card_gid in ("ga-card-naive", "ga-card-robust"):
        subtitle = _artist_bbox(fig, f"{card_gid}-subtitle")
        for index in range(4):
            assert not subtitle.overlaps(_artist_bbox(fig, f"{card_gid}-bar-{index}"))


def test_graphical_abstract_consensus_label_is_separate_from_glyph_bars() -> None:
    fig = _build_graphical_abstract_figure()
    fig.canvas.draw()

    label = _artist_bbox(fig, "ga-consensus-label")
    for index in range(4):
        assert not label.overlaps(_artist_bbox(fig, f"ga-consensus-bar-{index}"))

    card = _artist_bbox(fig, "ga-consensus-card")
    for index in range(5):
        assert not card.overlaps(_artist_bbox(fig, f"ga-agent-{index}"))
