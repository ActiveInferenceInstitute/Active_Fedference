"""Palette-discipline gate: one source of colour truth (``figures/_common.py``).

Every figure generator must draw its colours from the shared named palette in
``figures/_common.py`` (colour-blind-safe, naive vs robust distinct in greyscale)
rather than inlining hex literals. This keeps the figure set visually
coherent and recolourable from one place. The gate: no ``#rrggbb`` literal appears
in any generator except ``_common.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

_FIGURES = Path(__file__).resolve().parent.parent.parent / "src" / "figures"
# Hex in any web spelling (3/4/6/8 digit), checked at fullmatch for the palette
# constants and as a search for inline literals.
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
# Inline-literal offenders the shared palette is meant to replace, in every
# spelling matplotlib accepts (audit w9p4z6iv4 found a 6-digit-hex-only check let
# named colours / tab: / C0-9 / greyscale strings / rgb tuples slip past):
#   - hex of any length, rgb()/rgba() strings
#   - quoted matplotlib colour spellings: 'tab:red', 'C3', greyscale '0.5'
#   - common CSS colour NAMES (the structural white/black/none are allowed, as
#     are colormap names like 'viridis' which are not colours)
#   - numeric RGB(A) tuples passed to an explicit colour keyword (color=/facecolor=
#     /edgecolor=); xy=/xytext= numeric tuples are NOT colours and are left alone.
_INLINE_COLOUR = re.compile(
    r"#[0-9a-fA-F]{3,8}\b"
    r"|rgba?\s*\("
    r"|['\"](?:tab:[a-z]+|C[0-9])['\"]"
    r"|['\"]0?\.\d+['\"]"
    r"|['\"](?:crimson|red|green|blue|cyan|magenta|yellow|orange|purple|brown"
    r"|pink|gray|grey|navy|teal|maroon|olive|lime|aqua|fuchsia|silver|gold"
    r"|salmon|coral|indigo|violet|darkred|darkblue|firebrick|steelblue)['\"]"
    r"|(?:color|facecolor|edgecolor|markerfacecolor|markeredgecolor)\s*=\s*\(\s*[\d.]+\s*,"
)


def test_no_inline_colour_literals_outside_common() -> None:
    offenders: list[str] = []
    for gen in sorted(_FIGURES.glob("*.py")):
        if gen.name in ("__init__.py", "_common.py"):
            continue
        for i, line in enumerate(gen.read_text(encoding="utf-8").splitlines(), 1):
            if _INLINE_COLOUR.search(line):
                offenders.append(f"{gen.name}:{i} {line.strip()}")
    assert not offenders, (
        "inline colour literals (hex/rgb tuple/rgb()); use figures/_common palette:\n  "
        + "\n  ".join(offenders)
    )


def test_common_palette_defines_the_named_colours() -> None:
    from figures import _common

    for name in ("COLOR_NAIVE", "COLOR_ROBUST", "COLOR_ACCENT", "COLOR_MUTED",
                 "COLOR_AXIS", "COLOR_GRID"):
        value = getattr(_common, name)
        assert isinstance(value, str) and _HEX.fullmatch(value), f"{name} must be a hex colour"
    assert len(_common.ROBUST_CYCLE) >= 5  # enough distinct robust-method colours


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance of a #rrggbb colour (0=black .. 1=white)."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def test_naive_vs_robust_distinct_in_greyscale() -> None:
    # The palette claims naive-vs-robust distinctness in greyscale; assert it: the
    # two headline colours must differ enough in luminance to survive a B&W print.
    from figures import _common

    delta = abs(_relative_luminance(_common.COLOR_NAIVE)
                - _relative_luminance(_common.COLOR_ROBUST))
    assert delta >= 0.12, f"naive vs robust luminance gap too small for greyscale: {delta:.3f}"


def test_robust_cycle_colours_are_luminance_separated() -> None:
    # Robust-method curves should be distinguishable in greyscale. We check ALL
    # pairs (not just adjacent ones), so two same-luminance curves anywhere in the
    # cycle are caught. The bar (0.025) is deliberately gentler than the
    # naive-vs-robust headline gap (0.12): the cycle members are also separated by
    # hue (colour-blind-ordered) and by marker/legend, so greyscale is a secondary
    # cue here, whereas naive-vs-robust must survive greyscale on its own.
    from figures import _common

    lums = [_relative_luminance(c) for c in _common.ROBUST_CYCLE]
    for i in range(len(lums)):
        for j in range(i + 1, len(lums)):
            assert abs(lums[i] - lums[j]) >= 0.025, (
                f"ROBUST_CYCLE[{i}] vs [{j}] too close in luminance "
                f"({abs(lums[i] - lums[j]):.4f})"
            )
