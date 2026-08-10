"""{{TOKEN}} provenance lint: no hardcoded data numerals in manuscript prose.

The project's standing rule is "every prose number is a ``{{TOKEN}}``" — a
rendered result must come from the analysis pipeline, never be typed in. This
gate enforces it for the highest-risk literals: it strips the legitimately-numeric
contexts first — fenced code, inline ``code``, ``$math$``, ``{{TOKEN}}``
placeholders, and pandoc ``{#label ...}`` attributes — then asserts no bare
*data* literal survives in any of these spellings: Arabic decimals
(``\\d+\\.\\d+``), comma-decimals (``\\d+,\\d+``), bare-dot decimals (``.95``),
scientific notation (``1e-3``), percentages (``\\d+%``), and digit-plus-percent-word
spellings (``48 percent``).

**Documented scope (so the gate is not over-trusted):** plain integers (citation
years, ``Eq. 7``, counts already tokenized) are not flagged — they are mostly
already tokens and integers in prose are usually structural. Numbers spelled
entirely as *words* ("ninety-five percent") are out of regex scope; the standing
rule still forbids them, but this gate does not catch them (digit-plus-word
forms like "48 percent" ARE caught). The gate proves the *numeric*
spellings are clean, not that every conceivable hardcoded number is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

_MANUSCRIPT = Path(__file__).resolve().parent.parent / "manuscript"

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"\$[^$]*\$")
_TOKEN = re.compile(r"\{\{[A-Z0-9_]+\}\}")
_ATTR = re.compile(r"\{[#.][^}]*\}")  # {#label width=80%}, {.class}
# Data-literal spellings: scientific notation, Arabic + comma decimals, bare-dot
# decimals, and percentages. Order matters (scientific first so its mantissa is
# not split). Integers alone are not flagged (see the module docstring scope).
_DATA_LITERAL = re.compile(
    r"\d+(?:\.\d+)?[eE][-+]?\d+"   # scientific: 1e-3, 2.5E+04
    r"|\d+\.\d+"                    # Arabic decimal: 0.95
    r"|\d+,\d+"                     # comma decimal / thousands: 0,12 or 1,000
    r"|(?<![\w.])\.\d+"            # bare-dot decimal: .95
    r"|\d+\s?%"                     # percentage: 95% or 95 %
    r"|\d+[\s-]?[Pp]ercent\b"       # digits + percent word: 48 percent, 48-percent
)


def _strip(text: str) -> str:
    for pat in (_FENCED, _DISPLAY_MATH, _INLINE_MATH, _INLINE_CODE, _TOKEN, _ATTR):
        text = pat.sub(" ", text)
    return text


def test_no_hardcoded_decimal_or_percentage_in_prose() -> None:
    offenders: list[str] = []
    sections = sorted(_MANUSCRIPT.glob("[0-9]*.md")) + sorted(_MANUSCRIPT.glob("S[0-9]*.md"))
    for section in sections:
        stripped = _strip(section.read_text(encoding="utf-8"))
        for i, line in enumerate(stripped.splitlines(), 1):
            for m in _DATA_LITERAL.finditer(line):
                offenders.append(f"{section.name}:{i} '{m.group()}' in: {line.strip()[:90]}")
    assert not offenders, (
        "hardcoded data numerals in prose (should be {{TOKEN}}s):\n  "
        + "\n  ".join(offenders)
    )
