"""Tests for the language-acquisition KL-decay figure generator.

No mocks: a real declining KL curve is rendered to ``tmp_path`` (headless Agg);
we assert the PNG exists, is non-empty PNG bytes, and the error path raises.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from figures import FIGURE_METADATA, generate_language_kl_decay

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_ROOT = Path(__file__).resolve().parents[2]
_PROHIBITED_AFFIRMATIVE_TERM = re.compile(
    r"\b(?:confirm(?:ed|s|ing|ation)?|verdicts?|benefits?|advantages?|"
    r"superior(?:ity)?|success(?:ful(?:ly)?)?)\b",
    re.IGNORECASE,
)
_NEGATION_BEFORE_TERM = re.compile(
    r"\b(?:does?|did|is|are|was|were|can|cannot|could|would)\s+not\b|"
    r"\b(?:no|without)\b",
    re.IGNORECASE,
)
_NEGATION_AFTER_TERM = re.compile(
    r"\b(?:is|are|was|were)\s+not\b|\b(?:is|are)\s+not\s+established\b",
    re.IGNORECASE,
)


def _affirmative_claim_terms(text: str) -> list[str]:
    """Find prohibited affirmative framing while retaining no-claim boundaries."""
    issues: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        for match in _PROHIBITED_AFFIRMATIVE_TERM.finditer(sentence):
            before = sentence[max(0, match.start() - 120) : match.start()]
            after = sentence[match.end() : match.end() + 80]
            if _NEGATION_BEFORE_TERM.search(before) or _NEGATION_AFTER_TERM.search(after):
                continue
            issues.append(match.group(0))
    return issues


def _language_caption() -> str:
    manuscript = (_ROOT / "manuscript" / "17_results_language.md").read_text(encoding="utf-8")
    match = re.search(
        r"!\[(?P<caption>.*?)\]\([^)]*language_kl_decay\.png\)"
        r"\{#fig:language-kl\b",
        manuscript,
        re.DOTALL,
    )
    assert match is not None
    return match.group("caption")


def test_language_kl_decay_happy_path(tmp_path: Path) -> None:
    kl = [1.2, 0.9, 0.7, 0.5, 0.4, 0.3]
    path = generate_language_kl_decay(
        kl,
        trajectory_ci=([1.0, 0.7, 0.5, 0.35, 0.3, 0.2], [1.4, 1.1, 0.9, 0.7, 0.6, 0.5]),
        monotone_decreasing=True,
        n_seeds=4,
        project_root=tmp_path,
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_language_kl_decay_without_ci(tmp_path: Path) -> None:
    path = generate_language_kl_decay([2.0, 1.0, 0.5], project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_language_kl_decay_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([], project_root=tmp_path)


def test_language_kl_decay_rejects_bad_ci(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([1.0, 0.5], trajectory_ci=([0.5], [0.8]), project_root=tmp_path)


def test_language_kl_decay_rejects_nonfinite_and_negative_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([1.0, float("nan")], project_root=tmp_path)
    with pytest.raises(ValueError):
        generate_language_kl_decay([-0.1, 0.0], project_root=tmp_path)


def test_language_kl_decay_rejects_bad_pointwise_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([1.0, 0.5], trajectory_ci=([0.9, 0.6], [1.1, 0.55]), project_root=tmp_path)


def test_language_figure_surfaces_use_neutral_estimand_first_wording() -> None:
    """Block affirmative visual framing while retaining explicit no-claim prose."""
    generator_source = (_ROOT / "src" / "figures" / "language_kl_decay.py").read_text(encoding="utf-8")
    metadata = FIGURE_METADATA["language_kl_decay"]
    metadata_surface = " ".join(
        metadata.get(field, "")
        for field in (
            "alt_text",
            "long_description",
            "status",
            "estimand",
            "uncertainty",
            "replication_unit",
        )
    )
    surfaces = {
        "visible generator wording": generator_source,
        "figure metadata": metadata_surface,
        "manuscript caption": _language_caption(),
    }

    offenders = {
        surface: _affirmative_claim_terms(text)
        for surface, text in surfaces.items()
        if _affirmative_claim_terms(text)
    }
    assert not offenders

    assert _affirmative_claim_terms("Communication benefit confirmed.") == [
        "benefit",
        "confirmed",
    ]
    assert not _affirmative_claim_terms("This result does not establish a general communication benefit.")
