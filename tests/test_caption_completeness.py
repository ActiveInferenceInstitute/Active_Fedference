"""Figure-caption completeness gate (show, not tell, at the caption level).

Every figure embedded in the manuscript must carry a caption that states, at
minimum: (1) what the axes are, (2) the source relationship and estimand, and
(3) the uncertainty disposition — either an explicit error band / confidence
interval, or an explicit statement that the quantity is deterministic and
carries none. This is the caption-level form of
the project's "every quantitative claim is backed" rule: a reader must be able to
read a figure without guessing its axes or whether its values are noisy.

Each ``src/figures/*.py`` generator (bar ``_common``) must also be embedded at
least once, so no generator ships an orphan figure. No mocks: parses real prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MANUSCRIPT = _ROOT / "manuscript"
_FIGURES = _ROOT / "src" / "figures"

# A markdown image embed: ![caption](path){#fig:label ...}
_EMBED_RE = re.compile(
    r"!\[(?P<caption>.*?)\]\((?P<path>[^)]*?)\)\{#fig:(?P<label>[A-Za-z0-9_\-]+)",
    re.DOTALL,
)

_AXES_PATTERNS = (
    re.compile(r"x-axis", re.IGNORECASE),
)
_SECOND_AXIS = (
    re.compile(r"y-axis", re.IGNORECASE),
    re.compile(r"\brows?\b", re.IGNORECASE),  # heatmap: each row is one agent
)
_UNCERTAINTY = (
    re.compile(r"bootstrap CI", re.IGNORECASE),
    re.compile(r"percentile-bootstrap interval", re.IGNORECASE),
    re.compile(r"confidence interval", re.IGNORECASE),
    re.compile(r"error band", re.IGNORECASE),
    re.compile(r"error bar", re.IGNORECASE),
    re.compile(r"no resampling", re.IGNORECASE),
    re.compile(r"deterministic", re.IGNORECASE),
)


def _embeds() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    sections = sorted(_MANUSCRIPT.glob("[0-9]*.md")) + sorted(_MANUSCRIPT.glob("S[0-9]*.md"))
    for section in sections:
        for m in _EMBED_RE.finditer(section.read_text(encoding="utf-8")):
            out.append((m.group("label"), m.group("caption"), m.group("path")))
    return out


# A caption must carry real description, not a keyword-only placeholder. The
# shipped captions run 200-600 chars; a contentless "x-axis. y-axis. deterministic."
# example is roughly 40 characters.
_MIN_CAPTION_CHARS = 120

# Evidence-bearing figures must identify both the estimand/quantity and the
# unit of replication or resampling.  This prevents a caption gate from being
# satisfied by generic axis and "CI" keywords alone. Conceptual/formal and
# deterministic diagnostic schematics are intentionally outside this set.
_DATA_BEARING_LABELS = {
    "free-energy",
    "language-kl",
    "emergence-bmr",
    "robustness-sweep",
    "bnn-robustness",
    "contamination-gallery",
    "robustness-onset",
    "disjoint-fov-world",
    "hierarchical-pomdp",
    "cross-study-summary",
    "parameter-recovery",
}
_ESTIMAND = re.compile(
    r"accuracy|free[- ]energy|\bkl\b|delta\s*f|r-squared|absolute error|"
    r"influence|weight|surprise|probability|benefit",
    re.IGNORECASE,
)
_REPLICATION_UNIT = re.compile(
    r"\bn\s*=|trials?|seeds?|clients?|agents?|points?|bootstrap|resampl",
    re.IGNORECASE,
)
_SOURCE_RELATION = re.compile(r"source relation:", re.IGNORECASE)
_CAPTION_ESTIMAND = re.compile(r"estimand:", re.IGNORECASE)
_CAPTION_UNIT = re.compile(
    r"\bnats?\b|\bfraction(?:s)?\b|\bprobabilit(?:y|ies)\b|\bunitless\b|"
    r"\bconceptual\b|\bcategorical\b|\bsteps?\b|\bweight\b|\bstates?\b|"
    r"\bcomponents?\b|\bR-squared\b|\$R\^2\$",
    re.IGNORECASE,
)


def test_every_caption_states_axes_and_uncertainty() -> None:
    embeds = _embeds()
    assert embeds, "expected figure embeds in the manuscript"
    incomplete: list[str] = []
    for label, caption, _ in embeds:
        has_x = any(p.search(caption) for p in _AXES_PATTERNS)
        has_second = any(p.search(caption) for p in _SECOND_AXIS)
        has_unc = any(p.search(caption) for p in _UNCERTAINTY)
        has_source_relation = bool(_SOURCE_RELATION.search(caption))
        has_estimand = bool(_CAPTION_ESTIMAND.search(caption))
        has_unit = bool(_CAPTION_UNIT.search(caption))
        # axes keywords must co-occur with a verb ("x-axis is/indexes/...") rather
        # than appear as bare tokens, and the caption must be substantive — both
        # guard against a keyword-stuffed but contentless caption (audit finding).
        substantive = len(caption.strip()) >= _MIN_CAPTION_CHARS
        axis_described = re.search(r"x-axis\s*(is|are|indexes|shows|=|:)", caption, re.IGNORECASE)
        missing = []
        if not has_x:
            missing.append("x-axis")
        if not has_second:
            missing.append("y-axis/rows")
        if not has_unc:
            missing.append("uncertainty (CI/error-band/deterministic)")
        if not has_source_relation:
            missing.append("source relationship")
        if not has_estimand:
            missing.append("estimand")
        if not has_unit:
            missing.append("unit")
        if not substantive:
            missing.append(f"substance (caption < {_MIN_CAPTION_CHARS} chars)")
        if not axis_described:
            missing.append("axis description (bare 'x-axis' keyword, no 'is/indexes/...')")
        if missing:
            incomplete.append(f"fig:{label} missing: {', '.join(missing)}")
    assert not incomplete, "incomplete figure captions:\n  " + "\n  ".join(incomplete)


def test_data_bearing_captions_name_estimand_and_replication_unit() -> None:
    captions = {label: caption for label, caption, _ in _embeds()}
    missing = []
    for label in sorted(_DATA_BEARING_LABELS):
        caption = captions[label]
        if not _ESTIMAND.search(caption):
            missing.append(f"fig:{label}: estimand/quantity")
        if not _REPLICATION_UNIT.search(caption):
            missing.append(f"fig:{label}: sample or resampling unit")
    assert not missing, "under-specified data-bearing captions:\n  " + "\n  ".join(missing)


def test_every_generator_is_embedded() -> None:
    embedded_paths = " ".join(p for _, _, p in _embeds())
    orphans: list[str] = []
    for gen in sorted(_FIGURES.glob("*.py")):
        if gen.name in ("__init__.py", "_common.py", "_metadata.py"):
            continue
        # figures may be generated as PNG or PDF
        if (gen.stem + ".png") not in embedded_paths and (gen.stem + ".pdf") not in embedded_paths:
            orphans.append(gen.name)
    assert not orphans, f"figure generators with no manuscript embed: {orphans}"


def test_captions_do_not_embed_cross_reference_syntax() -> None:
    """Keep captions parseable in per-section HTML as well as combined output."""
    offenders = [
        label
        for label, caption, _ in _embeds()
        if "[@" in caption or re.search(r"\\(?:ref|eqref)\{", caption)
    ]
    assert not offenders, (
        "cross-reference syntax inside image alt text breaks per-section HTML: "
        f"{offenders}"
    )


def test_math_figure_alts_lead_with_plain_language() -> None:
    """Keep the renderer's first-sentence figure alternatives TeX-free."""
    captions = {label: caption for label, caption, _ in _embeds()}
    expected_first_sentences = {
        "language-kl": (
            "Seed-mean KL divergence from the true likelihood A "
            "to the learned likelihood A"
        ),
        "robustness-sweep": (
            "Consensus accuracy: probability mass assigned to the "
            "true hidden state"
        ),
    }
    expected_math = {
        "language-kl": r"$\mathrm{KL}(\text{true }A \,\|\, \text{learned }A)$",
        "robustness-sweep": r"$q(\text{true state})$",
    }
    for label, expected_first_sentence in expected_first_sentences.items():
        first_sentence, separator, remainder = captions[label].partition(". ")
        assert separator, f"fig:{label} needs a first caption sentence"
        assert first_sentence == expected_first_sentence
        assert "$" not in first_sentence and "\\" not in first_sentence
        assert expected_math[label] in remainder


def test_new_schematic_captions_are_claim_bounded() -> None:
    """Conceptual figures must declare their status and non-empirical scope."""
    captions = {label: caption.lower() for label, caption, _ in _embeds()}
    required = {"generative-model-schema", "message-passing", "pomdp-loop", "graphical-abstract"}
    assert required <= captions.keys()
    for label in required:
        caption = captions[label]
        assert any(word in caption for word in ("schematic", "formal", "mechanistic")), label
        assert "x-axis" in caption, label
        assert any(word in caption for word in ("y-axis", "rows")), label
        assert any(word in caption for word in ("deterministic", "no ci", "no error band")), label
    assert "non-transferable" in captions["graphical-abstract"]
    assert "recovery-limit" in captions["message-passing"]
    assert "moving-world extension" in captions["pomdp-loop"]


def test_robustness_caption_matches_uncertainty_aware_artifact() -> None:
    """The caption must not certify a deterministic plot when CI data are drawn."""
    report_path = _ROOT / "output" / "reports" / "robustness_sweep.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("per_rate_summary"):
        return
    captions = {label: caption for label, caption, _ in _embeds()}
    caption = captions["robustness-sweep"]
    # Keep the set-valued rate annotation valid in LaTeX/Pandoc: both braces
    # must be escaped around the injected token or standalone Beamer rendering
    # sees an unmatched math delimiter.
    assert r"$\{{{SWEEP_RATES}}\}$" in caption
    assert re.search(
        r"bootstrap(?:[^.]{0,40})CI|confidence interval|error bar",
        caption,
        re.IGNORECASE,
    )
    assert not re.search(
        r"no error (?:band|bar)|single deterministic.*(?:no|without)",
        caption,
        re.IGNORECASE,
    )

    # The rendered caption must use the same matched-trial profile means as the
    # plotted curves.  The deterministic single-colony table has a different
    # estimand and must not silently leak into this uncertainty-aware caption.
    source = (_MANUSCRIPT / "19_results_robustness.md").read_text(encoding="utf-8")
    assert "{{SWEEP_PROFILE_NAIVE_ACCURACY}}" in source
    assert "{{SWEEP_PROFILE_BEST_ROBUST_ACCURACY}}" in source
    resolved_path = _ROOT / "output" / "manuscript" / "19_results_robustness.md"
    if not resolved_path.exists():
        return
    resolved = resolved_path.read_text(encoding="utf-8")
    resolved_caption = next(
        match.group("caption")
        for match in _EMBED_RE.finditer(resolved)
        if match.group("label") == "robustness-sweep"
    )
    worst_key = f"{float(report['worst_rate']):g}"
    methods = report["per_rate_summary"][worst_key]["methods"]
    expected_naive = f"{float(methods['KLD']['mean']):.4f}"
    expected_best_cell = max(
        (cell for method, cell in methods.items() if method != 'KLD'),
        key=lambda cell: cell['mean'],
    )
    expected_best = f"{float(expected_best_cell['mean']):.4f}"
    deterministic_naive = f"{float(report['accuracy_by_method_and_rate']['KLD'][worst_key]):.4f}"
    assert expected_naive in resolved_caption
    assert expected_best in resolved_caption
    if deterministic_naive != expected_naive:
        assert deterministic_naive not in resolved_caption
