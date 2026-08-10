"""Cross-reference integrity gate for the manuscript (pandoc-crossref labels).

Every ``[@eq:/@fig:/@tbl:/@sec:]`` reference in the rendered section set must
resolve to a defined ``{#eq:/#fig:/#tbl:/#sec:}`` label — no dangling pointer
survives into the PDF. We scan the live numbered-section glob (so new sections
are covered automatically) and compute the set difference. Definition labels may
carry trailing attributes (``{#fig:x width=80%}``), so the definition pattern
stops at whitespace or ``}``. Reverse direction (defined-but-unreferenced) is
reported as an informational set, not gated — many H2/H3 anchors exist for
navigation, not cross-reference.
"""

from __future__ import annotations

import re
from pathlib import Path

_MANUSCRIPT = Path(__file__).resolve().parent.parent / "manuscript"
_PREFIXES = "eq|fig|tbl|sec|prop|thm|lem|cor|def"
_DEF_RE = re.compile(rf"\{{#((?:{_PREFIXES}):[A-Za-z0-9_\-]+)")
# raw LaTeX \label{...} used for theorem/proposition/equation environments
# that pandoc-crossref cannot see inside raw-LaTeX blocks (e.g. \begin{definition}).
_LABEL_RE = re.compile(rf"\\label\{{((?:{_PREFIXES}):[A-Za-z0-9_\-]+)\}}")
_REF_RE = re.compile(rf"@((?:{_PREFIXES}):[A-Za-z0-9_\-]+)")
_LATEX_REF_RE = re.compile(
    rf"\\(?:ref|eqref|autoref|cref)\{{((?:{_PREFIXES}):[A-Za-z0-9_\-]+)\}}"
)
_THEOREM_RE = re.compile(
    r"\\begin\{(?P<kind>theorem|lemma|proposition|corollary|definition)\}"
    r"(?P<head>[^\n]*)\n(?P<body>.*?)\\end\{(?P=kind)\}",
    re.DOTALL,
)
_DISPLAY_RE = re.compile(r"\$\$(?P<body>.*?)\$\$(?P<label>[^\n]*)", re.DOTALL)


def _section_paths() -> list[Path]:
    return sorted(_MANUSCRIPT.glob("[0-9]*.md")) + sorted(_MANUSCRIPT.glob("S[0-9]*.md"))


def _manuscript_text() -> str:
    return "\n".join(section.read_text(encoding="utf-8") for section in _section_paths())


def _scan() -> tuple[set[str], set[str]]:
    defs: set[str] = set()
    refs: set[str] = set()
    for section in _section_paths():
        text = section.read_text(encoding="utf-8")
        defs.update(_DEF_RE.findall(text))
        defs.update(_LABEL_RE.findall(text))
        refs.update(_REF_RE.findall(text))
        refs.update(_LATEX_REF_RE.findall(text))
    return defs, refs


def test_every_reference_resolves_to_a_definition() -> None:
    defs, refs = _scan()
    assert defs, "expected defined cross-reference labels in the manuscript"
    assert refs, "expected cross-references in the manuscript"
    dangling = sorted(refs - defs)
    assert not dangling, f"references with no matching label (dangling): {dangling}"


def test_new_aggregation_labels_are_wired() -> None:
    # the iteration-4 variational-aggregation labels must be both defined and used.
    defs, refs = _scan()
    for label in (
        "eq:agg-free-energy",
        "eq:agg-updates",
        "sec:method-variational",
        "sec:results-variational",
        "sec:supp-variational",
        "sec:supp-extended",
        "fig:aggregation-descent",
        "fig:bounded-influence",
        # V1 tempered-aggregation labels
        "sec:supp-tempered",
        "eq:tempered-family",
        "eq:tempered-updates",
        # V3 federation-transport labels
        "sec:supp-federation",
        # V4 moving-world labels
        "sec:results-moving",
        "fig:moving-world",
        "sec:supp-moving",
    ):
        assert label in defs, f"{label} is not defined"
        assert label in refs, f"{label} is never referenced"


def test_notation_objective_is_canonical_and_referenced() -> None:
    """The notation contract must not introduce an unregistered objective label."""
    defs, refs = _scan()
    label = "eq:notation-variational-objective"
    assert label in defs, f"{label} is not defined"
    assert label in refs, f"{label} is never referenced"
    registry = (_MANUSCRIPT / "SYNTAX.md").read_text(encoding="utf-8")
    assert "{#eq:notation-variational-objective}" in registry
    assert "`30_supplement_notation.md`" in registry


def test_rcce_loss_parameter_is_distinct_from_posterior_q() -> None:
    """Reserve plain q for posterior notation and q_loss for the RCCE control."""
    text = _manuscript_text()
    assert r"L_{q_{\rm loss}}" in text
    assert "L_q" not in text


def test_raw_latex_proposition_uses_print_reference_not_pandoc_citation() -> None:
    defs, _ = _scan()
    text = _manuscript_text()
    assert "prop:federation-bit-identity" in defs
    assert r"Proposition \ref{prop:federation-bit-identity}" in text
    assert "@prop:" not in text


def test_live_cross_reference_labels_are_unique() -> None:
    occurrences: dict[str, list[str]] = {}
    for section in _section_paths():
        text = section.read_text(encoding="utf-8")
        for match in (*_DEF_RE.finditer(text), *_LABEL_RE.finditer(text)):
            line = text[: match.start()].count("\n") + 1
            occurrences.setdefault(match.group(1), []).append(f"{section.name}:{line}")
    duplicates = {label: sites for label, sites in occurrences.items() if len(sites) > 1}
    assert not duplicates, f"duplicate live manuscript labels: {duplicates}"


def test_every_equation_figure_and_table_is_referenced() -> None:
    defs, refs = _scan()
    public_defs = {label for label in defs if label.startswith(("eq:", "fig:", "tbl:"))}
    unreferenced = sorted(public_defs - refs)
    assert not unreferenced, f"defined but never referenced: {unreferenced}"


def test_every_display_equation_has_exactly_one_label() -> None:
    unlabeled: list[str] = []
    for section in _section_paths():
        text = section.read_text(encoding="utf-8")
        for match in _DISPLAY_RE.finditer(text):
            tail = match.group("label")
            following = text[match.end() : match.end() + 80]
            labels = re.findall(r"\{#eq:[A-Za-z0-9_\-]+\}", tail)
            labels += re.findall(r"\\label\{eq:[A-Za-z0-9_\-]+\}", following)
            if len(labels) != 1:
                line = text[: match.start()].count("\n") + 1
                unlabeled.append(f"{section.name}:{line} has {len(labels)} labels")
    assert not unlabeled, "display-equation label failures:\n" + "\n".join(unlabeled)


def test_theorem_like_statements_have_stable_labels_and_no_raw_markdown_citations() -> None:
    failures: list[str] = []
    for section in _section_paths():
        text = section.read_text(encoding="utf-8")
        for match in _THEOREM_RE.finditer(text):
            line = text[: match.start()].count("\n") + 1
            block = match.group(0)
            if not _LABEL_RE.search(block):
                failures.append(f"{section.name}:{line} missing typed label")
            if "[@" in block:
                failures.append(f"{section.name}:{line} contains unrendered Markdown citation")
    assert not failures, "theorem-reference failures:\n" + "\n".join(failures)


def test_formalism_references_do_not_hardcode_counter_numbers() -> None:
    text = _manuscript_text()
    hardcoded = re.findall(
        r"\b(?:Theorem|Lemma|Proposition|Corollary|Definition)s?\s+[0-9]+\b",
        text,
    )
    assert not hardcoded, f"brittle formalism numbers: {sorted(set(hardcoded))}"


def test_formalism_references_avoid_renderer_unsafe_tilde_spacing() -> None:
    """The PDF Unicode remapper must not turn ``Name~\\ref`` into ``Nameabout``."""
    text = _manuscript_text()
    unsafe = re.findall(
        r"(?:Definition|Lemma|Proposition|Theorem|Corollary)~\\ref\{[^}]+\}",
        text,
    )
    assert not unsafe, f"renderer-unsafe formalism references: {sorted(set(unsafe))}"


def test_every_citation_key_exists_and_every_bibliography_entry_is_used() -> None:
    # Remove code examples so the explanatory literal `[@key]` is not a citation.
    text = re.sub(r"```.*?```", "", _manuscript_text(), flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    cited = {
        key
        for key in re.findall(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_:-]+)", text)
        if not key.startswith(("eq:", "fig:", "tbl:", "sec:", "prop:"))
    }
    bibliography = (_MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
    defined = set(
        re.findall(
            r"^@(?!comment\b)[A-Za-z]+\{([^,\s]+),",
            bibliography,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    )
    assert cited - defined == set(), f"citation keys absent from bibliography: {sorted(cited - defined)}"
    assert defined - cited == set(), f"bibliography entries never cited: {sorted(defined - cited)}"
