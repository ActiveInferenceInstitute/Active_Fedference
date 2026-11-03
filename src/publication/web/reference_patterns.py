"""Reference patterns for source-preserving HTML publication."""
from __future__ import annotations

import re

_XREF_PREFIXES = "sec|eq|fig|tbl|prop|thm|lem|cor|def"


_LEFT_BRACKET_PATTERN = (
    r"(?:\[|&(?:lbrack|lsqb);|&#0*91(?:;|(?![0-9]))|&#[xX]0*5[bB](?:;|(?![0-9A-Fa-f])))"
)


_RIGHT_BRACKET_PATTERN = (
    r"(?:\]|&(?:rbrack|rsqb);|&#0*93(?:;|(?![0-9]))|&#[xX]0*5[dD](?:;|(?![0-9A-Fa-f])))"
)


_RAW_XREF_RE = re.compile(
    rf"{_LEFT_BRACKET_PATTERN}@?(?P<key>(?:{_XREF_PREFIXES}):.+?){_RIGHT_BRACKET_PATTERN}"
)


_LATEX_REF_TEXT_RE = re.compile(
    rf"\\\(\\ref\{{(?P<key>(?:{_XREF_PREFIXES}):[^}}]+)\}}\\\)"
)


_LATEX_KIND_SUFFIX_RE = re.compile(
    r"(?:Theorems?|Lemmas?|Propositions?|Corollaries|Definitions?|Sections?|Equations?)~?\s*$"
)


_DUPLICATE_XREF_KIND_SUFFIX_RE = re.compile(
    r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Section|Equation)~\s*$"
)


_RAW_CITATION_RE = re.compile(
    rf"{_LEFT_BRACKET_PATTERN}"
    r"(?P<keys>[A-Za-z][A-Za-z0-9_-]*(?:\s*;\s*[A-Za-z][A-Za-z0-9_-]*)*)"
    rf"{_RIGHT_BRACKET_PATTERN}"
)
