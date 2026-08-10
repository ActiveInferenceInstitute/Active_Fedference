"""Docstring-coverage gate for the delivered Python API surface.

A research package whose contract is "the limit is the proof" must keep its
public surface self-documenting. This gate walks every module in
the source packages and script modules and asserts that every *public*
module-level function/class, and every public method/property of a public class,
carries a non-empty docstring. Nested (closure) functions and private
(``_``-prefixed) names are excluded — they are not the public API. No mocks: it
parses the real source.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# The delivered public surface: all source packages plus thin script
# orchestrators. Keeping this explicit prevents a new source package from
# quietly escaping the documentation gate.
_GATED_DIRS = (
    _ROOT / "src",
    _ROOT / "src" / "analysis",
    _ROOT / "src" / "fedference",
    _ROOT / "src" / "figures",
    _ROOT / "src" / "manuscript_vars",
    _ROOT / "src" / "publication",
    _ROOT / "scripts",
)


def _public(name: str) -> bool:
    return not name.startswith("_")


#: A docstring must carry real content, not a placeholder. Require a minimum of
#: non-whitespace characters AND at least two multi-letter words, so "." , "x",
#: and a single-token filler like "aaaaaaaaaaaa" (which clear a bare-existence or
#: one-word check) are rejected — closing the audit's low-substance bypasses.
_MIN_DOC_CHARS = 12
_WORD = re.compile(r"[A-Za-z]{3,}")


def _inadequate(node) -> bool:
    """True if *node*'s docstring is missing or too thin to document anything."""
    doc = ast.get_docstring(node)
    if not doc:
        return True
    cleaned = doc.strip()
    return len(cleaned) < _MIN_DOC_CHARS or len(_WORD.findall(cleaned)) < 2


def _missing_docstrings(path: Path) -> list[str]:
    """Public defs/classes/methods in *path* with a missing/thin docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for node in tree.body:  # module level only — nested closures are not public API
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(node.name) and _inadequate(node):
                missing.append(f"{path.name}:{node.lineno} {node.name}")
        elif isinstance(node, ast.ClassDef):
            if not _public(node.name):
                continue
            if _inadequate(node):
                missing.append(f"{path.name}:{node.lineno} class {node.name}")
            for item in node.body:  # public methods / properties of a public class
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _public(item.name) and _inadequate(item):
                        missing.append(
                            f"{path.name}:{item.lineno} {node.name}.{item.name}"
                        )
    return missing


def test_public_api_is_fully_documented() -> None:
    missing: list[str] = []
    for gated in _GATED_DIRS:
        assert gated.is_dir(), f"{gated} must exist"
        for module in sorted(gated.glob("*.py")):
            if module.name == "__init__.py":
                continue
            missing.extend(_missing_docstrings(module))
    assert not missing, "public symbols missing/thin docstrings:\n  " + "\n  ".join(missing)


def test_gate_sees_a_nontrivial_public_surface() -> None:
    # guard against a vacuous pass if the glob or parse silently finds nothing.
    total = 0
    for gated in _GATED_DIRS:
        for module in gated.glob("*.py"):
            if module.name == "__init__.py":
                continue
            tree = ast.parse(module.read_text(encoding="utf-8"))
            total += sum(
                1
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and _public(n.name)
            )
    assert total >= 35, f"expected a substantial public API, counted {total}"
