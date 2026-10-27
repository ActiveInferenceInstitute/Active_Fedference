"""Closed-source upstream Markdown parsing for explicit GNN extension families.

The real parser preserves named extensions; project admission interprets them.
No synthetic parser packages, source execution outside the pinned closure, or
fallback to a manual extractor are permitted (Friston et al., 2024).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .admission import GNNAdmissionError
from .loader import MODEL_DIR, UPSTREAM_REVISION, _json, _mapping, _sha

EXTENSION_MANIFEST: Final = MODEL_DIR / "extensions.parser.json"
_STANDARD: Final = frozenset(
    {
        "GNNSection",
        "GNNVersionAndFlags",
        "ModelName",
        "ModelAnnotation",
        "StateSpaceBlock",
        "Connections",
        "InitialParameterization",
        "Equations",
        "Time",
        "ActInfOntologyAnnotation",
        "ModelParameters",
        "Footer",
        "Signature",
    }
)
_RUNNER: Final = """
import sys, json, hashlib
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[3]]
from gnn.parsers.markdown_parser import MarkdownGNNParser
result = MarkdownGNNParser().parse_file(sys.argv[2])
files = sorted({str(Path(m.__file__).resolve().relative_to(Path(p).resolve()))
    for m in sys.modules.values() for p in (sys.argv[1], sys.argv[3])
    if getattr(m, "__file__", None)
    and Path(m.__file__).resolve().is_relative_to(Path(p).resolve())})
print(json.dumps(dict(success=result.success, errors=result.errors,
    warnings=result.warnings, sections=result.model.raw_sections,
    extensions=result.model.extensions, modules=files,
    source_digest=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()), allow_nan=False))
"""


@dataclass(frozen=True, slots=True)
class ParsedExtensions:
    """Immutable preserved payload and parser/source binding, not admission."""

    document: str
    sections: tuple[tuple[str, str], ...]
    extensions: tuple[tuple[str, str], ...]
    parser_digest: str
    source_digest: str
    modules: tuple[str, ...]

    def extension(self, name: str) -> str:
        """Return one explicitly preserved namespace."""
        for key, value in self.extensions:
            if key == name:
                return value
        raise GNNAdmissionError("GNN_EXTENSION", name)


def _sections(document: str, allowed_extensions: tuple[str, ...]) -> dict[str, str]:
    """Decode declared section boundaries for fresh parsing and offline integrity."""
    normalized = document.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?m)^## ([^\n]+)\n", normalized)
    sections: dict[str, str] = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        if name in sections or name not in _STANDARD.union(allowed_extensions):
            raise GNNAdmissionError("GNN_SECTIONS", name)
        sections[name] = body.strip()
    if not allowed_extensions or not set(allowed_extensions).issubset(sections):
        raise GNNAdmissionError("GNN_EXTENSION", "required namespaces")
    return sections


def parse_extensions(
    path: Path,
    *,
    upstream_root: Path,
    allowed_extensions: tuple[str, ...],
    python: Path | None = None,
    dependency_root: Path | None = None,
) -> ParsedExtensions:
    """Execute the pinned normal parser import from a verified-only snapshot.

    The caller explicitly authorizes this local execution. The optional Python
    must support the upstream parser (>=3.11); default core imports remain 3.10.
    """
    if dependency_root is None:
        raise GNNAdmissionError("GNN_PARSER_DEPENDENCY", "explicit dependency_root required")
    source_bytes = path.read_bytes()
    document = source_bytes.decode("utf-8")
    sections = _sections(document, allowed_extensions)
    manifest = _mapping(_json(EXTENSION_MANIFEST.read_text()), "parser manifest")
    if (
        set(manifest) != {"revision", "files", "dependencies", "dependency_versions", "capability"}
        or manifest["revision"] != UPSTREAM_REVISION
    ):
        raise GNNAdmissionError("GNN_PARSER", "manifest")
    files = _mapping(manifest["files"], "manifest.files")
    dependencies = _mapping(manifest["dependencies"], "manifest.dependencies")
    if (
        manifest["dependency_versions"] != {"defusedxml": "0.7.1", "PyYAML": "6.0.3"}
        or manifest["capability"] != "markdown-only-pure-python/1"
    ):
        raise GNNAdmissionError("GNN_PARSER", "dependency contract")
    if not files or not dependencies:
        raise GNNAdmissionError("GNN_PARSER", "empty closure")
    parser_digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    with tempfile.TemporaryDirectory(prefix="fedference-extensions-") as temporary:
        root = Path(temporary)
        for source, table, prefix in (
            (upstream_root, files, "source"),
            (dependency_root, dependencies, "dependency"),
        ):
            for relative, expected in table.items():
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise GNNAdmissionError("GNN_PARSER", relative)
                if not isinstance(expected, str) or re.fullmatch("[0-9a-f]{64}", expected) is None:
                    raise GNNAdmissionError("GNN_PARSER", relative)
                try:
                    payload = (source / relative).read_bytes()
                except OSError as error:
                    raise GNNAdmissionError("GNN_PARSER_DRIFT", relative) from error
                if _sha(payload) != expected:
                    raise GNNAdmissionError("GNN_PARSER_DRIFT", relative)
                target = root / "parser" / prefix / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        snapshot = root / "model.gnn.md"
        snapshot.write_bytes(source_bytes)
        command = [
            str(python or sys.executable),
            "-I",
            "-S",
            "-B",
            "-c",
            _RUNNER,
            str(root / "parser" / "source" / "src"),
            str(snapshot),
            str(root / "parser" / "dependency"),
        ]
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GNNAdmissionError("GNN_PROCESS", str(path)) from error
        if result.returncode:
            raise GNNAdmissionError("GNN_EXTRACTION", result.stdout + result.stderr)
    parsed = _mapping(_json(result.stdout), "parser result")
    if parsed.get("success") is not True or parsed.get("errors") or parsed.get("warnings"):
        raise GNNAdmissionError("GNN_EXTRACTION", result.stdout)
    if parsed.get("sections") != sections:
        raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", "raw_sections")
    if parsed.get("source_digest") != _sha(source_bytes):
        raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", "snapshot bytes")
    extensions = _mapping(parsed.get("extensions"), "extensions")
    for name in allowed_extensions:
        if extensions.get(name) != sections[name]:
            raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", name)
    modules = parsed.get("modules")
    if not isinstance(modules, list) or any(not isinstance(name, str) for name in modules):
        raise GNNAdmissionError("GNN_PARSER", "modules")
    module_names = tuple(name for name in modules if isinstance(name, str))
    if any("src/" + name not in files and name not in dependencies for name in module_names):
        raise GNNAdmissionError("GNN_PARSER", "unmanifested execution")
    for source, table in ((upstream_root, files), (dependency_root, dependencies)):
        for relative, expected in table.items():
            if _sha((source / relative).read_bytes()) != expected:
                raise GNNAdmissionError("GNN_PARSER_DRIFT", relative)
    return ParsedExtensions(
        document,
        tuple(sections.items()),
        tuple((name, sections[name]) for name in allowed_extensions),
        parser_digest,
        hashlib.sha256(source_bytes).hexdigest(),
        module_names,
    )
