#!/usr/bin/env python3
"""Validate and optionally render Mermaid blocks in the public Markdown surfaces.

The repository keeps Mermaid source in ``README.md`` and ``docs/**/*.md`` so
GitHub and the local documentation pipeline can consume the same diagrams. The
static pass is dependency-free; ``--render`` adds an actual Mermaid CLI probe
for every block and writes only to an explicit review directory.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_DIRECTION = r"(?:TB|BT|RL|LR|TD|DU)"
_DIAGRAM_HEADER = re.compile(
    rf"^(?:flowchart|graph)\s+{_DIRECTION}\s*$"
    r"|^(?:sequenceDiagram|stateDiagram(?:-v2)?|classDiagram|erDiagram|"
    r"journey|gantt|pie|quadrantChart|xychart-beta|mindmap|timeline|gitGraph)\b"
)
_UNQUOTED_PUNCTUATED_NODE = re.compile(
    r"^\s*[A-Za-z][A-Za-z0-9_-]*\[(?![\"'`])[^\]]*[()][^\]]*\]"
)


@dataclass(frozen=True)
class MermaidBlock:
    """One fenced Mermaid source block and its Markdown location."""

    path: Path
    start_line: int
    end_line: int
    source: str


def _markdown_paths(root: Path) -> tuple[Path, ...]:
    """Return the public Markdown surfaces that are allowed to contain diagrams."""
    readme = root / "README.md"
    docs = tuple(sorted((root / "docs").rglob("*.md")))
    return tuple(path for path in (readme, *docs) if path.exists())


def extract_mermaid_blocks(root: Path) -> tuple[MermaidBlock, ...]:
    """Extract fenced Mermaid blocks while preserving source line locations."""
    blocks: list[MermaidBlock] = []
    for path in _markdown_paths(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        in_block = False
        start_line = 0
        source_lines: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            marker = line.strip().lower()
            if not in_block and marker == "```mermaid":
                in_block = True
                start_line = line_number
                source_lines = []
                continue
            if in_block and marker == "```":
                blocks.append(
                    MermaidBlock(
                        path=path,
                        start_line=start_line,
                        end_line=line_number,
                        source="\n".join(source_lines).strip() + "\n",
                    )
                )
                in_block = False
                continue
            if in_block:
                source_lines.append(line)
        if in_block:
            blocks.append(
                MermaidBlock(
                    path=path,
                    start_line=start_line,
                    end_line=len(lines),
                    source="\n".join(source_lines).strip() + "\n",
                )
            )
    return tuple(blocks)


def validate_mermaid_blocks(root: Path) -> tuple[MermaidBlock, ...]:
    """Validate fence balance, diagram declarations, and safe node labels."""
    errors: list[str] = []
    blocks = extract_mermaid_blocks(root)
    for block in blocks:
        location = f"{block.path.relative_to(root)}:{block.start_line}"
        if block.end_line == len(block.path.read_text(encoding="utf-8").splitlines()):
            lines = block.path.read_text(encoding="utf-8").splitlines()
            if not any(
                line_number > block.start_line and line.strip() == "```"
                for line_number, line in enumerate(lines, start=1)
            ):
                errors.append(f"{location}: unclosed Mermaid fence")
        header = next((line.strip() for line in block.source.splitlines() if line.strip()), "")
        if not _DIAGRAM_HEADER.match(header):
            errors.append(f"{location}: unsupported or missing diagram declaration: {header!r}")
        if not block.source.strip():
            errors.append(f"{location}: empty Mermaid block")
        for offset, line in enumerate(block.source.splitlines(), start=1):
            if _UNQUOTED_PUNCTUATED_NODE.search(line):
                errors.append(
                    f"{location}+{offset}: quote node labels containing parentheses or similar punctuation"
                )
    if errors:
        raise ValueError("\n".join(errors))
    return blocks


def render_mermaid_blocks(
    root: Path,
    blocks: tuple[MermaidBlock, ...],
    output_dir: Path,
    renderer: str,
) -> None:
    """Render every validated block to SVG with Mermaid CLI."""
    if renderer == "npx":
        command_prefix = ["npx", "--yes", "@mermaid-js/mermaid-cli"]
    else:
        command_prefix = [renderer]
    if shutil.which(command_prefix[0]) is None:
        raise FileNotFoundError(
            f"Mermaid renderer {command_prefix[0]!r} is not on PATH; "
            "install Mermaid CLI or pass --renderer npx"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, block in enumerate(blocks, start=1):
        stem = block.path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        source_path = output_dir / f"{stem}__{index}_{block.start_line}.mmd"
        svg_path = source_path.with_suffix(".svg")
        source_path.write_text(block.source, encoding="utf-8")
        completed = subprocess.run(
            [*command_prefix, "-i", str(source_path), "-o", str(svg_path), "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Mermaid CLI error"
            raise RuntimeError(f"{block.path}:{block.start_line}: Mermaid render failed: {detail}")
        if not svg_path.exists() or svg_path.stat().st_size == 0:
            raise RuntimeError(f"{block.path}:{block.start_line}: Mermaid CLI produced no SVG")


def main(argv: list[str] | None = None) -> int:
    """Run the static Mermaid check and optional renderer probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--render", action="store_true", help="render every block to SVG")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".tmp/mermaid-render"),
        help="scratch directory used by --render",
    )
    parser.add_argument(
        "--renderer",
        default="mmdc",
        help="Mermaid CLI executable, or npx for @mermaid-js/mermaid-cli",
    )
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        blocks = validate_mermaid_blocks(root)
        if args.render:
            output_dir = args.output_dir
            if not output_dir.is_absolute():
                output_dir = root / output_dir
            render_mermaid_blocks(root, blocks, output_dir, args.renderer)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"mermaid: FAIL\n{error}", file=sys.stderr)
        return 1
    mode = "rendered" if args.render else "validated"
    print(f"mermaid_blocks: {len(blocks)}")
    print(f"mermaid: PASS ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
