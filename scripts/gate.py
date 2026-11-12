#!/usr/bin/env python3
"""One documented gate entry point with separately runnable arms.

The ``ci`` workflow calls these arms instead of keeping a second copy of each
command. Every arm is independent; ``all`` composes the measurable arms in
order. The commands are the same commands the workflow used to inline, so a
local arm and its CI counterpart cannot drift.

Arms that need the pinned GNN runtime consume the ``environment.json`` written
by ``bootstrap_gnn.py``. Run ``bootstrap`` first, then ``gnn`` (or ``all``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

Command = tuple[str, ...]
Env = dict[str, str]
Invocation = tuple[Command, Env]

_RUFF = (
    "uv", "run", "--locked", "--extra", "dev", "ruff", "check",
    "src/", "tests/", "scripts/", "examples/", "_fedference_build_backend.py",
)
_MYPY = ("uv", "run", "--locked", "--extra", "dev", "mypy", "src/")
_METADATA = ("uv", "run", "--locked", "--extra", "dev", "python", "scripts/emit_metadata.py", "--check")
_MERMAID = ("uv", "run", "--locked", "--extra", "dev", "python", "scripts/validate_mermaid.py")
_DOCS_CONTRACT = ("uv", "run", "--locked", "--extra", "dev", "pytest", "tests/test_docs_contract.py", "-q")
_FAST = (
    "uv", "run", "--locked", "--extra", "dev", "python", "-c",
    "import pytest; raise SystemExit(pytest.main(["
    "\"tests/\", \"--validation-profile=native\", \"-m\", "
    "\"not slow and not publication\", \"-q\", \"--durations=0\", "
    "\"--junitxml=.tmp/ci-timings/compatibility.xml\"]))",
)
_PLAN = (
    "uv", "run", "--locked", "--extra", "dev", "python", "scripts/ci_shards.py",
    "plan", "--plan", ".tmp/ci-plan/plan.json", "--count", "4",
)
_PIN_CHECK = (
    "uv", "run", "--locked", "--extra", "dev", "python", "-c",
    "import json, re, sys; from pathlib import Path; "
    "pin = json.loads(Path('.github/gnn-pin.json').read_text()); "
    "revision = pin.get('commit'); "
    "assert pin.get('schema') == 'gnn-importer-pin/v1'; "
    "assert isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision); "
    "print('gnn-pin=' + revision)",
)
_VERSION_CHECK = (
    "uv", "run", "--locked", "--extra", "dev", "python", "-c",
    "import re, sys; from pathlib import Path; "
    "source = Path(sys.argv[1], 'src/gnn/__init__.py').read_text(encoding='utf-8'); "
    "match = re.search(r'^__version__\\s*=\\s*\"([^\"]+)\"', source, re.MULTILINE); "
    "assert match is not None, 'pinned upstream source exposes no __version__'; "
    "print('gnn-version=' + match.group(1))",
)


def _shard(index: int) -> Command:
    return (
        "uv", "run", "--locked", "--extra", "dev", "python", "scripts/ci_shards.py",
        "run", "--plan", ".tmp/ci-plan/plan.json", "--index", str(index),
    )


_VERIFY = (
    ("uv", "run", "--locked", "--extra", "dev", "python", "scripts/ci_shards.py",
     "verify", "--plan", ".tmp/ci-plan/plan.json"),
    ("uv", "run", "--locked", "--extra", "dev", "coverage", "combine", "--keep",
     ".tmp/ci-shards/shard-*"),
    ("uv", "run", "--locked", "--extra", "dev", "coverage", "report", "--fail-under=90"),
    ("uv", "run", "--locked", "--extra", "dev", "coverage", "json",
     "-o", ".tmp/combined-coverage.json"),
)


def _load_environment(path: Path, *, required: bool = True) -> Env:
    """Load the bootstrap environment; an absent file is a clear setup failure."""
    if not path.is_file():
        if not required:
            return {}
        raise SystemExit(
            f"gate: GNN environment not found at {path}; run `scripts/gate.py bootstrap` first"
        )
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(value, str) for value in data.values()):
        raise SystemExit(f"gate: malformed GNN environment at {path}")
    return {key: value for key, value in data.items() if isinstance(key, str)}


def _bootstrap(upstream: Path, destination: Path) -> Command:
    return (
        "uv", "run", "--locked", "--extra", "dev", "python", "scripts/bootstrap_gnn.py",
        "--upstream-root", str(upstream), "--destination", str(destination),
        "--profile", "cpython312-cpu",
    )


def _environment_path(args: argparse.Namespace, project_root: Path) -> Path:
    """Resolve the bootstrap environment against the selected checkout."""
    path = Path(args.environment)
    return path if path.is_absolute() else project_root / path


def _commands(arm: str, args: argparse.Namespace, project_root: Path) -> list[Invocation]:
    """Return the exact command sequence for one arm."""
    base: Env = {}
    if arm == "lint":
        return [(_RUFF, base)]
    if arm == "types":
        return [(_MYPY, base)]
    if arm == "metadata":
        return [(_METADATA, base)]
    if arm == "docs":
        return [(_MERMAID, base), (_DOCS_CONTRACT, base)]
    if arm == "fast":
        return [(_FAST, base)]
    if arm == "plan":
        return [(_PLAN, base)]
    if arm == "shard":
        if args.index is None:
            raise SystemExit("gate: --index is required for the shard arm")
        return [(_shard(args.index), base)]
    if arm == "verify":
        return [(command, base) for command in _VERIFY]
    if arm == "bootstrap":
        return [
            (_PIN_CHECK, base),
            (_bootstrap(Path(args.upstream_root), Path(args.destination)), base),
        ]
    if arm == "gnn":
        environment = _load_environment(
            _environment_path(args, project_root), required=not args.dry_run
        )
        return [
            ((*_VERSION_CHECK, args.upstream_root), base),
            (_shard(4), environment),
        ]
    if arm == "full":
        environment = _load_environment(
            _environment_path(args, project_root), required=not args.dry_run
        )
        return (
            [(_PLAN, base)]
            + [(_shard(index), base) for index in range(4)]
            + [(_shard(4), environment)]
            + [(command, base) for command in _VERIFY]
        )
    if arm == "all":
        return [
            invocation
            for name in ("lint", "types", "metadata", "docs", "fast", "full")
            for invocation in _commands(name, args, project_root)
        ]
    raise SystemExit(f"gate: unknown arm {arm!r}")


def _expand(command: Command, project_root: Path) -> Command:
    """Expand the shard receipt glob; coverage rejects a literal wildcard."""
    expanded: list[str] = []
    for part in command:
        if part == ".tmp/ci-shards/shard-*":
            expanded.extend(
                sorted(str(path) for path in (project_root / ".tmp/ci-shards").glob("shard-*"))
            )
        else:
            expanded.append(part)
    return tuple(expanded)


def _run(command: Command, project_root: Path, extra_env: Env) -> int:
    command = _expand(command, project_root)
    print(f"$ {shlex.join(command)}", flush=True)
    env = dict(os.environ)
    env["ACTIVE_FEDFERENCE_PROJECT_ROOT"] = str(project_root)
    env.update(extra_env)
    return subprocess.run(command, cwd=project_root, env=env).returncode


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arm",
        choices=("lint", "types", "metadata", "docs", "fast", "plan", "shard",
                 "verify", "bootstrap", "gnn", "full", "all"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--upstream-root", default=".tmp/gnn-upstream")
    parser.add_argument("--destination", default=".tmp/fedference-gnn")
    parser.add_argument("--environment", default=".tmp/fedference-gnn/environment.json")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="standalone checkout in which the arm should run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one explicit gate arm."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    from project_paths import resolve_script_project_root

    project_root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)
    commands = _commands(args.arm, args, project_root)
    if not args.dry_run:
        if args.arm in {"fast", "all"}:
            (project_root / ".tmp/ci-timings").mkdir(parents=True, exist_ok=True)
        if args.arm in {"full", "all"}:
            # The shard planner and shard runner refuse to overwrite prior
            # receipts; a local full run owns only its own .tmp/ci-* directories.
            for owned in (Path(".tmp/ci-plan"), Path(".tmp/ci-shards")):
                target = project_root / owned
                if target.exists():
                    subprocess.run(["rm", "-rf", str(target)], check=True)
    failures: list[int] = []
    for command, extra_env in commands:
        if args.dry_run:
            print(f"$ {shlex.join(_expand(command, project_root))}")
            continue
        exit_code = _run(command, project_root, extra_env)
        if exit_code == 0:
            continue
        failures.append(exit_code)
        if not args.keep_going:
            return exit_code
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
