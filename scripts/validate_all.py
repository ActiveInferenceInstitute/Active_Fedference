#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

Command = tuple[str, ...]

_PROFILES: dict[str, tuple[Command, ...]] = {
    "quick": (
        (
            "uv",
            "run",
            "--extra",
            "dev",
            "pytest",
            "tests/test_docs_contract.py",
            "tests/test_caption_completeness.py",
            "tests/fedference/test_bnn_baseline_torch.py",
            "-q",
        ),
    ),
    "manuscript": (
        (
            "uv",
            "run",
            "pytest",
            "tests/test_xref_integrity.py",
            "tests/test_caption_completeness.py",
            "tests/test_token_provenance.py",
            "tests/test_token_tables.py",
            "tests/test_manuscript_variables.py",
            "-q",
        ),
    ),
    "package": (
        ("uv", "run", "python", "scripts/prepare_web_package.py"),
        ("uv", "run", "python", "scripts/validate_web_package.py"),
    ),
    "rendered": (
        ("uv", "run", "python", "scripts/validate_rendered_surfaces.py"),
    ),
    "freshness": (
        (
            "uv",
            "run",
            "--extra",
            "dev",
            "python",
            "scripts/validate_test_coverage.py",
            "--verify",
        ),
        ("uv", "run", "python", "scripts/validate_pipeline_freshness.py"),
    ),
    "torch": (
        (
            "uv",
            "run",
            "--extra",
            "dev",
            "pytest",
            "tests/fedference/test_bnn_baseline_torch.py",
            "-q",
        ),
    ),
    "source": (
        ("uv", "run", "ruff", "check", "src/", "tests/", "scripts/"),
        ("uv", "run", "mypy", "src/"),
        ("uv", "run", "python", "scripts/01_run_invariants.py"),
        ("sh", "-c", "! grep -rn 'import infrastructure' src/fedference/"),
        ("uv", "run", "python", "scripts/build_release.py"),
        ("uv", "run", "python", "scripts/build_release.py", "--verify"),
    ),
    "full": (
        ("uv", "run", "python", "scripts/validate_all.py", "quick"),
        ("uv", "run", "python", "scripts/validate_all.py", "manuscript"),
        ("uv", "run", "python", "scripts/validate_all.py", "package"),
        ("uv", "run", "python", "scripts/validate_all.py", "rendered"),
        ("uv", "run", "python", "scripts/validate_all.py", "freshness"),
        ("uv", "run", "python", "scripts/validate_all.py", "source"),
        (
            "uv",
            "run",
            "--extra",
            "dev",
            "pytest",
            "tests/",
            "--cov=src",
            "--cov-fail-under=90",
        ),
    ),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=sorted(_PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args(argv)


def _run(command: Command, *, dry_run: bool) -> int:
    print(f"$ {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=_PROJECT_ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    """Run one local validation profile."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    failures: list[int] = []
    for command in _PROFILES[args.profile]:
        exit_code = _run(command, dry_run=args.dry_run)
        if exit_code == 0:
            continue
        failures.append(exit_code)
        if not args.keep_going:
            return exit_code
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
