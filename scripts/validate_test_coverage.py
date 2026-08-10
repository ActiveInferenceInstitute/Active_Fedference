#!/usr/bin/env python3
"""Run the authoritative full test gate and write its source-bound receipt."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _junit_summary(path: Path) -> dict[str, int]:
    """Read pytest's JUnit aggregate without inferring results from stdout."""
    try:
        root = element_tree.parse(path).getroot()
        if root.tag == "testsuite":
            suites = (root,)
        elif root.tag == "testsuites":
            # pytest 9 emits an uncounted <testsuites> wrapper around one or
            # more counted <testsuite> elements. Sum only its direct children
            # so an aggregate wrapper cannot be mistaken for an empty run.
            suites = tuple(root.findall("testsuite"))
            if not suites:
                raise ValueError("JUnit testsuites wrapper has no test suites")
        else:
            raise ValueError(f"unsupported JUnit root element: {root.tag}")

        collected = sum(int(suite.attrib["tests"]) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", "0")) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    except (OSError, ValueError, element_tree.ParseError, KeyError) as exc:
        raise RuntimeError(f"invalid pytest JUnit report: {path}") from exc
    failed = failures + errors
    passed = collected - failed - skipped
    if min(collected, failures, errors, skipped, passed) < 0:
        raise RuntimeError(f"invalid pytest JUnit totals: {path}")
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def _coverage_percent(path: Path) -> float:
    """Read the achieved source-coverage percentage from coverage.py JSON."""
    import json

    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        value = payload["totals"]["percent_covered"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("percent_covered is not numeric")
        return float(value)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid coverage JSON report: {path}") from exc


def _project_root_from_args(args: argparse.Namespace) -> Path:
    if args.project_root is not None:
        root = args.project_root.resolve()
    else:
        from project_paths import resolve_env_project_root

        root = resolve_env_project_root(_PROJECT_ROOT)
    if not (root / "manuscript" / "config.yaml").is_file():
        raise RuntimeError(f"invalid project root (missing manuscript/config.yaml): {root}")
    return root


def main(argv: list[str] | None = None) -> int:
    """Run or verify the one successful full-suite validation receipt."""
    from publication.release_manifest import timestamp_from_source_date_epoch
    from publication.validation_receipt import (
        MINIMUM_COVERAGE_THRESHOLD,
        capture_validation_snapshot,
        require_fresh_validation_receipt,
        validation_environment,
        write_validation_receipt,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing receipt without executing the test suite",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = _project_root_from_args(args)
    if args.verify:
        receipt = require_fresh_validation_receipt(root)
        print(f"validation receipt: PASS ({receipt['test_summary']['collected']} tests)")
        return 0

    scratch_parent = root / ".tmp"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="test-coverage-receipt-",
        dir=scratch_parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        junit = temporary / "pytest-junit.xml"
        coverage = temporary / "coverage.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--cov=src",
            f"--cov-fail-under={MINIMUM_COVERAGE_THRESHOLD:g}",
            f"--junitxml={junit}",
            f"--cov-report=json:{coverage}",
        ]
        # Snapshot every input and the fresh analysis receipt *before* pytest
        # starts.  The writer captures the same boundary again after the suite
        # and refuses to issue a receipt if any source, test, manuscript,
        # config, lock, script, or analysis evidence changed in between.
        pre_run_snapshot = capture_validation_snapshot(root)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode:
            return completed.returncode
        summary = _junit_summary(junit)
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        timestamp = (
            timestamp_from_source_date_epoch(source_date_epoch) if source_date_epoch is not None else None
        )
        receipt = write_validation_receipt(
            root,
            command=command,
            test_summary=summary,
            coverage_percent=_coverage_percent(coverage),
            pre_run_snapshot=pre_run_snapshot,
            coverage_threshold=MINIMUM_COVERAGE_THRESHOLD,
            environment=validation_environment(),
            timestamp=timestamp,
        )
    print(f"validation receipt: {root / 'output/data/test_coverage_receipt.json'}")
    print(f"input_digest: {receipt['input_digest']}")
    print(f"analysis_output_digest: {receipt['analysis_stage']['output_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
