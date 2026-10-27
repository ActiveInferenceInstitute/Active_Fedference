#!/usr/bin/env python3
"""Bootstrap a fresh, pinned GNN runtime and qualify real execution and parsing.

Run with the locked project environment:
uv run --locked python scripts/bootstrap_gnn.py --upstream-root PATH --destination PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from fedference.gnn.backend import BackendConfig, run_pymdp, verify_backend_output
from fedference.gnn.backend_registry import load_backend_registry
from fedference.gnn.loader import PARSER_MANIFEST, load_sentinel, verify_sentinel_cache
from fedference.gnn.scientific import HYBRID_ASSET, load_scientific
from fedference.gnn.source import EXTENSION_MANIFEST
from project_paths import resolve_script_project_root


def main() -> int:
    """Provision only a new owned directory; preserve failures for inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--profile", choices=("cpython312-cpu",), default="cpython312-cpu")
    parser.add_argument("--python", help="Exact interpreter path; otherwise use the profile version")
    args = parser.parse_args()
    root = resolve_script_project_root(Path(__file__).resolve().parents[1], args.project_root)
    lock_path = args.lock or root / "src/fedference/config/gnn-backend-lock.json"
    registry = load_backend_registry(lock_path)
    profile = registry.profiles[args.profile]
    if profile.qualification != "fresh-provisioning-and-execution":
        raise ValueError("Runtime profile is not qualified for fresh provisioning")
    upstream = args.upstream_root.resolve(strict=True)
    destination = args.destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("GNN bootstrap requires a new destination")
    files = dict(registry.source_files)
    for manifest in (PARSER_MANIFEST, EXTENSION_MANIFEST):
        for relative, expected in json.loads(manifest.read_text())["files"].items():
            if relative in files and files[relative] != expected:
                raise ValueError("Conflicting pinned source hashes")
            files[relative] = expected
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe pinned source path")
        source = upstream / path
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned upstream source differs: {relative}")
    destination.mkdir(parents=True)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((upstream / relative).read_bytes())
    runtime_registry = destination / "registry.json"
    runtime_registry.write_bytes(lock_path.read_bytes())
    requirements = destination / "requirements.txt"
    requirements.write_text("".join(
        f"{name}=={version}\n" for name, version in sorted(profile.distributions.items())
    ))
    python = destination / ".venv/bin/python"
    commands = [
        ["uv", "venv", "--python", args.python or profile.python, str(destination / ".venv")],
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "-r", str(requirements)],
        ["uv", "pip", "check", "--python", str(python)],
    ]
    for index, command in enumerate(commands):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (destination / f"install-{index}.json").write_text(json.dumps({
            "argv": command, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
        }, indent=2))
        if completed.returncode:
            print(completed.stderr, file=sys.stderr)
            return completed.returncode
    config = BackendConfig(destination, python, lock_path=runtime_registry)
    model = load_sentinel()
    output = destination / "qualification"
    result = run_pymdp(model, config, output)
    verify_backend_output(model, config, output)
    runtime = json.loads((output / "runtime.json").read_text())
    if runtime["profile"] != args.profile:
        raise ValueError("Executed runtime differs from the requested profile")
    version = "python" + ".".join(profile.python.split(".")[:2])
    dependencies = destination / ".venv/lib" / version / "site-packages"
    verify_sentinel_cache(upstream_root=destination)
    load_scientific(HYBRID_ASSET, upstream_root=destination, dependency_root=dependencies)
    environment = {
        "GNN_ROOT": str(destination),
        "GNN_PARSER_DEPENDENCIES": str(dependencies),
        "GNN_PYTHON": str(python),
        "GNN_BACKEND_LOCK": str(runtime_registry),
    }
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (destination / "bootstrap.json").write_text(json.dumps({
        "profile": args.profile, "registry_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(runtime_registry.read_bytes()).hexdigest(),
        "source_files": files, "replay_max_error": result.replay_max_error,
        "receipt": str(result.receipt), "environment": environment,
    }, indent=2) + "\n")
    print(json.dumps(environment))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
