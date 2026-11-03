"""Read-only GNN model admission adapter; domain loaders own every validation rule."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

from fedference.gnn.admission import GNNAdmissionError, content_digest, dumps, loads
from fedference.gnn.loader import load_gnn
from fedference.gnn.scientific_admission import load_scientific
from fedference.gnn.sites import site_configuration
from fedference.gnn.structured_admission import load_structured
from fedference.gnn.tracking import tracking_config
from fedference.gnn.types import HybridGaussianModel, JSONValue, Model


class ModelValidation(TypedDict):
    """Admission result, explicitly distinct from execution or scientific evidence."""

    valid: Literal[True]
    format: str
    model_digest: str
    program_digest: str | None
    model: JSONValue
    native_configuration: Literal["validated", "not-provided"]


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value is not None else None


def _model_validation(args: argparse.Namespace) -> ModelValidation:
    path = Path(args.model)
    selected = args.format
    if selected == "auto":
        if path.suffix.lower() == ".json":
            selected = "native"
        else:
            # Select a loader only; that loader owns syntax, uniqueness, and admission.
            headers = path.read_text(encoding="utf-8").splitlines()
            families = [name for header, name in (
                ("## FedferenceStructured", "structured"),
                ("## FedferenceScientific", "scientific"),
            ) if header in headers]
            if len(families) > 1:
                raise GNNAdmissionError("GNN_SECTIONS", "ambiguous model family")
            selected = families[0] if families else "sentinel"
    model: Model
    program_digest: str | None = None
    native: Literal["validated", "not-provided"] = "not-provided"
    if selected == "native":
        if args.gnn_root or args.gnn_python or args.parser_dependencies:
            raise ValueError("native JSON admission does not consume --gnn-root or parser options")
        model = loads(path.read_text(encoding="utf-8"))
    else:
        if args.gnn_root is None:
            raise ValueError("GNN Markdown validation requires explicit --gnn-root")
        upstream = Path(args.gnn_root)
        python = _optional_path(args.gnn_python or os.environ.get("GNN_PYTHON"))
        dependencies = _optional_path(args.parser_dependencies or os.environ.get("GNN_PARSER_DEPENDENCIES"))
        if selected == "sentinel":
            if args.parser_dependencies is not None:
                raise ValueError("--parser-dependencies applies only to structured/scientific models")
            model = load_gnn(path, upstream_root=upstream, python=python)
        elif selected == "structured":
            structured = load_structured(
                path, upstream_root=upstream, python=python, dependency_root=dependencies,
            )
            model, program_digest = structured.model, structured.digest
            native = "validated"
        else:
            scientific = load_scientific(
                path, upstream_root=upstream, python=python, dependency_root=dependencies,
            )
            if isinstance(scientific.model, HybridGaussianModel):
                tracking_config(scientific)
            else:
                site_configuration(scientific)
            model, program_digest = scientific.model, scientific.digest
            native = "validated"
    return {
        "valid": True,
        "format": selected,
        "model_digest": content_digest(model),
        "program_digest": program_digest,
        "model": json.loads(dumps(model)),
        "native_configuration": native,
    }


def _model_validate_command(args: argparse.Namespace) -> int:
    result = _model_validation(args)
    if args.as_json:
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"Valid model: {result['model_digest']} ({result['format']}); "
              f"native configuration: {result['native_configuration']}; no execution")
    return 0
