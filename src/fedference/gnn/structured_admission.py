"""Source-bound structured family admission (Friston et al., 2024 subset).

Capability checks admit only distinct moving factors or native hierarchy chains.
Offline cache integrity uses this same source admission without claiming execution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from ._documents import MODEL_DIR, _json, _mapping, _sha
from .admission import GNNAdmissionError, dumps, loads
from .source import ParsedExtensions, parse_extensions
from .types import HierarchicalModel, JSONValue, MovingObserverModel, Tensor

if TYPE_CHECKING:
    from ..hierarchy_tasks import HierarchyTask


@dataclass(frozen=True, slots=True)
class StructuredProgram:
    """Immutable admitted model and source-bound native mapping."""

    model: MovingObserverModel | HierarchicalModel
    native_json: str

    @property
    def digest(self) -> str:
        return _sha((dumps(self.model) + self.native_json).encode())



MOVING_ASSET: Final = MODEL_DIR / "moving-observer.fed.md"



HIERARCHY_ASSET: Final = MODEL_DIR / "hierarchy.fed.md"



def _native(program: StructuredProgram) -> dict[str, JSONValue]:
    dumps(program.model)
    native = _mapping(_json(program.native_json), "native")
    match program.model:
        case MovingObserverModel() as model:
            if set(native) != {"capability", "roles", "agent_positions", "fov_width", "resize"}:
                raise GNNAdmissionError("GNN_FIELDS", "moving native")
            if native["capability"] != "native-moving-observer/1" or native["roles"] != {
                "threat": "latent-half",
                "position": "controlled-observer",
                "observation_map": "observation-given-threat-position",
            }:
                raise GNNAdmissionError("GNN_FACTOR_ROLE", "threat/position")
            if (
                model.threat.states != ("left", "right")
                or model.threat.observations != ("detected", "not_detected")
                or model.threat.actions != ("stay", "left", "right")
            ):
                raise GNNAdmissionError("GNN_LABEL", "moving factors")
            if model.positions != tuple(str(i) for i in range(len(model.positions))):
                raise GNNAdmissionError("GNN_LABEL", "positions")
            if native["resize"] != "reflecting-line-half-sensor/1":
                raise GNNAdmissionError("GNN_CAPABILITY", "resize")
            positions = native["agent_positions"]
            if (
                not isinstance(positions, list)
                or not positions
                or any(type(p) is not int or not 0 <= p < len(model.positions) for p in positions)
            ):
                raise GNNAdmissionError("GNN_POSITION", "agent_positions")
            if type(native["fov_width"]) is not int or native["fov_width"] < 1:
                raise GNNAdmissionError("GNN_POSITION", "fov_width")
            if any(value not in (0.0, 1.0) for value in model.position_transition.values):
                raise GNNAdmissionError("GNN_CAPABILITY", "native deterministic position controls")
            observation_map = _array(model.observation_map)
            if not np.array_equal(_array(model.threat.A), observation_map[:, :, 0]):
                raise GNNAdmissionError("GNN_FACTOR_ROLE", "threat reference likelihood")
            if not np.array_equal(_array(model.threat.B), np.repeat(np.eye(2)[:, :, None], 3, axis=2)) or any(
                model.threat.C.values
            ):
                raise GNNAdmissionError("GNN_CAPABILITY", "static threat; no preferences")
        case HierarchicalModel() as model:
            if (
                set(native) != {"capability", "messages", "tasks"}
                or native["capability"] != "native-hierarchy/1"
                or native["messages"] != ["top-down-mixture", "bottom-up-posterior-evidence"]
            ):
                raise GNNAdmissionError("GNN_CAPABILITY", "hierarchy messages")
            if len(model.layers[-1].categorical.states) != 9:
                raise GNNAdmissionError("GNN_CAPABILITY", "native nine-location leaf")
            for parent, child in zip(model.layers, model.layers[1:]):
                if child.parent != parent.name or child.conditioned_prior is None:
                    raise GNNAdmissionError("GNN_PARENT", "native chain")
            for layer in model.layers[:-1]:
                c = layer.categorical
                if (
                    c.observations != c.states
                    or c.actions != ("stay",)
                    or not np.array_equal(_array(c.A), np.eye(len(c.states)))
                    or any(c.C.values)
                ):
                    raise GNNAdmissionError("GNN_CAPABILITY", "unobserved context identity channel")
    return native



def _array(tensor: Tensor) -> NDArray[np.float64]:
    return np.array(tensor.values, dtype=np.float64).reshape(tensor.shape)



def _admit(parsed: ParsedExtensions) -> StructuredProgram:
    return _admit_sections(dict(parsed.sections), parsed.source_digest, parsed.parser_digest)



def _admit_sections(
    sections: dict[str, str], source_digest: str, parser_digest: str
) -> StructuredProgram:
    if (
        set(sections)
        != {"GNNSection", "GNNVersionAndFlags", "ModelName", "ModelAnnotation", "FedferenceStructured"}
        or sections["GNNSection"] != "ActInfPOMDP"
        or sections["GNNVersionAndFlags"] != "GNN v1.0"
    ):
        raise GNNAdmissionError("GNN_SECTIONS", "structured")
    payload = _mapping(_json(sections["FedferenceStructured"]), "structured")
    if set(payload) != {"schema", "model", "native"} or payload["schema"] != "fedference.structured/1":
        raise GNNAdmissionError("GNN_SCHEMA", "structured")
    raw = _mapping(payload["model"], "model")
    metadata = _mapping(raw.get("metadata"), "metadata")
    if "provenance" in metadata or metadata.get("name") != sections["ModelName"]:
        raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", "metadata")
    raw["metadata"] = {
        **metadata,
        "provenance": {
            "source": source_digest,
            "asset": source_digest,
            "parser": parser_digest,
            "backend": None,
            "native_result": None,
        },
    }
    model = loads(json.dumps(raw, allow_nan=False))
    if not isinstance(model, (MovingObserverModel, HierarchicalModel)):
        raise GNNAdmissionError("GNN_KIND", "structured")
    program = StructuredProgram(model, json.dumps(payload["native"], sort_keys=True, allow_nan=False))
    _native(program)
    if isinstance(program.model, HierarchicalModel):
        structured_tasks(program)
    return program



def load_structured(
    path: Path, *, upstream_root: Path, python: Path | None = None, dependency_root: Path | None = None
) -> StructuredProgram:
    """Run the real pinned parser before strict project extension admission."""
    return _admit(
        parse_extensions(
            path,
            upstream_root=upstream_root,
            allowed_extensions=("FedferenceStructured",),
            python=python,
            dependency_root=dependency_root,
        )
    )



def structured_tasks(program: StructuredProgram) -> tuple[HierarchyTask, ...]:
    """Load the two explicit grid task definitions; no ambient task defaults."""
    from ..hierarchy_tasks import HierarchyTask

    native = _native(program)
    if not isinstance(program.model, HierarchicalModel):
        raise GNNAdmissionError("GNN_KIND", "tasks")
    raw = native["tasks"]
    if not isinstance(raw, list) or len(raw) != 2:
        raise GNNAdmissionError("GNN_FIELDS", "tasks")
    tasks = []
    for value in raw:
        task = _mapping(value, "task")
        if set(task) != {"task_id", "grid", "start", "goal", "horizon", "key", "door"}:
            raise GNNAdmissionError("GNN_FIELDS", "task")

        def coordinate(name: str) -> tuple[int, int] | None:
            point = task[name]
            if point is None:
                return None
            if not isinstance(point, list) or len(point) != 2 or any(type(x) is not int for x in point):
                raise GNNAdmissionError("GNN_POSITION", name)
            x, y = point
            assert isinstance(x, int) and isinstance(y, int)
            return x, y

        grid, task_id, horizon = task["grid"], task["task_id"], task["horizon"]
        start, goal = coordinate("start"), coordinate("goal")
        if (
            not isinstance(grid, list)
            or any(not isinstance(row, str) for row in grid)
            or not isinstance(task_id, str)
            or type(horizon) is not int
            or start is None
            or goal is None
        ):
            raise GNNAdmissionError("GNN_FIELDS", "task types")
        try:
            tasks.append(
                HierarchyTask(
                    task_id,
                    tuple(row for row in grid if isinstance(row, str)),
                    start,
                    goal,
                    horizon,
                    coordinate("key"),
                    coordinate("door"),
                )
            )
        except ValueError as error:
            raise GNNAdmissionError("GNN_TASK", str(error)) from error
    if tuple(t.task_id for t in tasks) != ("four-rooms", "key-door"):
        raise GNNAdmissionError("GNN_LABEL", "task ids")
    return tuple(tasks)



def require_structured_backend(program: StructuredProgram, backend: str) -> None:
    """Refuse unsupported compositions before any backend output is created."""
    _native(program)
    if backend != "native":
        raise GNNAdmissionError("GNN_CAPABILITY", f"{program.model.kind}/{backend}")
