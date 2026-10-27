"""Source-bound structured native adapters (Friston et al., 2024).

The moving threat and controlled observer position remain distinct factors.
Hierarchy execution retains the project empirical-prior message algorithm;
this boundary does not assert upstream backend support or protocol equivalence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
from numpy.typing import NDArray

from .admission import GNNAdmissionError, dumps, loads
from .loader import MODEL_DIR, _json, _mapping, _sha
from .native import NativeWorld, native_world
from .source import EXTENSION_MANIFEST, ParsedExtensions, _sections, parse_extensions
from .types import CategoricalModel, HierarchicalModel, JSONValue, MovingObserverModel, Tensor

if TYPE_CHECKING:
    from ..hierarchy_tasks import HierarchyTask
    from ..pomdp import LayerSpec


class HierarchyWorld(TypedDict):
    L1: NativeWorld
    layers: list[LayerSpec]
    n_levels: int
    level_priors: list[NDArray[np.float64]]
    level_transitions: list[NDArray[np.float64]]
    conditioned_priors: list[list[NDArray[np.float64]]]
    model_digest: str


MOVING_ASSET: Final = MODEL_DIR / "moving-observer.gnn.md"
HIERARCHY_ASSET: Final = MODEL_DIR / "hierarchy.gnn.md"


@dataclass(frozen=True, slots=True)
class StructuredProgram:
    """Immutable admitted model and source-bound native mapping."""

    model: MovingObserverModel | HierarchicalModel
    native_json: str

    @property
    def digest(self) -> str:
        return _sha((dumps(self.model) + self.native_json).encode())


@dataclass(frozen=True, slots=True)
class MovingParameters:
    """Explicit topology parameters; numeric sensor profiles remain source-owned."""

    n_positions: int | None = None
    n_agents: int | None = None
    fov_width: int | None = None
    seed: int = 0


class MovingWorld(TypedDict):
    A: list[NDArray[np.float64]]
    B: NDArray[np.float64]
    D: NDArray[np.float64]
    observation_map: NDArray[np.float64]
    position_prior: NDArray[np.float64]
    agent_positions: list[int]
    n_actions: int
    actions: tuple[str, ...]
    n_states: int
    n_positions: int
    n_agents: int
    fov_width: int
    seed: int
    model_digest: str


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


def cached_structured(path: Path) -> StructuredProgram:
    """Load a generated IR only when bound source/parser/full payload hashes agree."""
    source_bytes = path.read_bytes()
    cache = _mapping(_json(path.with_suffix(".ir.json").read_text()), "cache")
    if set(cache) != {"model", "native", "digest"}:
        raise GNNAdmissionError("GNN_CACHE", "fields")
    model = loads(json.dumps(cache["model"], allow_nan=False))
    if not isinstance(model, (MovingObserverModel, HierarchicalModel)):
        raise GNNAdmissionError("GNN_KIND", "cache")
    manifest = _json(EXTENSION_MANIFEST.read_text())
    parser_digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    if (
        model.metadata.provenance.source != _sha(source_bytes)
        or model.metadata.provenance.asset != _sha(source_bytes)
        or model.metadata.provenance.parser != parser_digest
    ):
        raise GNNAdmissionError("GNN_CACHE", "source/parser binding")
    program = StructuredProgram(model, json.dumps(cache["native"], sort_keys=True, allow_nan=False))
    if program.digest != cache["digest"]:
        raise GNNAdmissionError("GNN_CACHE", "payload binding")
    # Offline integrity reuses typed admission; it does not claim parser execution.
    admitted = _admit_sections(
        _sections(source_bytes.decode("utf-8"), ("FedferenceStructured",)),
        _sha(source_bytes),
        parser_digest,
    )
    if dumps(admitted.model) != dumps(program.model) or admitted.native_json != program.native_json:
        raise GNNAdmissionError("GNN_CACHE", "source payload")
    _native(program)
    if isinstance(program.model, HierarchicalModel):
        structured_tasks(program)
    return program


def moving_world(program: StructuredProgram, *, seed: int = 0) -> MovingWorld:
    """Consume distinct threat and position tensors without flattening them."""
    native = _native(program)
    model = program.model
    if not isinstance(model, MovingObserverModel):
        raise GNNAdmissionError("GNN_KIND", "moving")
    raw_positions = native["agent_positions"]
    assert isinstance(raw_positions, list)
    positions = [p for p in raw_positions if type(p) is int]
    width = native["fov_width"]
    assert isinstance(width, int)
    observation_map = _array(model.observation_map)
    return {
        "A": [observation_map[:, :, p].copy() for p in positions],
        "B": _array(model.position_transition),
        "D": _array(model.threat.D),
        "observation_map": observation_map,
        "position_prior": _array(model.position_prior),
        "agent_positions": positions,
        "n_actions": len(model.threat.actions),
        "actions": model.threat.actions,
        "n_states": len(model.threat.states),
        "n_positions": len(model.positions),
        "n_agents": len(positions),
        "fov_width": width,
        "seed": seed,
        "model_digest": program.digest,
    }


def hierarchy_leaf(program: StructuredProgram) -> NativeWorld:
    """Expose the source leaf channel through the shared categorical adapter."""
    _native(program)
    if not isinstance(program.model, HierarchicalModel):
        raise GNNAdmissionError("GNN_KIND", "hierarchy")
    metadata = replace(program.model.metadata, capabilities=("fedference.gnn/categorical/1",))
    return native_world(CategoricalModel(metadata, program.model.layers[-1].categorical))


def parameterize_moving(program: StructuredProgram, parameters: MovingParameters) -> StructuredProgram:
    """Resize only the declared line template, retaining source sensor/prior values."""
    _native(program)
    model = program.model
    if not isinstance(model, MovingObserverModel):
        raise GNNAdmissionError("GNN_KIND", "moving")
    native = _native(program)
    raw_positions = native["agent_positions"]
    assert isinstance(raw_positions, list)
    n = len(model.positions) if parameters.n_positions is None else parameters.n_positions
    agents = len(raw_positions) if parameters.n_agents is None else parameters.n_agents
    width = native["fov_width"] if parameters.fov_width is None else parameters.fov_width
    if type(n) is not int or type(agents) is not int or agents < 1 or n < agents:
        raise GNNAdmissionError("GNN_POSITION", "need one position per agent")
    if type(width) is not int or width < 1:
        raise GNNAdmissionError("GNN_POSITION", "fov_width")
    native = _native(program)
    if n != len(model.positions):
        source_n = len(model.positions)
        expected_controls = np.zeros((source_n, source_n, 3))
        for position in range(source_n):
            targets = (position, max(0, position - 1), min(source_n - 1, position + 1))
            for action, target in enumerate(targets):
                expected_controls[target, position, action] = 1
        if not np.array_equal(_array(model.position_transition), expected_controls):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "non-template controls cannot resize")
        if not np.array_equal(_array(model.position_prior), np.full(source_n, 1 / source_n)):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "nonuniform position prior cannot resize")
        source = _array(model.observation_map)
        half = len(model.positions) // 2
        if (
            half < 1
            or not np.array_equal(source[:, :, :half], np.repeat(source[:, :, :1], half, axis=2))
            or not np.array_equal(
                source[:, :, half:],
                np.repeat(source[:, :, half : half + 1], len(model.positions) - half, axis=2),
            )
        ):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "heterogeneous sensors cannot resize")
        b = np.zeros((n, n, 3))
        for p in range(n):
            for action, target in enumerate((p, max(0, p - 1), min(n - 1, p + 1))):
                b[target, p, action] = 1
        observation_map = np.stack([source[:, :, 0 if p < n // 2 else half] for p in range(n)], axis=2)
        model = replace(
            model,
            positions=tuple(str(p) for p in range(n)),
            position_transition=Tensor(model.position_transition.axes, b.shape, tuple(b.ravel())),
            position_prior=Tensor(("position",), (n,), tuple(np.full(n, 1 / n))),
            observation_map=Tensor(
                model.observation_map.axes, observation_map.shape, tuple(observation_map.ravel())
            ),
            threat=replace(
                model.threat,
                A=Tensor(model.threat.A.axes, model.threat.A.shape, tuple(observation_map[:, :, 0].ravel())),
            ),
        )
    if parameters.n_positions is not None or parameters.n_agents is not None:
        native["agent_positions"] = [i * (n // agents) for i in range(agents)]
    native["fov_width"] = width
    result = StructuredProgram(model, json.dumps(native, sort_keys=True))
    _native(result)
    return result


def hierarchy_depth(program: StructuredProgram, depth: int) -> StructuredProgram:
    """Select a source-defined suffix; its new root keeps its declared default prior."""
    _native(program)
    model = program.model
    if (
        not isinstance(model, HierarchicalModel)
        or type(depth) is not int
        or not 2 <= depth <= len(model.layers)
    ):
        raise GNNAdmissionError("GNN_LAYERS", "depth")
    layers = model.layers[-depth:]
    model = replace(model, layers=(replace(layers[0], parent=None, conditioned_prior=None), *layers[1:]))
    result = replace(program, model=model)
    _native(result)
    return result


def hierarchy_world(program: StructuredProgram) -> HierarchyWorld:
    """Map source levels/conditional edges to the existing native inference contract."""
    from ..pomdp import LayerSpec

    _native(program)
    model = program.model
    if not isinstance(model, HierarchicalModel):
        raise GNNAdmissionError("GNN_KIND", "hierarchy")
    conditioned = []
    for layer in model.layers[1:]:
        assert layer.conditioned_prior is not None
        conditioned.append(list(_array(layer.conditioned_prior).T))
    priors = [_array(layer.categorical.D) for layer in model.layers]
    layers = [
        LayerSpec(
            n_states=len(layer.categorical.states),
            labels=layer.categorical.states,
            default_prior=priors[i],
            conditioned_priors=conditioned[i] if i < len(conditioned) else None,
        )
        for i, layer in enumerate(model.layers)
    ]
    return {
        "L1": hierarchy_leaf(program),
        "layers": layers,
        "n_levels": len(layers),
        "level_priors": priors,
        "level_transitions": [_array(layer.categorical.B) for layer in model.layers],
        "conditioned_priors": conditioned,
        "model_digest": program.digest,
    }


def infer_hierarchy(
    program: StructuredProgram, observation: int, *, n_iters: int = 4
) -> tuple[NDArray[np.float64], ...]:
    """Execute native message passing with the admitted source likelihood and gates."""
    from ..pomdp import nlevel_infer

    world = hierarchy_world(program)
    return tuple(nlevel_infer(world["L1"]["A"][0], observation, dict(world), n_iters=n_iters)["q_levels"])


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


@dataclass(frozen=True, slots=True)
class HierarchyParameters:
    """Explicit source-preserving hierarchy overrides; no PMF normalization."""

    depth: int = 4
    acuity: float | None = None
    goal_bonus: float | None = None
    priors: tuple[tuple[float, ...] | None, ...] | None = None
    conditioned_priors: tuple[tuple[tuple[float, ...], ...] | None, ...] | None = None


def parameterize_hierarchy(program: StructuredProgram, parameters: HierarchyParameters) -> StructuredProgram:
    """Bind all declared overrides into the admitted IR and its digest."""
    from .native import SentinelParameters, parameterize_sentinel

    selected = hierarchy_depth(program, parameters.depth)
    model = selected.model
    assert isinstance(model, HierarchicalModel)
    layers = list(model.layers)
    if parameters.priors is not None:
        if len(parameters.priors) != len(layers):
            raise GNNAdmissionError("GNN_LAYERS", "prior overrides")
        for i, prior in enumerate(parameters.priors):
            if prior is not None:
                c = layers[i].categorical
                layers[i] = replace(layers[i], categorical=replace(c, D=Tensor(c.D.axes, c.D.shape, prior)))
    if parameters.conditioned_priors is not None:
        if len(parameters.conditioned_priors) != len(layers) - 1:
            raise GNNAdmissionError("GNN_LAYERS", "conditioned overrides")
        for i, conditioned_prior in enumerate(parameters.conditioned_priors, 1):
            if conditioned_prior is not None:
                cp = layers[i].conditioned_prior
                assert cp is not None
                array = np.asarray(conditioned_prior, dtype=np.float64).T
                layers[i] = replace(
                    layers[i], conditioned_prior=Tensor(cp.axes, array.shape, tuple(array.ravel()))
                )
    leaf = parameterize_sentinel(
        CategoricalModel(
            replace(model.metadata, capabilities=("fedference.gnn/categorical/1",)), layers[-1].categorical
        ),
        SentinelParameters(parameters.acuity, parameters.goal_bonus),
    )
    layers[-1] = replace(layers[-1], categorical=leaf.categorical)
    result = replace(selected, model=replace(model, layers=tuple(layers)))
    _native(result)
    return result
