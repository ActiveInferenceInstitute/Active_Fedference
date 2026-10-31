"""Real parser and native structured execution; no backend substitution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fedference.gnn.admission import GNNAdmissionError
from fedference.gnn.structured import (
    HIERARCHY_ASSET,
    MOVING_ASSET,
    MovingParameters,
    cached_structured,
    hierarchy_depth,
    hierarchy_world,
    infer_hierarchy,
    load_structured,
    moving_world,
    parameterize_moving,
    require_structured_backend,
    structured_tasks,
)
from fedference.hierarchy_tasks import FOUR_ROOMS, KEY_DOOR, simulate_hierarchy_task
from fedference.pomdp import build_3level_world, build_moving_world, nlevel_infer


def _load(path: Path):
    return load_structured(
        path,
        upstream_root=Path(os.environ["GNN_ROOT"]),
        dependency_root=Path(os.environ["GNN_PARSER_DEPENDENCIES"]),
    )


def _mutate(tmp_path: Path, source: Path, mutate) -> Path:
    prefix, body = source.read_text().split("## FedferenceStructured\n")
    data = json.loads(body)
    mutate(data)
    target = tmp_path / source.name
    target.write_text(prefix + "## FedferenceStructured\n" + json.dumps(data, indent=2) + "\n")
    return target


def test_moving_and_depth_four_native_execution() -> None:
    # Given real upstream-extracted models and independently characterized native factories.
    moving = _load(MOVING_ASSET)
    hierarchy = _load(HIERARCHY_ASSET)
    expected = build_moving_world()
    # When mapped into the real native data/inference contracts.
    actual = moving_world(moving)
    posteriors = infer_hierarchy(hierarchy, 4)
    # Then distinct threat/position dimensions and native numeric values are retained.
    assert actual["B"].shape == (4, 4, 3)
    assert actual["observation_map"].shape == (2, 2, 4)
    for key in ("A", "B", "D"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(actual["A"][0], [[0.85, 0.10], [0.15, 0.90]])
    assert len(posteriors) == 4
    for q in posteriors:
        np.testing.assert_allclose(q.sum(), 1, atol=1e-14)
    assert posteriors[-1][4] > 0.9
    native3 = build_3level_world()
    for a, b in zip(
        infer_hierarchy(hierarchy_depth(hierarchy, 3), 4),
        nlevel_infer(native3["L1"]["A"][0], 4, native3)["q_levels"],
    ):
        np.testing.assert_array_equal(a, b)
    print(
        json.dumps({"moving": [a.tolist() for a in actual["A"]], "depth4": [q.tolist() for q in posteriors]})
    )


def test_threat_position_conflation_and_missing_conditioned_prior(tmp_path: Path) -> None:
    # Given malformed factor semantics and a missing required conditional edge.
    cases = (
        (MOVING_ASSET, lambda d: d["native"]["roles"].update(threat="controlled-observer")),
        (HIERARCHY_ASSET, lambda d: d["model"]["layers"][1].update(conditioned_prior=None)),
    )
    for source, mutation in cases:
        path = _mutate(tmp_path, source, mutation)
        original = path.read_bytes()
        # When real parsing and strict admission execute.
        with pytest.raises(GNNAdmissionError) as error:
            _load(path)
        # Then rejection preserves the caller source and produces no cache.
        assert path.read_bytes() == original
        assert not path.with_suffix(".ir.json").exists()
        print(error.value)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_depth_native_oracle(depth: int) -> None:
    # Given the canonical source-bound hierarchy at each supported depth.
    program = hierarchy_depth(cached_structured(HIERARCHY_ASSET), depth)
    world = hierarchy_world(program)
    # When the source-driven native algorithm executes.
    result = infer_hierarchy(program, 4, n_iters=1)
    # Then an independent one-sweep numerical oracle agrees.
    qs = [p.copy() for p in world["level_priors"]]
    cp = world["conditioned_priors"]
    prior = np.array(cp[-1]).T @ qs[-2]
    leaf = prior * world["L1"]["A"][0][4]
    qs[-1] = leaf / leaf.sum()
    for i in range(depth - 2, -1, -1):
        weight = np.array(cp[i]) @ qs[i + 1] * world["level_priors"][i]
        qs[i] = weight / weight.sum()
    for actual, expected in zip(result, qs):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_source_changes_drive_moving_output(tmp_path: Path) -> None:
    # Given a real raw source sensor mutation respecting the reference channel.
    def mutate(data):
        values = data["model"]["observation_map"]["values"]
        values[1], values[9] = 0.7, 0.3

    path = _mutate(tmp_path, MOVING_ASSET, mutate)
    # When upstream parsing feeds native conversion.
    actual = moving_world(_load(path))
    # Then a changed source cell appears in the native observation map.
    assert actual["observation_map"][0, 0, 1] == 0.7
    assert actual["model_digest"] != moving_world(cached_structured(MOVING_ASSET))["model_digest"]


def test_source_changes_drive_hierarchy_posterior(tmp_path: Path) -> None:
    # Given changed raw top-level prior.
    path = _mutate(
        tmp_path,
        HIERARCHY_ASSET,
        lambda d: d["model"]["layers"][0]["categorical"]["D"].update(values=[0.9, 0.1]),
    )
    # When real parser and native message passing run.
    actual = infer_hierarchy(_load(path), 4)
    # Then the source mutation affects real native posterior.
    assert not np.allclose(actual[0], infer_hierarchy(cached_structured(HIERARCHY_ASSET), 4)[0])


def test_source_tasks_load_and_execute() -> None:
    # Given the source-bound task definitions.
    tasks = structured_tasks(cached_structured(HIERARCHY_ASSET))
    # When the actual native grid task runner consumes them.
    results = [simulate_hierarchy_task(task, method="oracle", seed=1943) for task in tasks]
    # Then task geometry exactly matches the characterized controls.
    assert tasks == (FOUR_ROOMS, KEY_DOOR)
    assert len(results) == 2
    print(results)


def test_cache_tampering_and_backend_refusal(tmp_path: Path) -> None:
    # Given copied canonical source and cache with an altered likelihood.
    source = tmp_path / MOVING_ASSET.name
    source.write_bytes(MOVING_ASSET.read_bytes())
    cache = json.loads(MOVING_ASSET.with_suffix(".ir.json").read_text())
    cache["model"]["threat"]["D"]["values"] = [0.8, 0.2]
    source.with_suffix(".ir.json").write_text(json.dumps(cache))
    # When cache admission or an unsupported execution family is requested.
    with pytest.raises(GNNAdmissionError, match="GNN_CACHE"):
        cached_structured(source)
    with pytest.raises(GNNAdmissionError, match="GNN_CAPABILITY"):
        require_structured_backend(cached_structured(MOVING_ASSET), "pymdp")
    # Then no execution outputs are emitted.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [source.name, source.with_suffix(".ir.json").name]
    )


@pytest.mark.parametrize("n,agents", [(1, 1), (4, 2), (8, 3)])
def test_parameterized_moving_preserves_native_arrays(n: int, agents: int) -> None:
    # Given explicit topology parameters on the source model.
    program = parameterize_moving(cached_structured(MOVING_ASSET), MovingParameters(n, agents))
    # When native conversion runs.
    actual = moving_world(program)
    # Then characterized native control arrays and priors agree.
    expected = build_moving_world(n_positions=n, n_agents=agents)
    for key in ("A", "B", "D"):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize(
    "source,field,value",
    [
        ("moving", "native.capability", "other"),
        ("moving", "native.resize", "other"),
        ("moving", "native.agent_positions", []),
        ("moving", "native.agent_positions", [99]),
        ("moving", "native.fov_width", 0),
        ("moving", "model.positions", ["a", "b", "c", "d"]),
        ("moving", "model.threat.states", ["position0", "position1"]),
        ("moving", "model.threat.A.values", [0.8, 0.1, 0.2, 0.9]),
        ("moving", "model.threat.C.values", [1, 0]),
        ("moving", "model.position_transition.axes", ["next_state", "previous_state", "action"]),
        ("hierarchy", "native.messages", ["observe"]),
        ("hierarchy", "model.layers.2.parent", "regime"),
        ("hierarchy", "model.layers.1.categorical.C.values", [1, 0]),
        ("hierarchy", "schema", "bad"),
        ("hierarchy", "model.metadata.name", "wrong-name"),
    ],
)
def test_reject_unsupported_source_semantics(tmp_path: Path, source: str, field: str, value) -> None:
    # Given an invalid raw namespace, role, label, tensor axis, or message contract.
    def mutate(data):
        current = data
        keys = field.split(".")
        for key in keys[:-1]:
            current = current[int(key)] if isinstance(current, list) else current[key]
        current[keys[-1]] = value

    path = _mutate(tmp_path, MOVING_ASSET if source == "moving" else HIERARCHY_ASSET, mutate)
    # When the real parser is called.
    with pytest.raises(GNNAdmissionError):
        _load(path)
    # Then no generated cache is created.
    assert not path.with_suffix(".ir.json").exists()


def test_cache_consistent_digest_still_rejects_payload_change(tmp_path: Path) -> None:
    # Given a cache attacker that updates the self digest too.
    from fedference.gnn.admission import loads
    from fedference.gnn.structured import StructuredProgram

    source = tmp_path / MOVING_ASSET.name
    source.write_bytes(MOVING_ASSET.read_bytes())
    cache = json.loads(MOVING_ASSET.with_suffix(".ir.json").read_text())
    cache["model"]["threat"]["D"]["values"] = [0.8, 0.2]
    forged = StructuredProgram(loads(json.dumps(cache["model"])), json.dumps(cache["native"], sort_keys=True))
    cache["digest"] = forged.digest
    source.with_suffix(".ir.json").write_text(json.dumps(cache))
    # When offline integrity checks bind the full source payload.
    with pytest.raises(GNNAdmissionError, match="source payload"):
        cached_structured(source)
    # Then unchanged canonical source remains readable.
    assert source.read_bytes() == MOVING_ASSET.read_bytes()


def test_real_parser_closure_tamper_and_cleanup(tmp_path: Path) -> None:
    # Given a copy of the exact parser closure with one modified file.
    import shutil
    import tempfile

    from fedference.gnn.source import EXTENSION_MANIFEST

    manifest = json.loads(EXTENSION_MANIFEST.read_text())
    upstream = Path(os.environ["GNN_ROOT"])
    root = tmp_path / "parser"
    for relative in manifest["files"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / relative, destination)
    target = root / "src/gnn/parsers/markdown_parser.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    temporary_root = Path(tempfile.gettempdir())
    before = sorted(p.name for p in temporary_root.glob("fedference-extensions-*"))
    # When the genuine extraction boundary checks its closure before execution.
    with pytest.raises(GNNAdmissionError, match="GNN_PARSER_DRIFT"):
        load_structured(
            MOVING_ASSET, upstream_root=root, dependency_root=Path(os.environ["GNN_PARSER_DEPENDENCIES"])
        )
    # Then temporary parser snapshots are cleaned and upstream was not changed.
    assert sorted(p.name for p in temporary_root.glob("fedference-extensions-*")) == before
    assert (upstream / "src/gnn/parsers/markdown_parser.py").read_bytes() != target.read_bytes()


@pytest.mark.parametrize(
    "parameters",
    [MovingParameters(0, 1), MovingParameters(4, 0), MovingParameters(4, 5), MovingParameters(4, 2, 0)],
)
def test_invalid_parameterization(parameters: MovingParameters) -> None:
    # Given invalid explicit dimensions.
    program = cached_structured(MOVING_ASSET)
    # When parameterization crosses admission.
    with pytest.raises(GNNAdmissionError):
        parameterize_moving(program, parameters)
    # Then the immutable source program is unchanged.
    assert program == cached_structured(MOVING_ASSET)


def test_unsupported_family_and_depth() -> None:
    # Given valid models of the other family.
    moving, hierarchy = cached_structured(MOVING_ASSET), cached_structured(HIERARCHY_ASSET)
    # When a mismatched operation or unsupported depth is requested.
    for operation in (
        lambda: hierarchy_world(moving),
        lambda: hierarchy_depth(moving, 2),
        lambda: hierarchy_depth(hierarchy, 1),
        lambda: moving_world(hierarchy),
        lambda: parameterize_moving(hierarchy, MovingParameters()),
        lambda: structured_tasks(moving),
    ):
        with pytest.raises(GNNAdmissionError):
            operation()
    # Then the supported native family remains available.
    require_structured_backend(hierarchy, "native")


def test_hierarchy_explicit_parameters_and_strict_prior_boundary() -> None:
    # Given exact caller priors and conditional gates.
    from fedference.gnn.structured import HierarchyParameters, parameterize_hierarchy

    source = cached_structured(HIERARCHY_ASSET)
    parameters = HierarchyParameters(
        depth=2,
        acuity=0.75,
        goal_bonus=1.0,
        priors=((0.7, 0.3), None),
        conditioned_priors=(((1 / 9,) * 9, (1 / 9,) * 9),),
    )
    # When parameters become a fully digest-bound admitted IR.
    program = parameterize_hierarchy(source, parameters)
    world = hierarchy_world(program)
    # Then supplied values participate and rounded invalid GNN PMFs are refused.
    np.testing.assert_array_equal(world["level_priors"][0], [0.7, 0.3])
    np.testing.assert_allclose(np.diag(world["L1"]["A"][0]), 0.75)
    assert program.digest != source.digest
    for invalid in (
        HierarchyParameters(depth=2, priors=((0.5, 0.5),)),
        HierarchyParameters(depth=2, conditioned_priors=()),
        HierarchyParameters(depth=2, priors=((0.5, 0.500001), None)),
    ):
        with pytest.raises(GNNAdmissionError):
            parameterize_hierarchy(source, invalid)


def test_seeded_native_moving_trajectory() -> None:
    # Given the six-step seed-1943 native trajectory captured before factory routing.
    from fedference.belief_updating import infer_states
    from fedference.pomdp import efe_policy_select

    world = moving_world(cached_structured(MOVING_ASSET))
    qs = [world["D"].copy() for _ in range(2)]
    rng = np.random.default_rng(1943)
    observed_actions = []
    # When the real native policy and inference functions consume the GNN arrays.
    for _ in range(6):
        actions = efe_policy_select(qs, world)
        observed_actions.append(actions)
        for i, action in enumerate(actions):
            p = int(np.argmax(world["B"][:, world["agent_positions"][i], action]))
            world["agent_positions"][i] = p
            a = world["observation_map"][:, :, p]
            o = int(rng.choice(2, p=a[:, 1]))
            qs[i] = infer_states([a], o, np.log(qs[i]))
    # Then the full action sequence and final posterior equal the captured original.
    assert observed_actions == [[0, 0], [0, 1], [0, 2], [0, 1], [0, 0], [0, 0]]
    np.testing.assert_array_equal(
        qs, [[0.7397982231591628, 0.2602017768408372], [0.027027027027027032, 0.972972972972973]]
    )
    print({"actions": observed_actions, "posteriors": [q.tolist() for q in qs]})


def test_moving_policy_consumes_source_observation_map(tmp_path: Path) -> None:
    # Given a source-defined perfect sensor at position 1 only.
    from fedference.pomdp import efe_policy_select

    def mutate(data):
        values = data["model"]["observation_map"]["values"]
        values[1], values[5], values[9], values[13] = 1.0, 0.0, 0.0, 1.0

    path = _mutate(tmp_path, MOVING_ASSET, mutate)
    world = moving_world(_load(path))
    # When the actual native policy selects a position from uninformative priors.
    actions = efe_policy_select([world["D"], world["D"]], world)
    # Then moving right to the source-defined perfect sensor wins over staying.
    assert actions[0] == 2


@pytest.mark.parametrize("changed", ["controls", "prior"])
def test_resize_refuses_discarding_non_template_source(tmp_path: Path, changed: str) -> None:
    def mutate(data):
        if changed == "controls":
            b = np.repeat(np.eye(4)[:, :, None], 3, axis=2)
            data["model"]["position_transition"]["values"] = b.ravel().tolist()
        else:
            data["model"]["position_prior"]["values"] = [1.0, 0.0, 0.0, 0.0]

    program = _load(_mutate(tmp_path, MOVING_ASSET, mutate))
    unchanged = parameterize_moving(program, MovingParameters())
    np.testing.assert_array_equal(moving_world(unchanged)["B"], moving_world(program)["B"])
    np.testing.assert_array_equal(
        moving_world(unchanged)["position_prior"], moving_world(program)["position_prior"]
    )
    with pytest.raises(GNNAdmissionError, match="GNN_PARAMETERIZATION"):
        parameterize_moving(program, MovingParameters(n_positions=5))


def _write_bound_cache(source: Path, cache: dict) -> None:
    import hashlib

    from fedference.gnn.admission import loads
    from fedference.gnn.structured import StructuredProgram

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    cache["model"]["metadata"]["provenance"].update(source=digest, asset=digest)
    program = StructuredProgram(
        loads(json.dumps(cache["model"])), json.dumps(cache["native"], sort_keys=True)
    )
    cache["digest"] = program.digest
    source.with_suffix(".ir.json").write_text(json.dumps(cache))


@pytest.mark.parametrize("changed", ["boolean", "name", "duplicate"])
def test_cache_rechecks_typed_source_and_headers(tmp_path: Path, changed: str) -> None:
    prefix, body = MOVING_ASSET.read_text().split("## FedferenceStructured\n")
    data = json.loads(body)
    if changed == "boolean":
        values = data["model"]["position_transition"]["values"]
        values[values.index(1.0)] = True
    elif changed == "name":
        prefix = prefix.replace("## ModelName\n", "## ModelName\nchanged-")
    else:
        prefix += "## ModelName\nduplicate\n\n"
    source = tmp_path / MOVING_ASSET.name
    source.write_text(prefix + "## FedferenceStructured\n" + json.dumps(data))
    cache = json.loads(MOVING_ASSET.with_suffix(".ir.json").read_text())
    _write_bound_cache(source, cache)
    with pytest.raises(GNNAdmissionError, match="GNN_NUMBER|GNN_EXTRACTION_MISMATCH|GNN_SECTIONS"):
        cached_structured(source)


def test_cache_accepts_admitted_integer_probabilities(tmp_path: Path) -> None:
    prefix, body = MOVING_ASSET.read_text().split("## FedferenceStructured\n")
    data = json.loads(body)
    values = data["model"]["position_transition"]["values"]
    data["model"]["position_transition"]["values"] = [int(value) for value in values]
    source = tmp_path / MOVING_ASSET.name
    source.write_text(prefix + "## FedferenceStructured\n" + json.dumps(data))
    cache = json.loads(MOVING_ASSET.with_suffix(".ir.json").read_text())
    _write_bound_cache(source, cache)
    assert cached_structured(source).model.position_transition.values == tuple(values)


@pytest.mark.parametrize("override", [False, True])
def test_public_hierarchy_factories_consume_source_leaf(tmp_path: Path, override: bool) -> None:
    import shutil
    import subprocess
    import sys

    from fedference.gnn.admission import dumps

    def mutate(data):
        c = data["model"]["layers"][-1]["categorical"]
        b = np.zeros((9, 9, 3))
        b[0, :, :] = 1
        a = np.full((9, 9), 0.25 / 8)
        np.fill_diagonal(a, 0.75)
        c["A"]["values"] = a.ravel().tolist()
        c["B"]["values"] = b.ravel().tolist()
        c["C"]["values"] = [0, 0, 0, 0, 0, 7, 0, 0, 0]
        c["D"]["values"] = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    source = _mutate(tmp_path, HIERARCHY_ASSET, mutate)
    program = _load(source)
    package = tmp_path / "staged" / "fedference"
    shutil.copytree(
        HIERARCHY_ASSET.parents[1], package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    asset = package / "models" / HIERARCHY_ASSET.name
    asset.write_bytes(source.read_bytes())
    asset.with_suffix(".ir.json").write_text(json.dumps({
        "model": json.loads(dumps(program.model)),
        "native": json.loads(program.native_json),
        "digest": program.digest,
    }))
    code = """
import json,sys
from pathlib import Path
import fedference
from fedference.pomdp import build_hierarchical_world,build_3level_world
assert Path(fedference.__file__).resolve().is_relative_to(Path.cwd()/"staged")
kwargs={"acuity":0.6,"goal_bonus":3.0} if sys.argv[1]=="yes" else {}
worlds=[build_hierarchical_world(**kwargs),build_3level_world(**kwargs)]
print(json.dumps([{**{k:[v.tolist() for v in w["L1"][k]] for k in ("A","B","C","D")},
    "leaf_prior":w["level_priors"][-1].tolist() if "level_priors" in w else None} for w in worlds]))
"""
    env = {**os.environ, "PYTHONPATH": str(package.parent)}
    result = subprocess.run([sys.executable, "-c", code, "yes" if override else "no"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    worlds = json.loads(result.stdout)
    for world in worlds:
        np.testing.assert_array_equal(np.argmax(world["B"][0], axis=0), np.zeros((9, 3)))
        np.testing.assert_allclose(np.diag(world["A"][0]), 0.6 if override else 0.75)
        np.testing.assert_array_equal(
            np.array(world["C"][0]).ravel(), [0, 0, 0, 0, 0, 3 if override else 7, 0, 0, 0]
        )
        np.testing.assert_array_equal(np.array(world["D"][0]).ravel(), [1, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(worlds[1]["leaf_prior"], [1, 0, 0, 0, 0, 0, 0, 0, 0])
