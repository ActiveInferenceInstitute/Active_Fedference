"""Real canonical IR behavior; float64 admission atol=1e-12, rtol=0."""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from fedference.gnn.admission import GNNAdmissionError, admit, content_digest, dumps, loads
from fedference.gnn.types import HybridGaussianModel, SiteFactorModel


def tensor(axes, shape, values):
    return dict(axes=axes.split(), shape=shape, values=values)


def categorical():
    return dict(
        states=["safe", "threat"],
        observations=["quiet", "alarm"],
        actions=["stay"],
        A=tensor("observation state", [2, 2], [0.9, 0.2, 0.1, 0.8]),
        B=tensor("next_state previous_state action", [2, 2, 1], [1.0, 0.0, 0.0, 1.0]),
        C=tensor("observation", [2], [0.0, -2.0]),
        D=tensor("state", [2], [0.6, 0.4]),
    )


def fixture(kind="categorical"):
    data = dict(
        kind=kind,
        metadata=dict(
            name="tiny",
            schema="fedference.gnn/1",
            precision="float64-atol-1e-12-rtol-0",
            capabilities=[f"fedference.gnn/{kind}/1"],
            schedule=["site-add"] if kind == "site-factor" else ["predict", "observe"],
            provenance=dict(
                source="1" * 64, asset="2" * 64, parser="3" * 64, backend=None, native_result=None
            ),
        ),
    )
    if kind == "categorical":
        data["categorical"] = categorical()
    elif kind == "moving-observer":
        data.update(
            threat=categorical(),
            positions=["left", "right"],
            position_transition=tensor(
                "next_position previous_position action", [2, 2, 1], [0.0, 1.0, 1.0, 0.0]
            ),
            position_prior=tensor("position", [2], [1.0, 0.0]),
            observation_map=tensor(
                "observation threat position", [2, 2, 2], [0.9, 0.7, 0.2, 0.4, 0.1, 0.3, 0.8, 0.6]
            ),
        )
    elif kind == "hierarchical":
        data["layers"] = [dict(name="root", categorical=categorical(), parent=None, conditioned_prior=None)]
        for i in range(1, 4):
            data["layers"].append(
                dict(
                    name=f"level{i}",
                    categorical=categorical(),
                    parent="root" if i == 1 else f"level{i - 1}",
                    conditioned_prior=tensor("child_state parent_state", [2, 2], [0.8, 0.1, 0.2, 0.9]),
                )
            )
    else:
        data.update(coordinates=["x", "y"], units=["m", "m"])
        if kind == "hybrid-gaussian":
            data.update(
                categorical=categorical(),
                mean=tensor("coordinate", [2], [2.0, -1.0]),
                covariance=tensor("coordinate coordinate_dual", [2, 2], [2.0, 0.5, 0.5, 1.0]),
            )
        else:
            data.update(
                protocol="gaussian-natural-additive/1",
                precision_increment=tensor("coordinate coordinate_dual", [2, 2], [-0.5, 0.0, 0.0, 0.25]),
                information_increment=tensor("coordinate", [2], [-1.0, 2.0]),
            )
    return data


@pytest.mark.parametrize(
    "kind", ["categorical", "moving-observer", "hierarchical", "hybrid-gaussian", "site-factor"]
)
def test_typed_family_roundtrip_and_labels(kind):
    data = fixture(kind)
    before = copy.deepcopy(data)
    model = admit(data)
    assert json.loads(dumps(model)) == before
    assert loads(dumps(model)) == model
    assert model.kind == kind
    with pytest.raises(FrozenInstanceError):
        model.metadata.name = "mutated"
    data["metadata"]["name"] = "mutated"
    assert model.metadata.name == "tiny"


def test_shape_axis_duplicate_and_unsupported_kind():
    data = fixture()
    data["categorical"]["A"]["shape"] = [1, 4]
    with pytest.raises(GNNAdmissionError, match="GNN_SHAPE"):
        admit(data)
    data = fixture()
    data["categorical"]["B"]["axes"] = ["previous_state", "next_state", "action"]
    with pytest.raises(GNNAdmissionError, match="GNN_AXIS"):
        admit(data)
    with pytest.raises(GNNAdmissionError, match="GNN_DUPLICATE"):
        loads('{"kind":"categorical","kind":"site-factor"}')
    data["kind"] = "unknown"
    with pytest.raises(GNNAdmissionError, match="GNN_KIND"):
        admit(data)


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("nan", "NONFINITE"),
        ("inf", "NONFINITE"),
        ("negative", "PMF"),
        ("sum", "PMF"),
        ("empty", "LABEL"),
        ("duplicate", "DUPLICATE"),
        ("missing", "FIELDS"),
        ("extra", "FIELDS"),
        ("bool", "NUMBER"),
        ("ragged", "NUMBER"),
        ("shape-bool", "SHAPE"),
        ("overflow", "NONFINITE"),
        ("schema", "SCHEMA"),
        ("precision", "PRECISION"),
        ("capability", "CAPABILITY"),
        ("schedule", "SCHEDULE"),
        ("digest", "DIGEST"),
        ("c-nan", "NONFINITE"),
    ],
)
def test_malformed_admission(mutation, code):
    data = fixture()
    c = data["categorical"]
    if mutation == "nan":
        c["A"]["values"][0] = float("nan")
    elif mutation == "inf":
        c["A"]["values"][0] = float("inf")
    elif mutation == "negative":
        c["D"]["values"] = [-0.1, 1.1]
    elif mutation == "sum":
        c["D"]["values"] = [0.4, 0.4]
    elif mutation == "empty":
        c["states"] = []
    elif mutation == "duplicate":
        c["states"] = ["same", "same"]
    elif mutation == "missing":
        del c["D"]
    elif mutation == "extra":
        c["ignored"] = 1
    elif mutation == "bool":
        c["A"]["values"][0] = True
    elif mutation == "ragged":
        c["A"]["values"][0] = [0.9]
    elif mutation == "shape-bool":
        c["B"]["shape"][-1] = True
    elif mutation == "overflow":
        c["A"]["values"][0] = 10**400
    elif mutation == "schema":
        data["metadata"]["schema"] = "2"
    elif mutation == "precision":
        data["metadata"]["precision"] = "float32"
    elif mutation == "capability":
        data["metadata"]["capabilities"] += ["unknown/1"]
    elif mutation == "schedule":
        data["metadata"]["schedule"].reverse()
    elif mutation == "digest":
        data["metadata"]["provenance"]["source"] = "bad"
    elif mutation == "c-nan":
        c["C"]["values"][0] = float("nan")
    with pytest.raises(GNNAdmissionError) as exc:
        admit(data)
    assert exc.value.code == "GNN_" + code


def test_valid_mutation_changes_content_digest():
    data = fixture()
    original = admit(data)
    data["categorical"]["A"]["values"] = [0.7, 0.2, 0.3, 0.8]
    changed = admit(data)
    assert changed.categorical.A.values == (0.7, 0.2, 0.3, 0.8)
    assert content_digest(original) != content_digest(changed)
    assert original.categorical.A.values == (0.9, 0.2, 0.1, 0.8)


def test_covariance_and_signed_site_recovery():
    hybrid = admit(fixture("hybrid-gaussian"))
    site = admit(fixture("site-factor"))
    assert isinstance(hybrid, HybridGaussianModel)
    assert isinstance(site, SiteFactorModel)
    covariance = np.array(hybrid.covariance.values).reshape(2, 2)
    np.testing.assert_allclose(
        np.linalg.inv(covariance), [[4 / 7, -2 / 7], [-2 / 7, 8 / 7]], atol=1e-12, rtol=0
    )
    precision = np.eye(2) + np.array(site.precision_increment.values).reshape(2, 2)
    np.testing.assert_allclose(
        np.linalg.solve(precision, site.information_increment.values), [-2.0, 1.6], atol=1e-12, rtol=0
    )
    assert site.precision_increment.values[0] == -0.5


@pytest.mark.parametrize(
    "kind,mutation,code",
    [
        ("hybrid-gaussian", "singular", "COVARIANCE"),
        ("hybrid-gaussian", "asymmetric", "SYMMETRY"),
        ("hybrid-gaussian", "units", "UNIT"),
        ("site-factor", "protocol", "PROTOCOL"),
        ("site-factor", "asymmetric", "SYMMETRY"),
        ("hierarchical", "parent", "PARENT"),
        ("hierarchical", "duplicate", "DUPLICATE"),
        ("hierarchical", "empty", "LAYERS"),
        ("moving-observer", "axis", "AXIS"),
    ],
)
def test_family_refusals(kind, mutation, code):
    data = fixture(kind)
    if mutation == "singular":
        data["covariance"]["values"] = [1.0, 1.0, 1.0, 1.0]
    elif mutation == "asymmetric":
        data["covariance" if kind == "hybrid-gaussian" else "precision_increment"]["values"][1] = 0.3
    elif mutation == "units":
        data["units"][0] = "unknown"
    elif mutation == "protocol":
        data["protocol"] = "cavity-pmf"
    elif mutation == "parent":
        data["layers"][1]["parent"] = "level3"
    elif mutation == "duplicate":
        data["layers"][1]["name"] = "root"
    elif mutation == "empty":
        data["layers"] = []
    elif mutation == "axis":
        data["position_transition"]["axes"][0] = "next_state"
    with pytest.raises(GNNAdmissionError) as exc:
        admit(data)
    assert exc.value.code == "GNN_" + code


def test_json_and_nested_duplicate_rejection():
    with pytest.raises(GNNAdmissionError, match="GNN_JSON"):
        loads("{")
    with pytest.raises(GNNAdmissionError, match="GNN_DUPLICATE"):
        loads('{"metadata":{"name":"one","name":"two"}}')


def test_oversized_json_integer_has_stable_refusal():
    with pytest.raises(GNNAdmissionError) as oversized:
        loads("{\"kind\":" + "1" * 5000 + "}")
    assert oversized.value.code == "GNN_JSON"
    assert oversized.value.location == "document"
    with pytest.raises(GNNAdmissionError) as duplicate:
        loads("{\"kind\":\"categorical\",\"kind\":\"site-factor\"}")
    assert duplicate.value.code == "GNN_DUPLICATE"
