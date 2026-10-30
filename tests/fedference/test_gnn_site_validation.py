"""Native site configuration admission without executing a replacement round."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from fedference.gnn import sites
from fedference.gnn.admission import GNNAdmissionError, loads
from fedference.gnn.scientific_admission import SITES_ASSET, ScientificProgram
from fedference.gnn.types import SiteFactorModel


def site_program() -> ScientificProgram:
    """Use canonical resource values with explicit unit-test source bindings."""
    payload = json.loads(SITES_ASSET.read_text().split("## FedferenceScientific\n")[1])
    models = []
    for raw in [payload["model"], *payload["peers"]]:
        raw["metadata"]["provenance"] = {
            "source": "0" * 64, "asset": "0" * 64, "parser": "0" * 64,
            "backend": None, "native_result": None,
        }
        model = loads(json.dumps(raw))
        assert isinstance(model, SiteFactorModel)
        models.append(model)
    return ScientificProgram(models[0], json.dumps(payload["native"]), tuple(models[1:]))


def test_site_configuration_preserves_signed_factors_without_advancing() -> None:
    program = site_program()
    state, clients = sites.site_configuration(program)
    assert state.round_index == 0
    assert len(clients) == 2
    np.testing.assert_array_equal(state.sites[0].precision, [-0.5, 1.0])
    np.testing.assert_allclose(state.posterior().mean, [0.0, 1 / 22])
    assert sites.run_sites(program).updated.round_index == 1


@pytest.mark.parametrize("field,value", [
    ("prior_mean", [0.0]),
    ("prior_mean", [True, 0.0]),
    ("prior_mean", [float("nan"), 0.0]),
    ("prior_covariance", [[0.0, 0.0], [0.0, 1.0]]),
    ("prior_covariance", [[1.0, 0.1], [0.1, 1.0]]),
    ("client_means", [[0.0, 0.0]]),
    ("client_covariances", [[[1.0, 0.0], [0.0, -1.0]]] * 2),
    ("client_covariances", [[[1.0, 0.1], [0.1, 1.0]]] * 2),
    ("schedule", "sequential"),
])
def test_site_configuration_rejects_invalid_native(field: str, value: object) -> None:
    program = site_program()
    native = json.loads(program.native_json)
    native[field] = value
    malformed = replace(program, native_json=json.dumps(native))
    with pytest.raises(GNNAdmissionError):
        sites.site_configuration(malformed)
    with pytest.raises(GNNAdmissionError):
        sites.run_sites(malformed)


def test_site_configuration_rejects_improper_cavity() -> None:
    program = site_program()
    factor = replace(program.model, precision_increment=replace(
        program.model.precision_increment, values=(-2.1, 0.0, 0.0, 1.0)
    ))
    with pytest.raises(GNNAdmissionError, match="GNN_SITE_CONFIGURATION"):
        sites.site_configuration(replace(program, model=factor))
