"""Source-bound signed-site native checkpoint serialization."""
from __future__ import annotations

import json

import numpy as np

from ..bnn_fedgvi import FedGVIServerState
from ._documents import _json, _mapping
from .admission import GNNAdmissionError
from .scientific_admission import ScientificProgram
from .sites import run_sites


def checkpoint(program: ScientificProgram, state: FedGVIServerState) -> str:
    """Serialize an explicit native checkpoint bound to the full GNN source."""
    run = run_sites(program)
    if (
        not np.array_equal(state.prior.mean, run.initial.prior.mean)
        or not np.array_equal(state.prior.variance, run.initial.prior.variance)
        or len(state.sites) != len(run.initial.sites)
    ):
        raise GNNAdmissionError("GNN_CHECKPOINT_BINDING", "prior/sites")
    return json.dumps(
        {
            "schema": "fedference.gnn-site-checkpoint/1",
            "model_digest": run.model_digest,
            "state": state.as_dict(),
        },
        sort_keys=True,
        allow_nan=False,
    )



def restore_checkpoint(program: ScientificProgram, document: str) -> FedGVIServerState:
    """Reject stale model binding before returning a native server state."""
    raw = _mapping(_json(document), "checkpoint")
    if set(raw) != {"schema", "model_digest", "state"} or raw["schema"] != "fedference.gnn-site-checkpoint/1":
        raise GNNAdmissionError("GNN_CHECKPOINT", "schema")
    if raw["model_digest"] != program.digest:
        raise GNNAdmissionError("GNN_CHECKPOINT_BINDING", "model_digest")
    state = FedGVIServerState.from_dict(_mapping(raw["state"], "state"))
    expected = run_sites(program).initial
    if (
        not np.array_equal(state.prior.mean, expected.prior.mean)
        or not np.array_equal(state.prior.variance, expected.prior.variance)
        or len(state.sites) != len(expected.sites)
    ):
        raise GNNAdmissionError("GNN_CHECKPOINT_BINDING", "prior/sites")
    return state
