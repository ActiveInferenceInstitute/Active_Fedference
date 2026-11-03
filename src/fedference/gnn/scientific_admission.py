"""Admit source-bound scientific family settings before native execution.

Friston et al. (2024) and Mildner et al. (2025) motivate these bounded adapters;
admission does not establish complete scientific protocol replication.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ._documents import MODEL_DIR, _json, _mapping, _sha
from .admission import GNNAdmissionError, dumps, loads
from .source import ParsedExtensions, parse_extensions
from .types import HybridGaussianModel, JSONValue, Model, SiteFactorModel


@dataclass(frozen=True, slots=True)
class ScientificProgram:
    """Admitted family plus immutable source-bound native settings."""

    model: HybridGaussianModel | SiteFactorModel
    native_json: str
    peers: tuple[SiteFactorModel, ...] = ()

    @property
    def digest(self) -> str:
        """Bind model, parser, and full source including native settings."""
        return _sha(
            (dumps(self.model) + self.native_json + "".join(dumps(peer) for peer in self.peers)).encode()
        )



HYBRID_ASSET: Final = MODEL_DIR / "hybrid-tracking.gnn.md"



SITES_ASSET: Final = MODEL_DIR / "fedgvi-sites.gnn.md"



def load_scientific(
    path: Path, *, upstream_root: Path, python: Path | None = None, dependency_root: Path | None = None
) -> ScientificProgram:
    """Admit the payload preserved by the actual pinned Markdown parser."""
    parsed = parse_extensions(
        path,
        upstream_root=upstream_root,
        allowed_extensions=("FedferenceScientific",),
        python=python,
        dependency_root=dependency_root,
    )
    sections = dict(parsed.sections)
    if (
        set(sections)
        != {"GNNSection", "GNNVersionAndFlags", "ModelName", "ModelAnnotation", "FedferenceScientific"}
        or sections["GNNSection"] != "ActInfPOMDP"
        or sections["GNNVersionAndFlags"] != "GNN v1.0"
    ):
        raise GNNAdmissionError("GNN_SECTIONS", "scientific document")
    payload = _mapping(_json(parsed.extension("FedferenceScientific")), "scientific")
    if (
        set(payload) != {"schema", "model", "native", "peers"}
        or payload["schema"] != "fedference.scientific/1"
    ):
        raise GNNAdmissionError("GNN_SCHEMA", "scientific")
    raw = _mapping(payload["model"], "model")
    metadata = _mapping(raw.get("metadata"), "metadata")
    if metadata.get("name") != sections["ModelName"]:
        raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", "ModelName")
    model = _bound_model(raw, parsed)
    if not isinstance(model, (HybridGaussianModel, SiteFactorModel)):
        raise GNNAdmissionError("GNN_KIND", "scientific")
    raw_peers = payload["peers"]
    if not isinstance(raw_peers, list):
        raise GNNAdmissionError("GNN_FIELDS", "peers")
    peers = []
    for raw_peer in raw_peers:
        peer = _bound_model(_mapping(raw_peer, "peer"), parsed)
        if not isinstance(peer, SiteFactorModel):
            raise GNNAdmissionError("GNN_KIND", "peer")
        peers.append(peer)
    native = _mapping(payload["native"], "native")
    program = ScientificProgram(model, json.dumps(native, sort_keys=True, allow_nan=False), tuple(peers))
    _native(program)
    return program



def _bound_model(raw: dict[str, JSONValue], parsed: ParsedExtensions) -> Model:
    """Admit each canonical primary/peer model with the same exact source binding."""
    metadata = _mapping(raw.get("metadata"), "metadata")
    if "provenance" in metadata:
        raise GNNAdmissionError("GNN_FIELDS", "source-declared provenance")
    bound = dict(raw)
    bound["metadata"] = {
        **metadata,
        "provenance": {
            "source": parsed.source_digest,
            "asset": parsed.source_digest,
            "parser": parsed.parser_digest,
            "backend": None,
            "native_result": None,
        },
    }
    return loads(json.dumps(bound, allow_nan=False))



def _native(program: ScientificProgram) -> dict[str, JSONValue]:
    # Re-admission protects public dataclass construction and parameterization.
    dumps(program.model)
    native = _mapping(_json(program.native_json), "native")
    if isinstance(program.model, HybridGaussianModel):
        if program.peers:
            raise GNNAdmissionError("GNN_CAPABILITY", "tracking peers")
        expected = {"capability", "field_roles", "tracking"}
        roles = {
            "mean": "target-position-and-velocity-observation-bias",
            "covariance": "position-and-velocity-observation-noise",
        }
        if set(native) != expected or native.get("capability") != "native-hybrid-tracking/1":
            raise GNNAdmissionError("GNN_CAPABILITY", "hybrid native")
        if native["field_roles"] != roles:
            raise GNNAdmissionError("GNN_UNIT", "hybrid field roles")
    else:
        if not program.peers:
            raise GNNAdmissionError("GNN_CAPABILITY", "distributed sites require a peer")
        for peer in program.peers:
            dumps(peer)
            if (peer.coordinates, peer.units, peer.protocol) != (
                program.model.coordinates,
                program.model.units,
                program.model.protocol,
            ):
                raise GNNAdmissionError("GNN_UNIT", "peer coordinates/units/protocol")
        expected = {
            "capability",
            "representation",
            "coordinates",
            "units",
            "prior_mean",
            "prior_covariance",
            "client_means",
            "client_covariances",
            "schedule",
        }
        if set(native) != expected or native.get("capability") != "native-diagonal-site/1":
            raise GNNAdmissionError("GNN_CAPABILITY", "site native")
        if native["representation"] != "signed-natural":
            raise GNNAdmissionError("GNN_SITE_AS_PMF", "representation")
        if native["schedule"] != "parallel":
            raise GNNAdmissionError("GNN_SCHEDULE", "native.schedule")
        if native["coordinates"] != list(program.model.coordinates) or native["units"] != list(
            program.model.units
        ):
            raise GNNAdmissionError("GNN_UNIT", "prior/site coordinates")
    return native



def _array(value: JSONValue, shape: tuple[int, ...], location: str) -> NDArray[np.float64]:
    def numeric(item: JSONValue) -> bool:
        return all(numeric(child) for child in item) if isinstance(item, list) else type(item) in (int, float)

    if not numeric(value):
        raise GNNAdmissionError("GNN_NUMBER", location)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise GNNAdmissionError("GNN_SHAPE", location) from error
    if result.shape != shape or not np.isfinite(result).all():
        raise GNNAdmissionError("GNN_SHAPE", location)
    return result
