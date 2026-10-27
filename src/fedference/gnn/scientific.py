"""Explicit native Gaussian and signed-site GNN capabilities.

Friston et al. (2024) motivates belief sharing; Mildner et al. (2025)
motivates cavity/site replacement. These adapters prove bounded arithmetic and
execution contracts, not replication of either complete scientific protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..bnn_fedgvi import DiagonalGaussian, FedGVIServerState, GaussianSiteFactor
from ..hybrid_tracking import HybridTrackingConfig, run_hybrid_tracking
from ..protocol_parity import ParityRow, ProtocolParityMatrix
from .admission import GNNAdmissionError, dumps, loads
from .loader import MODEL_DIR, _json, _mapping, _sha
from .source import ParsedExtensions, parse_extensions
from .types import HybridGaussianModel, JSONValue, Model, SiteFactorModel

HYBRID_ASSET: Final = MODEL_DIR / "hybrid-tracking.gnn.md"
SITES_ASSET: Final = MODEL_DIR / "fedgvi-sites.gnn.md"


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


@dataclass(frozen=True, slots=True)
class GaussianPosterior:
    """Multivariate conditioning result in the model coordinate units."""

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class SiteRun:
    """Real cavity/replacement states and bounded protocol parity."""

    initial: FedGVIServerState
    cavity: DiagonalGaussian
    updated: FedGVIServerState
    parity: ProtocolParityMatrix
    model_digest: str


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


def gaussian_condition(
    model: HybridGaussianModel,
    observation: tuple[float, ...],
    observation_covariance: tuple[tuple[float, ...], ...],
) -> GaussianPosterior:
    """Condition a full Gaussian on a same-coordinate noisy observation.

    This is a conjugate arithmetic operation, separate from the limited native
    tracking configuration adapter. Every covariance entry participates.
    """
    dumps(model)
    n = len(model.coordinates)
    y = _array(list(observation), (n,), "observation")
    r = _array([list(row) for row in observation_covariance], (n, n), "observation_covariance")
    if not np.array_equal(r, r.T):
        raise GNNAdmissionError("GNN_SYMMETRY", "observation covariance")
    try:
        np.linalg.cholesky(r)
    except np.linalg.LinAlgError as error:
        raise GNNAdmissionError("GNN_COVARIANCE", "observation covariance") from error
    mean = np.array(model.mean.values)
    covariance = np.array(model.covariance.values).reshape(n, n)
    try:
        with np.errstate(over="raise", invalid="raise"):
            scale = max(float(np.max(np.abs(covariance))), float(np.max(np.abs(r))))
            scaled_prior, scaled_noise = covariance / scale, r / scale
            innovation = scaled_prior + scaled_noise
            gain = np.linalg.solve(innovation, scaled_prior).T
            prior_gain = np.linalg.solve(innovation, scaled_noise).T
            updated = prior_gain @ mean + gain @ y
            posterior = prior_gain @ covariance @ prior_gain.T + gain @ r @ gain.T
            posterior = posterior * 0.5 + posterior.T * 0.5
        if not np.isfinite(updated).all() or not np.isfinite(posterior).all():
            raise GNNAdmissionError("GNN_COVARIANCE", "unrepresentable posterior")
        np.linalg.cholesky(posterior)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise GNNAdmissionError("GNN_COVARIANCE", "posterior arithmetic") from error
    return GaussianPosterior(
        tuple(float(value) for value in updated),
        tuple(tuple(float(value) for value in row) for row in posterior),
    )


def tracking_config(program: ScientificProgram) -> HybridTrackingConfig:
    """Bind target/noise/cue inputs; refuse unsupported dynamical semantics."""
    native = _native(program)
    model = program.model
    if not isinstance(model, HybridGaussianModel):
        raise GNNAdmissionError("GNN_KIND", "tracking")
    if model.coordinates != ("target_position", "velocity_observation_bias") or model.units != ("m", "m/s"):
        raise GNNAdmissionError("GNN_UNIT", "tracking coordinates")
    covariance = np.array(model.covariance.values).reshape(2, 2)
    if covariance[0, 1] != 0 or covariance[1, 0] != 0 or model.mean.values[1] != 0:
        raise GNNAdmissionError("GNN_CAPABILITY", "correlation or velocity bias")
    c = model.categorical
    a = np.array(c.A.values).reshape(c.A.shape)
    if (
        c.states != ("constant-velocity", "accelerating")
        or c.observations != ("constant-cue", "accelerating-cue")
        or c.actions != ("native-controller",)
        or c.B.values != (1.0, 0.0, 0.0, 1.0)
        or c.C.values != (0.0, 0.0)
        or c.D.values != (0.5, 0.5)
        or a.shape != (2, 2)
        or not np.array_equal(a, a[::-1, ::-1])
    ):
        raise GNNAdmissionError("GNN_CAPABILITY", "tracking categorical protocol")
    settings = _mapping(native["tracking"], "tracking settings")
    derived = {"target_position", "position_observation_var", "velocity_observation_var", "context_accuracy"}
    expected = {field.name for field in fields(HybridTrackingConfig)} - derived
    if set(settings) != expected:
        raise GNNAdmissionError("GNN_FIELDS", "tracking settings")
    numbers: dict[str, float | int] = {}
    for key, value in settings.items():
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise GNNAdmissionError("GNN_NUMBER", key)
        numbers[key] = value
    if numbers["robustness"] != 0 or numbers["outlier_bias"] != 0:
        raise GNNAdmissionError("GNN_CAPABILITY", "clean naive tracking protocol")
    return HybridTrackingConfig(
        n_agents=_integer(numbers["n_agents"], "n_agents"),
        horizon=_integer(numbers["horizon"], "horizon"),
        target_position=model.mean.values[0],
        position_observation_var=float(covariance[0, 0]),
        velocity_observation_var=float(covariance[1, 1]),
        context_accuracy=float(a[0, 0]),
        process_var=numbers["process_var"],
        action_gain=numbers["action_gain"],
        max_acceleration=numbers["max_acceleration"],
        robustness=numbers["robustness"],
        outlier_bias=numbers["outlier_bias"],
    )


def _integer(value: float | int, name: str) -> int:
    if type(value) is not int:
        raise GNNAdmissionError("GNN_NUMBER", name)
    return int(value)


def run_tracking(program: ScientificProgram, *, seed: int) -> dict[str, JSONValue]:
    """Run the real native tracking consumer; seed is the explicit caller input."""
    config = tracking_config(program)
    result = _mapping(
        _json(
            json.dumps(
                run_hybrid_tracking(
                    seed,
                    method="naive",
                    contaminate_one_agent=False,
                    config=config,
                ),
                allow_nan=False,
            )
        ),
        "tracking result",
    )
    return {"model_digest": program.digest, "config": _json(json.dumps(asdict(config))), "result": result}


def run_sites(program: ScientificProgram) -> SiteRun:
    """Run signed diagonal prior × sites, proper cavity, and factor replacement."""
    native = _native(program)
    model = program.model
    if not isinstance(model, SiteFactorModel):
        raise GNNAdmissionError("GNN_KIND", "site runner")
    n = len(model.coordinates)
    sites = []
    for factor in (model, *program.peers):
        precision = np.array(factor.precision_increment.values).reshape(n, n)
        if not np.array_equal(precision, np.diag(np.diag(precision))):
            raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal site")
        sites.append(GaussianSiteFactor(np.array(factor.information_increment.values), np.diag(precision)))
    prior_mean = _array(native["prior_mean"], (n,), "prior mean")
    prior_covariance = _array(native["prior_covariance"], (n, n), "prior covariance")
    if not np.array_equal(prior_covariance, np.diag(np.diag(prior_covariance))):
        raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal prior")
    prior = DiagonalGaussian(prior_mean, np.diag(prior_covariance))
    client_means = _array(native["client_means"], (len(sites), n), "client means")
    client_covariances = _array(native["client_covariances"], (len(sites), n, n), "client covariances")
    clients = {}
    for index, covariance in enumerate(client_covariances):
        if not np.array_equal(covariance, np.diag(np.diag(covariance))):
            raise GNNAdmissionError("GNN_CAPABILITY", "non-diagonal client")
        clients[index] = DiagonalGaussian(client_means[index], np.diag(covariance))
    state = FedGVIServerState(prior, tuple(sites))
    cavity = state.cavity(0)
    updated = state.advance_round(clients, schedule="parallel")
    parity = ProtocolParityMatrix(
        "gnn-diagonal-site-smoke",
        "fedgvi-mildner-2025",
        "native signed-site arithmetic",
        (
            ParityRow(
                "server update",
                "cavity and site replacement",
                "cavity and site replacement",
                "matched",
                "fedference.bnn_fedgvi.FedGVIServerState",
            ),
            ParityRow(
                "client objective and optimization",
                "source loss and schedule",
                "declared Gaussian client posterior; no optimization",
                "unresolved",
                "GNN native client parameters; arithmetic smoke only",
            ),
        ),
    )
    return SiteRun(state, cavity, updated, parity, program.digest)


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
