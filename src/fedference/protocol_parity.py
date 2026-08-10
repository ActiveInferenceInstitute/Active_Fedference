"""Machine-readable source-to-project protocol parity matrices.

The matrices prevent a qualitative analogue from being relabelled as an exact
replication. ``exact replication`` is emitted only when every required row is
matched and no source value remains unresolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ParityStatus = Literal["matched", "deviation", "unresolved", "not-applicable"]


@dataclass(frozen=True)
class ParityRow:
    """One source parameter, project value, and explicit disposition."""

    field: str
    source_value: str
    project_value: str
    status: ParityStatus
    evidence: str

    def __post_init__(self) -> None:
        for name in ("field", "source_value", "project_value", "evidence"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.status not in (
            "matched",
            "deviation",
            "unresolved",
            "not-applicable",
        ):
            raise ValueError("status is not recognized")


@dataclass(frozen=True)
class ProtocolParityMatrix:
    """Versioned comparison for one named source protocol."""

    protocol_id: str
    source_id: str
    target: str
    rows: tuple[ParityRow, ...]
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for name in ("protocol_id", "source_id", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("protocol_id, source_id, and target must be non-empty")
        if not isinstance(self.rows, (tuple, list)):
            raise ValueError("parity matrix rows must be a sequence")
        object.__setattr__(self, "rows", tuple(self.rows))
        if not self.rows or any(not isinstance(row, ParityRow) for row in self.rows):
            raise ValueError("parity matrix rows must contain ParityRow values")
        fields = [row.field for row in self.rows]
        if len(set(fields)) != len(fields):
            raise ValueError("parity matrix fields must be unique")
        if self.schema_version != "1.0":
            raise ValueError("unsupported parity matrix schema")

    @property
    def exact(self) -> bool:
        """Whether every required row is source-matched."""
        relevant = [row for row in self.rows if row.status != "not-applicable"]
        return bool(relevant) and all(row.status == "matched" for row in relevant)

    @property
    def claim_label(self) -> str:
        """Reader-facing replication label implied by the matrix."""
        if self.exact:
            return "exact replication"
        if self.source_id.startswith("friston-"):
            return "paper-constrained reconstruction"
        return "source-constrained implementation"

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible comparison artifact."""
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "source_id": self.source_id,
            "target": self.target,
            "claim_label": self.claim_label,
            "exact": self.exact,
            "rows": [asdict(row) for row in self.rows],
        }


def fedgvi_bnn_parity_matrix() -> ProtocolParityMatrix:
    """Current parity disposition for the portable FedGVI BNN lane."""
    return ProtocolParityMatrix(
        protocol_id="fedgvi-bnn-fashion-mnist",
        source_id="fedgvi-source@5440352890037a81218285b8f4de81090861e9df",
        target="FashionMNIST contamination protocol",
        rows=(
            ParityRow(
                "dataset and architecture",
                "FashionMNIST; two hidden fully connected layers of width 100",
                "declared identically in source_5090; portable implementation not yet wired",
                "unresolved",
                "source run_fedgvi.py and research_registry.BNN_PROTOCOL_PROFILES",
            ),
            ParityRow(
                "variational family",
                "mean-field diagonal Gaussian BNN",
                "VariationalMLP diagonal Gaussian primitive",
                "matched",
                "src/fedference/bnn_variational_torch.py",
            ),
            ParityRow(
                "server update",
                "site factor, cavity, factor replacement",
                "diagonal-Gaussian site factor, cavity, factor replacement",
                "matched",
                "src/fedference/bnn_fedgvi.py",
            ),
            ParityRow(
                "client optimization",
                "source divergence/loss and stopping schedule",
                (
                    "cavity-conditioned MC beta-loss optimizer in synthetic pilot; "
                    "source schedule and datasets unresolved"
                ),
                "deviation",
                "src/fedference/single_machine.py; MAJ-2A active gate",
            ),
            ParityRow(
                "client split",
                "homogeneous three-client split",
                "declared identically in source_5090 and m4_confirmatory",
                "matched",
                "source fmnist_contamination.sh and run_fedgvi.py",
            ),
            ParityRow(
                "loss and client divergence",
                "AR(2.5) with GCE/RCCE 1.0 and 0.5; PVI uses KLD/NLL",
                (
                    "portable cavity-conditioned beta-loss executes; source AR/GCE/RCCE "
                    "schedule remains unresolved"
                ),
                "deviation",
                "src/fedference/single_machine.py; source fmnist_contamination.sh",
            ),
            ParityRow(
                "seed indexing",
                ("run indices 1..5 index [42, 676, 93, 215, 318, 242], yielding [676, 93, 215, 318, 242]"),
                "effective seeds and the source table/indices are recorded explicitly",
                "matched",
                "source fmnist_contamination.sh and run_fedgvi.py",
            ),
            ParityRow(
                "contamination grid",
                "0, 0.1, 0.2, 0.4, 0.6",
                "declared identically in source_5090 and m4_confirmatory",
                "matched",
                "research_registry.BNN_PROTOCOL_PROFILES",
            ),
            ParityRow(
                "server rounds",
                "25",
                "25 in source_5090 and m4_confirmatory",
                "matched",
                "research_registry.BNN_PROTOCOL_PROFILES",
            ),
            ParityRow(
                "local stopping",
                "maximum 2500 epochs with ELBO early-stopping patience 10",
                "exact in source_5090; M4 budget and stopping rule await pilot",
                "deviation",
                "portable profile is not source-scale",
            ),
            ParityRow(
                "local training budget",
                "up to 2500 epochs",
                "exact in source_5090; M4 budget locked by pilot",
                "deviation",
                "portable profile is not source-scale",
            ),
            ParityRow(
                "posterior predictive samples",
                "200",
                "200 in source_5090; M4 value awaits pilot",
                "deviation",
                "portable profile is not source-scale",
            ),
            ParityRow(
                "ELBO samples",
                "10",
                "10 in source_5090; M4 value awaits pilot",
                "deviation",
                "portable profile is not source-scale",
            ),
        ),
    )


def friston_protocol_parity_matrices() -> tuple[ProtocolParityMatrix, ...]:
    """Current explicit unknowns for Eq. 2 and source Figures 5, 7, and 9."""
    targets = ("Equation 2", "Figure 5", "Figure 7", "Figure 9")
    return tuple(
        ProtocolParityMatrix(
            protocol_id=f"friston-{target.lower().replace(' ', '-')}",
            source_id="friston-2024-belief-sharing",
            target=target,
            rows=(
                ParityRow(
                    "agent count",
                    "paper-defined",
                    "awaiting source-protocol extraction",
                    "unresolved",
                    "source table required before implementation",
                ),
                ParityRow(
                    "modalities and mappings",
                    "paper-defined A/B/C/D",
                    "awaiting complete native-scale mapping",
                    "unresolved",
                    "current categorical lane is an analogue",
                ),
                ParityRow(
                    "episode horizon",
                    "paper-defined",
                    "awaiting source-protocol extraction",
                    "unresolved",
                    "ordered trajectory points are not independent units",
                ),
                ParityRow(
                    "learning and policy schedule",
                    "paper/SPM routine-defined",
                    "Python reconstruction not yet implemented",
                    "unresolved",
                    "no MATLAB or Octave dependency is assumed",
                ),
                ParityRow(
                    "plotted estimand and native unit",
                    "source figure-specific",
                    "awaiting extraction before comparison",
                    "unresolved",
                    "qualitative direction is not numerical parity",
                ),
            ),
        )
        for target in targets
    )


def run_friston_protocol_audit() -> dict[str, Any]:
    """Emit the paper-constrained reconstruction audit and its negative control.

    The audit is intentionally executable even while source extraction is
    incomplete.  It prevents the current categorical analogue from being
    relabelled as a source-figure reproduction: mutating a required parity row
    must leave the matrix non-exact and the reader-facing label constrained.
    """
    matrices = friston_protocol_parity_matrices()
    negative_control = {
        "name": "analogue-relabeling-control",
        "expected": "required unresolved rows keep exact=False",
        "passed": all(
            not matrix.exact and matrix.claim_label == "paper-constrained reconstruction"
            for matrix in matrices
        ),
    }
    return {
        "status": "paper-constrained reconstruction",
        "protocols": [matrix.as_dict() for matrix in matrices],
        "targets": [matrix.target for matrix in matrices],
        "primary_estimand": "source-defined plotted quantity in native units",
        "independent_unit": "source-defined agent, episode, or seed",
        "negative_control": negative_control,
        "no_claim": (
            "unresolved source parameters, mappings, schedules, or native units "
            "prohibit exact replication and numerical source-figure claims"
        ),
    }


__all__ = [
    "ParityRow",
    "ParityStatus",
    "ProtocolParityMatrix",
    "fedgvi_bnn_parity_matrix",
    "friston_protocol_parity_matrices",
    "run_friston_protocol_audit",
]
