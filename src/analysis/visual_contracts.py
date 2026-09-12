"""Source-owned explanatory contracts for non-experimental manuscript figures.

The three payloads in this module describe application integrity, evidence
ownership, and publication provenance.  They are intentionally data-only and
contain no plotting or aggregation mathematics.  The analysis workflow writes
them through the same typed JSON boundary used by empirical reports, then the
corresponding figure generators render those exact declarations.

The source-to-render contract deliberately does not read the generated release
manifest: the figure documents the source-owned producer graph and therefore
cannot depend on a downstream artifact that it helps to describe.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Final, TypedDict, cast

# Complex figures require a structured long description in every generated
# registry and reader surface.  This analysis-owned inventory is deliberately
# independent of plotting metadata so schema and publication validation never
# need to import the figure package they are responsible for validating.
COMPLEX_FIGURE_GENERATORS: Final[frozenset[str]] = frozenset(
    {
        "application_integrity_flow",
        "complexity_scaling",
        "cross_study_summary",
        "evidence_replication_map",
        "generative_model_schema",
        "graphical_abstract",
        "hierarchical_pomdp",
        "message_passing",
        "pomdp_loop",
        "robustness_review_grid",
        "source_render_provenance",
    }
)


class FlowNode(TypedDict):
    """One labeled node in a source-owned explanatory flow."""

    id: str
    label: str
    role: str


class FlowEdge(TypedDict):
    """One directed relation in a source-owned explanatory flow."""

    source: str
    target: str
    label: str
    disposition: str


class VerificationLevel(TypedDict):
    """One application-receipt verification level and its interpretation."""

    id: str
    label: str
    establishes: str
    does_not_establish: str


class ApplicationIntegrityFlowReport(TypedDict):
    """Typed payload for the application integrity and solver-health figure."""

    schema_version: str
    study_status: str
    source_relation: str
    panels: list[dict[str, object]]
    solver_statuses: list[dict[str, object]]
    verification_levels: list[VerificationLevel]
    no_claims: list[str]


class EvidenceLane(TypedDict):
    """One claim-owner lane in the evidence and replication map."""

    id: str
    headline: str
    evidence_class: str
    estimand: str
    unit: str
    replication_unit: str
    nesting: str
    permitted_interpretation: str
    prohibited_generalization: str
    display: dict[str, str]


class EvidenceReplicationMapReport(TypedDict):
    """Typed payload for the evidence-class and replication-unit map."""

    schema_version: str
    study_status: str
    source_relation: str
    evidence_classes: list[dict[str, str]]
    lanes: list[EvidenceLane]
    nesting: list[FlowNode]
    nesting_edges: list[FlowEdge]
    no_claims: list[str]


class SourceRenderProvenanceReport(TypedDict):
    """Typed payload for the source-to-render provenance and invalidation map."""

    schema_version: str
    study_status: str
    source_relation: str
    inputs: list[FlowNode]
    stages: list[FlowNode]
    terminals: list[dict[str, object]]
    producer_edges: list[FlowEdge]
    invalidation_edges: list[FlowEdge]
    authorization_boundary: str
    manifest_cycle_boundary: str
    no_claims: list[str]


# Exact source-owned inventories bind explanatory data to every rendered arrow.
# Human-readable labels may evolve, but node ownership, dependency endpoints,
# disposition, and order cannot drift independently across schema, figure, and
# documentation tests.
APPLICATION_PANEL_NODE_INVENTORY: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "validation",
        (
            "request",
            "strict_validation",
            "canonical_request",
            "destination_safety",
            "aggregate_result",
            "invalid_request",
            "unsafe_destination",
        ),
    ),
    ("solver_health", ("rich_result", "solver_status", "non_nominal")),
    ("receipt", ("request_json", "result_json", "receipt_json", "verification", "mismatch")),
)

APPLICATION_FLOW_EDGE_INVENTORY: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("validation", "request", "strict_validation", "normal"),
    ("validation", "strict_validation", "canonical_request", "normal"),
    ("validation", "strict_validation", "invalid_request", "rejected"),
    ("validation", "canonical_request", "destination_safety", "normal"),
    ("validation", "destination_safety", "unsafe_destination", "rejected"),
    ("validation", "destination_safety", "aggregate_result", "normal"),
    ("solver_health", "aggregate_result", "rich_result", "data_dependency"),
    ("solver_health", "rich_result", "solver_status", "normal"),
    ("solver_health", "solver_status", "non_nominal", "retained_warning"),
    ("receipt", "canonical_request", "request_json", "data_dependency"),
    ("receipt", "rich_result", "result_json", "data_dependency"),
    ("receipt", "request_json", "receipt_json", "data_dependency"),
    ("receipt", "result_json", "receipt_json", "data_dependency"),
    ("receipt", "receipt_json", "verification", "normal"),
    ("receipt", "verification", "mismatch", "rejected"),
)

APPLICATION_ARTIFACT_WRITE_ORDER: Final[tuple[str, ...]] = (
    "request_json",
    "result_json",
    "receipt_json",
)

SOURCE_RENDER_INPUT_INVENTORY: Final[tuple[str, ...]] = (
    "source",
    "config",
    "lockfile",
    "registry",
    "template_renderer",
)

SOURCE_RENDER_STAGE_INVENTORY: Final[tuple[str, ...]] = (
    "reports",
    "figures",
    "provisional",
    "coverage",
    "final_hydration",
    "render",
    "surfaces",
    "surface_validation",
    "provenance",
    "release_manifest",
)

SOURCE_RENDER_PRODUCER_EDGE_INVENTORY: Final[tuple[tuple[str, str, str], ...]] = (
    ("source", "reports", "producer"),
    ("config", "reports", "producer"),
    ("lockfile", "reports", "producer"),
    ("source", "figures", "producer"),
    ("reports", "figures", "producer"),
    ("registry", "figures", "producer"),
    ("source", "provisional", "producer"),
    ("config", "provisional", "producer"),
    ("reports", "provisional", "producer"),
    ("source", "coverage", "gate"),
    ("config", "coverage", "gate"),
    ("lockfile", "coverage", "gate"),
    ("provisional", "coverage", "gate"),
    ("coverage", "final_hydration", "gate"),
    ("source", "final_hydration", "producer"),
    ("config", "final_hydration", "producer"),
    ("reports", "final_hydration", "producer"),
    ("final_hydration", "render", "producer"),
    ("figures", "render", "producer"),
    ("template_renderer", "render", "producer"),
    ("render", "surfaces", "producer"),
    ("surfaces", "surface_validation", "gate"),
    ("surface_validation", "provenance", "receipt"),
    ("surfaces", "release_manifest", "receipt"),
    ("provenance", "release_manifest", "receipt"),
    ("release_manifest", "github", "authorization"),
    ("github", "zenodo", "authorization"),
)

SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY: Final[tuple[tuple[str, str, str], ...]] = tuple(
    (source, target, "invalidation")
    for source, target, disposition in SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
    if disposition != "authorization"
)


_APPLICATION_INTEGRITY_FLOW: ApplicationIntegrityFlowReport = {
    "schema_version": "1.0",
    "study_status": "source_owned_explanatory_contract",
    "source_relation": (
        "explanatory map of the implemented labeled aggregation, solver-health, "
        "artifact-writing, and receipt-verification boundaries"
    ),
    "panels": [
        {
            "id": "validation",
            "label": "A  Validate and canonicalize",
            "nodes": [
                {"id": "request", "label": "Labeled JSON or Python request", "role": "input"},
                {
                    "id": "strict_validation",
                    "label": "Strict schema + one-dimensional shape validation",
                    "role": "gate",
                },
                {
                    "id": "canonical_request",
                    "label": "Canonical semantic request + SHA-256",
                    "role": "artifact",
                },
                {
                    "id": "destination_safety",
                    "label": "Validate destination safety and required source provenance",
                    "role": "gate",
                },
                {
                    "id": "aggregate_result",
                    "label": "Delegate once to aggregate_result",
                    "role": "domain",
                },
                {
                    "id": "invalid_request",
                    "label": "Exit 2: invalid request; no directory created",
                    "role": "rejected",
                },
                {
                    "id": "unsafe_destination",
                    "label": (
                        "Exit 2: unsafe destination or required provenance unavailable; "
                        "no directory created"
                    ),
                    "role": "rejected",
                },
            ],
            "edges": [
                {
                    "source": "request",
                    "target": "strict_validation",
                    "label": "parse",
                    "disposition": "normal",
                },
                {
                    "source": "strict_validation",
                    "target": "canonical_request",
                    "label": "valid",
                    "disposition": "normal",
                },
                {
                    "source": "strict_validation",
                    "target": "invalid_request",
                    "label": "invalid request",
                    "disposition": "rejected",
                },
                {
                    "source": "canonical_request",
                    "target": "destination_safety",
                    "label": "validated semantics",
                    "disposition": "normal",
                },
                {
                    "source": "destination_safety",
                    "target": "unsafe_destination",
                    "label": "unsafe path or provenance failure",
                    "disposition": "rejected",
                },
                {
                    "source": "destination_safety",
                    "target": "aggregate_result",
                    "label": "single delegation",
                    "disposition": "normal",
                },
            ],
        },
        {
            "id": "solver_health",
            "label": "B  Preserve numerical health",
            "nodes": [
                {
                    "id": "rich_result",
                    "label": "Consensus + raw/normalized weights + histories",
                    "role": "result",
                },
                {
                    "id": "solver_status",
                    "label": "Convergence × fallback → four-state solver status",
                    "role": "health",
                },
                {
                    "id": "non_nominal",
                    "label": "Exit 1: retain valid artifacts for non-nominal solver status",
                    "role": "retained_warning",
                },
            ],
            "edges": [
                {
                    "source": "aggregate_result",
                    "target": "rich_result",
                    "label": "finite result",
                    "disposition": "data_dependency",
                },
                {
                    "source": "rich_result",
                    "target": "solver_status",
                    "label": "classify health",
                    "disposition": "normal",
                },
                {
                    "source": "solver_status",
                    "target": "non_nominal",
                    "label": "fallback or nonconvergence",
                    "disposition": "retained_warning",
                },
            ],
        },
        {
            "id": "receipt",
            "label": "C  Write atomically and verify by level",
            "nodes": [
                {"id": "request_json", "label": "1  request.json", "role": "artifact"},
                {"id": "result_json", "label": "2  result.json", "role": "artifact"},
                {"id": "receipt_json", "label": "3  receipt.json", "role": "receipt"},
                {
                    "id": "verification",
                    "label": "Artifact integrity → source equivalence → optional nominal health",
                    "role": "verification",
                },
                {
                    "id": "mismatch",
                    "label": "Verification mismatch: report exact finding; do not reinterpret result",
                    "role": "rejected",
                },
            ],
            "edges": [
                {
                    "source": "canonical_request",
                    "target": "request_json",
                    "label": "semantic input",
                    "disposition": "data_dependency",
                },
                {
                    "source": "rich_result",
                    "target": "result_json",
                    "label": "semantic output",
                    "disposition": "data_dependency",
                },
                {
                    "source": "request_json",
                    "target": "receipt_json",
                    "label": "bind request hash",
                    "disposition": "data_dependency",
                },
                {
                    "source": "result_json",
                    "target": "receipt_json",
                    "label": "bind result hash + provenance",
                    "disposition": "data_dependency",
                },
                {
                    "source": "receipt_json",
                    "target": "verification",
                    "label": "verify",
                    "disposition": "normal",
                },
                {
                    "source": "verification",
                    "target": "mismatch",
                    "label": "mismatch",
                    "disposition": "rejected",
                },
            ],
            "write_order": list(APPLICATION_ARTIFACT_WRITE_ORDER),
        },
    ],
    "solver_statuses": [
        {"id": "nominal", "converged": True, "fallbacks": False, "label": "nominal"},
        {
            "id": "converged_with_fallback",
            "converged": True,
            "fallbacks": True,
            "label": "converged_with_fallback",
        },
        {"id": "not_converged", "converged": False, "fallbacks": False, "label": "not_converged"},
        {
            "id": "not_converged_with_fallback",
            "converged": False,
            "fallbacks": True,
            "label": "not_converged_with_fallback",
        },
    ],
    "verification_levels": [
        {
            "id": "artifact_integrity",
            "label": "Artifact integrity",
            "establishes": "recorded request/result bytes match the receipt",
            "does_not_establish": "equivalence to another source checkout or nominal solver health",
        },
        {
            "id": "source_equivalence",
            "label": "Source equivalence",
            "establishes": "requested source inputs match recorded source provenance",
            "does_not_establish": "calibration, domain suitability, or scientific validity",
        },
        {
            "id": "nominal_solver",
            "label": "Nominal solver health",
            "establishes": "the recorded run converged without fallback",
            "does_not_establish": "a downstream decision, acceptance, or truth of an interpretation",
        },
    ],
    "no_claims": [
        "calibration",
        "domain suitability",
        "scientific validity",
        "downstream decision correctness",
        "acceptance",
    ],
}


_EVIDENCE_REPLICATION_MAP: EvidenceReplicationMapReport = {
    "schema_version": "1.0",
    "study_status": "source_owned_evidence_classification",
    "source_relation": (
        "headline-result and claim-owner map separating mathematical identities, source-conditional "
        "results, conditional experiments, implementation checks, and open work"
    ),
    "evidence_classes": [
        {
            "id": "formal_executable",
            "label": "Formal / executable",
            "meaning": "proved or exactly tested in the project model",
        },
        {
            "id": "source_conditional",
            "label": "Source-conditional",
            "meaning": "inherits only under the cited paper's assumptions",
        },
        {
            "id": "conditional_empirical",
            "label": "Conditional empirical",
            "meaning": "estimated in the declared finite design",
        },
        {
            "id": "scoped_implementation",
            "label": "Scoped implementation",
            "meaning": "checks one software or protocol boundary",
        },
        {"id": "open", "label": "Open", "meaning": "not established by the current evidence"},
    ],
    "lanes": [
        {
            "id": "project_identity",
            "headline": "Project identity at c=0",
            "evidence_class": "formal_executable",
            "estimand": "robust_aggregate(c=0) versus log_linear_pool",
            "unit": "categorical posterior vector",
            "replication_unit": "not applicable; exact executable identity",
            "nesting": "validated categorical inputs → paired function outputs",
            "permitted_interpretation": "the project functions agree at the declared limit",
            "prohibited_generalization": "complete Friston protocol reconstruction",
            "display": {
                "estimand_unit": "paired outputs [categorical posterior]",
                "replication": "exact identity",
                "nesting": "input → paired functions",
                "permitted": "project limit",
                "prohibited": "full source protocol",
            },
        },
        {
            "id": "client_loss",
            "headline": "Client-loss robustness",
            "evidence_class": "source_conditional",
            "estimand": "client generalized posterior under the cited loss/divergence assumptions",
            "unit": "client update or posterior",
            "replication_unit": "not applicable; theorem-defined source statement",
            "nesting": "source assumptions → client update → posterior",
            "permitted_interpretation": "source theorem under its stated assumptions",
            "prohibited_generalization": (
                "server robustness, empirical performance, or universal calibration"
            ),
            "display": {
                "estimand_unit": "generalized posterior [client update]",
                "replication": "theorem; no empirical unit",
                "nesting": "assumptions → update → posterior",
                "permitted": "source theorem",
                "prohibited": "server theorem; performance; calibration",
            },
        },
        {
            "id": "belief_sharing_free_energy",
            "headline": "Paired free-energy contrast",
            "evidence_class": "conditional_empirical",
            "estimand": "within-seed isolated-minus-communicating colony-mean variational free energy",
            "unit": "nats",
            "replication_unit": "independent configured seed",
            "nesting": "study → seed → communication condition → agents",
            "permitted_interpretation": (
                "lower free energy with communication in the reduced declared protocol"
            ),
            "prohibited_generalization": "general communication benefit or exact source-protocol replication",
            "display": {
                "estimand_unit": "isolated − shared colony F [nats]",
                "replication": "configured seed",
                "nesting": "study → seed → condition → agents",
                "permitted": "paired protocol contrast",
                "prohibited": "general benefit; exact replication",
            },
        },
        {
            "id": "likelihood_learning",
            "headline": "Dirichlet likelihood learning",
            "evidence_class": "conditional_empirical",
            "estimand": "seed-mean KL from true to learned categorical likelihood by ordered step",
            "unit": "nats",
            "replication_unit": "independent configured seed",
            "nesting": "study → seed → ordered learning steps → likelihood rows",
            "permitted_interpretation": "learning trajectory in the executed conjugate categorical design",
            "prohibited_generalization": "universal convergence rate or continuous-model learning claim",
            "display": {
                "estimand_unit": "KL(true A || learned A) [nats]",
                "replication": "configured seed",
                "nesting": "study → seed → steps → A rows",
                "permitted": "executed conjugate path",
                "prohibited": "universal rate; continuous model",
            },
        },
        {
            "id": "bmr_sign_control",
            "headline": "Configured BMR sign control",
            "evidence_class": "formal_executable",
            "estimand": "free-energy difference for redundant- and supported-column pruning",
            "unit": "nats",
            "replication_unit": "not applicable; one deterministic closed-form comparison",
            "nesting": "fixed posterior → candidate pruning → signed free-energy difference",
            "permitted_interpretation": "the configured redundant/supported sign control passes",
            "prohibited_generalization": "universal structure emergence or true-structure recovery",
            "display": {
                "estimand_unit": "pruning delta F [nats]",
                "replication": "closed-form comparison",
                "nesting": "posterior → pruning → signed delta",
                "permitted": "configured sign control",
                "prohibited": "universal emergence",
            },
        },
        {
            "id": "client_loss_baseline",
            "headline": "Exploratory client baseline",
            "evidence_class": "conditional_empirical",
            "estimand": (
                "held-out accuracy for joint NLL/L2=0.05 and RCCE/L2=0.10 "
                "configurations by contamination rate"
            ),
            "unit": "accuracy fraction",
            "replication_unit": "independent synthetic-data seed",
            "nesting": "synthetic sweep → seed → client → observation",
            "permitted_interpretation": (
                "finite exploratory behavior of the two declared joint configurations"
            ),
            "prohibited_generalization": (
                "an RCCE-only effect, leakage-free calibration, universal robustness, or "
                "posterior-uncertainty BNN replication"
            ),
            "display": {
                "estimand_unit": "joint-config accuracy [fraction]",
                "replication": "synthetic seed",
                "nesting": "sweep → seed → client → row",
                "permitted": "composite configuration contrast",
                "prohibited": "RCCE-only; calibration; BNN claims",
            },
        },
        {
            "id": "server_heuristic",
            "headline": "Server reweighting heuristic",
            "evidence_class": "conditional_empirical",
            "estimand": "robust-minus-naive consensus quantity in a declared attack cell",
            "unit": "accuracy fraction or true-state probability mass",
            "replication_unit": "configured seed or matched trial, as declared per report",
            "nesting": "study → seed/world → matched trial → agents and contamination condition",
            "permitted_interpretation": (
                "finite, mechanism-specific behavior and the exact zero-strength recovery"
            ),
            "prohibited_generalization": "bounded influence, Byzantine tolerance, or a universal winner",
            "display": {
                "estimand_unit": "robust − naive [p or fraction]",
                "replication": "seed or matched trial",
                "nesting": "study → seed/world → trial → agents",
                "permitted": "finite mechanism behavior",
                "prohibited": "influence bound; Byzantine tol.; universal win",
            },
        },
        {
            "id": "variational_server",
            "headline": "Variational server rule",
            "evidence_class": "formal_executable",
            "estimand": (
                "declared free-energy descent, effective-weight property, and tempered-family "
                "recovery/API transfer"
            ),
            "unit": "nats, normalized server weight, or categorical posterior",
            "replication_unit": "not applicable for identity; seeded path for diagnostics",
            "nesting": "configuration → initialization → ordered block-coordinate iterates",
            "permitted_interpretation": (
                "objective-backed properties and exact tested transfer points on the categorical rule"
            ),
            "prohibited_generalization": "global optimality or universal robustness",
            "display": {
                "estimand_unit": "descent, weights, transfer [mixed]",
                "replication": "identity or seeded path",
                "nesting": "config → start → iterates",
                "permitted": "objective + tested transfer",
                "prohibited": "global or universal claim",
            },
        },
        {
            "id": "moving_disjoint_fov",
            "headline": "Moving / disjoint-FOV contrasts",
            "evidence_class": "conditional_empirical",
            "estimand": "communication or policy accuracy and free-energy contrasts in declared worlds",
            "unit": "accuracy fraction, nats, or steps",
            "replication_unit": "configured seed where reported; otherwise deterministic world",
            "nesting": "world → seed when present → condition → agents → steps",
            "permitted_interpretation": "condition-specific communication behavior in executed worlds",
            "prohibited_generalization": "universal communication benefit or physical multi-host result",
            "display": {
                "estimand_unit": "policy/sharing [fraction, nats, steps]",
                "replication": "seed or fixed world",
                "nesting": "world → seed → condition → agents/steps",
                "permitted": "executed-world behavior",
                "prohibited": "universal claim; physical multi-host",
            },
        },
        {
            "id": "hierarchy_sensitivity",
            "headline": "Hierarchy / sensitivity diagnostics",
            "evidence_class": "conditional_empirical",
            "estimand": "posterior, accuracy-gap, and information-gating quantities by hierarchy design",
            "unit": "probability, accuracy fraction, or nats",
            "replication_unit": "configured seed when reported; otherwise deterministic diagnostic",
            "nesting": "world → seed when present → hierarchy level → agent/trial → ordered state",
            "permitted_interpretation": "behavior of the executed two-/three-level and sensitivity designs",
            "prohibited_generalization": (
                "universal hierarchy benefit, depth recovery, or structure emergence"
            ),
            "display": {
                "estimand_unit": "posterior/gap/gate [probability, fraction, nats]",
                "replication": "seed or fixed diagnostic",
                "nesting": "world → seed → level → agents/states",
                "permitted": "executed hierarchy design",
                "prohibited": "universal benefit, depth, or emergence",
            },
        },
        {
            "id": "parameter_recovery",
            "headline": "Acuity recovery diagnostic",
            "evidence_class": "conditional_empirical",
            "estimand": "recovered acuity, absolute error, and mean-recovery coefficient of determination",
            "unit": "probability units or unitless R-squared",
            "replication_unit": "independent synthetic trial within each true-acuity grid point",
            "nesting": "grid point → trial → observations → recovered estimate",
            "permitted_interpretation": "finite-grid acuity recoverability in the executed design",
            "prohibited_generalization": "global structural identifiability",
            "display": {
                "estimand_unit": "acuity, error, R² [p or unitless]",
                "replication": "trial within acuity cell",
                "nesting": "grid point → trial → rows → estimate",
                "permitted": "finite-grid recovery",
                "prohibited": "global structural identifiability",
            },
        },
        {
            "id": "protocol_integrity",
            "headline": "Protocol-v1 / replay integrity",
            "evidence_class": "scoped_implementation",
            "estimand": "byte/envelope consistency and deterministic replay recomputation",
            "unit": "frame, worker round, or replay event",
            "replication_unit": "configured local process or loopback round",
            "nesting": "round → ordered workers → frames → payload fields",
            "permitted_interpretation": "local transport and replay integrity for protocol v1",
            "prohibited_generalization": (
                "encryption, mTLS identity, physical multi-host operation, or Byzantine tolerance"
            ),
            "display": {
                "estimand_unit": "frame/replay match [frame or round]",
                "replication": "local round",
                "nesting": "round → workers → frames → fields",
                "permitted": "protocol-v1 local integrity",
                "prohibited": "encryption/mTLS; multi-host; Byzantine tol.",
            },
        },
        {
            "id": "application_provenance",
            "headline": "Application receipt",
            "evidence_class": "scoped_implementation",
            "estimand": "hash and provenance agreement for one labeled aggregation run",
            "unit": "canonical request, result, and receipt artifact",
            "replication_unit": "one completed application invocation",
            "nesting": "invocation → request/result artifacts → receipt verification levels",
            "permitted_interpretation": "declared artifact integrity and, when requested, source equivalence",
            "prohibited_generalization": (
                "scientific validity, domain suitability, decision correctness, or acceptance"
            ),
            "display": {
                "estimand_unit": "hash/source agreement [artifact]",
                "replication": "application invocation",
                "nesting": "invocation → artifacts → levels",
                "permitted": "hashes; requested source match",
                "prohibited": "validity; decision; acceptance",
            },
        },
        {
            "id": "external_confirmation",
            "headline": "External calibration / confirmation",
            "evidence_class": "open",
            "estimand": "future preregistered paired log-score contrasts on sealed external test partitions",
            "unit": "nats per observation, summarized within dataset",
            "replication_unit": "dataset as highest external unit; seeds nested",
            "nesting": "dataset → seed → clients/conditions → observations",
            "permitted_interpretation": "none before source-bound frozen design and confirmatory report",
            "prohibited_generalization": (
                "retrospective positive claim or pooled dataset-population inference"
            ),
            "display": {
                "estimand_unit": "paired log score [nats/observation]",
                "replication": "dataset highest; seeds nested",
                "nesting": "dataset → seed → clients → rows",
                "permitted": "none before design + report",
                "prohibited": "retrospective or pooled-dataset claim",
            },
        },
    ],
    "nesting": [
        {"id": "study", "label": "Configured study or source-bound run", "role": "highest configured scope"},
        {
            "id": "seed",
            "label": "Seed when present; independence is lane-specific",
            "role": "possible nested unit",
        },
        {"id": "trial", "label": "Matched trial or condition", "role": "nested comparison"},
        {"id": "agent", "label": "Agent / contamination role", "role": "nested unit"},
        {"id": "step", "label": "Ordered step / posterior state", "role": "repeated state"},
    ],
    "nesting_edges": [
        {"source": "study", "target": "seed", "label": "contains", "disposition": "nesting"},
        {"source": "seed", "target": "trial", "label": "contains or pairs", "disposition": "nesting"},
        {"source": "trial", "target": "agent", "label": "contains", "disposition": "nesting"},
        {"source": "agent", "target": "step", "label": "repeats over", "disposition": "nesting"},
    ],
    "no_claims": [
        "guarantees do not transfer between client, server, protocol, and application lanes",
        "nested agents, trials, and time steps are not promoted to independent datasets",
        "open external confirmation is not implied by formal or software evidence",
    ],
}


_SOURCE_RENDER_PROVENANCE: SourceRenderProvenanceReport = {
    "schema_version": "1.0",
    "study_status": "source_owned_pipeline_contract",
    "source_relation": (
        "producer-order and stale-invalidation map derived from source-owned pipeline contracts"
    ),
    "inputs": [
        {"id": "source", "label": "Source + tests + manuscript", "role": "upstream input"},
        {"id": "config", "label": "Configuration + claim contracts", "role": "upstream input"},
        {"id": "lockfile", "label": "uv.lock + runtime metadata", "role": "upstream input"},
        {"id": "registry", "label": "Figure / accessibility registry source", "role": "upstream input"},
        {
            "id": "template_renderer",
            "label": "Exact clean Template renderer commit",
            "role": "upstream input",
        },
    ],
    "stages": [
        {"id": "reports", "label": "Typed analysis reports", "role": "producer"},
        {"id": "figures", "label": "Figure pairs + accessibility metadata", "role": "producer"},
        {"id": "provisional", "label": "Provisional hydration", "role": "producer"},
        {"id": "coverage", "label": "Full tests + coverage receipt", "role": "gate"},
        {"id": "final_hydration", "label": "Final receipt-backed hydration", "role": "producer"},
        {"id": "render", "label": "Clean source-locked Template render", "role": "external producer"},
        {"id": "surfaces", "label": "PDF + HTML + reveal.js + Beamer", "role": "rendered surface"},
        {
            "id": "surface_validation",
            "label": "Structure + text + visual + accessibility validation",
            "role": "gate",
        },
        {"id": "provenance", "label": "Pipeline provenance + artifact manifests", "role": "receipt"},
        {"id": "release_manifest", "label": "Release manifest + staged checksums", "role": "receipt"},
    ],
    "terminals": [
        {
            "id": "github",
            "label": "GitHub tag and release",
            "role": "authorization-gated terminal",
            "authorization_required": True,
        },
        {
            "id": "zenodo",
            "label": "Zenodo publication",
            "role": "irreversible authorization-gated terminal",
            "authorization_required": True,
        },
    ],
    "producer_edges": [
        {"source": "source", "target": "reports", "label": "execute", "disposition": "producer"},
        {"source": "config", "target": "reports", "label": "configure", "disposition": "producer"},
        {"source": "lockfile", "target": "reports", "label": "bind environment", "disposition": "producer"},
        {"source": "source", "target": "figures", "label": "run generators", "disposition": "producer"},
        {"source": "reports", "target": "figures", "label": "render typed values", "disposition": "producer"},
        {"source": "registry", "target": "figures", "label": "bind semantics", "disposition": "producer"},
        {
            "source": "source",
            "target": "provisional",
            "label": "hydrate manuscript",
            "disposition": "producer",
        },
        {"source": "config", "target": "provisional", "label": "bind tokens", "disposition": "producer"},
        {"source": "reports", "target": "provisional", "label": "supply values", "disposition": "producer"},
        {"source": "source", "target": "coverage", "label": "test source", "disposition": "gate"},
        {"source": "config", "target": "coverage", "label": "test configuration", "disposition": "gate"},
        {"source": "lockfile", "target": "coverage", "label": "lock test runtime", "disposition": "gate"},
        {
            "source": "provisional",
            "target": "coverage",
            "label": "test provisional surface",
            "disposition": "gate",
        },
        {"source": "coverage", "target": "final_hydration", "label": "bind receipt", "disposition": "gate"},
        {
            "source": "source",
            "target": "final_hydration",
            "label": "hydrate final source",
            "disposition": "producer",
        },
        {
            "source": "config",
            "target": "final_hydration",
            "label": "bind final tokens",
            "disposition": "producer",
        },
        {
            "source": "reports",
            "target": "final_hydration",
            "label": "supply final values",
            "disposition": "producer",
        },
        {
            "source": "final_hydration",
            "target": "render",
            "label": "render manuscript",
            "disposition": "producer",
        },
        {"source": "figures", "target": "render", "label": "embed figures", "disposition": "producer"},
        {
            "source": "template_renderer",
            "target": "render",
            "label": "execute exact renderer",
            "disposition": "producer",
        },
        {"source": "render", "target": "surfaces", "label": "produce", "disposition": "producer"},
        {"source": "surfaces", "target": "surface_validation", "label": "validate", "disposition": "gate"},
        {"source": "surface_validation", "target": "provenance", "label": "record", "disposition": "receipt"},
        {
            "source": "surfaces",
            "target": "release_manifest",
            "label": "inventory bytes",
            "disposition": "receipt",
        },
        {
            "source": "provenance",
            "target": "release_manifest",
            "label": "seal receipts",
            "disposition": "receipt",
        },
        {
            "source": "release_manifest",
            "target": "github",
            "label": "review + authorize",
            "disposition": "authorization",
        },
        {
            "source": "github",
            "target": "zenodo",
            "label": "verify + fresh approval",
            "disposition": "authorization",
        },
    ],
    "invalidation_edges": [
        {
            "source": source,
            "target": target,
            "label": "change invalidates",
            "disposition": "invalidation",
        }
        for source, target, _ in SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY
    ],
    "authorization_boundary": (
        "green engineering and rendering gates are necessary inputs to review; they do not "
        "automatically authorize a merge, tag, GitHub release, or Zenodo publication"
    ),
    "manifest_cycle_boundary": (
        "this figure is generated from this source-owned contract and never reads the "
        "downstream release manifest that later inventories it"
    ),
    "no_claims": [
        "a green build is not scientific validation",
        "tagged PDF structure is not PDF/UA conformance",
        "HTML accessibility enhancements are not a WCAG conformance claim",
        "a GitHub release or DOI does not enlarge the scientific claim surface",
    ],
}


def application_integrity_flow_contract() -> ApplicationIntegrityFlowReport:
    """Return a defensive copy of the application integrity flow contract."""

    return cast(ApplicationIntegrityFlowReport, deepcopy(_APPLICATION_INTEGRITY_FLOW))


def evidence_replication_map_contract() -> EvidenceReplicationMapReport:
    """Return a defensive copy of the evidence and replication map contract."""

    return cast(EvidenceReplicationMapReport, deepcopy(_EVIDENCE_REPLICATION_MAP))


def source_render_provenance_contract() -> SourceRenderProvenanceReport:
    """Return a defensive copy of the source-to-render provenance contract."""

    return cast(SourceRenderProvenanceReport, deepcopy(_SOURCE_RENDER_PROVENANCE))


__all__ = [
    "APPLICATION_ARTIFACT_WRITE_ORDER",
    "APPLICATION_FLOW_EDGE_INVENTORY",
    "APPLICATION_PANEL_NODE_INVENTORY",
    "ApplicationIntegrityFlowReport",
    "COMPLEX_FIGURE_GENERATORS",
    "EvidenceReplicationMapReport",
    "SOURCE_RENDER_INPUT_INVENTORY",
    "SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY",
    "SOURCE_RENDER_PRODUCER_EDGE_INVENTORY",
    "SOURCE_RENDER_STAGE_INVENTORY",
    "SourceRenderProvenanceReport",
    "application_integrity_flow_contract",
    "evidence_replication_map_contract",
    "source_render_provenance_contract",
]
