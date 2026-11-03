"""Figure evidence dependencies report contracts."""

from __future__ import annotations

from analysis.schemas.primitives import FigureDependencyContract

_FIGURE_EVIDENCE_DEPENDENCIES: dict[str, tuple[FigureDependencyContract, ...]] = {
    "application_integrity_flow": (
        FigureDependencyContract(
            report_name="application_integrity_flow",
            required_fields={
                "panels": "list",
                "solver_statuses": "list",
                "verification_levels": "list",
                "no_claims": "list",
                "source_relation": "str",
            },
        ),
    ),
    "evidence_replication_map": (
        FigureDependencyContract(
            report_name="evidence_replication_map",
            required_fields={
                "evidence_classes": "list",
                "lanes": "list",
                "nesting": "list",
                "nesting_edges": "list",
                "no_claims": "list",
                "source_relation": "str",
            },
        ),
    ),
    "source_render_provenance": (
        FigureDependencyContract(
            report_name="source_render_provenance",
            required_fields={
                "inputs": "list",
                "stages": "list",
                "terminals": "list",
                "producer_edges": "list",
                "invalidation_edges": "list",
                "authorization_boundary": "str",
                "manifest_cycle_boundary": "str",
                "no_claims": "list",
            },
        ),
    ),
}
