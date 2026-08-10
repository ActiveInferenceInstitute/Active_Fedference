"""Static API reference markdown + API-doc generation workflow.

The static reference template and :func:`build_api_reference_markdown` are pure
(no I/O). :func:`run_api_doc_generation` is the generated-output *workflow*
entry point that the thin ``scripts/generate_api_docs.py`` wrapper calls: it
owns the output-directory creation, the infrastructure glossary build, the file
writes, and the degradation handling. Keeping it here (not in the script) holds
the thin-orchestrator contract — scripts only wire paths and print.
"""

from __future__ import annotations

from pathlib import Path

try:  # package context
    from ._runtime import get_logger
except ImportError:  # standalone load (no package context) — mirror the src/ import pattern
    from _runtime import get_logger  # type: ignore[no-redef]

_logger = get_logger(__name__)

API_REFERENCE_TEMPLATE = """# Active Fedference API Reference

This document provides API reference for the Active Fedference project's
federated active inference modules.

## Core packages

### `fedference.divergences`

Statistical divergences over categorical distributions for FedGVI:
- `kl_divergence(q, p)` — KL divergence used by the project's naive-pool diagnostics.
- `reverse_kl(q, p)` — Reverse KL, FedGVI RKL client divergence.
- `renyi_divergence(q, p, alpha)` — standard Rényi; `alpha -> 1` recovers KL.
- `alpha_renyi_divergence(q, p, alpha)` — FedGVI Alpha-Rényi normalization.
- `total_variation(q, p)` — Total variation distance.

### `fedference.losses`

Loss functions for generalized Bayesian inference:
- `nll(q, outcome)` — Negative log likelihood (standard Bayes).
- `beta_loss(q, outcome, beta)` — Beta-loss; `beta -> 0` recovers NLL.
- `rcce(q, outcome, q_reg)` — Robust categorical cross-entropy.

### `fedference.aggregation`

Belief aggregation primitives for federated consensus:
- `log_linear_pool(local_posteriors, base_weights=None)` — Project product-of-experts;
  a categorical posterior-log-potential specialization of Friston Eq. 7 under
  documented assumptions, not a full protocol reconstruction.
- `robust_aggregate(local_posteriors, robustness, base_weights=None)` — Divergence-reweighted pooling.
- `variational_aggregate(local_posteriors, robustness, base_weights=None)` —
  Objective-backed conservative rule.
- `aggregation_free_energy(consensus_posterior, raw_effective_weights,`
  `local_posteriors, robustness)` — Variational free energy.

### `fedference.experiments`

Reproducible study harness:
- `run_belief_sharing(seed)` — categorical source-mechanism analogue related to Friston Fig. 5.
- `run_language_acquisition(seed)` — categorical language-learning trajectory related to Friston Fig. 7.
- `run_emergence(seed)` — categorical BMR diagnostic related to Friston Fig. 9.
- `run_robustness_sweep(seed)` — Contamination robustness sweep.
- `run_contamination_gallery(seed)` — Multi-mechanism contamination comparison.

See the full docstrings in each module for parameter details.
"""


def build_api_reference_markdown() -> str:
    """Return markdown API reference for Active Fedference."""
    return API_REFERENCE_TEMPLATE


def run_api_doc_generation(project_root: Path) -> dict[str, str | None]:
    """Generate the glossary-style API index and the static API reference.

    Writes ``output/docs/api_glossary.md`` (best-effort — degrades to ``None``
    if the infrastructure index build fails) and ``output/docs/api_reference.md``
    (always). Returns a label -> path mapping for the calling script to print.
    """
    output_dir = project_root / "output" / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    glossary_path: Path | None = None
    try:
        # Lazy import: the infrastructure glossary generator is a monorepo
        # adapter, absent in the standalone fork. The guard degrades to None
        # rather than failing the workflow when infrastructure is unavailable.
        from infrastructure.documentation.glossary_gen import (
            build_api_index,
            generate_markdown_table,
        )

        entries = build_api_index(str(project_root / "src"))
        glossary_path = output_dir / "api_glossary.md"
        glossary_path.write_text(generate_markdown_table(entries), encoding="utf-8")
    except (OSError, ImportError, ValueError, SyntaxError) as exc:
        _logger.warning("API index generation failed: %s", exc)
        glossary_path = None

    api_ref_path = output_dir / "api_reference.md"
    api_ref_path.write_text(build_api_reference_markdown(), encoding="utf-8")

    return {
        "api_reference": str(api_ref_path),
        "glossary": str(glossary_path) if glossary_path is not None else None,
    }


__all__ = [
    "API_REFERENCE_TEMPLATE",
    "build_api_reference_markdown",
    "run_api_doc_generation",
]
