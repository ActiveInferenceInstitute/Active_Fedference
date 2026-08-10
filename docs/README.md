# docs/ — Active Fedference documentation hub

Operational rulebook for the Active Fedference project.
Enforcement varies by file — see [`AGENTS.md`](AGENTS.md)'s REQUIRED vs
AESTHETIC table for what is gate-checked versus convention.

**Contract & identity:** [`../ISA.md`](../ISA.md) · [`../README.md`](../README.md) · [`../STANDALONE.md`](../STANDALONE.md)

**Published release:** [Zenodo DOI `10.5281/zenodo.21864004`](https://doi.org/10.5281/zenodo.21864004) · [public GitHub repository](https://github.com/ActiveInferenceInstitute/Active_Fedference) · [Zenodo record](https://zenodo.org/records/21864004)

## Subfolders

| Folder | Purpose |
| --- | --- |
| [`core/`](core/README.md) | Science narrative, architecture, experiments & artifacts |
| [`development/`](development/README.md) | Agent rules, quickstart, testing, style |
| [`manuscript/`](manuscript/README.md) | PDF/web pipeline, accessibility boundary, tokens & labels |
| [`operations/`](operations/README.md) | Output layout, troubleshooting, FAQ |
| [`reference/`](reference/README.md) | Verification, API/schema stability, and Zenodo release boundary |
| [`research/`](research/literature-audit.md) | Source-audited scholarship, claim boundaries, simulation-design rationale, the [dated extended statistical audit](research/extended-statistical-audit-2026-07-14.md), the [first-principles/RedTeam review](research/first-principles-redteam-review-2026-07-16.md), and the [runtime-surface/composability follow-up](research/runtime-surface-composability-review-2026-07-17.md) |
| [`security/`](security/README.md) | Runtime/development trust boundaries, abuse paths, mitigations, and no-claim constraints |
| [`todo/`](todo/README.md) | Scoped forward roadmap pages linked from `TODO.md` |

## Read order (agents)

1. [`development/agent_instructions.md`](development/agent_instructions.md) — repository editing rules and visual QA
2. [`core/architecture.md`](core/architecture.md) — layer boundaries
3. [`development/testing_philosophy.md`](development/testing_philosophy.md) — zero mocks + coverage
4. [`manuscript/rendering_pipeline.md`](manuscript/rendering_pipeline.md) — before editing outputs
5. [`manuscript/accessibility.md`](manuscript/accessibility.md) — before accessibility or publication-surface claims
6. [`development/style_guide.md`](development/style_guide.md) — before editing source
7. [`manuscript/tokens-and-labels.md`](manuscript/tokens-and-labels.md) — before editing manuscript prose
8. [`reference/zenodo-release.md`](reference/zenodo-release.md) — before reserving, updating, uploading, or publishing a DOI deposition

## Read order (new contributors)

1. [`../README.md`](../README.md) — project pitch and run commands
2. [`development/quickstart.md`](development/quickstart.md) — first green run
3. [`core/conceptual-foundations.md`](core/conceptual-foundations.md) — why this project exists
4. [`operations/faq.md`](operations/faq.md) — common questions

## Reader paths

- **Active-inference readers:** begin with the Friston belief-sharing mechanism
  and categorical recovery identity in
  [`core/conceptual-foundations.md`](core/conceptual-foundations.md), then follow
  the POMDP and belief-sharing modules in
  [`core/architecture.md`](core/architecture.md).
- **Federated-learning readers:** begin with PVI/FedGVI client objectives,
  source-protocol parity, site factors/cavities, and the three distinct server
  aggregation rules in [`core/architecture.md`](core/architecture.md), then
  inspect the registry/receipt boundary in
  [`core/experiments-and-artifacts.md`](core/experiments-and-artifacts.md).

## Before modifying…

| Task | Read first |
| --- | --- |
| `src/fedference/*` | [`core/architecture.md`](core/architecture.md), [`development/style_guide.md`](development/style_guide.md) |
| `tests/*` | [`development/testing_philosophy.md`](development/testing_philosophy.md) |
| `manuscript/*.md` | [`manuscript/tokens-and-labels.md`](manuscript/tokens-and-labels.md), [`../manuscript/SYNTAX.md`](../manuscript/SYNTAX.md) |
| `scripts/*` | [`core/architecture.md`](core/architecture.md), [`../scripts/AGENTS.md`](../scripts/AGENTS.md) |
| Federation transport / deployment | [`security/active_fedference-threat-model.md`](security/active_fedference-threat-model.md), [`todo/true-multi-machine-federation.md`](todo/true-multi-machine-federation.md) |
| Pipeline / `output/` | [`operations/output-layout.md`](operations/output-layout.md) |
| Roadmap / TODO scope | [`todo/README.md`](todo/README.md), [`../TODO.md`](../TODO.md) |

## Quick verification

```bash
uv run --locked pytest tests/ \
  --cov=src --cov-fail-under=90 -q
```

Full probe list: [`reference/verification-commands.md`](reference/verification-commands.md).

Mermaid diagrams are deliberately kept in the Markdown source for GitHub and
local readers. Every block in `README.md` and `docs/` must pass the static
contract and, before a release render, an actual Mermaid CLI SVG probe:

```bash
uv run --locked python scripts/validate_mermaid.py
uv run --locked python scripts/validate_mermaid.py --render --renderer npx \
  --output-dir .tmp/mermaid-render
```

## Index

Technical file inventory and REQUIRED vs AESTHETIC table:
[`AGENTS.md`](AGENTS.md).
