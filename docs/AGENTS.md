# docs/ — Agent-facing documentation hub

Technical index for `docs/`. Navigation hub:
[`README.md`](README.md).

## File inventory

| Path | Purpose |
| --- | --- |
| **Hub** | |
| `README.md` | Navigation, read order, task routing |
| `AGENTS.md` | This index; REQUIRED vs AESTHETIC |
| **core/** | |
| `core/README.md` | Core-document index and routing to science, architecture, and artifact contracts |
| `core/conceptual-foundations.md` | Friston × FedGVI bridge; three robustness axes; protocol, hierarchy, and transport boundaries |
| `core/architecture.md` | Layer contract; domain, protocol, evidence, CLI, optional-Torch, and publication module map |
| `core/experiments-and-artifacts.md` | Existing studies, planned registry lanes, artifacts, receipts, and producer graph |
| **development/** | |
| `development/README.md` | Development-document index and editing/test routing |
| `development/agent_instructions.md` | Repository editing rules, visual QA, and pre-submit checklist |
| `development/quickstart.md` | First green run |
| `development/testing_philosophy.md` | Zero-mock; test inventory |
| `development/style_guide.md` | Seven coding/manuscript rules |
| `development/modularity.md` | Cross-layer extension and orchestration contract |
| **manuscript/** | |
| `manuscript/README.md` | Manuscript-pipeline index and authoritative registry signposts |
| `manuscript/accessibility.md` | Canonical HTML accessibility contract, PDF/UA no-claim boundary, and manual release review |
| `manuscript/rendering_pipeline.md` | Analysis → variables → PDF |
| `manuscript/tokens-and-labels.md` | Token groups; links to SYNTAX.md |
| **operations/** | |
| `operations/README.md` | Operations-document index and output/review routing |
| `operations/output-layout.md` | Producer-owned tracked reviewer snapshot; disposable review scratch under `.tmp/` |
| `operations/troubleshooting.md` | Symptom → fix |
| `operations/faq.md` | FAQ; link to STANDALONE |
| **reference/** | |
| `reference/verification-commands.md` | ISA probes; copy-paste commands |
| `reference/api-stability.md` | Public API/schema compatibility and deprecation policy |
| `reference/zenodo-release.md` | Published v1.0.3 DOI record and safe new-version/update/upload/verify/publish boundary for future versions |
| `reference/README.md` | Reference-document index and metadata/release boundary signposts |
| **research/** | |
| `research/*.md` | Source-audited literature, statistical, complexity, claim, visual, and composability reviews |
| **source package docs** | |
| `../src/fedference_cli/README.md` | Installed CLI module map, stable facade, and extension recipe |
| `../src/fedference_cli/AGENTS.md` | Package-local CLI boundary and testing rules |
| **security/** | |
| `security/README.md` | Security navigation and current no-deployment boundary |
| `security/active_fedference-threat-model.md` | Repository-grounded assets, trust boundaries, attacker capabilities, threats, controls, and MAJ-4A review paths |
| **todo/** | |
| `todo/README.md` | Scoped forward roadmap index linked from `../TODO.md` |
| `todo/*.md` | One decision-complete page per open TODO item |

## REQUIRED vs AESTHETIC

| Path | Status | Enforcing gate / source of truth |
| --- | --- | --- |
| `src/fedference/*.py` | REQUIRED | Coverage ≥90%; ISC probes in [`../ISA.md`](../ISA.md) |
| `src/analysis/workflow.py` | REQUIRED | Pipeline stage 4; `tests/analysis/test_workflow.py` |
| `src/analysis/artifacts.py` | REQUIRED | Canonical Stage-02 artifact declaration; `tests/analysis/test_artifacts.py` and `tests/test_scripts_smoke.py` |
| `src/analysis/report_schemas.py` | REQUIRED | Report/figure write-boundary schemas + figure dependency contracts; `tests/analysis/test_report_schemas.py` |
| `src/fedference_cli/` | REQUIRED | Installed CLI facade, command dispatch, output isolation, and receipt contracts; `tests/test_fedference_cli.py` |
| `src/figures/*.py` | REQUIRED | Figure generators (one module per figure); `tests/figures/` |
| `README.md`, `docs/**/*.md` Mermaid blocks | REQUIRED | `scripts/validate_mermaid.py`; renderer probe for release review |
| `src/manuscript_variables.py` | REQUIRED | `tests/test_manuscript_variables.py`; live tokens |
| `src/experiment_config.py` | REQUIRED | `tests/test_experiment_config.py` |
| `src/fedference/config/hierarchical_layers.yaml` | REFERENCE | Human-readable mirror of the defaults in `build_3level_world`; `tests/fedference/test_hierarchical_layers_yaml.py` gates acuity, goal bonus, and canonical prior drift |
| `tests/` (all `test_*.py`) | REQUIRED | Coverage gate |
| `tests/conftest.py` | REQUIRED | `MPLBACKEND=Agg` + `sys.path` |
| `scripts/02_run_analysis.py` | REQUIRED | Pipeline stage 4 entry |
| `scripts/z_generate_manuscript_variables.py` | REQUIRED | Token hydration before PDF |
| `scripts/01_run_invariants.py` | AESTHETIC | Local invariant runner |
| `scripts/00_preflight.py` | AESTHETIC | Pre-render warnings |
| `scripts/generate_api_docs.py` | AESTHETIC | `output/docs/api_reference.md` |
| `manuscript/config.yaml` | REQUIRED | Render + experiment config |
| `manuscript/*.md` | REQUIRED | PDF source (after substitution) |
| `manuscript/references.bib` | REQUIRED | Citations |
| `manuscript/preamble.md` | REQUIRED | LaTeX preamble |
| `manuscript/SYNTAX.md` | AESTHETIC (partly gated) | Authoritative label registry (humans/agents); `tests/test_docs_contract.py::test_manuscript_syntax_registry_references_live_files` gates referenced-file existence and one stale-string check |
| `docs/todo/**` | REQUIRED | `tests/test_docs_contract.py` link and scope checks |
| `docs/**` | AESTHETIC (partly gated) | Docs contract catches critical drift; still update docs when behavior changes |
| `../ISA.md` | REQUIRED (contract) | Acceptance criteria |
| `../README.md`, `../AGENTS.md` | AESTHETIC (load-bearing) | Onboarding |

"AESTHETIC" means no domain-specific gate fails when the file rots — still
update it when behaviour changes. The documentation contract nevertheless
discovers every `docs/**/*.md` page and checks its local links, stale-claim
language, future multi-machine wording, and Mermaid fences.

## Verification commands

See [`reference/verification-commands.md`](reference/verification-commands.md).

## Cross-references

- [`../AGENTS.md`](../AGENTS.md) — project technical reference (slim)
- [`../manuscript/AGENTS.md`](../manuscript/AGENTS.md) — manuscript editing
- [`../src/AGENTS.md`](../src/AGENTS.md) — source tree contract
- [`../tests/AGENTS.md`](../tests/AGENTS.md) — test tree contract
- [`../scripts/AGENTS.md`](../scripts/AGENTS.md) — script contract
