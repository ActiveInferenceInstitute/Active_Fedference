# Testing philosophy

Active Fedference enforces the template **no-mocks** policy and a **≥90%**
coverage gate on `src/`.

## Zero mocks (absolute)

Forbidden anywhere under `tests/`:

- `unittest.mock`, `MagicMock`, `@patch`, `create_autospec`, `pytest-mock`

Tests use real NumPy computations with fixed seeds, real temp files (`tmp_path`),
and subprocess smoke for auxiliary scripts.

## Coverage gate

Configured in [`pyproject.toml`](../../pyproject.toml). Authoritative command:

```bash
uv run --locked pytest tests/ \
  --cov=src \
  --cov-fail-under=90
```

The raw pytest command above is authoritative for the line-coverage threshold.
For a release-facing hydration, run
`uv run --locked --extra dev python scripts/validate_test_coverage.py`: it executes that
same full coverage gate and writes the source-bound
`output/data/test_coverage_receipt.json` required by final non-draft hydration.

Pytest profiles are explicit and complementary:

```bash
uv run --locked pytest tests/ -m "not slow" -q       # fast seeded feedback
uv run --locked pytest tests/ -m integration -q       # transport and cross-module paths
uv run --locked pytest tests/ -m publication -q       # reports, scripts, and release surfaces
```

The full coverage command remains authoritative. `slow` marks expensive real
experiments and publication subprocesses; `integration` and `publication` are
selection labels, not weaker test substitutes.

## Schema and contract test layer

Beyond unit and integration tests, a dedicated layer pins the *shapes and
claims* of the project's surfaces:

- **Report schemas** — `tests/analysis/test_report_schemas.py` exercises the
  write boundary: every `output/reports/*.json` payload and the figure
  registry are validated by `report_schemas.validate_report` inside
  `workflow._write_json` before anything touches disk. Accept/reject cases are
  derived mechanically from the schema table in
  `src/analysis/report_schemas.py`, so every declared schema is covered
  automatically; figure generators' consumed fields are pinned via
  `FIGURE_DEPENDENCY_CONTRACTS`.
- **Docs contract** — `tests/test_docs_contract.py` discovers every source-owned
  `AGENTS.md` and `README.md` and checks that doc links
  resolve, documented `scripts/*.py` commands name real files, stale claim
  language does not reappear, and the figure registry agrees with manuscript
  embeds and generator defaults.
- **Claim boundaries** — `tests/test_manuscript_claim_audit.py` rejects
  reintroduced unbounded claim phrases; `tests/test_token_provenance.py` and
  `tests/test_token_sourcing.py` guarantee manuscript numbers are derived
  tokens, never hardcoded or green-by-construction literals.
- **Release integrity** — `tests/test_release_manifest.py` covers the release
  bundle's per-file digests and its source/manuscript/documentation/producer
  provenance fingerprint, metadata, and changed-input diagnostics, including
  `--verify` rejecting a stale or tampered bundle.

## Proof of detection

A gate that has never been seen to fail proves nothing. For every new
contract-style gate, run a negative control: inject a known-bad input (a
malformed payload, a stale claim marker, a tampered file), confirm the gate
fails with a *named* error, then restore the input byte-exact and confirm
green. Record the probe in [`../../ISA.md`](../../ISA.md). Examples in the
suite: the write boundary rejects a deliberately malformed report with a
named-report/named-field `ReportSchemaError`; `tests/test_publication_metadata.py`
flags a tampered metadata surface; `tests/fedference/test_statistics.py` pins
the feasibility floor of the power analysis with a saturated-effect control.

## Test layout (mirrors `src/`)

### `tests/fedference/` — domain core

| File | Covers |
| --- | --- |
| `test_core_identities.py` | Central aggregation/Bayes recovery limits |
| `test_core_edges.py` | Numerical edge cases |
| `test_validation.py` | Shared finite-simplex validators (`as_pmf`, `as_pmf_matrix`, weights) |
| `test_gaussian_divergences.py` | KL, Rényi, TV dispatch |
| `test_belief_updating.py` | Variational inference + VFE |
| `test_dirichlet_learning.py` | Language acquisition |
| `test_expected_free_energy.py` | EFE decomposition identity |
| `test_bayesian_model_reduction.py` | BMR $\Delta F$ |
| `test_pomdp.py` | Sentinel world |
| `test_agents.py` | `SentinelEnsemble` |
| `test_contamination.py` | Saboteur models |
| `test_contamination_gallery.py` | Gallery sweep |
| `test_statistics.py` | Wilcoxon + BH-FDR |
| `test_statistics_multiseed.py` | Multi-seed statistical probes |
| `test_experiments.py` | Seeded source-mechanism analogues (including the V4 moving world) |
| `test_experiments_sensitivity.py` | Sensitivity sweep probes |
| `test_property_based.py` | Hypothesis property tests: invariants over divergences, losses, aggregation, statistics |
| `test_tempered_aggregation.py` | V1 tempered aggregation |
| `test_tempered_aggregation_objective.py` | V1: tempered-family fixed point, effective-weight bound, λ-grid, c→0 recovery |
| `test_aggregation_objective.py` | Aggregation objective probes |
| `test_aggregation_config.py` | Public config validation, rich dispatch, legacy parity, and conflict rejection |
| `test_aggregation_comparators.py` | Linear and CLR-median experimental controls |
| `test_server_theory.py` | Orientation plus exact scoped raw-log-pool and normalized-weight no-go witnesses; no universal-theorem overclaim |
| `test_calibration.py` | Proper-score selection, deterministic ties, and calibration/evaluation separation |
| `test_evidence.py` | Strict experiment/dataset/receipt schemas and artifact-tamper detection |
| `test_research_registry.py` | Source revisions, profiles, confirmatory design fields, and pinned dataset declarations |
| `test_external_data.py` | Real ZIP parsing with archive, shape, class, and schema failure controls |
| `test_federation_end_to_end.py` | Queue/process transport, strict belief payloads, configured dispatch, and bit-identical end-to-end federation |
| `test_socket_transport.py` | MAJ-4 foundation: enforced single-host loopback TCP, framed messages, strict replay event order/schema, process-local and restart-durable local round guards, and bit-identical consensus |
| `test_transport_envelope.py` | Version, round, worker, config hash, payload digest, exact float64 probability schema, and tamper/replay metadata |
| `test_moving_world.py` | V4: disjoint-FOV consensus only via communication; EFE policy |
| `test_moving_world_v4.py` | V4: EFE navigation test |
| `test_bnn_baseline.py` | FedGVI logistic regression |
| `test_bnn_baseline_torch.py` | FedGVI PyTorch point-mass MLP complement |
| `test_robustness_onset.py` | Per-mechanism onset rate probes |
| `test_parameter_recovery.py` | Parameter recovery probes |
| `test_hierarchical_pomdp.py` | V2: `LayerSpec` validation; `build_hierarchical_world` / `build_3level_world` shapes; `hierarchical_infer` / `nlevel_infer` PMF validity; `run_hierarchical_world` / `run_3level_world` smoke + determinism |
| `test_hierarchical_layers_yaml.py` | Documented 3-level YAML defaults match code defaults |
| `test_nlevel_depth.py` | MAJ-5: arbitrary-depth stacks (`depth >= 2`); depth-4 hierarchical reduction |
| `test_benchmark.py` | MAJ-6: external tabular-benchmark harness — real CSV + real fits |
| `test_continuous_recovery.py` | MAJ-3: continuous 1-D Gaussian generalized-Bayes recovery limits |
| `test_bnn_variational_torch.py` | MAJ-2: mean-field `VariationalMLP` recovery limits (real torch) |
| `test_bnn_fedgvi.py` | Site factors, cavity, factor replacement, schedules, and checkpoint/resume |
| `test_torch_bnn.py` | CPU/MPS determinism, device receipts, and explicit fallback behavior |
| `test_protocol_parity.py` | FedGVI source-constrained and Friston paper-constrained exactness labeling |
| `test_hybrid_tracking.py` | Seeded tracking metrics, zero-robustness recovery, and outlier behavior |
| `test_heuristic_characterization.py` | MAJ-1: empirical characterization of `robust_aggregate` — real runs |
| `test_hierarchical_bmr.py` | MAJ-7: hierarchical Bayesian model reduction — real inference |

### `tests/figures/` — figure generators

Each figure generator is covered by a dedicated or shared test module under
`tests/figures/` (PNG written to temp/output); most `src/figures/*.py`
generators map 1:1 to a `test_<generator>.py`, but `test_introductory_figures.py`
covers both `system_overview.py` and `graphical_abstract.py` jointly.

### Root-level integration

| File | Covers |
| --- | --- |
| `test_manuscript_variables.py` | Token hydration contract |
| `test_experiment_config.py` | `config.yaml` loading |
| `test_invariants.py` | Project invariants |
| `test_xref_integrity.py` | All `{#eq:}`, `{#fig:}`, `{#tbl:}`, `{#sec:}`, `{#prop:}` labels defined in manuscript |
| `test_caption_completeness.py` | Every figure carries an axes/uncertainty caption |
| `test_token_provenance.py` | Every `{{TOKEN}}` in prose is emitted by `generate_variables()` |
| `test_token_sourcing.py` | Tokens are derived from executed runs, never asserted literals |
| `test_documentation.py` | API doc helpers |
| `test_docs_contract.py` | Docs consistency gate, SYNTAX registry, stale language |
| `test_manuscript_claim_audit.py` | Claim-boundary regressions; unbounded claim phrases stay out |
| `test_docstrings.py` | Docstring coverage and style |
| `test_publication_metadata.py` | Publication metadata sourcing + tamper detection |
| `test_release_manifest.py` | Release bundle digests, provenance fingerprint, `--verify` round trip |
| `test_runtime_surface.py` | No retired runtime markers or test-double APIs |
| `test_fedference_cli.py` | Installed CLI list/run/benchmark/verify/replay contracts and output isolation |
| `test_web_publication_contract.py` | Web publication metadata contract |
| `test_token_tables.py` | Token table structure |
| `test_scripts_smoke.py` | Subprocess smoke for auxiliary scripts; writes go to a temp scaffold via `ACTIVE_FEDFERENCE_PROJECT_ROOT`, never the committed `output/` tree |
| `conftest.py` | `MPLBACKEND=Agg`, `sys.path` to `src/` |

### `tests/analysis/`

| File | Covers |
| --- | --- |
| `test_workflow.py` | End-to-end analysis pipeline paths and JSON/figure outputs |
| `test_report_schemas.py` | Write-boundary validation: schema-table-derived accept/reject cases, figure dependency contracts |

## Determinism

Every stochastic step uses `np.random.default_rng(seed)` with seeds from
`manuscript/config.yaml`. Never rely on global `np.random`.

## Adding tests for new code

1. Place unit tests beside the domain under `tests/fedference/test_<module>.py`.
2. Pin **exact numeric expectations** for recovery limits and identities.
3. Add error-path tests for invalid inputs (raises `ValueError`, etc.).
4. Extend [`../../ISA.md`](../../ISA.md) with a named probe.

## See also

- Agent checklist: [`agent_instructions.md`](agent_instructions.md)
- Verification commands: [`../reference/verification-commands.md`](../reference/verification-commands.md)
- Test directory contract: [`../../tests/AGENTS.md`](../../tests/AGENTS.md)
